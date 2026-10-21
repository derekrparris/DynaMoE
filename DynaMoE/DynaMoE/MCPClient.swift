//
//  MCPClient.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  JSON-RPC 2.0 over stdio and Streamable HTTP transports.
//
//  Two process models meet here. DynaMoE's existing tool runners are strictly
//  one-shot: `ControlledProcessRunner` and `AgentHarness.runProcess` launch a
//  command, wait for it to exit, and read its pipes only at termination. An MCP
//  server is the opposite: a long-lived process whose stdout is a message
//  stream and whose stdin must stay open. So this file owns the app's first
//  persistent child process, and the rules that come with it:
//
//    - stdout is drained continuously by a readability handler and split into
//      newline-delimited JSON frames;
//    - stderr is drained continuously into a bounded ring buffer, because an
//      unread stderr pipe blocks the child at 64 KB and the server would look
//      like a hang rather than a failure;
//    - a single frame larger than `maxMessageBytes` is refused instead of
//      growing the buffer without bound.
//

import Foundation

// MARK: - Errors

public nonisolated enum MCPClientError: LocalizedError {
    case launchFailed(command: String, reason: String)
    case connectionClosed(String)
    case timeout(method: String, seconds: Double)
    case remote(code: Int, message: String)
    case unsupportedTransport(String)
    case unresolvedTokens([String])
    case missingEndpoint
    case httpStatus(code: Int, body: String)
    case serverCapabilityMissing(String)

    public var errorDescription: String? {
        switch self {
        case .launchFailed(let command, let reason):
            return "Could not launch '\(command)': \(reason)"
        case .connectionClosed(let detail):
            return detail.isEmpty ? "The MCP server closed the connection." : "The MCP server closed the connection. \(detail)"
        case .timeout(let method, let seconds):
            return "MCP request '\(method)' timed out after \(Int(seconds))s."
        case .remote(let code, let message):
            return "MCP server error \(code): \(message)"
        case .unsupportedTransport(let detail):
            return "Unsupported MCP transport: \(detail)"
        case .unresolvedTokens(let tokens):
            return "Unresolved credential token(s) \(tokens.joined(separator: ", ")). Store the secret in the Keychain from Settings, or set the referenced environment variable."
        case .missingEndpoint:
            return "The MCP server config declares neither 'command' nor 'url'."
        case .httpStatus(let code, let body):
            let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty
                ? "MCP endpoint returned HTTP \(code)."
                : "MCP endpoint returned HTTP \(code): \(trimmed)"
        case .serverCapabilityMissing(let capability):
            return "The MCP server does not advertise the '\(capability)' capability."
        }
    }
}

// MARK: - JSON Value

/// A Sendable, Equatable stand-in for decoded JSON. `[String: Any]` cannot
/// cross an actor boundary, and MCP tool schemas are arbitrary JSON, so the
/// wire layer decodes once into this and the bridge converts to `AnyCodable`
/// at the `ToolDefinition` boundary.
public nonisolated indirect enum MCPJSONValue: Sendable, Equatable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([MCPJSONValue])
    case object([String: MCPJSONValue])

    public init(any value: Any) {
        if value is NSNull {
            self = .null
            return
        }
        if let string = value as? String {
            self = .string(string)
            return
        }
        if let number = value as? NSNumber {
            // JSONSerialization reports booleans as NSNumber, and `as? Bool`
            // would also accept 1/0, so booleans are identified by CF type.
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                self = .bool(number.boolValue)
                return
            }
            let doubleValue = number.doubleValue
            if doubleValue == doubleValue.rounded(), abs(doubleValue) < 9_007_199_254_740_992 {
                self = .int(number.intValue)
            } else {
                self = .double(doubleValue)
            }
            return
        }
        if let array = value as? [Any] {
            self = .array(array.map { MCPJSONValue(any: $0) })
            return
        }
        if let object = value as? [String: Any] {
            self = .object(object.mapValues { MCPJSONValue(any: $0) })
            return
        }
        self = .string(String(describing: value))
    }

    public init?(data: Data) {
        guard let object = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) else {
            return nil
        }
        self = MCPJSONValue(any: object)
    }

    public var anyValue: Any {
        switch self {
        case .null: return NSNull()
        case .bool(let value): return value
        case .int(let value): return value
        case .double(let value): return value
        case .string(let value): return value
        case .array(let values): return values.map { $0.anyValue }
        case .object(let values): return values.mapValues { $0.anyValue }
        }
    }

    public subscript(key: String) -> MCPJSONValue? {
        guard case .object(let values) = self else { return nil }
        return values[key]
    }

    public var stringValue: String? {
        if case .string(let value) = self { return value }
        return nil
    }

    public var boolValue: Bool? {
        if case .bool(let value) = self { return value }
        return nil
    }

    public var intValue: Int? {
        if case .int(let value) = self { return value }
        return nil
    }

    public var arrayValue: [MCPJSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }

    public var objectValue: [String: MCPJSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    public var isNull: Bool {
        if case .null = self { return true }
        return false
    }

    public var prettyJSONString: String {
        guard let data = try? JSONSerialization.data(
            withJSONObject: anyValue,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        ), let string = String(data: data, encoding: .utf8) else {
            return String(describing: anyValue)
        }
        return string
    }
}

// MARK: - Wire Format

public nonisolated enum MCPWire {
    /// Newest protocol revision this client asks for. Servers answer with the
    /// revision they support and the client accepts whatever comes back.
    public static let protocolVersion = "2025-06-18"
    public static let supportedProtocolVersions = ["2025-06-18", "2025-03-26", "2024-11-05"]

    public static var clientVersion: String {
        (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String) ?? "1.0"
    }

    public static func request(id: Int, method: String, params: [String: Any]?) throws -> Data {
        var payload: [String: Any] = ["jsonrpc": "2.0", "id": id, "method": method]
        if let params { payload["params"] = params }
        return try encode(payload)
    }

    public static func notification(method: String, params: [String: Any]?) throws -> Data {
        var payload: [String: Any] = ["jsonrpc": "2.0", "method": method]
        if let params { payload["params"] = params }
        return try encode(payload)
    }

    /// `id` is the JSON value exactly as it arrived so a reply can echo the
    /// original type and value: JSON-RPC ids may be strings, and a server that
    /// sent `"health-check"` or `"007"` cannot correlate a numeric echo.
    public static func successResponse(id: MCPJSONValue, result: [String: Any]) throws -> Data {
        try encode(["jsonrpc": "2.0", "id": id.anyValue, "result": result])
    }

    public static func errorResponse(id: MCPJSONValue, code: Int, message: String) throws -> Data {
        try encode(["jsonrpc": "2.0", "id": id.anyValue, "error": ["code": code, "message": message]])
    }

    public static func cancelledParams(requestId: Int, reason: String) -> [String: Any] {
        ["requestId": requestId, "reason": reason]
    }

    private static func encode(_ payload: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: payload, options: [.withoutEscapingSlashes])
    }
}

/// One decoded frame arriving from a server: a response, a server-initiated
/// request, or a notification.
public nonisolated struct MCPIncomingMessage: Sendable {
    /// Numeric view of the request id, used to match responses to this
    /// client's own outbound requests (whose ids are always integers). A
    /// string id that parses as an integer still matches, but `"007"` and
    /// `"health-check"` are not interchangeable with a server's request id.
    public let id: Int?
    /// The request id exactly as it arrived (integer, string, or any other
    /// JSON value). Replies to server-initiated requests echo this verbatim.
    public let rawID: MCPJSONValue?
    public let method: String?
    public let result: [String: MCPJSONValue]?
    public let errorCode: Int?
    public let errorMessage: String?

    public init?(data: Data) {
        guard let value = MCPJSONValue(data: data), case .object(let object) = value else { return nil }

        let rawID = object["id"]
        self.rawID = rawID
        if case .int(let numeric)? = rawID {
            self.id = numeric
        } else if case .string(let raw)? = rawID, let numeric = Int(raw) {
            self.id = numeric
        } else {
            self.id = nil
        }

        self.method = object["method"]?.stringValue

        if case .object(let result)? = object["result"] {
            self.result = result
        } else {
            self.result = nil
        }

        if case .object(let error)? = object["error"] {
            if case .int(let code)? = error["code"] {
                self.errorCode = code
            } else {
                self.errorCode = nil
            }
            self.errorMessage = error["message"]?.stringValue
        } else {
            self.errorCode = nil
            self.errorMessage = nil
        }
    }
}

public nonisolated struct MCPResponseEnvelope: Sendable {
    public let result: [String: MCPJSONValue]?
    public let errorCode: Int?
    public let errorMessage: String?
}

// MARK: - Message Framing

/// Splits a byte stream into newline-delimited frames. MCP stdio frames are
/// one JSON object per line, and SSE frames arrive as text lines, so both
/// transports share this.
nonisolated struct MCPLineBuffer {
    private(set) var buffer = Data()
    let limit: Int
    private(set) var overflowed = false
    /// How many bytes of `buffer` are known to contain no newline. Appends
    /// resume the search here, so feeding a long frame one byte at a time (the
    /// SSE reader's `AsyncBytes` loop) stays linear instead of rescanning the
    /// whole partial line on every byte.
    private var scanOffset = 0

    init(limit: Int) {
        self.limit = limit
    }

    /// Returns complete lines with the newline removed. A trailing `\r` is
    /// stripped so `\r\n` framing works. A frame longer than `limit` is
    /// refused whether it arrives unterminated or terminated by the current
    /// chunk: the former would be caught by the tail check, the latter only by
    /// checking each complete line before it is copied or decoded.
    mutating func append(_ chunk: Data) -> [String] {
        guard !overflowed else { return [] }
        buffer.append(chunk)

        var lines: [String] = []
        while true {
            let resume = buffer.index(buffer.startIndex, offsetBy: scanOffset)
            guard let newline = buffer[resume...].firstIndex(of: 0x0A) else {
                scanOffset = buffer.count
                break
            }
            // The newline is consumed but a trailing `\r` is not, so the byte
            // distance includes it, matching the unterminated-tail check below.
            if buffer.distance(from: buffer.startIndex, to: newline) > limit {
                overflowed = true
                buffer.removeAll()
                scanOffset = 0
                return lines
            }
            var lineData = Data(buffer[buffer.startIndex..<newline])
            if lineData.last == 0x0D { lineData.removeLast() }
            if let line = String(data: lineData, encoding: .utf8) {
                lines.append(line)
            }
            buffer.removeSubrange(buffer.startIndex...newline)
            scanOffset = 0
        }

        if buffer.count > limit {
            overflowed = true
            buffer.removeAll()
            scanOffset = 0
        }
        return lines
    }

    /// Flushes any unterminated trailing line (a server that omits the final
    /// newline still deserves its last frame).
    mutating func flush() -> String? {
        guard !overflowed, !buffer.isEmpty else { return nil }
        var lineData = buffer
        buffer.removeAll()
        scanOffset = 0
        if lineData.last == 0x0D { lineData.removeLast() }
        return String(data: lineData, encoding: .utf8)
    }
}

/// Text/event-stream frame assembler: `data:` lines accumulate until a blank
/// line dispatches the event.
nonisolated struct MCPSSEFrameParser {
    /// Largest event this parser will assemble. The line buffer bounds each
    /// line, but a server can send any number of short `data:` lines before the
    /// blank line that dispatches them, so the event itself needs a bound.
    let limit: Int
    private(set) var overflowed = false
    private var dataLines: [String] = []
    private var dataBytes = 0
    private var eventName: String?

    init(limit: Int = MCPHTTPTransport.maxMessageBytes) {
        self.limit = limit
    }

    mutating func consume(line: String) -> (event: String?, data: String)? {
        guard !overflowed else { return nil }
        if line.isEmpty {
            return dispatch()
        }
        if line.hasPrefix(":") {
            return nil
        }
        guard let separator = line.firstIndex(of: ":") else {
            // A bare field name with no colon is legal and carries an empty value.
            return nil
        }
        let field = String(line[line.startIndex..<separator])
        var value = String(line[line.index(after: separator)...])
        if value.hasPrefix(" ") { value.removeFirst() }

        switch field {
        case "data":
            // +1 per line for the newline `dispatch` inserts between them.
            dataBytes += value.utf8.count + 1
            if dataBytes > limit {
                overflowed = true
                dataLines.removeAll()
                eventName = nil
                return nil
            }
            dataLines.append(value)
        case "event":
            eventName = value
        default:
            break
        }
        return nil
    }

    mutating func finish() -> (event: String?, data: String)? {
        guard !overflowed else { return nil }
        return dispatch()
    }

    private mutating func dispatch() -> (event: String?, data: String)? {
        defer {
            dataLines.removeAll()
            dataBytes = 0
            eventName = nil
        }
        guard !overflowed, !dataLines.isEmpty else { return nil }
        return (eventName, dataLines.joined(separator: "\n"))
    }
}

// MARK: - Transport

public nonisolated protocol MCPTransport: AnyObject, Sendable {
    func start() async throws
    /// Writes one JSON-RPC frame. Must not await the matching response: the
    /// client's read loop owns delivery.
    func send(_ payload: Data) async throws
    /// Whole JSON frames in arrival order. Finishing the stream means the
    /// connection is gone; `closureReason` explains why.
    var messages: AsyncStream<Data> { get }
    var stderrTail: String { get }
    var closureReason: String? { get }
    func close()
    /// Records the revision `initialize` agreed on so later requests can
    /// advertise it. Transports without a version header ignore it.
    func setNegotiatedProtocolVersion(_ version: String)
}

public extension MCPTransport {
    func setNegotiatedProtocolVersion(_ version: String) {}
}

// MARK: - stdio Transport

public final class MCPStdioTransport: MCPTransport, @unchecked Sendable {
    /// Refuses a single frame larger than this rather than growing without
    /// bound. A 10 MB tool result belongs in a file, not a JSON-RPC frame.
    public static let maxMessageBytes = 8 * 1024 * 1024
    /// Enough stderr to explain a crash, bounded so a chatty server cannot
    /// exhaust memory.
    public static let maxStderrBytes = 16 * 1024

    private let launchCommand: String
    private let arguments: [String]
    private let environment: [String: String]
    private let workingDirectory: URL?
    private let label: String

    private let lock = NSLock()
    private let writeLock = NSLock()
    private var process: Process?
    private var stdinHandle: FileHandle?
    private var stdoutLines: MCPLineBuffer
    private var stderrBuffer = Data()
    private var didFinish = false
    private var _closureReason: String?
    private var _stderrTail = ""

    private let stream: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation

    public var messages: AsyncStream<Data> { stream }

    public var stderrTail: String {
        lock.lock()
        defer { lock.unlock() }
        return _stderrTail
    }

    public var closureReason: String? {
        lock.lock()
        defer { lock.unlock() }
        return _closureReason
    }

    public init(
        command: String,
        arguments: [String],
        environment: [String: String],
        workingDirectory: URL?,
        label: String
    ) {
        self.launchCommand = command
        self.arguments = arguments
        self.environment = environment
        self.workingDirectory = workingDirectory
        self.label = label
        self.stdoutLines = MCPLineBuffer(limit: Self.maxMessageBytes)
        let pair = AsyncStream<Data>.makeStream(bufferingPolicy: .bufferingNewest(256))
        self.stream = pair.stream
        self.continuation = pair.continuation
    }

    public func start() async throws {
        // `/usr/bin/env` execs the target in place, so the child pid *is* the
        // server and PATH from `toolEnvironment()` resolves `npx`, `uvx`, and
        // `cargo` the same way `shell_run` does.
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [launchCommand] + arguments

        var env = ControlledProcessRunner.toolEnvironment()
        for (key, value) in environment {
            env[key] = value
        }
        process.environment = env
        if let workingDirectory {
            process.currentDirectoryURL = workingDirectory
        }

        let stdinPipe = Pipe()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        install(process: process, stdinHandle: stdinPipe.fileHandleForWriting)

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in            let chunk = handle.availableData
            guard let self else { return }
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                self.finish(reason: "stdout closed")
            } else {
                self.ingestStdout(chunk)
            }
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard let self else { return }
            if chunk.isEmpty {
                handle.readabilityHandler = nil
            } else {
                self.ingestStderr(chunk)
            }
        }

        process.terminationHandler = { [weak self] finished in
            self?.finish(reason: "server exited with status \(finished.terminationStatus)")
        }

        // A close() that landed between install and run must not orphan a child.
        if hasFinished() {
            finish(reason: "closed before launch")
            throw MCPClientError.connectionClosed("The transport was closed before the server could start.")
        }

        do {
            try process.run()
        } catch {
            finish(reason: "launch failed: \(error.localizedDescription)")
            throw MCPClientError.launchFailed(command: launchCommand, reason: error.localizedDescription)
        }

        if hasFinished() {
            let pid = process.processIdentifier
            if pid > 0 {
                kill(-pid, SIGTERM)
                kill(pid, SIGTERM)
                DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
                    if process.isRunning {
                        kill(-pid, SIGKILL)
                        kill(pid, SIGKILL)
                    }
                }
            }
            throw MCPClientError.connectionClosed("The transport was closed while the server was starting.")
        }

        DiagLog.info("mcp", "server=\(label) launched pid=\(process.processIdentifier)")
    }

    public func send(_ payload: Data) async throws {
        let snapshot = writeState()
        guard snapshot.isRunning, let handle = snapshot.handle else {
            throw MCPClientError.connectionClosed(
                Self.describeClosure(reason: snapshot.reason, tail: snapshot.tail)
            )
        }

        var framed = payload
        framed.append(0x0A)
        do {
            try writeFramed(framed, to: handle)
        } catch {
            throw MCPClientError.connectionClosed(
                "Write to server stdin failed: \(error.localizedDescription)\n\(currentStderrTail())"
            )
        }
    }

    /// A separate write lock keeps two concurrent messages from interleaving on
    /// the same stdin handle; the state lock is never held across a write.
    private func writeFramed(_ framed: Data, to handle: FileHandle) throws {
        writeLock.lock()
        defer { writeLock.unlock() }
        try handle.write(contentsOf: framed)
    }

    /// Locking is confined to this synchronous helper: `NSLock` is unavailable
    /// from async contexts and becomes an error under the Swift 6 language mode.
    private func install(process: Process, stdinHandle: FileHandle) {
        lock.lock()
        defer { lock.unlock() }
        self.process = process
        self.stdinHandle = stdinHandle
    }

    private func hasFinished() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return didFinish
    }

    private func writeState() -> (handle: FileHandle?, isRunning: Bool, tail: String, reason: String?) {
        lock.lock()
        defer { lock.unlock() }
        return (stdinHandle, process?.isRunning ?? false, _stderrTail, _closureReason)
    }

    private func currentStderrTail() -> String {
        lock.lock()
        defer { lock.unlock() }
        return _stderrTail
    }

    public func close() {
        lock.lock()
        let process = self.process
        let handle = self.stdinHandle
        self.stdinHandle = nil
        self.process = nil
        lock.unlock()

        // EOF first: a well-behaved server shuts down on its own.
        try? handle?.close()

        guard let process, process.isRunning else {
            finish(reason: "closed")
            return
        }
        let pid = process.processIdentifier
        if pid > 0 {
            kill(-pid, SIGTERM)
            kill(pid, SIGTERM)
        }
        process.terminate()

        DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) { [weak self] in
            if process.isRunning {
                if pid > 0 {
                    kill(-pid, SIGKILL)
                    kill(pid, SIGKILL)
                }
            }
            self?.finish(reason: "closed")
        }
        finish(reason: "closed")
    }

    // MARK: Ingestion

    private func ingestStdout(_ chunk: Data) {
        lock.lock()
        let lines = stdoutLines.append(chunk)
        let overflowed = stdoutLines.overflowed
        lock.unlock()

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard let data = line.data(using: .utf8) else { continue }
            continuation.yield(data)
        }

        if overflowed {
            finish(reason: "a single MCP frame exceeded \(Self.maxMessageBytes) bytes")
        }
    }

    private func ingestStderr(_ chunk: Data) {
        lock.lock()
        stderrBuffer.append(chunk)
        if stderrBuffer.count > Self.maxStderrBytes {
            let excess = stderrBuffer.count - Self.maxStderrBytes
            stderrBuffer.removeFirst(excess)
        }
        _stderrTail = String(decoding: stderrBuffer, as: UTF8.self)
        lock.unlock()
    }

    private func finish(reason: String) {
        lock.lock()
        if didFinish {
            lock.unlock()
            return
        }
        didFinish = true
        _closureReason = reason
        let tail = _stderrTail
        lock.unlock()

        DiagLog.info("mcp", "server=\(label) closed: \(reason)\(tail.isEmpty ? "" : " stderr=\(tail.suffix(240))")")
        continuation.finish()
    }

    static func describeClosure(reason: String?, tail: String) -> String {
        var parts: [String] = []
        if let reason, !reason.isEmpty { parts.append(reason) }
        let trimmedTail = tail.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedTail.isEmpty { parts.append("stderr: \(trimmedTail.suffix(600))") }
        return parts.joined(separator: "\n")
    }
}

// MARK: - Streamable HTTP Transport

public final class MCPHTTPTransport: MCPTransport, @unchecked Sendable {
    public static let maxMessageBytes = 8 * 1024 * 1024

    private let endpoint: URL
    private let baseHeaders: [String: String]
    private let session: URLSession
    private let label: String

    private let lock = NSLock()
    private let writeLock = NSLock()
    private var sessionId: String?
    /// Revision advertised on every request. Starts at the newest revision this
    /// client asks for and switches to whatever `initialize` negotiated: a
    /// server that picked `2025-03-26` may reject a `2025-06-18` header.
    private var protocolVersion = MCPWire.protocolVersion
    private var inFlight: [UUID: Task<Void, Never>] = [:]
    private var didClose = false
    private var _closureReason: String?

    private let stream: AsyncStream<Data>
    private let continuation: AsyncStream<Data>.Continuation

    public var messages: AsyncStream<Data> { stream }
    public var stderrTail: String { "" }

    public var closureReason: String? {
        lock.lock()
        defer { lock.unlock() }
        return _closureReason
    }

    public init(
        endpoint: URL,
        headers: [String: String],
        requestTimeout: TimeInterval,
        label: String
    ) {
        self.endpoint = endpoint
        self.baseHeaders = headers
        self.label = label

        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = requestTimeout
        configuration.timeoutIntervalForResource = max(requestTimeout * 2, requestTimeout + 30)
        configuration.waitsForConnectivity = false
        configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
        configuration.httpShouldSetCookies = false
        self.session = URLSession(configuration: configuration)

        let pair = AsyncStream<Data>.makeStream(bufferingPolicy: .bufferingNewest(256))
        self.stream = pair.stream
        self.continuation = pair.continuation
    }

    public func start() async throws {
        // No handshake happens at the transport layer: `initialize` is the first
        // POST. Server-initiated notifications would need a long-lived GET SSE
        // stream; this client discovers tools on connect and on `tools/list`
        // refreshes instead, so it does not open one.
        DiagLog.info("mcp", "server=\(label) using Streamable HTTP at \(endpoint.host ?? "?")")
    }

    public func send(_ payload: Data) async throws {
        let snapshot = connectionState()
        guard !snapshot.isClosed else {
            throw MCPClientError.connectionClosed("The MCP session is closed.")
        }

        let request = makeRequest(payload: payload, sessionId: snapshot.sessionId)

        let bytes: URLSession.AsyncBytes
        let response: URLResponse
        do {
            (bytes, response) = try await session.bytes(for: request)
        } catch {
            let reason = "HTTP request failed: \(error.localizedDescription)"
            // A cancellation belongs to an abandoned request and a timeout to a
            // slow server; neither means the session is gone. An endpoint that
            // cannot be reached or negotiated with is a terminal closure, and
            // the stream ending is how the client and its manager find out.
            if Self.isTerminalTransportError(error) {
                finish(reason: reason)
            }
            throw MCPClientError.connectionClosed(reason)
        }

        guard let http = response as? HTTPURLResponse else {
            let reason = "The MCP endpoint returned a non-HTTP response."
            finish(reason: reason)
            throw MCPClientError.connectionClosed(reason)
        }

        if let newSession = http.value(forHTTPHeaderField: "Mcp-Session-Id") {
            storeSessionId(newSession)
        }

        if http.statusCode == 202 {
            return
        }
        if http.statusCode == 404, snapshot.sessionId != nil {
            storeSessionId(nil)
            let reason = "The MCP session expired (HTTP 404). Reconnect the server."
            finish(reason: reason)
            throw MCPClientError.connectionClosed(reason)
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = await Self.drain(bytes, limit: 4096)
            throw MCPClientError.httpStatus(code: http.statusCode, body: Self.summarizeErrorBody(body))
        }

        let contentType = http.value(forHTTPHeaderField: "Content-Type") ?? ""
        let taskID = UUID()
        let task = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            if contentType.contains("text/event-stream") {
                await self.consumeSSE(bytes, request: payload)
            } else {
                await self.consumeJSON(bytes, request: payload)
            }
            self.untrack(taskID)
        }
        track(task, id: taskID)
    }

    /// Synchronous lock helpers: `NSLock` is unavailable from async contexts and
    /// is an error under the Swift 6 language mode.
    private func connectionState() -> (isClosed: Bool, sessionId: String?) {
        lock.lock()
        defer { lock.unlock() }
        return (didClose, sessionId)
    }

    /// Builds the POST body this transport sends. Internal so a test can assert
    /// the advertised revision without a live endpoint.
    func makeRequest(payload: Data, sessionId: String?) -> URLRequest {
        var request = URLRequest(url: endpoint)
        request.httpMethod = "POST"
        request.httpBody = payload
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("application/json, text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue(advertisedProtocolVersion(), forHTTPHeaderField: "MCP-Protocol-Version")
        for (key, value) in baseHeaders {
            request.setValue(value, forHTTPHeaderField: key)
        }
        if let sessionId {
            request.setValue(sessionId, forHTTPHeaderField: "Mcp-Session-Id")
        }
        return request
    }

    public func setNegotiatedProtocolVersion(_ version: String) {
        let trimmed = version.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        lock.lock()
        protocolVersion = trimmed
        lock.unlock()
    }

    private func advertisedProtocolVersion() -> String {
        lock.lock()
        defer { lock.unlock() }
        return protocolVersion
    }

    private func storeSessionId(_ value: String?) {
        lock.lock()
        defer { lock.unlock() }
        sessionId = value
    }

    private func track(_ task: Task<Void, Never>, id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        inFlight[id] = task
    }

    private func untrack(_ id: UUID) {
        lock.lock()
        defer { lock.unlock() }
        inFlight.removeValue(forKey: id)
    }

    /// Ends this transport's message stream because the session cannot be used
    /// any more. The stream finishing is what makes `MCPClient` mark itself
    /// closed, fail its pending requests, and report a terminal failure to the
    /// manager, so a dead HTTP endpoint cannot keep a Ready row and its tools
    /// alive. Idempotent, and a later `close()` keeps the useful reason.
    private func finish(reason: String) {
        lock.lock()
        if didClose {
            lock.unlock()
            return
        }
        didClose = true
        _closureReason = reason
        lock.unlock()
        DiagLog.error("mcp", "server=\(label) connection ended: \(reason)")
        continuation.finish()
    }

    /// Whether a URLSession failure means the endpoint is gone rather than
    /// slow or abandoned. A timeout or a cancellation leaves the session
    /// usable, so those must not tear the connection down.
    static func isTerminalTransportError(_ error: Error) -> Bool {
        guard let urlError = error as? URLError else { return false }
        switch urlError.code {
        case .cannotConnectToHost, .cannotFindHost, .dnsLookupFailed,
             .networkConnectionLost, .badServerResponse, .secureConnectionFailed,
             .serverCertificateUntrusted:
            return true
        default:
            return false
        }
    }

    public func close() {
        lock.lock()
        if !didClose { _closureReason = "closed" }
        didClose = true
        let sessionId = self.sessionId
        let version = protocolVersion
        let tasks = inFlight.values
        inFlight.removeAll()
        lock.unlock()

        tasks.forEach { $0.cancel() }
        continuation.finish()

        if let sessionId {
            var request = URLRequest(url: endpoint)
            request.httpMethod = "DELETE"
            request.setValue(sessionId, forHTTPHeaderField: "Mcp-Session-Id")
            request.setValue(version, forHTTPHeaderField: "MCP-Protocol-Version")
            for (key, value) in baseHeaders {
                request.setValue(value, forHTTPHeaderField: key)
            }
            // Uses the shared session: this transport's own session is about
            // to be invalidated, and a cancelled DELETE would leave the server
            // holding the session until it times out.
            Task.detached {
                _ = try? await URLSession.shared.data(for: request)
            }
        }
        session.invalidateAndCancel()
        DiagLog.info("mcp", "server=\(label) HTTP session closed")
    }

    // MARK: Body consumption

    private func consumeJSON(_ bytes: URLSession.AsyncBytes, request: Data) async {
        var data = Data()
        do {
            for try await byte in bytes {
                data.append(byte)
                if data.count > Self.maxMessageBytes {
                    DiagLog.error("mcp", "server=\(label) HTTP response exceeded \(Self.maxMessageBytes) bytes")
                    deliverResponseError(
                        for: request,
                        message: "The MCP response exceeded \(Self.maxMessageBytes) bytes and was discarded."
                    )
                    return
                }
            }
        } catch {
            DiagLog.warning("mcp", "server=\(label) HTTP body error: \(error.localizedDescription)")
            if !connectionState().isClosed {
                deliverResponseError(
                    for: request,
                    message: "The MCP response could not be read: \(error.localizedDescription)"
                )
            }
            return
        }
        guard !data.isEmpty else { return }

        // A response may be a single object or a batch array.
        if let value = MCPJSONValue(data: data), case .array(let frames) = value {
            for frame in frames {
                if let encoded = try? JSONSerialization.data(withJSONObject: frame.anyValue) {
                    continuation.yield(encoded)
                }
            }
            return
        }
        continuation.yield(data)
    }

    private func consumeSSE(_ bytes: URLSession.AsyncBytes, request: Data) async {
        var lineBuffer = MCPLineBuffer(limit: Self.maxMessageBytes)
        var frames = MCPSSEFrameParser()

        func dispatch(_ event: (event: String?, data: String)?) {
            guard let event else { return }
            // `event: endpoint` frames carry an upload URL in old revisions;
            // only JSON payloads can be JSON-RPC.
            guard event.data.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{")
                || event.data.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("[") else { return }
            guard let data = event.data.data(using: .utf8) else { return }
            continuation.yield(data)
        }

        do {
            for try await byte in bytes {
                var single = Data()
                single.append(byte)
                for line in lineBuffer.append(single) {
                    dispatch(frames.consume(line: line))
                }
                if lineBuffer.overflowed || frames.overflowed { break }
            }
            if !lineBuffer.overflowed && !frames.overflowed {
                if let trailing = lineBuffer.flush() {
                    dispatch(frames.consume(line: trailing))
                }
                dispatch(frames.finish())
            }
        } catch {
            DiagLog.warning("mcp", "server=\(label) SSE stream error: \(error.localizedDescription)")
            if !connectionState().isClosed {
                deliverResponseError(
                    for: request,
                    message: "The MCP response could not be read: \(error.localizedDescription)"
                )
            }
            return
        }
        if lineBuffer.overflowed || frames.overflowed {
            // Failing the waiter now is the difference between a clear "too
            // large" error and a call that sits out the whole tool deadline and
            // then claims it timed out.
            DiagLog.error("mcp", "server=\(label) SSE event exceeded \(Self.maxMessageBytes) bytes")
            deliverResponseError(
                for: request,
                message: "The MCP response exceeded \(Self.maxMessageBytes) bytes and was discarded."
            )
        }
    }

    /// Hands the client a JSON-RPC error for the request whose response could
    /// not be delivered. The transport cannot throw into a pending request, so
    /// this frame is the only way to settle it without waiting for the deadline.
    private func deliverResponseError(for request: Data, message: String) {
        guard let frame = Self.responseErrorFrame(request: request, message: message) else { return }
        continuation.yield(frame)
    }

    /// Builds that error frame for the request's own id. A notification has no
    /// id and therefore no waiter to fail. Internal so a test can pin that the
    /// id's type and value come back verbatim.
    static func responseErrorFrame(request: Data, message: String) -> Data? {
        guard let value = MCPJSONValue(data: request),
              case .object(let object) = value,
              let id = object["id"],
              !id.isNull else { return nil }
        let payload: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id.anyValue,
            "error": [
                "code": -32603,
                "message": message
            ]
        ]
        return try? JSONSerialization.data(withJSONObject: payload, options: [.withoutEscapingSlashes])
    }

    /// Gateways answer a malformed request with a full HTML page. Showing that
    /// page in a settings row buries the one useful word ("Bad Request") under
    /// kilobytes of markup, so tags are stripped, whitespace collapsed, and the
    /// result bounded.
    static func summarizeErrorBody(_ body: String, limit: Int = 400) -> String {
        var text = body
        if text.contains("<") {
            text = text.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        }
        text = text
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if text.isEmpty { return "" }
        if text.count > limit {
            text = String(text.prefix(limit)) + "..."
        }
        return text
    }

    private static func drain(_ bytes: URLSession.AsyncBytes, limit: Int) async -> String {
        var data = Data()
        do {
            for try await byte in bytes {
                data.append(byte)
                if data.count >= limit { break }
            }
        } catch {
            return ""
        }
        return String(decoding: data, as: UTF8.self)
    }
}

// MARK: - Resolved Server

/// A server config with every `${...}` token expanded, ready to become a
/// transport. Built at connect time, never persisted.
public nonisolated struct MCPResolvedServer: Sendable {
    public let key: String
    public let kind: MCPServerConfig.TransportKind
    public let command: String?
    public let arguments: [String]
    public let environment: [String: String]
    public let workingDirectory: URL?
    public let endpoint: URL?
    public let headers: [String: String]
    public let startupTimeoutSec: Double
    public let toolTimeoutSec: Double
    public let rawConfig: MCPServerConfig
}

public nonisolated enum MCPConfigResolver {
    /// Expands tokens and builds transport inputs. Returns validation errors
    /// as values so the caller can surface them without throwing.
    public static func resolve(
        key: String,
        config: MCPServerConfig,
        workspacePath: String?
    ) -> (server: MCPResolvedServer?, issues: [MCPConfigIssue]) {
        var issues = MCPConfigValidator.validate(server: config, key: key)
        guard !MCPConfigValidator.hasErrors(issues) else { return (nil, issues) }

        var unresolved: [String] = []

        let expandedArgs = MCPTokenExpander.expand(config.args ?? [], workspacePath: workspacePath)
        unresolved.append(contentsOf: expandedArgs.unresolved)

        let expandedEnv = MCPTokenExpander.expand(config.env ?? [:], workspacePath: workspacePath)
        unresolved.append(contentsOf: expandedEnv.unresolved)

        let expandedHeaders = MCPTokenExpander.expand(config.headers ?? [:], workspacePath: workspacePath)
        unresolved.append(contentsOf: expandedHeaders.unresolved)

        let expandedCWD: String? = {
            guard let raw = config.cwd, !raw.isEmpty else { return nil }
            let resolution = MCPTokenExpander.expand(raw, workspacePath: workspacePath)
            unresolved.append(contentsOf: resolution.unresolved)
            return resolution.value
        }()

        let expandedURL: URL? = {
            guard let raw = config.url, !raw.isEmpty else { return nil }
            let resolution = MCPTokenExpander.expand(raw, workspacePath: workspacePath)
            unresolved.append(contentsOf: resolution.unresolved)
            return URL(string: resolution.value)
        }()

        var expandedCommand: String?
        if let raw = config.command, !raw.isEmpty {
            let resolution = MCPTokenExpander.expand(raw, workspacePath: workspacePath)
            unresolved.append(contentsOf: resolution.unresolved)
            expandedCommand = resolution.value
        }

        if !unresolved.isEmpty {
            issues.append(MCPConfigIssue(
                severity: .error,
                serverKey: key,
                message: "Unresolved credential token(s): \(Set(unresolved).sorted().joined(separator: ", "))"
            ))
            return (nil, issues)
        }

        let kind = config.transportKind
        guard let kind else {
            issues.append(MCPConfigIssue(severity: .error, serverKey: key, message: "Missing transport endpoint."))
            return (nil, issues)
        }

        if kind == .http, expandedURL == nil {
            issues.append(MCPConfigIssue(severity: .error, serverKey: key, message: "'url' did not expand into a valid URL."))
            return (nil, issues)
        }

        let resolved = MCPResolvedServer(
            key: key,
            kind: kind,
            command: expandedCommand,
            arguments: expandedArgs.values,
            environment: expandedEnv.values,
            workingDirectory: expandedCWD.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) },
            endpoint: expandedURL,
            headers: expandedHeaders.values,
            startupTimeoutSec: config.resolvedStartupTimeoutSec,
            toolTimeoutSec: config.resolvedToolTimeoutSec,
            rawConfig: config
        )
        return (resolved, issues)
    }
}

// MARK: - Discovered Tools

public nonisolated struct MCPToolDescriptor: Sendable, Equatable {
    public let name: String
    public let description: String
    public let inputSchema: MCPJSONValue
    public let readOnlyHint: Bool
    public let destructiveHint: Bool

    public init?(json: MCPJSONValue) {
        guard case .object(let object) = json,
              let name = object["name"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines),
              !name.isEmpty else { return nil }

        self.name = name

        var summary = object["description"]?.stringValue ?? ""
        if summary.isEmpty, let title = object["title"]?.stringValue {
            summary = title
        }
        self.description = summary

        self.inputSchema = object["inputSchema"] ?? .object([
            "type": .string("object"),
            "properties": .object([:])
        ])

        let annotations = object["annotations"]
        self.readOnlyHint = annotations?["readOnlyHint"]?.boolValue ?? false
        self.destructiveHint = annotations?["destructiveHint"]?.boolValue ?? false
    }
}

public nonisolated struct MCPCallResult: Sendable, Equatable {
    public let text: String
    public let isError: Bool
    public let contentBlockCount: Int
    public let imageBlockCount: Int
    public let structuredContent: MCPJSONValue?

    public init(json: [String: MCPJSONValue]) {
        var parts: [String] = []
        var blocks = 0
        var images = 0

        if let content = json["content"]?.arrayValue {
            for block in content {
                blocks += 1
                switch block["type"]?.stringValue {
                case "text":
                    if let text = block["text"]?.stringValue {
                        parts.append(text)
                    }
                case "image":
                    images += 1
                    parts.append(Self.imagePlaceholder(block))
                case "audio":
                    parts.append("[audio \(block["mimeType"]?.stringValue ?? "unknown")]")
                case "resource":
                    if let resource = block["resource"] {
                        parts.append(Self.resourceDescription(resource))
                    } else {
                        parts.append("[resource]")
                    }
                case "resource_link":
                    parts.append("[resource_link \(block["uri"]?.stringValue ?? "")]")
                case .some(let other):
                    parts.append("[\(other) content block]")
                case .none:
                    parts.append("[content block]")
                }
            }
        }

        let structured = json["structuredContent"]
        if parts.isEmpty, let structured, !structured.isNull {
            // Servers that return only structuredContent would otherwise look
            // like an empty result to the model.
            parts.append(structured.prettyJSONString)
        }

        self.text = parts.joined(separator: "\n\n")
        self.isError = json["isError"]?.boolValue ?? false
        self.contentBlockCount = blocks
        self.imageBlockCount = images
        self.structuredContent = structured
    }

    static func imagePlaceholder(_ block: MCPJSONValue) -> String {
        let mime = block["mimeType"]?.stringValue ?? "image/unknown"
        // `data` is base64, so decoded size is 3/4 of the encoded length.
        let base64Count = block["data"]?.stringValue?.count ?? 0
        let decodedBytes = base64Count * 3 / 4
        if decodedBytes > 0 {
            return "[image \(mime) \(ByteCountFormatter.string(fromByteCount: Int64(decodedBytes), countStyle: .file))]"
        }
        return "[image \(mime)]"
    }

    static func resourceDescription(_ resource: MCPJSONValue) -> String {
        let uri = resource["uri"]?.stringValue ?? ""
        if let text = resource["text"]?.stringValue {
            return "[resource \(uri)]\n\(text)"
        }
        if let mime = resource["mimeType"]?.stringValue {
            return "[resource \(uri) \(mime)]"
        }
        return "[resource \(uri)]"
    }
}

// MARK: - Client

public actor MCPClient {
    public nonisolated let serverKey: String
    public nonisolated let config: MCPResolvedServer

    private let transport: any MCPTransport
    private let startupTimeout: Double
    private let toolTimeout: Double
    private let onToolsChanged: (@Sendable () -> Void)?
    /// Fired once when the transport's stream ends on its own. Deliberate
    /// teardowns mark the client closed first and never reach it.
    private let onClosed: (@Sendable (MCPClient, String) -> Void)?
    private let incoming: AsyncStream<Data>

    private var readTask: Task<Void, Never>?
    /// One entry per request whose response has not arrived yet. `dispatched`
    /// is the seam between the two abandonment behaviors: a request the
    /// server already received gets a best-effort `notifications/cancelled`,
    /// while a request whose write never completed has its send task
    /// cancelled so it cannot go out after abandonment.
    private var pending: [Int: PendingRequest] = [:]
    private var nextRequestID = 1
    private var isClosed = false
    private var didStart = false

    private struct PendingRequest {
        let method: String
        let continuation: CheckedContinuation<MCPResponseEnvelope, any Error>
        let sendTask: Task<Void, Never>?
        /// The deadline timer. Cancelled as soon as the request settles, so a
        /// session of many calls does not leave one sleeper per call behind.
        var deadlineTask: Task<Void, Never>?
        var dispatched = false
    }

    public private(set) var negotiatedProtocolVersion: String?
    public private(set) var serverName: String?
    public private(set) var serverVersion: String?
    public private(set) var serverInstructions: String?

    public init(
        config: MCPResolvedServer,
        onToolsChanged: (@Sendable () -> Void)? = nil,
        onClosed: (@Sendable (MCPClient, String) -> Void)? = nil
    ) throws {
        switch config.kind {
        case .stdio:
            guard let command = config.command, !command.isEmpty else {
                throw MCPClientError.missingEndpoint
            }
            self.init(
                config: config,
                transport: MCPStdioTransport(
                    command: command,
                    arguments: config.arguments,
                    environment: config.environment,
                    workingDirectory: config.workingDirectory,
                    label: config.key
                ),
                onToolsChanged: onToolsChanged,
                onClosed: onClosed
            )
        case .http:
            guard let endpoint = config.endpoint else {
                throw MCPClientError.missingEndpoint
            }
            self.init(
                config: config,
                transport: MCPHTTPTransport(
                    endpoint: endpoint,
                    headers: config.headers,
                    requestTimeout: min(max(config.toolTimeoutSec, 30), MCPServerConfig.maximumTimeoutSec),
                    label: config.key
                ),
                onToolsChanged: onToolsChanged,
                onClosed: onClosed
            )
        }
    }

    /// Injects a transport instead of building one. Lets tests drive the full
    /// JSON-RPC exchange against an in-memory server without spawning a process.
    public init(
        config: MCPResolvedServer,
        transport: any MCPTransport,
        onToolsChanged: (@Sendable () -> Void)? = nil,
        onClosed: (@Sendable (MCPClient, String) -> Void)? = nil
    ) {
        self.serverKey = config.key
        self.config = config
        self.startupTimeout = config.startupTimeoutSec
        self.toolTimeout = config.toolTimeoutSec
        self.onToolsChanged = onToolsChanged
        self.onClosed = onClosed
        self.transport = transport
        self.incoming = transport.messages
    }

    public var stderrTail: String { transport.stderrTail }
    public var closureReason: String? { transport.closureReason }
    public var isConnectionClosed: Bool { isClosed }

    // MARK: Lifecycle

    public func start() async throws {
        guard !didStart else { return }
        guard !isClosed else {
            throw MCPClientError.connectionClosed(MCPStdioTransport.describeClosure(reason: transport.closureReason, tail: transport.stderrTail))
        }

        let stream = incoming
        readTask = Task.detached(priority: .userInitiated) { [weak self] in
            for await frame in stream {
                guard let self else { return }
                await self.handle(frame)
            }
            await self?.handleStreamEnded()
        }

        try await transport.start()

        let params: [String: Any] = [
            "protocolVersion": MCPWire.protocolVersion,
            "capabilities": [String: Any](),
            "clientInfo": [
                "name": "DynaMoE",
                "version": MCPWire.clientVersion
            ]
        ]
        let result = try await request("initialize", params: params, timeout: startupTimeout)

        let protocolVersion = result["protocolVersion"]?.stringValue ?? ""
        if !protocolVersion.isEmpty {
            negotiatedProtocolVersion = protocolVersion
            // Every later request must advertise the revision the server chose.
            transport.setNegotiatedProtocolVersion(protocolVersion)
            if !MCPWire.supportedProtocolVersions.contains(protocolVersion) {
                DiagLog.warning("mcp", "server=\(serverKey) negotiated unknown protocol \(protocolVersion); continuing")
            }
        }
        if case .object(let info)? = result["serverInfo"] {
            serverName = info["name"]?.stringValue
            serverVersion = info["version"]?.stringValue
        }
        serverInstructions = result["instructions"]?.stringValue

        try await sendNotification("notifications/initialized", params: nil)
        didStart = true

        DiagLog.info(
            "mcp",
            "server=\(serverKey) initialized protocol=\(negotiatedProtocolVersion ?? "?") name=\(serverName ?? "?") version=\(serverVersion ?? "?")"
        )
    }

    public func stop() async {
        guard !isClosed else {
            transport.close()
            return
        }
        isClosed = true
        transport.close()
        readTask?.cancel()
        readTask = nil
        failAll(MCPClientError.connectionClosed("The MCP session was stopped."))
    }

    /// Closes the transport without awaiting actor isolation. Used on app
    /// termination, where a detached task would not be guaranteed to run before
    /// the process exits and the server would be orphaned. `transport` is an
    /// immutable `Sendable` reference, so this is safe from any thread.
    public nonisolated func forceCloseTransport() {
        transport.close()
    }

    // MARK: Requests

    public func listTools() async throws -> [MCPToolDescriptor] {
        var descriptors: [MCPToolDescriptor] = []
        var cursor: String?
        var page = 0

        repeat {
            var params: [String: Any]?
            if let cursor, !cursor.isEmpty {
                params = ["cursor": cursor]
            }
            let result = try await request("tools/list", params: params, timeout: toolTimeout)

            if let tools = result["tools"]?.arrayValue {
                for tool in tools {
                    if let descriptor = MCPToolDescriptor(json: tool) {
                        descriptors.append(descriptor)
                    }
                }
            }
            cursor = result["nextCursor"]?.stringValue
            page += 1
            if descriptors.count >= MCPConfigValidator.maxToolsPerServer { break }
        } while (cursor?.isEmpty == false) && page < 32

        return descriptors
    }

    public func callTool(name: String, arguments: [String: Any]) async throws -> MCPCallResult {
        let params: [String: Any] = [
            "name": name,
            "arguments": arguments
        ]
        let result = try await request("tools/call", params: params, timeout: toolTimeout)
        return MCPCallResult(json: result)
    }

    /** Sends `notifications/cancelled` for a request that was abandoned. */
    public func notifyCancelled(requestID: Int, reason: String = "cancelled by client") {
        Task.detached(priority: .utility) { [weak self] in
            guard let self else { return }
            let payload = try? MCPWire.notification(
                method: "notifications/cancelled",
                params: MCPWire.cancelledParams(requestId: requestID, reason: reason)
            )
            if let payload {
                try? await self.transport.send(payload)
            }
        }
    }

    private func sendNotification(_ method: String, params: [String: Any]?) async throws {
        let payload = try MCPWire.notification(method: method, params: params)
        try await transport.send(payload)
    }

    private func request(_ method: String, params: [String: Any]?, timeout: Double) async throws -> [String: MCPJSONValue] {
        guard !isClosed else {
            throw MCPClientError.connectionClosed(MCPStdioTransport.describeClosure(reason: transport.closureReason, tail: transport.stderrTail))
        }

        let id = nextRequestID
        nextRequestID += 1
        let payload = try MCPWire.request(id: id, method: method, params: params)

        let envelope = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<MCPResponseEnvelope, any Error>) in
                let sendTask = Task.detached(priority: .userInitiated) { [weak self] in
                    // An abandoned queued send must not go out: cancellation is
                    // observed before the write, and the transports' own awaits
                    // observe it mid-write where they can.
                    guard let self, !Task.isCancelled else { return }
                    do {
                        try await self.transport.send(payload)
                        await self.markDispatched(id: id)
                    } catch {
                        await self.failPending(id: id, error: error)
                    }
                }

                self.pending[id] = PendingRequest(
                    method: method,
                    continuation: continuation,
                    sendTask: sendTask
                )

                // Late deadlines are harmless: `abandon` is a no-op once the
                // request has completed or already been abandoned.
                let deadlineTask = Task.detached(priority: .utility) { [weak self] in
                    // A configured timeout is arbitrary JSON, so clamp it to
                    // the timer's range before the `UInt64` conversion traps.
                    let seconds = MCPServerConfig.timerSeconds(forTimeout: timeout)
                    let nanoseconds = UInt64(seconds * 1_000_000_000)
                    try? await Task.sleep(nanoseconds: nanoseconds)
                    await self?.abandon(
                        id: id,
                        error: MCPClientError.timeout(method: method, seconds: timeout),
                        reason: "client timed out waiting for the response"
                    )
                }
                // Settling the request cancels this timer, so completed calls
                // leave no sleeper behind for the rest of the timeout.
                self.pending[id]?.deadlineTask = deadlineTask
            }
        } onCancel: {
            Task.detached(priority: .utility) { [weak self] in
                await self?.abandon(id: id, error: CancellationError(), reason: "cancelled by client")
            }
        }

        if let code = envelope.errorCode {
            throw MCPClientError.remote(code: code, message: envelope.errorMessage ?? "no message")
        }
        return envelope.result ?? [:]
    }

    /// Records that a request's write completed, so abandonment switches from
    /// "stop the send" to "tell the server we gave up".
    private func markDispatched(id: Int) {
        pending[id]?.dispatched = true
    }

    /// Fails a request this client is giving up on: task cancellation or
    /// deadline. Abandonment is the only failure path with side effects beyond
    /// the waiter. A request the server already received gets a best-effort
    /// `notifications/cancelled` so state-changing work can stop remotely —
    /// except `initialize`, which carries no cancellable server work — and a
    /// request whose write never completed has its send task cancelled so it
    /// cannot be sent after abandonment. Completed requests have no entry
    /// left, so late abandons are ignored.
    private func abandon(id: Int, error: any Error, reason: String) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        entry.deadlineTask?.cancel()
        if entry.dispatched {
            if entry.method != "initialize", !isClosed {
                notifyCancelled(requestID: id, reason: reason)
            }
        } else {
            entry.sendTask?.cancel()
        }
        entry.continuation.resume(throwing: error)
    }

    private func failPending(id: Int, error: any Error) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        entry.deadlineTask?.cancel()
        entry.continuation.resume(throwing: error)
    }

    private func failAll(_ error: any Error) {
        let entries = pending
        pending.removeAll()
        for (_, entry) in entries {
            entry.sendTask?.cancel()
            entry.deadlineTask?.cancel()
            entry.continuation.resume(throwing: error)
        }
    }

    // MARK: Inbound

    private func handle(_ data: Data) async {
        guard let message = MCPIncomingMessage(data: data) else {
            DiagLog.warning("mcp", "server=\(serverKey) dropped a non-JSON-RPC frame")
            return
        }

        if let method = message.method {
            // Server-initiated request. Only `ping` is meaningful to a
            // client with no roots/sampling support; answer it, and tell
            // the server plainly that everything else is unsupported.
            // JSON-RPC request ids may be strings, so the reply echoes the
            // raw id verbatim; an absent or null id cannot be correlated
            // back, so it gets no reply (like a notification).
            if let rawID = message.rawID, !rawID.isNull {
                let reply: Data?
                if method == "ping" {
                    reply = try? MCPWire.successResponse(id: rawID, result: [:])
                } else {
                    reply = try? MCPWire.errorResponse(id: rawID, code: -32601, message: "Method '\(method)' is not supported by this client.")
                }
                if let reply {
                    try? await transport.send(reply)
                }
            } else if method == "notifications/tools/list_changed" {
                onToolsChanged?()
            }
            return
        }

        guard let id = message.id,
              let entry = pending.removeValue(forKey: id) else { return }
        entry.deadlineTask?.cancel()
        entry.continuation.resume(returning: MCPResponseEnvelope(
            result: message.result,
            errorCode: message.errorCode,
            errorMessage: message.errorMessage
        ))
    }

    private func handleStreamEnded() {
        guard !isClosed else { return }
        isClosed = true
        let detail = MCPStdioTransport.describeClosure(reason: transport.closureReason, tail: transport.stderrTail)
        DiagLog.info("mcp", "server=\(serverKey) connection ended: \(detail.isEmpty ? "no detail" : detail)")
        failAll(MCPClientError.connectionClosed(detail))
        // A server that exits on its own is a terminal failure: the manager
        // owns the connection entry and the registered tools, so it has to
        // learn about the closure or they outlive the process. An explicit
        // `stop()` marks the client closed first and never gets here.
        onClosed?(self, detail)
    }
}
