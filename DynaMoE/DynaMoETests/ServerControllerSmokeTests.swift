//
//  ServerControllerSmokeTests.swift
//  DynaMoE
//
//  In-process smoke coverage for the local OpenAI-compatible server: starts
//  ServerController, probes /health, /v1/models and the completions error
//  path (no model loaded in CI — the 503 envelope is asserted instead).
//  RestClient dependencies: none beyond URLSession hitting 127.0.0.1.
//

import XCTest
@testable import DynaMoE

final class ServerControllerSmokeTests: XCTestCase {

    private var savedServerEnabled: Any?
    private var savedServerPort: Any?
    private var savedServerAPIKey: Any?
    private var savedCaptureBodies: Any?
    private var logStore: ServerRequestLogStore!

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults.standard
        savedServerEnabled = defaults.object(forKey: ServerController.enabledKey)
        savedServerPort = defaults.object(forKey: ServerController.portKey)
        savedServerAPIKey = defaults.object(forKey: ServerController.apiKeyKey)
        savedCaptureBodies = defaults.object(forKey: ServerRequestLogRecorder.captureBodiesKey)
        defaults.set(true, forKey: ServerController.enabledKey)
        defaults.set(8787, forKey: ServerController.portKey)
        // The suite sends unauthenticated requests, so a key saved in the app's
        // settings (the developer's own machine) must not turn every expectation
        // into a 401. `testAPIKeyRejectsUnauthorizedRequests` sets its own key.
        defaults.removeObject(forKey: ServerController.apiKeyKey)
        defaults.set(false, forKey: ServerRequestLogRecorder.captureBodiesKey)
        // Route the server's request logging into an in-memory store so this
        // suite never writes to the developer's real request log, and every
        // case reads rows from a clean slate.
        logStore = ServerRequestLogStore.inMemory()
        ServerRequestLogRecorder.shared.useOverrideStore(logStore)
    }

    override func tearDown() {
        let defaults = UserDefaults.standard
        if let enabled = savedServerEnabled {
            defaults.set(enabled, forKey: ServerController.enabledKey)
        } else {
            defaults.removeObject(forKey: ServerController.enabledKey)
        }
        if let port = savedServerPort {
            defaults.set(port, forKey: ServerController.portKey)
        } else {
            defaults.removeObject(forKey: ServerController.portKey)
        }
        if let apiKey = savedServerAPIKey {
            defaults.set(apiKey, forKey: ServerController.apiKeyKey)
        } else {
            defaults.removeObject(forKey: ServerController.apiKeyKey)
        }
        if let savedCaptureBodies {
            defaults.set(savedCaptureBodies, forKey: ServerRequestLogRecorder.captureBodiesKey)
        } else {
            defaults.set(false, forKey: ServerRequestLogRecorder.captureBodiesKey)
        }
        ServerRequestLogRecorder.shared.useOverrideStore(nil)
        logStore = nil
        super.tearDown()
    }

    private func ensureServerRunning() async {
        let alreadyRunning = await MainActor.run { ServerController.shared.isRunning }
        if !alreadyRunning {
            await MainActor.run { ServerController.shared.start() }
            try? await Task.sleep(nanoseconds: 800_000_000)
        }
    }

    func testHealthEndpointServesStatusJSON() async throws {
        await ensureServerRunning()
        let running = await MainActor.run { ServerController.shared.isRunning }
        XCTAssertTrue(running, "ServerController.start() must leave the server running")

        let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:8787/health")!)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(body["status"] as? String, "ok")
        XCTAssertNotNil(body["model_loaded"])
    }

    func testModelsEndpointServesOpenAIListShape() async throws {
        await ensureServerRunning()
        let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:8787/v1/models")!)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(body["object"] as? String, "list")
        XCTAssertTrue(body["data"] is [Any])
        if let info = await MainActor.run(body: { CompletionEngine.shared.loadedModelInfo }) {
            let entries = try XCTUnwrap(body["data"] as? [[String: Any]])
            XCTAssertEqual(entries.first?["id"] as? String, info.displayName)
        }
    }

    func testCompletionsWithoutModelReturns503ErrorEnvelope() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let statsBefore = ServerController.stats.snapshot()
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 503)
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try XCTUnwrap(body["error"] as? [String: Any])
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
        XCTAssertEqual(
            ServerController.stats.snapshot().totalRequests,
            statsBefore.totalRequests + 1,
            "the 503 completion request must be recorded in the server stats"
        )
    }

    func testStreamingCompletionsWithoutModelEmitsSSEErrorThenDone() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the SSE error path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","stream":true,"messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        XCTAssertEqual(http.value(forHTTPHeaderField: "Content-Type"), "text/event-stream")

        var sawErrorFrame = false
        var sawDoneMarker = false
        for try await line in bytes.lines {
            if line.hasPrefix("data: ") {
                let payload = String(line.dropFirst("data: ".count))
                if payload == "[DONE]" {
                    sawDoneMarker = true
                    break
                }
                if let frameData = payload.data(using: .utf8),
                   let frame = try? JSONSerialization.jsonObject(with: frameData) as? [String: Any],
                   frame["error"] != nil {
                    sawErrorFrame = true
                }
            }
        }
        XCTAssertTrue(sawErrorFrame || sawDoneMarker, "stream must terminate with an error frame and [DONE]")
    }

    func testMessagesWithoutModelReturns503ErrorEnvelope() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let statsBefore = ServerController.stats.snapshot()
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 503)
        XCTAssertFalse(data.isEmpty)
        XCTAssertEqual(
            ServerController.stats.snapshot().totalRequests,
            statsBefore.totalRequests + 1,
            "the 503 messages request must be recorded in the server stats"
        )
    }

    func testMessagesWithoutMaxTokensReturns400() async throws {
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 400, "max_tokens is required by the Anthropic spec")
        // The error body must be Anthropic-shaped: {"type":"error","error":{...}}.
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(body["type"] as? String, "error")
        let error = try XCTUnwrap(body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
    }

    func testCompletionsErrorResponsesAreOpenAIShaped() async throws {
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"messages":"not an array"}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 400, "an unparseable body must be rejected")
        // The error body must be OpenAI-shaped: {"error":{"message","type"}}.
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try XCTUnwrap(body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
    }

    func testStreamingMessagesWithoutModelEmitsSSEErrorEvent() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the SSE error path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200)
        XCTAssertEqual(http.value(forHTTPHeaderField: "Content-Type"), "text/event-stream")

        var sawMessageStart = false
        var sawErrorEvent = false
        var sawJSONPayloadOnError = false
        var currentEventName: String?
        for try await line in bytes.lines {
            if line.hasPrefix("event: ") {
                currentEventName = String(line.dropFirst("event: ".count))
                if currentEventName == "message_start" { sawMessageStart = true }
            } else if line.hasPrefix("data: ") {
                let payload = String(line.dropFirst("data: ".count))
                if currentEventName == "error" {
                    sawErrorEvent = true
                    let parsed = try? JSONSerialization.jsonObject(with: Data(payload.utf8)) as? [String: Any]
                    sawJSONPayloadOnError = parsed?["error"] != nil
                }
                currentEventName = nil
            }
            if sawErrorEvent { break }
        }
        XCTAssertTrue(sawMessageStart, "the stream must open with message_start")
        XCTAssertTrue(sawErrorEvent, "the stream must surface the launch failure as an error event")
        XCTAssertTrue(sawJSONPayloadOnError, "an SSE parser must receive the JSON payload on the data: line")
    }

    func testAPIKeyRejectsUnauthorizedRequests() async throws {
        let defaults = UserDefaults.standard
        let savedKey = defaults.object(forKey: ServerController.apiKeyKey)
        defer {
            if let savedKey {
                defaults.set(savedKey, forKey: ServerController.apiKeyKey)
            } else {
                defaults.removeObject(forKey: ServerController.apiKeyKey)
            }
        }
        defaults.set("secret-test-key", forKey: ServerController.apiKeyKey)
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/models")!)
        request.httpMethod = "GET"
        let (_, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 401, "requests without the bearer token must be rejected")

        var apiKeyRequest = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/models")!)
        apiKeyRequest.httpMethod = "GET"
        apiKeyRequest.setValue("secret-test-key", forHTTPHeaderField: "x-api-key")
        let (_, apiKeyResponse) = try await URLSession.shared.data(for: apiKeyRequest)
        let apiKeyHttp = try XCTUnwrap(apiKeyResponse as? HTTPURLResponse)
        XCTAssertEqual(apiKeyHttp.statusCode, 200, "the x-api-key header must be accepted as an alternative presentation shape")
    }

    // MARK: - Request logging

    /// The logging pipeline appends on a background queue, so assertions poll
    /// briefly instead of sleeping a fixed interval.
    private func waitForLoggedRow(matching predicate: @escaping (ServerRequestLog) -> Bool) async throws -> ServerRequestLog? {
        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline {
            let rows = await logStore.fetch(limit: 50)
            if let hit = rows.first(where: predicate) {
                return hit
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        let rows = await logStore.fetch(limit: 50)
        XCTFail("no matching row appeared; store holds \(rows.map { "\(String(describing: $0.method)) \(String(describing: $0.path)) \(String(describing: $0.httpStatus))" })")
        return nil
    }

    func testHealthProbesLeaveNoRequestLogRow() async throws {
        await ensureServerRunning()
        let (data, _) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:8787/health")!)
        _ = data
        // Give the (asynchronous) store append a moment, then assert absence.
        try await Task.sleep(nanoseconds: 300_000_000)
        let rows = await logStore.fetch(limit: 50)
        XCTAssertFalse(rows.contains { $0.path == "/health" }, "health polling must never write a persisted row")
    }

    func testRequestLogModelsEndpointRowHasWireFields() async throws {
        await ensureServerRunning()
        let before = await logStore.count()
        let (data, response) = try await URLSession.shared.data(from: URL(string: "http://127.0.0.1:8787/v1/models")!)
        XCTAssertEqual(try XCTUnwrap(response as? HTTPURLResponse).statusCode, 200)
        _ = data

        let logged = try await waitForLoggedRow { $0.path == "/v1/models" }
        let row = try XCTUnwrap(logged)
        XCTAssertEqual(row.httpStatus, 200)
        XCTAssertEqual(row.method, "GET")
        XCTAssertEqual(row.proto, .other)
        XCTAssertFalse(row.isError)
        XCTAssertNil(row.requestBody, "GET rows carry no captured body")
        let after = await logStore.count()
        XCTAssertEqual(after, before + 1)
    }

    func testRequestLogRecordsDecodeFailureAsErrorRow() async throws {
        await ensureServerRunning()
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"messages":"not an array"}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual(try XCTUnwrap(response as? HTTPURLResponse).statusCode, 400)
        _ = data

        let logged = try await waitForLoggedRow { $0.path == "/v1/chat/completions" && $0.isError }
        let row = try XCTUnwrap(logged)
        XCTAssertEqual(row.httpStatus, 400)
        XCTAssertEqual(row.proto, .openai)
        XCTAssertFalse(row.wasStreaming)
        XCTAssertTrue(
            row.errorMessage?.hasPrefix("Invalid request body") == true,
            "the decode failure must surface its cause in the row (got: \(row.errorMessage ?? "nil"))"
        )
    }

    func testRequestLogRecordsUnauthorizedRowWithoutCredentialMaterial() async throws {
        let defaults = UserDefaults.standard
        defaults.set("secret-test-key", forKey: ServerController.apiKeyKey)
        defer { defaults.removeObject(forKey: ServerController.apiKeyKey) }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/models")!)
        request.httpMethod = "GET"
        request.setValue("Bearer secret-test-key-wrong", forHTTPHeaderField: "Authorization")
        let (_, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual(try XCTUnwrap(response as? HTTPURLResponse).statusCode, 401)

        let logged = try await waitForLoggedRow { $0.path == "/v1/models" && $0.isError }
        let row = try XCTUnwrap(logged)
        XCTAssertEqual(row.httpStatus, 401)
        XCTAssertEqual(row.errorMessage, "Invalid or missing API key")
        XCTAssertFalse(
            ServerRequestLogRecorder.composeLine(row).contains("secret"),
            "the presented credential must never reach the unified-log line"
        )
    }

    func testRequestLogRecordsStreamingErrorFrameWithTermination() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the SSE error path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","stream":true,"messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        XCTAssertEqual(try XCTUnwrap(response as? HTTPURLResponse).statusCode, 200)
        for try await _ in bytes.lines {}

        let logged = try await waitForLoggedRow { $0.path == "/v1/chat/completions" && $0.wasStreaming }
        let row = try XCTUnwrap(logged)
        XCTAssertEqual(row.httpStatus, 200, "streaming HTTP status stays 200; the error travels as frames")
        XCTAssertEqual(row.streamTermination, .errorFrame, "the launch failure must read as an error frame")
        XCTAssertTrue(row.isError)
        XCTAssertNotNil(row.errorMessage)
        XCTAssertEqual(row.proto, .openai)
        XCTAssertNil(row.model, "no model is loaded, so the plan never resolved and the row carries no model rather than the client's alias label")
    }

    func testRequestLogNonStreaming503RowHasWithheldBodyMarker() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        XCTAssertEqual(try XCTUnwrap(response as? HTTPURLResponse).statusCode, 503)
        _ = data

        let logged = try await waitForLoggedRow { $0.path == "/v1/chat/completions" && $0.httpStatus == 503 }
        let row = try XCTUnwrap(logged)
        XCTAssertTrue(row.isError)
        XCTAssertNil(row.model, "the plan never resolved, so the row has no model")
        XCTAssertEqual(row.requestBody, ServerRequestLog.contentWithheldMarker, "body capture is off; the row must say withheld, not silence")
        XCTAssertNil(row.streamTermination)
    }

    // MARK: - Lifecycle transitions

    func testToggleBackOnDuringShutdownLeavesTheServerRunning() async throws {
        // Turning the server off and straight back on used to leave it stopped:
        // the "on" request observed `isRunning == true` (the shutdown had not
        // drained yet) and did nothing, and the shutdown then published
        // "stopped". The controller reconciles to the persisted request after
        // draining, so the newest request decides.
        await ensureServerRunning()
        let defaults = UserDefaults.standard
        defaults.set(false, forKey: ServerController.enabledKey)

        let disabling = Task { await ServerController.shared.applyRequestedState() }
        // Let the main-actor controller reach the shutdown's first suspension,
        // then flip the request back on while that shutdown is still draining.
        await MainActor.run {}
        defaults.set(true, forKey: ServerController.enabledKey)
        await disabling.value

        let running = await MainActor.run { ServerController.shared.isRunning }
        XCTAssertTrue(running, "the last requested state was on, so the server must be running once the shutdown drains")
    }

    func testRestartAfterADisableLeavesTheServerStopped() async throws {
        // `restartServer` stops and then reconciles instead of starting
        // unconditionally, so a disable that landed while the shutdown was
        // draining is not undone by the restart.
        await ensureServerRunning()
        UserDefaults.standard.set(false, forKey: ServerController.enabledKey)

        await ServerController.shared.stop()
        await ServerController.shared.applyRequestedState()

        let running = await MainActor.run { ServerController.shared.isRunning }
        XCTAssertFalse(running, "a restart must not force the server back on after the user disabled it")
    }
}
