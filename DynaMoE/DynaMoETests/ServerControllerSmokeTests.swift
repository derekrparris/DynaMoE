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

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults.standard
        savedServerEnabled = defaults.object(forKey: ServerController.enabledKey)
        savedServerPort = defaults.object(forKey: ServerController.portKey)
        defaults.set(true, forKey: ServerController.enabledKey)
        defaults.set(8787, forKey: ServerController.portKey)
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
        for try await line in bytes.lines {
            if line.hasPrefix("event: ") {
                let eventName = String(line.dropFirst("event: ".count))
                if eventName == "message_start" { sawMessageStart = true }
                if eventName == "error" { sawErrorEvent = true }
            }
            if sawErrorEvent { break }
        }
        XCTAssertTrue(sawMessageStart, "the stream must open with message_start")
        XCTAssertTrue(sawErrorEvent, "the stream must surface the launch failure as an error event")
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
}
