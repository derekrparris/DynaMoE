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

    // MARK: - Shared models path, two protocols

    /// `GET /v1/models` is one path and two bodies, so the request's headers are the
    /// only evidence the route has. A client that sends neither Anthropic header must
    /// still get the shape this route has always answered.
    func testTheModelsListAnswersInTheShapeTheClientsHeadersAskFor() async throws {
        await ensureServerRunning()

        let plain = try await getModels("/v1/models", headers: [:])
        XCTAssertEqual(plain.status, 200)
        XCTAssertEqual(plain.body["object"] as? String, "list")
        XCTAssertNil(plain.body["has_more"], "an OpenAI body must not carry Anthropic pagination")

        // Either Anthropic header is enough: the SDKs send the version on every
        // request and the key header whether or not this server requires a key.
        for headers in [["anthropic-version": "2023-06-01"], ["x-api-key": "unused-key"]] {
            let anthropic = try await getModels("/v1/models", headers: headers)
            XCTAssertEqual(anthropic.status, 200)
            XCTAssertNil(anthropic.body["object"], "an Anthropic body is not an OpenAI list")
            XCTAssertEqual(anthropic.body["has_more"] as? Bool, false)
            if let info = await MainActor.run(body: { CompletionEngine.shared.loadedModelInfo }) {
                let entries = try XCTUnwrap(anthropic.body["data"] as? [[String: Any]])
                let first = try XCTUnwrap(entries.first)
                XCTAssertEqual(first["type"] as? String, "model")
                XCTAssertEqual(first["id"] as? String, info.displayName)
                XCTAssertEqual(anthropic.body["first_id"] as? String, info.displayName)
                XCTAssertEqual(anthropic.body["last_id"] as? String, info.displayName)
                let createdAt = try XCTUnwrap(first["created_at"] as? String)
                XCTAssertTrue(createdAt.hasSuffix("Z"), "the protocol's date is RFC 3339, not an epoch second")
                XCTAssertNotNil(first["display_name"] as? String)
            }
        }

        // Both shapes are recorded, and the row says which protocol answered, so one
        // path is still two readable protocols in the dashboard.
        let rows = await logStore.fetch(limit: 50).filter { $0.path == "/v1/models" }
        XCTAssertTrue(rows.contains { $0.proto == .openai }, "the plain request is an OpenAI models call")
        XCTAssertTrue(rows.contains { $0.proto == .anthropic }, "the headed requests are Anthropic models calls")
    }

    func testTheModelsRetrieveRouteAnswersBothShapesAndRefusesAnUnknownId() async throws {
        await ensureServerRunning()
        guard let info = await MainActor.run(body: { CompletionEngine.shared.loadedModelInfo }) else {
            throw XCTSkip("No model is loaded; there is no served id to retrieve")
        }

        let openai = try await getModels("/v1/models/\(info.displayName)", headers: [:])
        XCTAssertEqual(openai.status, 200)
        XCTAssertEqual(openai.body["object"] as? String, "model")
        XCTAssertEqual(openai.body["id"] as? String, info.displayName)

        let anthropic = try await getModels(
            "/v1/models/\(info.displayName)", headers: ["anthropic-version": "2023-06-01"]
        )
        XCTAssertEqual(anthropic.status, 200)
        XCTAssertEqual(anthropic.body["type"] as? String, "model")
        XCTAssertNil(anthropic.body["has_more"], "a retrieve answers the object, not a page")

        // An id this server does not serve is a 404 in the caller's own envelope:
        // OpenAI keeps `invalid_request_error` and names the condition in `code`,
        // Anthropic has a `not_found_error`.
        let missing = try await getModels("/v1/models/not-a-served-model", headers: [:])
        XCTAssertEqual(missing.status, 404)
        let openAIError = try XCTUnwrap(missing.body["error"] as? [String: Any])
        XCTAssertEqual(openAIError["type"] as? String, "invalid_request_error")
        XCTAssertEqual(openAIError["code"] as? String, "model_not_found")

        let missingAnthropic = try await getModels(
            "/v1/models/not-a-served-model", headers: ["anthropic-version": "2023-06-01"]
        )
        XCTAssertEqual(missingAnthropic.status, 404)
        XCTAssertEqual(missingAnthropic.body["type"] as? String, "error")
        XCTAssertEqual(
            (missingAnthropic.body["error"] as? [String: Any])?["type"] as? String, "not_found_error"
        )
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

    // MARK: - Request fields this server cannot honour

    /// A request asking for JSON, for two choices or for log-probabilities is
    /// refused BY NAME, and refused before the model gate: the proof is a 400 where
    /// the same body shape answers 503. Before `#88` these keys were dropped by the
    /// decoder and the turn ran as if they had not been sent.
    func testUnsupportedResponseShapeFieldsAreRefusedBeforeTheModelGate() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the pre-gate refusal cannot be isolated from generation")
        }
        await ensureServerRunning()

        for field in [
            #""response_format":{"type":"json_object"}"#,
            #""response_format":{"type":"json_schema","json_schema":{"name":"x"}}"#,
            #""n":2"#,
            #""logprobs":true"#,
            #""top_logprobs":3"#,
        ] {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("""
            {"model":"any",\(field),"messages":[{"role":"user","content":"hello"}]}
            """.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            let http = try XCTUnwrap(response as? HTTPURLResponse)
            XCTAssertEqual(http.statusCode, 400, "\(field) must be refused, not ignored")
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            let error = try XCTUnwrap(body["error"] as? [String: Any])
            XCTAssertEqual(error["type"] as? String, "invalid_request_error")
            let message = try XCTUnwrap(error["message"] as? String)
            let named = field.split(separator: "\"").first.map(String.init) ?? ""
            XCTAssertTrue(
                message.contains(named),
                "the refusal must name \(named), got: \(message)"
            )
        }
    }

    /// The same refusal on the streaming route keeps that route's own contract: the
    /// response is a 200 and the refusal travels as an SSE error frame, exactly as
    /// `#80` pinned for a picture against a model with no tower.
    func testUnsupportedResponseShapeIsAnSSEErrorOnTheStreamingRoute() async throws {
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","stream":true,"response_format":{"type":"json_schema","json_schema":{"name":"x"}},
         "messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 200, "a streaming refusal wears a 200 by contract")
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(
            text.contains("\"error\""),
            "the refusal must arrive as an SSE error frame, got: \(text)"
        )
        XCTAssertTrue(text.contains("response_format"), "the frame must name the field")
    }

    // MARK: - The legacy text-completions route

    /// `POST /v1/completions` exists and is the chat route's sibling, not a 404: with
    /// no model loaded it answers `503` for the same reason `/v1/chat/completions`
    /// does. Before this route existed the request log already classified the path as
    /// OpenAI, so the vocabulary named a path the router did not serve.
    func testTheCompletionsRouteExistsAndIsGatedOnALoadedModel() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the no-model gate cannot be isolated from generation")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"model":"any","prompt":"The capital of France is","max_tokens":4}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(
            http.statusCode, 503,
            "a route that exists answers 503 without a model, not 404"
        )
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let error = try XCTUnwrap(body["error"] as? [String: Any], "the 503 must wear the OpenAI error envelope")
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
    }

    /// The legacy route refuses the fields it cannot honour by name, and refuses them
    /// BEFORE the model gate: the proof is a 400 where the same body shape answers 503.
    /// The sampler fields are named here even though the chat route still ignores them,
    /// because this route is new and refusing them cannot break a working client.
    func testTheCompletionsRouteRefusesUnrunnableFieldsBeforeTheModelGate() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the pre-gate refusal cannot be isolated from generation")
        }
        await ensureServerRunning()

        func post(_ field: String, stream: Bool = false) async throws -> (status: Int, text: String) {
            let streaming = stream ? #","stream":true"# : ""
            var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("""
            {"model":"any","max_tokens":4,\(field)\(streaming),"prompt":"hello"}
            """.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            return (try XCTUnwrap((response as? HTTPURLResponse)?.statusCode), String(decoding: data, as: UTF8.self))
        }

        for field in [
            #""suffix":"tail""#,
            #""n":2"#,
            #""logprobs":1"#,
            #""best_of":2"#,
            #""logit_bias":{"50256":-100}"#,
            #""frequency_penalty":0.5"#,
        ] {
            let (status, text) = try await post(field)
            XCTAssertEqual(status, 400, "\(field) must be refused, not ignored: \(text)")
            let body = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
            let error = try XCTUnwrap(body["error"] as? [String: Any])
            XCTAssertEqual(error["type"] as? String, "invalid_request_error")
            let message = try XCTUnwrap(error["message"] as? String)
            let named = field.split(separator: "\"").first.map(String.init) ?? ""
            XCTAssertTrue(message.contains(named), "the refusal must name \(named), got: \(message)")
        }

        // A batch of prompts is a body shape rather than a field, so it gets its own
        // request instead of a field spliced into the loop's fixed body.
        var batch = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/completions")!)
        batch.httpMethod = "POST"
        batch.setValue("application/json", forHTTPHeaderField: "Content-Type")
        batch.httpBody = Data(#"{"model":"any","max_tokens":4,"prompt":["a","b"]}"#.utf8)
        let (batchData, batchResponse) = try await URLSession.shared.data(for: batch)
        XCTAssertEqual((batchResponse as? HTTPURLResponse)?.statusCode, 400, "one prompt per request")
        XCTAssertTrue(String(decoding: batchData, as: UTF8.self).contains("2 prompts"))

        // The streaming route keeps its own contract: a 200 whose refusal travels as an
        // SSE error frame, exactly as `#80` pinned for a picture against a model with no
        // tower.
        let streamed = try await post(#""suffix":"tail""#, stream: true)
        XCTAssertEqual(streamed.status, 200, "a streaming refusal wears a 200 by contract")
        XCTAssertTrue(streamed.text.contains("\"error\""), "got: \(streamed.text)")
        XCTAssertTrue(streamed.text.contains("suffix"), "the frame must name the field")
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

    // MARK: - Token counting

    /// Anthropic's pre-flight, on the real server. The body carries no
    /// `max_tokens` — the protocol's count_tokens body has no such field — so a
    /// 503 from the model gate is also the proof that the route does not demand
    /// one: a planner that still required it would answer 400 first.
    func testCountTokensWithoutModelReturns503ErrorEnvelope() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages/count_tokens")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("""
        {"model":"any","messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let statsBefore = ServerController.stats.snapshot()
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 503, "with no model there is no prompt to render")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(body["type"] as? String, "error", "the pre-flight answers the Anthropic envelope")
        XCTAssertFalse(((body["error"] as? [String: Any])?["message"] as? String ?? "").isEmpty)
        XCTAssertEqual(
            ServerController.stats.snapshot().totalRequests,
            statsBefore.totalRequests + 1,
            "the pre-flight must be counted in the server stats like any other request"
        )

        let loggedRow = try await waitForLoggedRow { $0.path == "/v1/messages/count_tokens" }
        let logged = try XCTUnwrap(loggedRow)
        XCTAssertEqual(logged.method, "POST")
        XCTAssertEqual(
            logged.proto, .anthropic,
            "the pre-flight shares the messages route's protocol, so a dashboard reads it as Anthropic"
        )
        XCTAssertEqual(logged.httpStatus, 503)
        XCTAssertTrue(logged.isError)
    }

    func testCountTokensErrorResponsesAreAnthropicShaped() async throws {
        await ensureServerRunning()

        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages/count_tokens")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(#"{"messages":"not an array"}"#.utf8)
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        XCTAssertEqual(http.statusCode, 400, "an unparseable body must be rejected before the model gate")
        let body = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(body["type"] as? String, "error")
        let error = try XCTUnwrap(body["error"] as? [String: Any])
        XCTAssertEqual(error["type"] as? String, "invalid_request_error")
        XCTAssertFalse((error["message"] as? String ?? "").isEmpty)
    }

    /// The pre-flight is a client request like any other, so it carries the same
    /// credential either a bearer token or Anthropic's `x-api-key`.
    func testCountTokensRequiresTheAPIKey() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the past-the-gate path cannot be exercised")
        }
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

        var anonymous = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/messages/count_tokens")!)
        anonymous.httpMethod = "POST"
        anonymous.setValue("application/json", forHTTPHeaderField: "Content-Type")
        anonymous.httpBody = Data("""
        {"model":"any","messages":[{"role":"user","content":"hello"}]}
        """.utf8)
        let (_, anonymousResponse) = try await URLSession.shared.data(for: anonymous)
        XCTAssertEqual(
            try XCTUnwrap(anonymousResponse as? HTTPURLResponse).statusCode, 401,
            "a request with no credential must be refused"
        )

        var keyed = anonymous
        keyed.setValue("secret-test-key", forHTTPHeaderField: "x-api-key")
        let (_, keyedResponse) = try await URLSession.shared.data(for: keyed)
        XCTAssertEqual(
            try XCTUnwrap(keyedResponse as? HTTPURLResponse).statusCode, 503,
            "x-api-key must be accepted, leaving the model gate as the only refusal"
        )
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

    /// Issues a models-route GET with the given headers and returns the status and
    /// the decoded body, which is all the shape assertions need.
    private func getModels(
        _ path: String,
        headers: [String: String]
    ) async throws -> (status: Int, body: [String: Any]) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787\(path)")!)
        for (field, value) in headers {
            request.setValue(value, forHTTPHeaderField: field)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = try XCTUnwrap((response as? HTTPURLResponse)?.statusCode)
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        return (status, body)
    }

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
        XCTAssertEqual(row.proto, .openai, "a header-less models call is the OpenAI shape, and the row says so")
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
