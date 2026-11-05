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
            XCTAssertEqual(entries.first?["id"] as? String, info.servedID)
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
                XCTAssertEqual(first["id"] as? String, info.servedID)
                XCTAssertEqual(anthropic.body["first_id"] as? String, info.servedID)
                XCTAssertEqual(anthropic.body["last_id"] as? String, info.servedID)
                let createdAt = try XCTUnwrap(first["created_at"] as? String)
                XCTAssertTrue(createdAt.hasSuffix("Z"), "the protocol's date is RFC 3339, not an epoch second")
                // The id is a path segment and the display name is the human
                // spelling; the two must not collapse into each other again.
                let displayName = try XCTUnwrap(first["display_name"] as? String)
                XCTAssertFalse(displayName.isEmpty)
                XCTAssertFalse(info.servedID.contains(" "), "an id with a space is not a path segment")
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

        let openai = try await getModels("/v1/models/\(info.servedID)", headers: [:])
        XCTAssertEqual(openai.status, 200)
        XCTAssertEqual(openai.body["object"] as? String, "model")
        XCTAssertEqual(openai.body["id"] as? String, info.servedID)

        let anthropic = try await getModels(
            "/v1/models/\(info.servedID)", headers: ["anthropic-version": "2023-06-01"]
        )
        XCTAssertEqual(anthropic.status, 200)
        XCTAssertEqual(anthropic.body["type"] as? String, "model")
        XCTAssertNil(anthropic.body["has_more"], "a retrieve answers the object, not a page")

        // The snapshot directory's own name was the served id before it became the
        // checkpoint's repository name. A client configured against the old server
        // keeps working rather than meeting a 404 on a rename it never made.
        let legacy = try await getModels("/v1/models/\(info.displayName)", headers: [:])
        XCTAssertEqual(legacy.status, 200)
        XCTAssertEqual(
            legacy.body["id"] as? String, info.servedID,
            "the alias answers the served object, not the name it was asked by"
        )

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
            #""modalities":["text","audio"]"#,
            #""audio":{"voice":"alloy","format":"mp3"}"#,
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

        // A reply budget asked for with BOTH spellings and two different values is a
        // 400 naming both, before the model gate, rather than a choice the server made
        // for the client. `max_completion_tokens` alone is a field this server honours,
        // so it must reach the gate (503) instead.
        func postFields(_ fields: String) async throws -> (status: Int, text: String) {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("""
            {"model":"any",\(fields),"messages":[{"role":"user","content":"hello"}]}
            """.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            return (
                try XCTUnwrap((response as? HTTPURLResponse)?.statusCode),
                String(decoding: data, as: UTF8.self)
            )
        }

        let (conflictStatus, conflictText) = try await postFields(
            #""max_tokens":64,"max_completion_tokens":512"#
        )
        XCTAssertEqual(conflictStatus, 400, "two disagreeing budgets are a bad request: \(conflictText)")
        XCTAssertTrue(conflictText.contains("max_completion_tokens"), "got: \(conflictText)")

        let (newerStatus, newerText) = try await postFields(#""max_completion_tokens":512"#)
        XCTAssertEqual(newerStatus, 503, "the newer reply-budget field is honoured, not refused: \(newerText)")

        // `["text"]` is the protocol's default, so a client that says it out loud is
        // asking for exactly what it would have got by staying silent, and must reach
        // the model gate rather than a new 400.
        let (textOnlyStatus, textOnlyBody) = try await postFields(#""modalities":["text"]"#)
        XCTAssertEqual(textOnlyStatus, 503, "text is the modality this server has: \(textOnlyBody)")
    }

    /// OpenAI's deprecated `functions` / `function_call` pair was absent from the
    /// decoder, so a client on that surface was answered in prose with no call and no
    /// error. It is refused by name, and refused before the model gate: the proof is a
    /// 400 where the same request spelled with `tools` answers 503. Mapping the pair
    /// was the alternative and is worse, because the reply would carry `tool_calls`
    /// while a legacy client parses `function_call`.
    func testTheDeprecatedFunctionCallingSurfaceIsRefusedBeforeTheModelGate() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the pre-gate refusal cannot be isolated from generation")
        }
        await ensureServerRunning()

        func post(_ fields: String) async throws -> (status: Int, text: String) {
            var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = Data("""
            {"model":"any",\(fields),"messages":[{"role":"user","content":"hello"}]}
            """.utf8)
            let (data, response) = try await URLSession.shared.data(for: request)
            return (
                try XCTUnwrap((response as? HTTPURLResponse)?.statusCode),
                String(decoding: data, as: UTF8.self)
            )
        }

        let (legacyStatus, legacyText) = try await post(#""functions":[{"name":"get_weather"}]"#)
        XCTAssertEqual(legacyStatus, 400, "functions must be refused, not ignored: \(legacyText)")
        XCTAssertTrue(legacyText.contains("functions"), "got: \(legacyText)")
        XCTAssertTrue(legacyText.contains("tools"), "the refusal must name the modern field: \(legacyText)")
        XCTAssertTrue(legacyText.contains("invalid_request_error"), "got: \(legacyText)")

        let (callStatus, callText) = try await post(#""function_call":"auto""#)
        XCTAssertEqual(callStatus, 400, "function_call must be refused, not ignored: \(callText)")
        XCTAssertTrue(callText.contains("function_call"), "got: \(callText)")
        XCTAssertTrue(callText.contains("tool_choice"), "the refusal must name the modern field: \(callText)")

        // The modern spelling is served, so a client sending both reaches the model gate
        // rather than a new 400: the deprecated field is then redundant, not lost.
        let (modernStatus, modernText) = try await post(
            #""tools":[{"type":"function","function":{"name":"get_weather"}}],"functions":[{"name":"get_weather"}]"#
        )
        XCTAssertEqual(modernStatus, 503, "tools is the vocabulary this server serves: \(modernText)")

        let (choiceStatus, choiceText) = try await post(
            #""tools":[{"type":"function","function":{"name":"get_weather"}}],"tool_choice":"auto","function_call":"none""#
        )
        XCTAssertEqual(choiceStatus, 503, "tool_choice is the vocabulary this server serves: \(choiceText)")
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
    /// The sampler fields are no longer on this list: both routes resolve them into the
    /// sampler, so they are asserted to reach the gate instead.
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

        // The sampler fields are honoured, so they must reach the MODEL gate (503)
        // rather than the refusal (400): a value this server can apply is not a bad
        // request, and the opposite answer would break a client that always sends it.
        let (samplerStatus, samplerText) = try await post(
            #""frequency_penalty":0.5,"logit_bias":{"50256":-100}"#
        )
        XCTAssertEqual(samplerStatus, 503, "a field the sampler honours is not a 400: \(samplerText)")

        // A value outside the range OpenAI defines is still refused by name.
        let (rangeStatus, rangeText) = try await post(#""logit_bias":{"50256":-500}"#)
        XCTAssertEqual(rangeStatus, 400, "an out-of-range bias is a bad request: \(rangeText)")
        XCTAssertTrue(rangeText.contains("logit_bias"), "the refusal must name logit_bias, got: \(rangeText)")

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

    /// Issues a chat-completions POST with the given body and returns the status.
    /// With no model loaded every arm answers 503, which is exactly enough to reach
    /// the row the request logging writes.
    private func postChat(_ body: [String: Any]) async throws -> Int {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:8787/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        _ = data
        return try XCTUnwrap((response as? HTTPURLResponse)?.statusCode)
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

    /// `store: false` is a promise the client is entitled to have kept, and it is
    /// honoured per request: with body capture ON, that request's row still carries
    /// the withheld marker and the flag that says the client asked for it. The other
    /// two arms are the control and the asymmetry — a client that says nothing or
    /// allows retention gets captured bodies, and `store: true` cannot force capture
    /// once the operator's switch is off, because retention here is the operator's
    /// decision. Each arm carries its own `max_tokens` so the rows cannot be
    /// confused: content is exactly what the withheld arms do not have to match on.
    func testAClientRetentionRefusalWithholdsItsRowAndATrueCannotForceCapture() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()
        // `captureBodiesKey` is main-actor state, so it is read once here rather than
        // from this nonisolated context at each use. `tearDown` restores the
        // developer's saved value; this case needs the switch ON for its first two
        // arms and OFF for the third.
        let captureBodiesKey = await MainActor.run { ServerRequestLogRecorder.captureBodiesKey }
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: captureBodiesKey)

        func row(maxTokens: Int) async throws -> ServerRequestLog {
            let logged = try await waitForLoggedRow {
                $0.path == "/v1/chat/completions" && $0.maxTokens == maxTokens
            }
            return try XCTUnwrap(logged)
        }

        let refusedStatus = try await postChat([
            "model": "any",
            "store": false,
            "max_tokens": 11,
            "messages": [["role": "user", "content": "a prompt the client asked not to keep"]]
        ])
        XCTAssertEqual(refusedStatus, 503)
        let refused = try await row(maxTokens: 11)
        XCTAssertEqual(refused.retentionWithheldByClient, true, "the row must say who withheld it, so this never reads as the switch failing")
        XCTAssertEqual(refused.requestBody, ServerRequestLog.contentWithheldMarker)
        XCTAssertFalse(refused.hasCapturedContent, "no content field may survive a store: false")

        let capturedStatus = try await postChat([
            "model": "any",
            "max_tokens": 22,
            "messages": [["role": "user", "content": "kept because the client allowed it"]]
        ])
        XCTAssertEqual(capturedStatus, 503)
        let captured = try await row(maxTokens: 22)
        XCTAssertNil(captured.retentionWithheldByClient, "no retention promise was sent, and nil is not the same claim as store: true")
        let capturedBody = try XCTUnwrap(captured.requestBody)
        XCTAssertTrue(
            capturedBody.contains("kept because the client allowed it"),
            "the control arm proves capture is on, so the refused arm was proving the flag"
        )
        XCTAssertFalse(
            capturedBody.contains("\"store\""),
            "a request that sent no retention promise records none, which is what keeps absent distinguishable from store: true"
        )

        defaults.set(false, forKey: captureBodiesKey)
        let allowingStatus = try await postChat([
            "model": "any",
            "store": true,
            "max_tokens": 33,
            "messages": [["role": "user", "content": "the client allowed retention but the switch is off"]]
        ])
        XCTAssertEqual(allowingStatus, 503)
        let stillWithheld = try await row(maxTokens: 33)
        XCTAssertNil(stillWithheld.retentionWithheldByClient, "the operator withheld this one, not the client, so no client refusal is recorded")
        XCTAssertEqual(
            stillWithheld.requestBody, ServerRequestLog.contentWithheldMarker,
            "store: true cannot force capture: the operator's switch is the stronger authority on their own Mac"
        )
    }

    /// The client's own labels are the one thing a fully withheld row can still say, so
    /// they survive the Store Request Content switch rather than riding on it: a request
    /// carrying `metadata` records it with the switch OFF (where its body reads
    /// "withheld") and with the switch ON, and the third arm is the control that proves
    /// the field rather than the switch is what puts them there. The pair limit is a
    /// recording bound, applied on the way in, so a client over OpenAI's own limit still
    /// gets its answer.
    func testAClientsLabelsSurviveAWithheldRow() async throws {
        guard await MainActor.run(body: { !CompletionEngine.shared.hasLoadedModel }) else {
            throw XCTSkip("A model is loaded; the 503 path cannot be exercised")
        }
        await ensureServerRunning()
        let captureBodiesKey = await MainActor.run { ServerRequestLogRecorder.captureBodiesKey }
        let defaults = UserDefaults.standard

        func row(maxTokens: Int) async throws -> ServerRequestLog {
            let logged = try await waitForLoggedRow {
                $0.path == "/v1/chat/completions" && $0.maxTokens == maxTokens
            }
            return try XCTUnwrap(logged)
        }

        defaults.set(false, forKey: captureBodiesKey)
        let withheldStatus = try await postChat([
            "model": "any",
            "max_tokens": 44,
            "metadata": ["ticket": "ABC-123", "team": "infra"],
            "messages": [["role": "user", "content": "which request was this"]]
        ])
        XCTAssertEqual(withheldStatus, 503)
        let withheld = try await row(maxTokens: 44)
        XCTAssertEqual(withheld.requestBody, ServerRequestLog.contentWithheldMarker, "the switch is off, so the bodies are withheld")
        XCTAssertEqual(
            withheld.clientMetadata, ["ticket": "ABC-123", "team": "infra"],
            "the labels are metadata rather than content, so a withheld row still says which request the caller meant"
        )

        // Control: the same switch state, no labels sent, nothing recorded.
        let controlStatus = try await postChat([
            "model": "any",
            "max_tokens": 55,
            "messages": [["role": "user", "content": "no labels sent"]]
        ])
        XCTAssertEqual(controlStatus, 503)
        let control = try await row(maxTokens: 55)
        XCTAssertNil(control.clientMetadata, "no labels sent records none")

        // Over the pair limit the answer is unaffected and the recorded copy is bounded.
        defaults.set(true, forKey: captureBodiesKey)
        var many: [String: String] = [:]
        for index in 0..<20 { many["key\(String(format: "%02d", index))"] = "v\(index)" }
        let manyStatus = try await postChat([
            "model": "any",
            "max_tokens": 66,
            "metadata": many,
            "messages": [["role": "user", "content": "too many labels"]]
        ])
        XCTAssertEqual(manyStatus, 503, "an over-long metadata map is recorded in part, never refused")
        let bounded = try await row(maxTokens: 66)
        XCTAssertEqual(bounded.clientMetadata?.count, ServerRequestLog.maxClientMetadataPairs)
        XCTAssertNil(bounded.clientMetadata?["key19"], "the sorted-key cut is what drops the last four")
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
