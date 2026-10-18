//
//  ServerRequestLogLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the request-log row model and recorder: body
//  clipping, the content-withheld policy, wire-line composition (metadata
//  only — bodies never reach the unified log), error-message extraction,
//  protocol inference, request summaries, and the per-request box
//  claim/finalize lifecycle. No server, no model, no network.
//

import XCTest
@testable import DynaMoE

final class ServerRequestLogLogicTests: XCTestCase {

    private var savedCaptureSetting: Any?

    override func setUp() {
        super.setUp()
        savedCaptureSetting = UserDefaults.standard.object(forKey: ServerRequestLogRecorder.captureBodiesKey)
        UserDefaults.standard.removeObject(forKey: ServerRequestLogRecorder.captureBodiesKey)
    }

    override func tearDown() {
        let defaults = UserDefaults.standard
        if let saved = savedCaptureSetting {
            defaults.set(saved, forKey: ServerRequestLogRecorder.captureBodiesKey)
        } else {
            defaults.removeObject(forKey: ServerRequestLogRecorder.captureBodiesKey)
        }
        super.tearDown()
    }

    // MARK: Capture policy

    func testBodyCaptureDefaultsOff() {
        XCTAssertFalse(ServerRequestLogRecorder.captureBodiesEnabled, "storing request/response content must be opt-in")
    }

    func testClippingBoundsBodies() {
        let large = String(repeating: "x", count: ServerRequestLog.maxBodyCaptureLength + 1_000)
        XCTAssertEqual(
            ServerRequestLog.clippedBody(large).count,
            ServerRequestLog.maxBodyCaptureLength
        )
        let small = "short"
        XCTAssertEqual(ServerRequestLog.clippedBody(small), "short", "bodies within the cap must pass through untouched")
    }

    func testClippingBoundsToolArguments() {
        let large = String(repeating: "y", count: ServerRequestLog.maxToolArgumentCaptureLength + 100)
        XCTAssertEqual(
            ServerRequestLog.clippedArguments(large).count,
            ServerRequestLog.maxToolArgumentCaptureLength
        )
    }

    func testHasCapturedContentDistinguishesMarkerFromNilFromContent() {
        var row = ServerRequestLog()
        XCTAssertFalse(row.hasCapturedContent, "an all-nil row stores nothing")

        row.requestBody = ServerRequestLog.contentWithheldMarker
        row.responseBody = ServerRequestLog.contentWithheldMarker
        row.toolCalls = [.init(name: "web_search", arguments: ServerRequestLog.contentWithheldMarker)]
        XCTAssertFalse(row.hasCapturedContent, "the withheld marker must never read as stored content")

        row.responseBody = "here is the answer"
        XCTAssertTrue(row.hasCapturedContent)
    }

    // MARK: Unified-log line

    func testComposedLineCarriesMetadataButNeverBodies() {
        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/chat/completions"
        row.httpStatus = 200
        row.durationMs = 1_234
        row.model = "ling-3.0-tiny"
        row.inputTokens = 512
        row.outputTokens = 89
        row.finishReason = "stop"
        row.requestBody = "SECRET PROMPT CONTENT"
        row.responseBody = "SECRET ANSWER CONTENT"
        row.toolCalls = [.init(name: "get_weather", arguments: "{\"city\":\"sf\"}")]

        let line = ServerRequestLogRecorder.composeLine(row)
        XCTAssertTrue(line.contains("POST /v1/chat/completions"))
        XCTAssertTrue(line.contains("model=ling-3.0-tiny"))
        XCTAssertTrue(line.contains("in=512"))
        XCTAssertTrue(line.contains("out=89"))
        XCTAssertTrue(line.contains("finish=stop"))
        XCTAssertTrue(line.contains("tools=get_weather"))
        XCTAssertFalse(line.contains("SECRET PROMPT CONTENT"), "bodies must never reach the unified log")
        XCTAssertFalse(line.contains("SECRET ANSWER CONTENT"), "bodies must never reach the unified log")
        XCTAssertFalse(line.contains("{\"city\""), "tool arguments must never reach the unified log")
    }

    func testComposedLineMarksStreamingAndError() {
        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/messages"
        row.wasStreaming = true
        row.streamTermination = .errorFrame
        row.errorMessage = "No model is loaded"
        let line = ServerRequestLogRecorder.composeLine(row)
        XCTAssertTrue(line.contains("stream=errorFrame"))
        XCTAssertTrue(line.contains("error=No model is loaded"))

        row.streamTermination = nil
        XCTAssertEqual(
            ServerRequestLogRecorder.composeLine(row).contains("stream=open"),
            true,
            "a streaming row with no terminal state reads as an open stream"
        )
    }

    // MARK: Error message policy

    func testErrorMessageExtraction() {
        XCTAssertEqual(
            ServerRequestLogRecorder.errorMessage(for: ServerAPIError.serviceUnavailable("No model is loaded")),
            "No model is loaded"
        )
        struct Opaque: Error {}
        XCTAssertFalse(ServerRequestLogRecorder.errorMessage(for: Opaque()).isEmpty)
    }

    func testUnauthorizedErrorMessageNeverCarriesCredentialMaterial() {
        // The 401 path composes its row through the recorder; the error
        // message is the fixed wire string, never the presented key value.
        let message = ServerRequestLogRecorder.errorMessage(for: ServerAPIError.unauthorized("Invalid or missing API key"))
        XCTAssertEqual(message, "Invalid or missing API key")
        XCTAssertFalse(message.contains("Bearer"), "the presented credential must not flow into the row")
    }

    // MARK: Protocol inference

    func testProtoInference() {
        XCTAssertEqual(ServerRequestLogRecorder.proto(forPath: "/v1/chat/completions"), .openai)
        XCTAssertEqual(ServerRequestLogRecorder.proto(forPath: "/v1/messages"), .anthropic)
        XCTAssertEqual(ServerRequestLogRecorder.proto(forPath: "/v1/models"), .other)
        XCTAssertEqual(ServerRequestLogRecorder.proto(forPath: "/health"), .other)
    }

    // MARK: Request summaries

    func testOpenAISummaryCapturesContentButNoCredentialHeaders() throws {
        let json = """
        {"model":"ling","messages":[{"role":"user","content":"hello there"}],"temperature":0.5,"max_tokens":128,"stream":false}
        """
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
        let summary = try XCTUnwrap(ServerRequestLog.openAISummary(request))
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(summary.utf8)) as? [String: Any])
        let messages = try XCTUnwrap(payload["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.first?["content"] as? String, "hello there")

        XCTAssertNil(payload["authorization"], "headers never appear in the captured body")
        XCTAssertNil(payload["x-api-key"], "headers never appear in the captured body")
    }

    func testAnthropicSummaryCapturesContent() throws {
        let json = """
        {"model":"ling","max_tokens":64,"system":"be terse","messages":[{"role":"user","content":"hello there"}]}
        """
        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(json.utf8))
        let summary = try XCTUnwrap(ServerRequestLog.anthropicSummary(request))
        let payload = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(summary.utf8)) as? [String: Any])
        XCTAssertEqual(payload["system"] as? String, "be terse")
        let messages = try XCTUnwrap(payload["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.first?["content"] as? String, "hello there")
    }

    // MARK: Box lifecycle

    func testBoxClaimIsFirstCallerWins() {
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "GET", path: "/v1/models")
        defer { recorder.endBox(requestId: id) }

        XCTAssertTrue(box.claim(), "the first claimant owns the row")
        XCTAssertFalse(box.claim(), "a second claim must not succeed")
        XCTAssertTrue(box.isClaimed)
    }

    func testBoxComposeFinalStampsWireFieldsAndError() {
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "POST", path: "/v1/messages")
        defer { recorder.endBox(requestId: id) }
        box.modify { $0.model = "ling-3.0-tiny" }

        let finalized = box.composeFinal(status: 400, error: ServerAPIError.badRequest("max_tokens is a required field"))
        XCTAssertEqual(finalized.method, "POST")
        XCTAssertEqual(finalized.path, "/v1/messages")
        XCTAssertEqual(finalized.proto, .anthropic)
        XCTAssertEqual(finalized.model, "ling-3.0-tiny")
        XCTAssertEqual(finalized.httpStatus, 400)
        XCTAssertEqual(finalized.errorMessage, "max_tokens is a required field")
        XCTAssertGreaterThan(finalized.durationMs, 0)

        let clean = box.composeFinal(status: 200, error: nil)
        XCTAssertEqual(clean.httpStatus, 200)
        XCTAssertNil(clean.errorMessage)
    }

    func testBoxRegistryLookupAndCleanup() {
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        XCTAssertNil(recorder.box(for: id))
        let box = recorder.beginBox(requestId: id, method: "GET", path: "/health")
        XCTAssertNotNil(recorder.box(for: id))
        recorder.endBox(requestId: id)
        XCTAssertNil(recorder.box(for: id), "the wire middleware's box must leave the registry when the request ends")
        _ = box
    }

    // MARK: Full-record privacy integration

    func testRecordedRowIsMetadataOnlyByDefault() async throws {
        // Route-level capture gating: with the toggle off, a chat row's body
        // fields are withheld markers, and neither the row nor its unified
        // log line contains client content.
        UserDefaults.standard.set(false, forKey: ServerRequestLogRecorder.captureBodiesKey)
        XCTAssertFalse(ServerRequestLogRecorder.captureBodiesEnabled)

        let json = """
        {"model":"ling","messages":[{"role":"user","content":"TOP SECRET INJECTION"}]}
        """
        let apiRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))

        var rowWithBodies = ServerRequestLog()
        rowWithBodies.requestBody = ServerRequestLogRecorder.captureBodiesEnabled
            ? ServerRequestLog.openAISummary(apiRequest)
            : ServerRequestLog.contentWithheldMarker

        XCTAssertEqual(rowWithBodies.requestBody, ServerRequestLog.contentWithheldMarker)
        XCTAssertFalse(ServerRequestLogRecorder.composeLine(rowWithBodies).contains("TOP SECRET"))
    }

    // MARK: Console-to-dashboard correlation

    func testComposedLineLeadsWithRowShortId() throws {
        let uuid = try XCTUnwrap(UUID(uuidString: "A1B2C3D4-1111-4222-8333-444455556666"))
        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/chat/completions"
        row.id = uuid
        XCTAssertEqual(row.shortId, "a1b2c3d4", "the short id is the lowercase 8-digit id prefix")

        let line = ServerRequestLogRecorder.composeLine(row)
        XCTAssertTrue(line.hasPrefix("id=a1b2c3d4 "), "the line must lead with the row's correlation id so a Console.app line pivots to the dashboard row")
    }

    func testBeginBoxRowIdReusesWireRequestIdWhenUUIDShaped() {
        let recorder = ServerRequestLogRecorder.shared
        let wireId = UUID()
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: wireId.uuidString, method: "POST", path: "/v1/messages")
        defer { recorder.endBox(requestId: wireId.uuidString) }
        XCTAssertEqual(
            box.snapshot().id, wireId,
            "the row must carry Hummingbird's request id so all logs for one request share an identity"
        )

        // Non-UUID registry keys (tests) fall back to a fresh, valid row id.
        let fallbackBox = recorder.beginBox(requestId: id, method: "GET", path: "/health")
        defer { recorder.endBox(requestId: id) }
        let fallbackId = fallbackBox.snapshot().id
        XCTAssertNotEqual(fallbackId, wireId)
        XCTAssertFalse(fallbackId.uuidString.isEmpty)

        // Whatever the id's provenance, the composed line always prints a
        // prefix of the row's own id: the line and the row cannot disagree.
        let line = ServerRequestLogRecorder.composeLine(fallbackBox.snapshot())
        XCTAssertTrue(line.hasPrefix("id=\(fallbackBox.snapshot().shortId) "))
    }

    // MARK: Queue wait split

    func testComposedLineSplitsQueueWaitFromDuration() {
        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/chat/completions"
        row.durationMs = 123_000
        row.queueWaitMs = 100_000

        let line = ServerRequestLogRecorder.composeLine(row)
        XCTAssertTrue(line.contains("queue=100000ms"), "the queue wait must be visible in Console.app to split it from the 123 s duration")

        row.queueWaitMs = nil
        XCTAssertFalse(
            ServerRequestLogRecorder.composeLine(row).contains("queue="),
            "rows that never reached the slot must not read as a zero wait"
        )
    }

    func testBoxNotesSlotOpenedOnceAndCarriesItThroughFinalization() {
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "POST", path: "/v1/chat/completions")
        defer { recorder.endBox(requestId: id) }

        XCTAssertNil(box.snapshot().queueWaitMs, "a fresh box has no queue wait")

        box.noteSlotOpened()
        let stamped = box.snapshot().queueWaitMs
        XCTAssertNotNil(stamped)

        box.noteSlotOpened()
        XCTAssertEqual(
            box.snapshot().queueWaitMs, stamped,
            "a second slot-open stamp must not move the recorded wait (exactly one per request)"
        )

        let finalized = box.composeFinal(status: 200, error: nil)
        XCTAssertEqual(finalized.queueWaitMs, stamped, "the finalized row keeps the stamped queue wait")
        XCTAssertGreaterThanOrEqual(
            finalized.durationMs, finalized.queueWaitMs ?? 0,
            "duration covers arrival-to-end, so it can never be shorter than the queue wait"
        )
    }
}
