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
import Hummingbird
import SQLite3
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

    func testClippingBoundsBodiesByUTF8Bytes() {
        // CJK bodies: one grapheme per ideograph, but three UTF-8 bytes each,
        // so a grapheme-counted cap lets severalfold the byte budget
        // through.
        let cjk = String(repeating: "语", count: ServerRequestLog.maxBodyCaptureLength)
        let clippedCJK = ServerRequestLog.clippedBody(cjk)
        XCTAssertGreaterThan(clippedCJK.utf8.count, ServerRequestLog.maxBodyCaptureLength / 4, "precondition: the byte budget must actually bite here")
        XCTAssertLessThanOrEqual(
            clippedCJK.utf8.count, ServerRequestLog.maxBodyCaptureLength,
            "CJK bodies must be clipped by bytes, not grapheme count"
        )
        XCTAssertGreaterThanOrEqual(
            clippedCJK.utf8.count, ServerRequestLog.maxBodyCaptureLength - 4,
            "the clip must fill the budget with whole scalars, not stop early"
        )
        XCTAssertEqual(
            String(decoding: clippedCJK.data(using: .utf8)!, as: UTF8.self), clippedCJK,
            "the clip must never split a Unicode scalar"
        )
    }

    func testClippingBitesOnCombiningMarkGraphemesThatAGraphemeCapBypasses() {
        // One grapheme built from a huge run of combining marks: a
        // grapheme-counted "cap" lets it pass whole (count 1 <= limit); the
        // byte budget must still apply, scalar-aligned.
        var megaGrapheme = "e"
        megaGrapheme += String(repeating: "\u{0301}", count: ServerRequestLog.maxBodyCaptureLength)
        let clippedBody = ServerRequestLog.clippedBody(megaGrapheme)
        XCTAssertEqual(clippedBody.count, 1, "the clip stays scalar-aligned, so the uncut base-plus-marks grapheme remains one cluster")
        XCTAssertLessThanOrEqual(
            clippedBody.utf8.count, ServerRequestLog.maxBodyCaptureLength,
            "the byte budget must bite even when the whole sample is a single grapheme"
        )

        let clippedArguments = ServerRequestLog.clippedArguments(megaGrapheme)
        XCTAssertLessThanOrEqual(clippedArguments.utf8.count, ServerRequestLog.maxToolArgumentCaptureLength)
        XCTAssertEqual(
            clippedArguments.unicodeScalars.count,
            (ServerRequestLog.maxToolArgumentCaptureLength + 1) / 2,
            "the smaller budget keeps the 1-byte base plus as many whole 2-byte marks as fit"
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

    func testRecordedStatusReadsAnyHTTPResponseError() {
        // The row's status must mirror what the client actually received:
        // Hummingbird's own `HTTPError` renders its own status on the wire, so
        // reading only `ServerAPIError` would record it as a misleading 500.
        XCTAssertEqual(ServerController.recordedStatus(for: HTTPError(.tooManyRequests, message: "slow down")), 429)
        XCTAssertEqual(ServerController.recordedStatus(for: ServerAPIError.requestTimeout("timed out waiting in queue")), 408,
                       "ServerAPIError is itself an HTTPResponseError and keeps its status")
        struct Opaque: Error {}
        XCTAssertEqual(ServerController.recordedStatus(for: Opaque()), 500,
                       "non-wire errors keep the 500 fallback")
    }

    func testReflectedRequestValuesStayOnTheWireAndOutOfRecordedErrors() throws {
        // Wire diagnostics echo the offending value back so the client can
        // debug its own request, but the recorded error must omit it: with
        // body capture off the error message is the only request-derived
        // content a row carries, and `composeLine` prints it to the public
        // unified log. Each reflected-value rejection carries a log-safe
        // variant; everything else records its full client-facing message.
        let roleError = try XCTUnwrap(extractError {
            try AnthropicAdapter.splitMessages([
                AnthropicMessagesRequest.Message(role: "wizard", contentText: "hi")
            ])
        }, "role rejection is a ServerAPIError")
        XCTAssertTrue(roleError.message.contains("wizard"), "the wire error keeps echoing the value for the client")
        XCTAssertFalse(
            ServerRequestLogRecorder.errorMessage(for: roleError).contains("wizard"),
            "the recorded message must omit the echoed request value"
        )

        let openAIToolChoiceError = try XCTUnwrap(extractError {
            try OpenAIAdapter.toolUseEnabled(.function(name: "dangerous_client_name"))
        }, "tool_choice rejection is a ServerAPIError")
        XCTAssertTrue(openAIToolChoiceError.message.contains("dangerous_client_name"))
        XCTAssertFalse(ServerRequestLogRecorder.errorMessage(for: openAIToolChoiceError).contains("dangerous_client_name"))

        let anthropicToolChoiceError = try XCTUnwrap(extractError {
            try AnthropicAdapter.toolUseEnabled(AnthropicMessagesRequest.ToolChoice(type: "tool", name: "exfiltrate"))
        }, "tool_choice rejection is a ServerAPIError")
        XCTAssertTrue(anthropicToolChoiceError.message.contains("exfiltrate"))
        XCTAssertFalse(ServerRequestLogRecorder.errorMessage(for: anthropicToolChoiceError).contains("exfiltrate"))

        // End to end through the box: the finalized row — and therefore the
        // unified-log line — records only the log-safe text.
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "POST", path: "/v1/messages")
        defer { recorder.endBox(requestId: id) }
        do {
            _ = try AnthropicAdapter.splitMessages([
                AnthropicMessagesRequest.Message(role: "sorcerer", contentText: "hi")
            ])
            XCTFail("an invalid role must be rejected")
        } catch {
            let finalized = box.composeFinal(status: 400, error: error)
            XCTAssertNotNil(finalized.errorMessage)
            XCTAssertFalse(finalized.errorMessage!.contains("sorcerer"), "the persisted row must omit the echoed request value")
            XCTAssertFalse(
                ServerRequestLogRecorder.composeLine(finalized).contains("sorcerer"),
                "the unified-log line must print only the log-safe error text"
            )
        }
    }

    /// Runs a throwing body and returns the `ServerAPIError` it must throw
    /// (failing the test when it does not throw or throws another type).
    private func extractError(_ body: () throws -> Any) -> ServerAPIError? {
        do {
            _ = try body()
            XCTFail("rejections must throw")
            return nil
        } catch let apiError as ServerAPIError {
            return apiError
        } catch {
            XCTFail("rejections are ServerAPIError, got \(error)")
            return nil
        }
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

    // MARK: Streaming partials (failure or disconnect after output)

    func testStreamedPartialOutcomeKeepsPartialTextAndTokenCount() {
        UserDefaults.standard.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "POST", path: "/v1/chat/completions")
        defer { recorder.endBox(requestId: id) }

        ServerController.recordStreamedPartialOutcome(
            logBox: box,
            frame: Self.streamedFrame(thinking: "partial thinking", response: "partial answer", tokens: 3),
            tokenCount: 9
        )
        let enriched = box.snapshot()
        XCTAssertEqual(enriched.outputTokens, 9, "the failure's exact token count must win over the frame's")
        XCTAssertEqual(enriched.responseBody, "partial thinking\n\npartial answer")

        UserDefaults.standard.set(false, forKey: ServerRequestLogRecorder.captureBodiesKey)
        ServerController.recordStreamedPartialOutcome(
            logBox: box,
            frame: Self.streamedFrame(thinking: "t", response: "r", tokens: 1),
            tokenCount: 2
        )
        XCTAssertEqual(
            box.snapshot().responseBody, ServerRequestLog.contentWithheldMarker,
            "a partial streamed body follows the capture toggle like any other"
        )
    }

    private static func streamedFrame(thinking: String, response: String, tokens: Int) -> CompletionFrame {
        CompletionFrame(
            rawText: thinking + response,
            thinkingText: thinking,
            responseText: response,
            isActivelyThinking: false,
            tokensGenerated: tokens,
            elapsedMs: 1,
            tokensPerSecond: 1,
            timeToFirstTokenSeconds: nil,
            thinkingDurationSeconds: nil,
            jetSpecActive: false,
            jetSpecMeanTau: 0,
            jetSpecDraftAccepted: 0,
            jetSpecDraftProposed: 0,
            residentMemoryGB: 0,
            residentExpertCount: 0,
            totalExpertCount: 0,
            cacheHitRatePercent: 0,
            prefetchEfficiencyPercent: 0,
            pagingLatencyMs: 0
        )
    }

    func testFailedNonStreamingTurnKeepsGeneratedTokenCount() {
        // The non-streaming pipeline throws on `.failedTurn` before any
        // success-path stamping (`recordTurnOutcome` runs only on success),
        // so the token count generated before the failure must be stamped
        // on the box up front — the mirror of `recordStreamedPartialOutcome`
        // for a path that never held a frame.
        let recorder = ServerRequestLogRecorder.shared
        let id = "test-\(UUID().uuidString)"
        let box = recorder.beginBox(requestId: id, method: "POST", path: "/v1/chat/completions")
        defer { recorder.endBox(requestId: id) }

        let failure = TurnFailure(
            kind: .forwardPassFailed,
            message: "inference forward pass failed after output",
            cancelled: false,
            tokensGenerated: 17
        )
        ServerController.recordFailedTurnTokenCount(logBox: box, failure: failure)
        let finalized = box.composeFinal(status: 500, error: ServerAPIError.internalError(failure.message))
        XCTAssertEqual(finalized.outputTokens, 17, "a failure after output must keep its generated-token count")
        XCTAssertEqual(finalized.errorMessage, "inference forward pass failed after output")

        // Cancelled failures carry their count too, and a missing box (no
        // logging on the turn) must be tolerated.
        let cancelled = TurnFailure(kind: .cancelled, message: "stopped", cancelled: true, tokensGenerated: 4)
        ServerController.recordFailedTurnTokenCount(logBox: box, failure: cancelled)
        XCTAssertEqual(box.snapshot().outputTokens, 4, "a cancellation after output keeps its token count")
        ServerController.recordFailedTurnTokenCount(logBox: nil, failure: cancelled)
    }

    // MARK: Disk reclamation (the disk-bound tests: the in-memory store has
    // no files to reclaim, so temp-directory stores prove the truncation)

    private func fileSize(atPath path: String) -> Int? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attributes[.size] as? NSNumber else { return nil }
        return size.intValue
    }

    func testClearAndMassPruneReleaseWALAndDatabaseFileSpace() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-store-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let dbPath = directory.appendingPathComponent("requests.sqlite").path
        let walPath = dbPath + "-wal"
        let store = ServerRequestLogStore(path: dbPath)

        func combinedFootprint() -> Int {
            (fileSize(atPath: walPath) ?? 0) + (fileSize(atPath: dbPath) ?? 0)
        }

        func appendRows(_ count: Int, bodyBytes: Int) async {
            let body = String(repeating: "x", count: bodyBytes)
            for _ in 0..<count {
                var row = ServerRequestLog()
                row.method = "POST"
                row.path = "/v1/chat/completions"
                row.responseBody = body
                store.append(row)
            }
            // count() runs on the store queue, so it drains the pending
            // fire-and-forget appends first.
            _ = await store.count()
        }

        await appendRows(8, bodyBytes: 128 * 1024)
        let walAfterAppends = try XCTUnwrap(fileSize(atPath: walPath))
        XCTAssertGreaterThan(walAfterAppends, 0, "freshly appended rows live in the WAL before any checkpoint")
        let peak = combinedFootprint()

        await store.clear()
        let walAfterClear = try XCTUnwrap(fileSize(atPath: walPath))
        XCTAssertEqual(walAfterClear, 0, "clear must TRUNCATE-checkpoint the WAL so cleared rows release disk space")
        XCTAssertLessThanOrEqual(
            combinedFootprint(), peak / 4,
            "clear must also VACUUM the database file: DELETE alone leaves the pages on the freelist, so the main file keeps its peak footprint forever"
        )

        // A retention pass below the compaction threshold compacts nothing;
        // one at or above it must.
        let compactionRows = ServerRequestLogStore.compactAfterRowsDeleted + 1
        await appendRows(compactionRows, bodyBytes: 2_048)
        XCTAssertGreaterThan(combinedFootprint(), peak / 4, "post-clear appends repopulate the database")

        await store.pruneForTesting(now: Date().addingTimeInterval(400 * 86_400))
        XCTAssertEqual(
            try XCTUnwrap(fileSize(atPath: walPath)), 0,
            "the mass retention prune must checkpoint, so expired rows release WAL space"
        )
        XCTAssertLessThanOrEqual(
            combinedFootprint(), peak / 4,
            "a mass prune must VACUUM so expired rows leave the database file, not just its freelist"
        )
    }

    func testSchemaVersionSurvivesStoreOpens() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-schema-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let dbPath = directory.appendingPathComponent("requests.sqlite").path

        let first = ServerRequestLogStore(path: dbPath)
        var opener = ServerRequestLog()
        opener.method = "GET"
        opener.path = "/v1/models"
        first.append(opener)
        _ = await first.count()
        let firstVersion = await first.schemaVersionForTesting()
        XCTAssertEqual(firstVersion, 1, "after the schema pass the file must read back user_version = 1")

        let second = ServerRequestLogStore(path: dbPath)
        second.append(opener)
        _ = await second.count()
        let secondVersion = await second.schemaVersionForTesting()
        XCTAssertEqual(
            secondVersion, 1,
            "a fresh open of an already-migrated file must read the persisted schema version instead of re-running the schema pass"
        )
    }

    func testSchemaRejectionNeverWritesToOrPrunesANewerDatabase() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-schema-reject-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let dbPath = directory.appendingPathComponent("requests.sqlite").path

        // Seed the file as a newer build would have left it: a row this app
        // has no business touching, stamped with a schema version beyond
        // what it understands.
        var newerDB: OpaquePointer?
        XCTAssertEqual(sqlite3_open(dbPath, &newerDB), SQLITE_OK)
        var statement: OpaquePointer?
        func rawExec(_ sql: String) {
            XCTAssertEqual(sqlite3_exec(newerDB, sql, nil, nil, nil), SQLITE_OK, "raw exec failed: \(sql)")
        }
        rawExec("""
            CREATE TABLE requests (
                id TEXT PRIMARY KEY, ts_ms INTEGER NOT NULL, path TEXT NOT NULL,
                http_status INTEGER NOT NULL, is_error INTEGER NOT NULL,
                was_streaming INTEGER NOT NULL, model TEXT, payload TEXT NOT NULL
            )
            """)
        rawExec("INSERT INTO requests VALUES ('seeded-by-newer-build', 1, '/v1/models', 200, 0, 0, NULL, '{}')")
        rawExec("PRAGMA user_version = 2")

        func rawRowCount() -> Int {
            XCTAssertEqual(sqlite3_prepare_v2(newerDB, "SELECT COUNT(*) FROM requests", -1, &statement, nil), SQLITE_OK)
            XCTAssertEqual(sqlite3_step(statement), SQLITE_ROW)
            let count = Int(sqlite3_column_int64(statement, 0))
            sqlite3_finalize(statement)
            statement = nil
            return count
        }
        XCTAssertEqual(rawRowCount(), 1)

        // Every operation against the rejected database must no-op: no
        // appends, no reads, and — the dangerous one — no clear or retention
        // pass may delete the newer build's rows.
        let store = ServerRequestLogStore(path: dbPath)
        var rejectedRow = ServerRequestLog()
        rejectedRow.method = "POST"
        rejectedRow.path = "/v1/chat/completions"
        store.append(rejectedRow)
        let fetched = await store.fetch(limit: 10)
        XCTAssertTrue(fetched.isEmpty, "a rejected database must never serve rows to this build")
        let visibleCount = await store.count()
        XCTAssertEqual(visibleCount, 0)

        await store.clear()
        await store.pruneForTesting(now: Date().addingTimeInterval(400 * 86_400))
        XCTAssertEqual(
            rawRowCount(), 1,
            "neither clear nor retention may delete rows out of a schema the store rejected"
        )

        // Export must fail loudly rather than report a 0-row success.
        let exportURL = directory.appendingPathComponent("export.jsonl")
        do {
            _ = try await store.exportJSONL(to: exportURL)
            XCTFail("export from a rejected database must throw, not report a 0-row success")
        } catch {}
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: exportURL.path),
            "a failed export must not leave a file behind"
        )

        sqlite3_close(newerDB)
    }

    // MARK: JSONL export

    private func makeExportStore() throws -> (store: ServerRequestLogStore, directory: URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-export-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let store = ServerRequestLogStore(path: directory.appendingPathComponent("requests.sqlite").path)
        return (store, directory)
    }

    private func decodedExportLines(_ url: URL) throws -> [ServerRequestLog] {
        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(separator: "\n").filter { !$0.isEmpty }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        return try lines.map { try decoder.decode(ServerRequestLog.self, from: Data($0.utf8)) }
    }

    func testExportJSONLWritesFilteredRowsNewestFirst() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Distinct, spread timestamps so the newest-first order is asserted
        // on real values instead of rowid tie-breaking.
        var firstRow = ServerRequestLog()
        firstRow.method = "POST"
        firstRow.path = "/v1/chat/completions"
        firstRow.model = "ling-3.0-tiny"
        firstRow.timestamp = Date(timeIntervalSinceNow: -3_600)
        var secondRow = ServerRequestLog()
        secondRow.method = "POST"
        secondRow.path = "/v1/messages"
        secondRow.httpStatus = 400
        secondRow.timestamp = Date(timeIntervalSinceNow: -1_800)
        var thirdRow = ServerRequestLog()
        thirdRow.method = "GET"
        thirdRow.path = "/v1/models"
        thirdRow.timestamp = Date(timeIntervalSinceNow: -60)
        for row in [firstRow, secondRow, thirdRow] {
            store.append(row)
            _ = await store.count()
        }

        let url = directory.appendingPathComponent("all.jsonl")
        let exported = try await store.exportJSONL(to: url)
        XCTAssertEqual(exported, 3, "an empty filter exports every stored row")
        XCTAssertEqual(
            try decodedExportLines(url).map(\.path),
            ["/v1/models", "/v1/messages", "/v1/chat/completions"],
            "exports preserve the dashboard's newest-first order"
        )
        XCTAssertEqual(try decodedExportLines(url).map(\.id), [thirdRow.id, secondRow.id, firstRow.id],
                      "each line must round-trip to the very row that was stored")

        var errorsFilter = ServerRequestLogStore.Filter()
        errorsFilter.errorsOnly = true
        let errorURL = directory.appendingPathComponent("errors.jsonl")
        let errorExported = try await store.exportJSONL(filter: errorsFilter, to: errorURL)
        XCTAssertEqual(errorExported, 1, "the Errors Only switch must scope the export")
        XCTAssertEqual(try decodedExportLines(errorURL).first?.id, secondRow.id)

        var textFilter = ServerRequestLogStore.Filter()
        textFilter.text = "ling-3.0-tiny"
        let textURL = directory.appendingPathComponent("search.jsonl")
        let textExported = try await store.exportJSONL(filter: textFilter, to: textURL)
        XCTAssertEqual(textExported, 1, "the search box must scope the export too")
        XCTAssertEqual(try decodedExportLines(textURL).first?.id, firstRow.id)
    }

    func testExportJSONLIsOneJSONObjectPerLine() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/chat/completions"
        row.requestBody = "multi\nline content that must stay inside one JSON string"
        store.append(row)
        for _ in 0..<2 {
            var filler = ServerRequestLog()
            filler.method = "GET"
            filler.path = "/v1/models"
            store.append(filler)
        }
        _ = await store.count()

        let url = directory.appendingPathComponent("lines.jsonl")
        let exported = try await store.exportJSONL(to: url)
        XCTAssertEqual(exported, 3)

        let text = try String(contentsOf: url, encoding: .utf8)
        let lines = text.split(separator: "\n", omittingEmptySubsequences: true)
        XCTAssertEqual(lines.count, 3, "JSONL must stay strictly one object per line even when bodies contain newlines")
        XCTAssertEqual(try decodedExportLines(url).count, 3)
        XCTAssertEqual(
            try decodedExportLines(url).first { $0.path == "/v1/chat/completions" }?.requestBody,
            "multi\nline content that must stay inside one JSON string",
            "bodies must survive the export byte-exact"
        )
    }

    func testExportJSONLThrowsWhenTheDatabaseCannotOpen() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-export-fail-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        // The user's previous export must not silently read as "Exported 0
        // rows": a failed export reports failure and leaves it untouched.
        let previousExport = "the user's last good export must survive a failed one"
        let url = directory.appendingPathComponent("requests.jsonl")
        try previousExport.write(to: url, atomically: true, encoding: .utf8)

        // The database's parent directory does not exist, so the store cannot
        // open it.
        let store = ServerRequestLogStore(
            path: directory
                .appendingPathComponent("missing-subdir")
                .appendingPathComponent("requests.sqlite")
                .path
        )
        do {
            _ = try await store.exportJSONL(to: url)
            XCTFail("an unopenable database must fail the export, not report a 0-row success")
        } catch {
            // Expected: the dashboard's error handler reports this path.
        }
        XCTAssertEqual(
            try String(contentsOf: url, encoding: .utf8),
            previousExport,
            "a failed export must leave the previous export file untouched"
        )
    }

    func testExportJSONLAtomicallyReplacesAnExistingExport() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        func appendRow(path: String) async {
            var row = ServerRequestLog()
            row.method = "POST"
            row.path = path
            store.append(row)
            _ = await store.count()
        }

        let url = directory.appendingPathComponent("requests.jsonl")
        await appendRow(path: "/v1/models")
        let firstExport = try await store.exportJSONL(to: url)
        XCTAssertEqual(firstExport, 1)
        XCTAssertEqual(try decodedExportLines(url).map(\.path), ["/v1/models"])

        // Re-exporting over the same destination must replace it via one
        // atomic rename — no delete-first window, no stale bytes, and no
        // staging file left behind.
        await appendRow(path: "/v1/messages")
        let secondExport = try await store.exportJSONL(to: url)
        XCTAssertEqual(secondExport, 2)
        XCTAssertEqual(try decodedExportLines(url).map(\.path).count, 2, "the second export must fully replace the first")
        XCTAssertTrue(try decodedExportLines(url).contains { $0.path == "/v1/messages" })
        XCTAssertTrue(leftoverStagingFiles(in: directory).isEmpty, "a successful export must not leave staging files behind")
    }

    func testExportJSONLFailingCommitLeavesDestinationAndDirectoryClean() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/messages"
        store.append(row)
        _ = await store.count()

        // A destination that is a directory cannot be replaced by an atomic
        // rename, so the commit fails — loudly, with the destination intact
        // and no partial export left standing. The old delete-first flow
        // would have removed the directory outright before failing.
        let destination = directory.appendingPathComponent("requests.jsonl", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: false)
        let sentinel = destination.appendingPathComponent("user-notes.txt")
        try "user data the export must never destroy".write(to: sentinel, atomically: true, encoding: .utf8)

        do {
            _ = try await store.exportJSONL(to: destination)
            XCTFail("a commit onto an un-replaceable destination must throw")
        } catch {
            // Expected: the dashboard's error handler reports the failure.
        }
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: sentinel.path),
            "a failed commit must leave the destination untouched"
        )
        XCTAssertTrue(leftoverStagingFiles(in: directory).isEmpty, "a failed commit must clean up its staging file")
    }

    private func leftoverStagingFiles(in directory: URL) -> [String] {
        (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?
            .filter { $0.contains("-export.tmp") } ?? []
    }

    // MARK: Change observation (the viewer listens instead of polling)

    private func waitUntil(timeout: TimeInterval = 2, _ condition: @escaping @Sendable () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await condition() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Condition not met within \(timeout)s")
    }

    /// Lock-guarded counter for cross-task signal assertions (a plain
    /// captured var would race between the consumer task and the test task).
    private final class SignalCountBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func increment() { lock.lock(); value += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return value }
    }

    func testChangeSignalFiresOnAppendAndClearButNotWhenIdle() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Open the store (and its one-time open prune) before subscribing so
        // every signal from here on maps 1:1 to an append or a clear.
        var opener = ServerRequestLog()
        opener.method = "GET"
        opener.path = "/v1/models"
        store.append(opener)
        _ = await store.count()

        let signals = SignalCountBox()
        let stream = store.changes()
        let consumer = Task {
            for await _ in stream { signals.increment() }
        }
        defer { consumer.cancel() }

        var row = ServerRequestLog()
        row.method = "POST"
        row.path = "/v1/chat/completions"
        store.append(row)
        _ = await store.count()
        try await waitUntil { signals.count >= 1 }

        await store.clear()
        try await waitUntil { signals.count >= 2 }

        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertEqual(
            signals.count, 2,
            "idle time must not signal: the viewer refetches only when the log actually changed"
        )
    }

    func testChangeSignalTerminationUnregistersTheObserver() async throws {
        let (store, directory) = try makeExportStore()
        defer { try? FileManager.default.removeItem(at: directory) }

        // Same open-before-subscribe ordering as above.
        var opener = ServerRequestLog()
        opener.method = "GET"
        opener.path = "/v1/models"
        store.append(opener)
        _ = await store.count()

        let signals = SignalCountBox()
        let stream = store.changes()
        let consumer = Task {
            for await _ in stream { signals.increment() }
        }

        var row = ServerRequestLog()
        row.method = "GET"
        row.path = "/v1/models"
        store.append(row)
        _ = await store.count()
        try await waitUntil { signals.count >= 1 }

        consumer.cancel()
        try await Task.sleep(nanoseconds: 100_000_000)

        store.append(row)
        _ = await store.count()
        try await Task.sleep(nanoseconds: 250_000_000)
        XCTAssertEqual(signals.count, 1, "a terminated subscription must hear nothing further")
    }

    // MARK: In-app (chatUI) rows

    func testLegacyRowsDecodeWithNoSource() throws {
        let legacyJSON = """
        {"id":"11111111-2222-3333-4444-555555555555","timestamp":1000,"method":"POST","path":"/v1/chat/completions","proto":"openai","wasStreaming":false,"httpStatus":200,"durationMs":5}
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        let row = try decoder.decode(ServerRequestLog.self, from: Data(legacyJSON.utf8))
        XCTAssertEqual(row.httpStatus, 200)
        XCTAssertNil(row.source, "rows persisted before the source split must still decode, as legacy server rows")
    }

    func testSourceRoundTripsThroughThePersistedPayload() throws {
        var row = ServerRequestLog()
        row.method = "TURN"
        row.path = "chat"
        row.source = .chatUI
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .millisecondsSince1970
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .millisecondsSince1970
        let round = try decoder.decode(ServerRequestLog.self, from: try encoder.encode(row))
        XCTAssertEqual(round.source, .chatUI)
    }

    func testInAppTurnRowWithholdsContentByDefaultAndRecordsExactlyOnce() async throws {
        let store = ServerRequestLogStore.inMemory()
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        defer { ServerRequestLogRecorder.shared.useOverrideStore(nil) }

        let box = ServerRequestLog.chatTurnLog(
            model: "ling-3.0-tiny",
            inputTokens: 64,
            isAgentTurn: true,
            prompt: "PROMPT SECRET"
        )
        let staged = box.snapshot()
        XCTAssertEqual(staged.source, .chatUI)
        XCTAssertEqual(staged.method, "TURN")
        XCTAssertEqual(staged.path, "agent", "an agent turn must be distinguishable from a plain chat turn")
        XCTAssertEqual(staged.model, "ling-3.0-tiny")
        XCTAssertEqual(staged.inputTokens, 64)
        XCTAssertEqual(
            staged.requestBody,
            ServerRequestLog.contentWithheldMarker,
            "capture is off: the row must say withheld, not silence"
        )

        box.noteFirstToken()
        try await Task.sleep(nanoseconds: 30_000_000)
        box.recordChatTurn(
            status: 200,
            errorMessage: nil,
            outputTokens: 32,
            finishReason: "toolFreeze",
            thinking: "THINK SECRET",
            response: "ANSWER SECRET",
            toolCalls: Self.parsedCalls()
        )
        let rows = await store.fetch()
        XCTAssertEqual(rows.count, 1, "the in-app turn must land in the same store as server rows")
        let recorded = try XCTUnwrap(rows.first)
        XCTAssertEqual(recorded.source, .chatUI)
        XCTAssertEqual(recorded.outputTokens, 32)
        XCTAssertEqual(recorded.finishReason, "toolFreeze")
        XCTAssertEqual(
            recorded.toolCalls?.map(\.name),
            ["shell_run", "file_read"],
            "tool names are metadata and are recorded with capture off"
        )
        XCTAssertEqual(
            recorded.toolCalls?.map { $0.arguments == ServerRequestLog.contentWithheldMarker },
            [true, true],
            "in-app tool arguments are content and follow the capture toggle"
        )
        XCTAssertEqual(recorded.responseBody, ServerRequestLog.contentWithheldMarker)
        XCTAssertFalse(recorded.hasCapturedContent, "with capture off the row stores no content")
        XCTAssertFalse(
            ServerRequestLogRecorder.composeLine(recorded).contains("SECRET"),
            "bodies never reach the unified-log line, capture or not"
        )
        XCTAssertFalse(recorded.isError)
        XCTAssertNotNil(recorded.firstTokenMs)
        XCTAssertLessThanOrEqual(recorded.firstTokenMs ?? 0, recorded.durationMs)

        // The boxed claim makes recording one-shot: a late duplicate call
        // from the generation loop cannot write a second row.
        box.recordChatTurn(
            status: 500,
            errorMessage: "duplicate terminal event",
            outputTokens: 1,
            finishReason: nil,
            thinking: "",
            response: "",
            toolCalls: []
        )
        let countAfterDuplicate = await store.count()
        XCTAssertEqual(countAfterDuplicate, 1, "recordChatTurn must be one-shot per turn")
    }

    func testInAppTurnRowCapturesPromptResponseAndToolArgumentsWhenEnabled() async throws {
        UserDefaults.standard.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        let store = ServerRequestLogStore.inMemory()
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        defer { ServerRequestLogRecorder.shared.useOverrideStore(nil) }

        let box = ServerRequestLog.chatTurnLog(
            model: "ling-3.0-tiny",
            inputTokens: 64,
            isAgentTurn: false,
            prompt: "USER: PROMPT SECRET"
        )
        box.recordChatTurn(
            status: 200,
            errorMessage: nil,
            outputTokens: 12,
            finishReason: "eos",
            thinking: "THINK SECRET",
            response: "ANSWER SECRET",
            toolCalls: Self.parsedCalls()
        )
        let rows = await store.fetch()
        let recorded = try XCTUnwrap(rows.first)
        XCTAssertEqual(
            recorded.requestBody,
            "USER: PROMPT SECRET",
            "the in-app request body is the rendered prompt the model actually saw"
        )
        XCTAssertEqual(
            recorded.responseBody,
            "THINK SECRET\n\nANSWER SECRET",
            "reasoning and answer travel together, as they do on server rows"
        )
        XCTAssertEqual(recorded.toolCalls?.first?.arguments, #"{"command":"ls"}"#)
        XCTAssertTrue(recorded.hasCapturedContent)
        XCTAssertFalse(
            ServerRequestLogRecorder.composeLine(recorded).contains("SECRET"),
            "captured bodies stay in the persisted row only, never in the unified log"
        )
    }

    func testCapturedBodiesDropEmptyHalvesAndKeepTheWithheldMarkerOff() {
        UserDefaults.standard.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        XCTAssertEqual(ServerRequestLog.capturedResponseBody(thinking: "", response: "answer"), "answer")
        XCTAssertEqual(
            ServerRequestLog.capturedResponseBody(thinking: "reasoning", response: ""),
            "reasoning",
            "a reasoning-only turn must not leave a dangling separator"
        )
        XCTAssertNil(
            ServerRequestLog.capturedResponseBody(thinking: "", response: ""),
            "a turn that generated nothing has no body to store, even with capture on"
        )
        XCTAssertEqual(ServerRequestLog.capturedRequestBody("prompt"), "prompt")

        UserDefaults.standard.set(false, forKey: ServerRequestLogRecorder.captureBodiesKey)
        XCTAssertEqual(
            ServerRequestLog.capturedResponseBody(thinking: "", response: ""),
            ServerRequestLog.contentWithheldMarker,
            "with capture off the row says withheld, so a silent failure reads as 'not stored', not 'empty'"
        )
        XCTAssertEqual(ServerRequestLog.capturedRequestBody("prompt"), ServerRequestLog.contentWithheldMarker)
        XCTAssertNil(
            ServerRequestLog.capturedRequestBody(nil),
            "a caller with no prompt records nil rather than a marker"
        )
    }

    func testFailedInAppTurnRecordsErrorWhileCancelledRecordsInfo() async throws {
        let store = ServerRequestLogStore.inMemory()
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        defer { ServerRequestLogRecorder.shared.useOverrideStore(nil) }

        let failedBox = ServerRequestLog.chatTurnLog(model: "ling", inputTokens: 10, isAgentTurn: false, prompt: "p")
        failedBox.recordChatTurn(
            status: 500,
            errorMessage: "prefill failed: out of memory",
            outputTokens: 0,
            finishReason: nil,
            thinking: "",
            response: "",
            toolCalls: []
        )
        let cancelledBox = ServerRequestLog.chatTurnLog(model: "ling", inputTokens: 10, isAgentTurn: false, prompt: "p")
        cancelledBox.recordChatTurn(
            status: 200,
            errorMessage: "Generation cancelled",
            outputTokens: 7,
            finishReason: nil,
            thinking: "partial thinking",
            response: "partial answer",
            toolCalls: []
        )
        let rows = await store.fetch()
        XCTAssertEqual(rows.count, 2)
        let failed = try XCTUnwrap(rows.first { $0.errorMessage == "prefill failed: out of memory" })
        XCTAssertTrue(failed.isError, "a failed in-app turn is an error row")
        let cancelled = try XCTUnwrap(rows.first { $0.errorMessage == "Generation cancelled" })
        XCTAssertFalse(cancelled.isError, "a cancelled turn is a user action, not an error")
        XCTAssertEqual(cancelled.outputTokens, 7, "the partial generation must still be visible in the row")
    }

    func testAppleFoundationModelTurnRowsCoverTheFullLifecycle() async throws {
        // `startAppleFoundationModelGeneration` stages and records its rows
        // exactly this way — the same in-app contract as a weights turn.
        // On-device specifics: approximate token counts (the macOS 26
        // streaming API reports none), no separate reasoning channel, no
        // tools, and a request body of the instructions + user message the
        // framework actually sees.
        UserDefaults.standard.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        let store = ServerRequestLogStore.inMemory()
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        defer { ServerRequestLogRecorder.shared.useOverrideStore(nil) }

        func afmBox() -> ServerRequestLogBox {
            ServerRequestLog.chatTurnLog(
                model: AppleFoundationModelService.displayName,
                inputTokens: AppleFoundationModelService.approximateTokenCount(for: "persona and history instructions")
                    + AppleFoundationModelService.approximateTokenCount(for: "hello"),
                isAgentTurn: false,
                prompt: "persona and history instructions\n\nhello"
            )
        }

        // Success: the final text and its approximate token count.
        let finishedBox = afmBox()
        finishedBox.noteFirstToken()
        finishedBox.recordChatTurn(
            status: 200,
            errorMessage: nil,
            outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: "final on-device answer"),
            finishReason: nil,
            thinking: "",
            response: "final on-device answer",
            toolCalls: []
        )

        // Stop: informational (200) with the partial text, never an error row.
        let stoppedBox = afmBox()
        stoppedBox.noteFirstToken()
        stoppedBox.recordChatTurn(
            status: 200,
            errorMessage: "⏹ Generation stopped by user.",
            outputTokens: AppleFoundationModelService.approximateTokenCount(for: "partial answer"),
            finishReason: nil,
            thinking: "",
            response: "partial answer",
            toolCalls: []
        )

        // Failure before the first partial: the displayed error, no body, and a
        // genuine zero token count (the empty-text guard, not the UI floor).
        let failedBox = afmBox()
        failedBox.recordChatTurn(
            status: 500,
            errorMessage: "⚠️ Apple Foundation Model error: generation failed",
            outputTokens: AppleFoundationModelService.recordedOutputTokenCount(for: ""),
            finishReason: nil,
            thinking: "",
            response: "",
            toolCalls: []
        )

        let rows = await store.fetch()
        XCTAssertEqual(rows.count, 3)
        for row in rows {
            XCTAssertEqual(row.source, .chatUI)
            XCTAssertEqual(row.method, "TURN")
            XCTAssertEqual(row.path, "chat", "on-device turns are plain chat turns")
            XCTAssertEqual(row.model, AppleFoundationModelService.displayName)
            XCTAssertEqual(
                row.inputTokens,
                AppleFoundationModelService.approximateTokenCount(for: "persona and history instructions")
                    + AppleFoundationModelService.approximateTokenCount(for: "hello"),
                "input tokens stay the framework-agnostic approximation"
            )
        }
        XCTAssertEqual(
            rows.map { $0.requestBody },
            [String](repeating: "persona and history instructions\n\nhello", count: 3),
            "the request body is what the framework actually saw: instructions plus the user message"
        )

        let finished = try XCTUnwrap(rows.first { $0.errorMessage == nil })
        XCTAssertFalse(finished.isError)
        XCTAssertEqual(finished.responseBody, "final on-device answer")
        XCTAssertEqual(finished.outputTokens, AppleFoundationModelService.approximateTokenCount(for: "final on-device answer"))
        XCTAssertEqual(finished.finishReason, nil, "the streaming API reports no stop reason")
        XCTAssertNil(finished.toolCalls, "plain chat records no tool calls — not an empty list")
        XCTAssertNotNil(finished.firstTokenMs, "the first partial must stamp the row")

        let stopped = try XCTUnwrap(rows.first { $0.errorMessage == "⏹ Generation stopped by user." })
        XCTAssertFalse(stopped.isError, "a stop is a user action, matching the weights path's cancelled contract")
        XCTAssertEqual(stopped.responseBody, "partial answer")
        XCTAssertEqual(stopped.outputTokens, AppleFoundationModelService.approximateTokenCount(for: "partial answer"))

        let failed = try XCTUnwrap(rows.first { $0.errorMessage == "⚠️ Apple Foundation Model error: generation failed" })
        XCTAssertEqual(failed.httpStatus, 500)
        XCTAssertTrue(failed.isError, "a failed on-device turn is an error row")
        XCTAssertEqual(failed.outputTokens, 0, "an empty response must not record a phantom output token")
        XCTAssertNil(failed.responseBody, "a failure before the first partial has nothing to capture")
        XCTAssertNil(failed.firstTokenMs)
    }

    /// Two parsed calls whose arguments are recognizable content: the capture
    /// tests assert those strings appear only behind the toggle.
    private static func parsedCalls() -> [ParsedToolCall] {
        [
            ParsedToolCall(
                name: "shell_run",
                arguments: ["command": "ls"],
                rawArguments: #"{"command":"ls"}"#,
                rawText: "raw1"
            ),
            ParsedToolCall(
                name: "file_read",
                arguments: ["path": "/tmp/secret"],
                rawArguments: #"{"path":"/tmp/secret"}"#,
                rawText: "raw2"
            ),
        ]
    }
}
