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

    // MARK: Disk reclamation (the one disk-bound test: the in-memory store
    // has no WAL to reclaim, so a temp-directory store proves the truncation)

    private func walFileSize(atPath path: String) -> Int? {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attributes[.size] as? NSNumber else { return nil }
        return size.intValue
    }

    func testClearAndPruneCheckpointWALSoDeletedRowsReleaseDiskSpace() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-store-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let dbPath = directory.appendingPathComponent("requests.sqlite").path
        let walPath = dbPath + "-wal"
        let store = ServerRequestLogStore(path: dbPath)

        func appendRows(_ count: Int) async {
            let body = String(repeating: "x", count: 128 * 1024)
            for index in 0..<count {
                var row = ServerRequestLog()
                row.method = "POST"
                row.path = "/v1/chat/completions"
                row.durationMs = Double(index)
                row.responseBody = body
                store.append(row)
            }
            // count() runs on the store queue, so it drains the pending
            // fire-and-forget appends first.
            _ = await store.count()
        }

        await appendRows(8)
        let walAfterAppends = try XCTUnwrap(walFileSize(atPath: walPath))
        XCTAssertGreaterThan(walAfterAppends, 0, "freshly appended rows live in the WAL before any checkpoint")

        await store.clear()
        let walAfterClear = try XCTUnwrap(walFileSize(atPath: walPath))
        XCTAssertEqual(walAfterClear, 0, "clear must TRUNCATE-checkpoint the WAL so cleared rows release disk space")

        await appendRows(4)
        XCTAssertGreaterThan(try XCTUnwrap(walFileSize(atPath: walPath)), 0, "post-clear appends repopulate the WAL")

        await store.pruneForTesting(now: Date().addingTimeInterval(400 * 86_400))
        XCTAssertEqual(
            try XCTUnwrap(walFileSize(atPath: walPath)), 0,
            "the retention prune must also checkpoint, so mass-expired rows release disk space"
        )
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
}
