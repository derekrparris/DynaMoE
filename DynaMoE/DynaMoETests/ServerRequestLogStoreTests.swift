//
//  ServerRequestLogStoreTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the persisted request-log store: append/fetch
//  round-trips through the JSON payload, filtering over the indexed
//  columns, clear, and both retention policies (age and count). Runs
//  against an in-memory SQLite store, so it needs no server, no model,
//  and never touches the developer's on-disk request log.
//

import XCTest
@testable import DynaMoE

final class ServerRequestLogStoreTests: XCTestCase {

    private var store: ServerRequestLogStore!

    override func setUp() {
        super.setUp()
        store = ServerRequestLogStore.inMemory()
    }

    override func tearDown() {
        store = nil
        super.tearDown()
    }

    private func row(
        id: UUID = UUID(),
        path: String = "/v1/chat/completions",
        status: Int = 200,
        streaming: Bool = false,
        model: String? = "test-model",
        timestamp: Date = Date(),
        error: String? = nil
    ) -> ServerRequestLog {
        var row = ServerRequestLog()
        row.id = id
        row.path = path
        row.httpStatus = status
        row.wasStreaming = streaming
        row.model = model
        row.timestamp = timestamp
        row.errorMessage = error
        return row
    }

    // MARK: Round-trip

    func testAppendThenFetchPreservesRow() async {
        var original = row()
        original.method = "POST"
        original.proto = .openai
        original.durationMs = 1234.5
        original.inputTokens = 512
        original.outputTokens = 89
        original.temperature = 0.7
        original.maxTokens = 4096
        original.finishReason = "stop"
        original.toolCalls = [.init(name: "get_weather", arguments: "{\"city\":\"sf\"}")]
        original.requestBytes = 1024
        original.requestBody = "{\"messages\":[]}"
        original.responseBody = "hello there"
        original.firstTokenMs = 250
        original.wasStreaming = true
        original.streamTermination = .completed
        store.append(original)

        let fetched = await store.fetch(limit: 10)
        XCTAssertEqual(fetched.count, 1)
        XCTAssertEqual(fetched.first?.id, original.id)
        XCTAssertEqual(fetched.first?.toolCalls, original.toolCalls)
        XCTAssertEqual(fetched.first?.requestBody, original.requestBody)
        XCTAssertEqual(fetched.first?.responseBody, original.responseBody)
        XCTAssertEqual(fetched.first?.inputTokens, 512)
        XCTAssertEqual(fetched.first?.durationMs ?? 0, original.durationMs, accuracy: 0.001)
    }

    func testFetchOrdersNewestFirst() async {
        let base = Date(timeIntervalSince1970: 1_700_000_000)
        for offset in 0..<5 {
            var older = row(timestamp: base.addingTimeInterval(TimeInterval(offset)))
            older.model = "model-\(offset)"
            store.append(older)
        }
        let fetched = await store.fetch(limit: 10)
        XCTAssertEqual(fetched.map(\.model), ["model-4", "model-3", "model-2", "model-1", "model-0"])
    }

    // MARK: Filters

    func testErrorsOnlyFilterKeepsErrorRows() async {
        store.append(row(status: 200))
        store.append(row(status: 503, error: "No model is loaded"))
        store.append(row(status: 401, error: "Invalid or missing API key"))

        var filter = ServerRequestLogStore.Filter()
        filter.errorsOnly = true
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.count, 2)
        XCTAssertTrue(fetched.allSatisfy(\.isError))
    }

    func testStreamErrorFrameCountsAsError() async {
        var streamRow = row(status: 200, streaming: true)
        streamRow.streamTermination = .errorFrame
        streamRow.errorMessage = "Request timed out waiting in the generation queue"
        store.append(streamRow)
        store.append(row(status: 200, streaming: true)) // completed stream, not an error

        var filter = ServerRequestLogStore.Filter()
        filter.errorsOnly = true
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.count, 1)
        XCTAssertEqual(fetched.first?.streamTermination, .errorFrame)
    }

    func testStreamingOnlyFilter() async {
        store.append(row(streaming: false))
        store.append(row(streaming: true))

        var filter = ServerRequestLogStore.Filter()
        filter.streamingOnly = true
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.count, 1)
        XCTAssertEqual(fetched.first?.wasStreaming, true)
    }

    func testPathFilter() async {
        store.append(row(path: "/v1/chat/completions"))
        store.append(row(path: "/v1/messages"))

        var filter = ServerRequestLogStore.Filter()
        filter.path = "/v1/messages"
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.map(\.path), ["/v1/messages"])
    }

    func testTextSearchMatchesPayloadAndEscapesWildcards() async {
        var rowWithBody = row()
        rowWithBody.responseBody = "the quick brown fox"
        var errorRow = row()
        errorRow.errorMessage = "prompt tokenization produced 0 tokens"
        store.append(rowWithBody)
        store.append(errorRow)
        store.append(row())

        var filter = ServerRequestLogStore.Filter()
        filter.text = "brown fox"
        var fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.count, 1)
        XCTAssertEqual(fetched.first?.responseBody, "the quick brown fox")

        // A search term consisting of a bare wildcard must not match
        // everything (LIKE's % would).
        filter.text = "%"
        fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertTrue(fetched.isEmpty)

        filter.text = "tokenization"
        fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(fetched.first?.errorMessage, "prompt tokenization produced 0 tokens")
    }

    func testTextSearchMatchesStoredJSONEncodingOfDisplayText() async {
        // The payload column is JSON, so a term matches only in its encoded
        // form: the displayed error `invalid "name"` persists as
        // `invalid \"name\"` inside the row. Searching what the dashboard
        // displays must find the row.
        var quotedRow = row(error: #"invalid "name""#)
        quotedRow.model = "quoted-model"
        store.append(quotedRow)
        store.append(row(error: "plain failure"))

        var filter = ServerRequestLogStore.Filter()
        filter.text = #"invalid "name""#
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(
            fetched.map(\.model),
            ["quoted-model"],
            "a search term must match its JSON-encoded form in the payload"
        )

        // Plain terms still match plainly-encoded fields.
        filter.text = "plain failure"
        let plainFetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(plainFetched.first?.errorMessage, "plain failure")
    }

    func testTextSearchMatchesPromptTextInsideCapturedRequestBodies() async throws {
        // Captured API request bodies embed one more JSON layer inside the
        // row's payload: `openAISummary`/`anthropicSummary` serialize the
        // prompt itself, and the row encoder then embeds that serialized
        // summary as a string value. Displayed prompt text such as
        // `invalid "name"` persists double-escaped there, so the search must
        // match its twice-encoded form to find those rows.
        // Raw literals (`#"..."#`): the JSON itself must carry the backslash
        // escapes, so `\"` in the fixture must reach the decoder literally.
        let openAIJSON = #"""
        {"model":"ling","messages":[{"role":"user","content":"error reads: invalid \"name\""}]}
        """#
        let openAIRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(openAIJSON.utf8))
        var openAIRow = row()
        openAIRow.model = "openai-summary-row"
        openAIRow.requestBody = try XCTUnwrap(ServerRequestLog.openAISummary(openAIRequest))

        let anthropicJSON = #"""
        {"model":"ling","max_tokens":64,"system":"verify nested escapes","messages":[{"role":"user","content":"also contains invalid \"name\""}]}
        """#
        let anthropicRequest = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(anthropicJSON.utf8))
        var anthropicRow = row(path: "/v1/messages")
        anthropicRow.model = "anthropic-summary-row"
        anthropicRow.requestBody = try XCTUnwrap(ServerRequestLog.anthropicSummary(anthropicRequest))

        store.append(openAIRow)
        store.append(anthropicRow)
        store.append(row(error: "unrelated single-encoded failure")) // decoy: similar words, no quotes

        var filter = ServerRequestLogStore.Filter()
        filter.text = "invalid \"name\""
        var fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(
            Set(fetched.map(\.model)),
            ["openai-summary-row", "anthropic-summary-row"],
            "the term must match its doubly-JSON-encoded form inside both protocols' captured request bodies"
        )

        // The same clauses scope the JSONL export, so filtered exports cannot
        // exclude matching request bodies either.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-nested-search-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("matches.jsonl")
        let exported = try await store.exportJSONL(filter: filter, to: url)
        XCTAssertEqual(exported, 2, "the search box must scope the export exactly as it scopes the viewer")
    }

    func testTextSearchMatchesToolArgumentsInsideCapturedRequestBodies() async throws {
        // Captured tool arguments travel one JSON layer deeper than prompt
        // text: OpenAI `function.arguments` and Anthropic tool-use `input`
        // are themselves JSON strings before the request summary and the row
        // payload each encode them again, so an argument value such as
        // `invalid "name"` persists triple-escaped there. The ladder must
        // climb to that third encoded level or rows whose only mention lives
        // inside tool arguments are invisible to both the viewer and the
        // filtered export.
        let openAIJSON = #"""
        {"model":"ling","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"run_query","arguments":"{\"query\":\"invalid \\\"name\\\"\"}"}}]}]}
        """#
        let openAIRequest = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(openAIJSON.utf8))
        var openAIRow = row()
        // The row's message content stays empty: the term occurs ONLY inside
        // the captured tool arguments, so a match proves the deep encoding
        // level is searched, not the shallow ones.
        openAIRow.requestBody = try XCTUnwrap(ServerRequestLog.openAISummary(openAIRequest))
        XCTAssertNil(openAIRequest.messages.first?.contentText, "fixture sanity: no prompt text to match shallower")

        let anthropicJSON = #"""
        {"model":"ling","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"run_query","input":{"query":"invalid \"name\""}}]}]}
        """#
        let anthropicRequest = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(anthropicJSON.utf8))
        var anthropicRow = row(path: "/v1/messages")
        anthropicRow.requestBody = try XCTUnwrap(ServerRequestLog.anthropicSummary(anthropicRequest))
        XCTAssertNil(anthropicRequest.messages.first?.contentText, "fixture sanity: no prompt text to match shallower")

        openAIRow.model = "openai-tool-argument-row"
        anthropicRow.model = "anthropic-tool-argument-row"
        store.append(openAIRow)
        store.append(anthropicRow)
        store.append(row(error: "unrelated invalid name failure without quotes")) // decoy: same words, no quotes

        var filter = ServerRequestLogStore.Filter()
        filter.text = "invalid \"name\""
        let fetched = await store.fetch(filter: filter, limit: 10)
        XCTAssertEqual(
            Set(fetched.map(\.model)),
            ["openai-tool-argument-row", "anthropic-tool-argument-row"],
            "the term must match its triply-JSON-encoded form inside captured tool arguments"
        )

        // The same clauses scope the JSONL export.
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-log-tool-argument-search-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("matches.jsonl")
        let exported = try await store.exportJSONL(filter: filter, to: url)
        XCTAssertEqual(exported, 2, "the search box must scope the export exactly as it scopes the viewer")
    }

    // MARK: Clear + count

    func testClearRemovesEverything() async {
        store.append(row())
        store.append(row())
        store.append(row())
        let before = await store.count()
        XCTAssertEqual(before, 3)

        await store.clear()
        let after = await store.count()
        XCTAssertEqual(after, 0)
        let fetched = await store.fetch(limit: 10)
        XCTAssertTrue(fetched.isEmpty)
    }

    // MARK: Retention

    func testPruneDropsRowsOlderThanRetentionWindow() async {
        let now = Date()
        var fresh = row(timestamp: now.addingTimeInterval(-86_400)) // yesterday
        fresh.model = "fresh"
        var stale = row(timestamp: now.addingTimeInterval(-45.0 * 86_400)) // 45 days old
        stale.model = "stale"
        store.append(fresh)
        store.append(stale)

        await store.pruneForTesting(now: now)
        let fetched = await store.fetch(limit: 10)
        XCTAssertEqual(fetched.map(\.model), ["fresh"], "the 45-day-old row must be pruned; the 1-day-old row must survive")
    }

    func testPruneCapsTotalRowCountKeepingNewest() async {
        // Timestamps must stay inside the 30-day window or the age prune
        // (which runs first) empties the store before the cap trim can be
        // observed.
        let base = Date().addingTimeInterval(-86_400)
        let total = ServerRequestLogStore.maxRows + 50
        for index in 0..<total {
            var candidate = row(timestamp: base.addingTimeInterval(TimeInterval(index)))
            candidate.model = "model-\(index)"
            store.append(candidate)
        }
        await store.pruneForTesting()

        let count = await store.count()
        XCTAssertEqual(count, ServerRequestLogStore.maxRows, "the prune must trim the store to its row cap")
        let fetched = await store.fetch(limit: total)
        XCTAssertEqual(fetched.first?.model, "model-\(total - 1)", "the newest rows must survive the cap trim")
        let oldestSurvivor = Int(fetched.last?.model?.replacingOccurrences(of: "model-", with: "") ?? "-1") ?? -1
        XCTAssertEqual(oldestSurvivor, 50, "the oldest rows (0..49) must be dropped")
    }
}
