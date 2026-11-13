//
//  CloudChatClient.swift
//  DynaMoE
//
//  Outbound streaming chat client for connected cloud providers.
//
//  This is the mirror of the app's inbound OpenAI/Anthropic adapters: the
//  same wire protocol, driven from the inside. The transport machine is
//  modeled on MCPHTTPTransport's streamable-HTTP path (a shared ephemeral
//  URLSession, byte-at-a-time SSE consumption through the shared MCPLineBuffer
//  and MCPSSEFrameParser, bounded events, error-body draining) with the
//  OpenAI `[DONE]` sentinel added on top.
//
//  Flavor seam: the OpenAI-compatible surface is implemented end to end;
//  the Anthropic-compatible surface is a stub that fails explicitly, so a
//  later branch can implement it without touching the registry, Settings, or
//  ContentView wiring.
//

import Foundation

// MARK: - Connection Test

public nonisolated enum CloudConnectionTestResult: Equatable, Sendable {
    case success(modelCount: Int, sampleModelIds: [String])
    case failure(CloudTurnFailure)
}

// MARK: - Service Protocol

nonisolated protocol CloudChatServicing: Sendable {
    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?
    ) -> AsyncStream<CloudTurnEvent>

    /// GET the provider's model listing. Used by the Settings connection test
    /// (and by the future model-registration branch).
    func fetchModelIds() async throws -> [String]
}

// MARK: - Session Cache

/// One URLSession per (endpoint, timeout) pair. An ephemeral session created
/// per turn cannot reuse connections, so a multi-step agent run would open a
/// fresh TLS connection for every request. Auth rides each request's header,
/// so the key does not have to key the session.
private nonisolated final class CloudSessionCache: @unchecked Sendable {
    static let shared = CloudSessionCache()

    private let lock = NSLock()
    private var sessions: [String: URLSession] = [:]

    func session(for key: String, make: () -> URLSession) -> URLSession {
        lock.lock()
        defer { lock.unlock() }
        if let existing = sessions[key] { return existing }
        let created = make()
        sessions[key] = created
        return created
    }
}

// MARK: - Cache Hint Memory

/// Remembers that an optional cache field was refused, per endpoint AND model.
///
/// A provider that rejects `prompt_cache_key` rejects it on every request, and
/// paying a rejected round trip per agent step would cost a turn more than the
/// hint could ever save. The key is the model as well as the endpoint because a
/// compatible gateway fronts several backends under one base URL (the built-in
/// Synthetic provider says so in its own description): one model answering 400
/// says nothing about the model beside it, and latching the whole endpoint
/// switched the hint off for every one of them with no way back — built-ins are
/// read-only in Settings, so the control that clears a latch is not even
/// reachable there.
///
/// One entry per (endpoint, model), process lifetime, and a lock only on the
/// miss path. A rejection is only recorded once a request WITHOUT the field has
/// been accepted, because an unrelated 400 looks identical (see `streamTurn`).
/// Internal rather than private so a test can pin the latch.
nonisolated final class CloudCacheHintMemory: @unchecked Sendable {
    static let shared = CloudCacheHintMemory()

    private let lock = NSLock()
    private var declined: [String: Set<String>] = [:]
    /// Endpoints that rejected the session-affinity HEADERS. Keyed by endpoint
    /// alone: a header is answered by the gateway in front of every model, so
    /// one model's refusal is not evidence about its neighbours the way
    /// `prompt_cache_key`'s body-field refusal is.
    private var declinedAffinity: Set<String> = []

    func acceptsPromptCacheKey(endpoint: String, model: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return declined[endpoint]?.contains(model) != true
    }

    func declinePromptCacheKey(endpoint: String, model: String) {
        lock.lock()
        declined[endpoint, default: []].insert(model)
        lock.unlock()
    }

    func acceptsAffinityHeaders(endpoint: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !declinedAffinity.contains(endpoint)
    }

    func declineAffinityHeaders(endpoint: String) {
        lock.lock()
        declinedAffinity.insert(endpoint)
        lock.unlock()
    }

    /// Forgets one endpoint's refusals of both cache hints.
    ///
    /// A latch is a statement about what a SERVER accepts, and the operator can
    /// know better: changing the cache hint in Settings, or pointing the
    /// provider at a different base URL, describes a different server and
    /// therefore different answers. Without this a provider that refused a hint
    /// once kept it switched off for the life of the process, and the control
    /// that turns it back on did nothing until relaunch.
    func allowCacheHints(endpoint: String) {
        lock.lock()
        declined.removeValue(forKey: endpoint)
        declinedAffinity.remove(endpoint)
        lock.unlock()
    }

    /// Forgets every endpoint. Process lifetime is the shipping behaviour, and a
    /// service can now hold its OWN instance (`OpenAICloudChatService(cacheHintMemory:)`),
    /// so a test should prefer injection over this: a case that latched the
    /// shared latch would otherwise leave the next case — and a
    /// repeat-on-failure rerun of itself — starting from a state it did not set
    /// up.
    func reset() {
        lock.lock()
        declined.removeAll()
        declinedAffinity.removeAll()
        lock.unlock()
    }
}

// MARK: - OpenAI-Compatible Service

nonisolated final class OpenAICloudChatService: CloudChatServicing, @unchecked Sendable {
    /// Stream frames are small; bounding them tightly keeps a hostile or
    /// broken endpoint from assembling an unbounded event in memory.
    static let maxEventBytes = 4 * 1024 * 1024

    private let chatURL: URL
    private let modelsURL: URL
    private let apiKey: String
    private let session: URLSession
    private let cacheHints: CloudCacheHints
    /// The latch this service consults and updates. Injected rather than
    /// reached through the singleton so a case can hold a private one: a shared
    /// latch keyed by endpoint leaks state between tests and between cases of
    /// one test, which is why the singleton needed a `reset()` seam at all.
    private let cacheHintMemory: CloudCacheHintMemory

    init(
        chatURL: URL,
        modelsURL: URL,
        apiKey: String,
        requestTimeout: TimeInterval,
        session: URLSession? = nil,
        cacheHints: CloudCacheHints = .affinity,
        cacheHintMemory: CloudCacheHintMemory = .shared
    ) {
        self.chatURL = chatURL
        self.modelsURL = modelsURL
        self.apiKey = apiKey
        self.cacheHints = cacheHints
        self.cacheHintMemory = cacheHintMemory

        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.timeoutIntervalForRequest = requestTimeout
            // A cloud generation can legitimately run for minutes; the resource
            // timeout only has to catch a wedged connection, not a long answer.
            configuration.timeoutIntervalForResource = max(requestTimeout * 20, 900)
            configuration.waitsForConnectivity = false
            configuration.requestCachePolicy = .reloadIgnoringLocalCacheData
            configuration.httpShouldSetCookies = false
            let cacheKey = "\(chatURL.absoluteString)|\(requestTimeout)"
            self.session = CloudSessionCache.shared.session(for: cacheKey) {
                URLSession(configuration: configuration)
            }
        }
    }

    // MARK: Request construction

    private struct Body: Encodable {
        struct StreamOptions: Encodable {
            let includeUsage: Bool
            enum CodingKeys: String, CodingKey { case includeUsage = "include_usage" }
        }

        let model: String
        let messages: [CloudChatMessage]
        let stream: Bool
        let streamOptions: StreamOptions?
        let temperature: Double?
        let topP: Double?
        let presencePenalty: Double?
        let maxTokens: Int?
        let tools: [ToolDefinition]?
        let toolChoice: String?
        let promptCacheKey: String?

        enum CodingKeys: String, CodingKey {
            case model, messages, stream, temperature, tools
            case streamOptions = "stream_options"
            case topP = "top_p"
            case presencePenalty = "presence_penalty"
            case maxTokens = "max_tokens"
            case toolChoice = "tool_choice"
            case promptCacheKey = "prompt_cache_key"
        }
    }

    /// Headers every chat request carries. `x-session-id` and
    /// `x-session-affinity` are read by routers that keep one conversation on
    /// one replica; a provider that does not know them ignores them by the
    /// HTTP contract, so they are sent unless the provider opts out — or until
    /// this endpoint refuses them once and the retry proves that was the
    /// reason (see `shouldRetryWithoutAffinity`).
    ///
    /// An empty key sends nothing: an empty header value is not "no affinity",
    /// it is a value routers are free to treat as a real (and therefore
    /// shared) session, and `prompt_cache_key` is already withheld on the same
    /// condition.
    private func applyCacheAffinity(
        to request: inout URLRequest,
        params: CloudChatGenerationParams,
        sendsAffinity: Bool
    ) {
        guard sendsAffinity,
              let sessionKey = params.sessionKey,
              !sessionKey.isEmpty else { return }
        request.setValue(sessionKey, forHTTPHeaderField: "x-session-id")
        request.setValue(sessionKey, forHTTPHeaderField: "x-session-affinity")
    }

    /// Whether the next request would carry the session-affinity headers.
    ///
    /// The same resolved decision `makeRequest` applies, so the retry predicate
    /// can test the request it actually sent. A latch keyed by endpoint is
    /// process-wide, which is exactly why the answer is taken before the send
    /// rather than re-derived from the response.
    private func willSendAffinityHeaders() -> Bool {
        cacheHints.sendsAffinityHeaders
            && cacheHintMemory.acceptsAffinityHeaders(endpoint: chatURL.absoluteString)
    }

    /// Whether the next request would actually carry `prompt_cache_key`.
    ///
    /// Shared by `makeRequest` and the retry predicate, which is the point: a
    /// retry can only be about the field if the rejected request carried it.
    /// After the process has latched the field off, every request is already
    /// key-free, so a 4xx must fail rather than re-upload the identical body.
    private func willSendCacheKey(params: CloudChatGenerationParams, model: String) -> Bool {
        Self.sendsCacheKey(
            hints: cacheHints,
            sessionKey: params.sessionKey,
            endpoint: chatURL.absoluteString,
            model: model,
            memory: cacheHintMemory
        )
    }

    /// The same rule as a pure function of its inputs, so a test can pin it
    /// without a live endpoint or a URLSession. `memory` is a parameter for the
    /// same reason the service holds one: a case that latches must not be seen
    /// by the next case or by the app's own requests. `model` is part of the
    /// latch key, not of the decision about the hint: one model on a shared
    /// gateway can refuse the field while its neighbours accept it.
    nonisolated static func sendsCacheKey(
        hints: CloudCacheHints,
        sessionKey: String?,
        endpoint: String,
        model: String,
        memory: CloudCacheHintMemory = .shared
    ) -> Bool {
        hints.sendsPromptCacheKey
            && !(sessionKey ?? "").isEmpty
            && memory.acceptsPromptCacheKey(endpoint: endpoint, model: model)
    }

    /// Internal so a test can pin the request shape without a live endpoint.
    ///
    /// `cacheKeyIsSent` is the already-resolved decision. `streamTurn` passes
    /// the answer it took before the send so its retry predicate can test the
    /// request it actually made; re-deriving it there re-read a process-wide
    /// latch a concurrent request can flip in between, which suppressed the
    /// retry this request had earned and failed the turn instead.
    func makeRequest(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]? = nil,
        includeUsage: Bool = true,
        includeCacheKey: Bool = true,
        includeAffinity: Bool = true,
        cacheKeyIsSent: Bool? = nil,
        affinityIsSent: Bool? = nil
    ) throws -> URLRequest {
        let wantsCacheKey = cacheKeyIsSent
            ?? (includeCacheKey && willSendCacheKey(params: params, model: model))
        let sendsAffinity = affinityIsSent ?? (includeAffinity && willSendAffinityHeaders())
        let body = Body(
            model: model,
            messages: messages,
            stream: true,
            streamOptions: includeUsage ? Body.StreamOptions(includeUsage: true) : nil,
            temperature: params.temperature,
            topP: params.topP,
            presencePenalty: params.presencePenalty,
            maxTokens: params.maxTokens,
            tools: (tools?.isEmpty == false) ? tools : nil,
            toolChoice: (tools?.isEmpty == false) ? "auto" : nil,
            promptCacheKey: wantsCacheKey ? params.sessionKey : nil
        )
        var request = URLRequest(url: chatURL)
        request.httpMethod = "POST"
        // Sorted keys is a cache requirement, not a style choice: the body is
        // the provider's token prefix once templated, and an unsorted
        // dictionary (`ToolDefinition.parameters` is a Swift dictionary) can
        // reorder a schema between requests and invalidate the whole prompt.
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        request.httpBody = try encoder.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        applyCacheAffinity(to: &request, params: params, sendsAffinity: sendsAffinity)
        return request
    }

    // MARK: Streaming

    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?
    ) -> AsyncStream<CloudTurnEvent> {
        AsyncStream { continuation in
            // Framing and JSON decoding are CPU work on a per-token hot path;
            // detached keeps them off the main actor no matter the caller.
            let task = Task.detached(priority: .userInitiated) {
                continuation.yield(.started)

                var parser = CloudStreamParser()
                var sawFailure = false

                do {
                    // `stream_options.include_usage` is an OpenAI extension that
                    // some compatible endpoints reject with a 400. Retry once
                    // without it rather than failing the whole turn over usage
                    // stats (the turn then reports estimated token counts).
                    // `prompt_cache_key` gets the same treatment: a strict
                    // endpoint may reject the unknown body field, and the hint
                    // is worth one retry, never a failed turn.
                    var includeUsage = true
                    var includeCacheKey = true
                    var includeAffinity = true
                    // Remembered only when the retry WITHOUT the field
                    // succeeds. A 400 is not proof the field was the reason:
                    // an unknown model name or a malformed tools schema is
                    // answered the same way, and latching on one of those
                    // would switch the hint off for every later turn on this
                    // provider. The retry is free; the memory is not.
                    var retriedWithoutCacheKey = false
                    var cacheKeyRejectionStatus: Int?
                    var retriedWithoutAffinity = false
                    var affinityRejectionStatus: Int?
                    // Four attempts: the initial one plus one per optional
                    // hint, because an endpoint can refuse `stream_options`
                    // and then a header, or a header and then the body field.
                    for _ in 0..<4 {
                        // Resolved BEFORE the send and reused by the retry
                        // predicate below: the latch is process-wide, so a
                        // concurrent request declaring this endpoint
                        // indifferent between now and the response would
                        // otherwise suppress the retry this request earned and
                        // fail the turn on a hint it had, in fact, sent.
                        let sentCacheKey = includeCacheKey && self.willSendCacheKey(params: params, model: model)
                        let sentAffinity = includeAffinity && self.willSendAffinityHeaders()
                        let request = try self.makeRequest(
                            model: model,
                            messages: messages,
                            params: params,
                            tools: tools,
                            includeUsage: includeUsage,
                            includeCacheKey: includeCacheKey,
                            includeAffinity: includeAffinity,
                            cacheKeyIsSent: sentCacheKey,
                            affinityIsSent: sentAffinity
                        )
                        let (bytes, response) = try await self.session.bytes(for: request)
                        guard let http = response as? HTTPURLResponse else {
                            continuation.yield(.failed(.invalidResponse("The provider returned a non-HTTP response.")))
                            continuation.finish()
                            return
                        }
                        guard (200..<300).contains(http.statusCode) else {
                            let body = await Self.drain(bytes, limit: 4096)
                            // Stop may have arrived while the error body was
                            // draining. Either retry below re-uploads the whole
                            // prompt, which nobody is waiting for any more.
                            try Task.checkCancellation()
                            if includeUsage,
                               Self.shouldRetryWithoutStreamOptions(status: http.statusCode, body: body) {
                                includeUsage = false
                                continue
                            }
                            // Checked BEFORE the cache-key predicate: a body
                            // that names a header is about the headers, and
                            // the cache-key rule's "does not mention another
                            // part" fallback would otherwise claim it and drop
                            // the wrong hint.
                            if sentAffinity,
                               Self.shouldRetryWithoutAffinity(status: http.statusCode, body: body) {
                                includeAffinity = false
                                retriedWithoutAffinity = true
                                affinityRejectionStatus = http.statusCode
                                continue
                            }
                            if sentCacheKey,
                               Self.shouldRetryWithoutCacheKey(status: http.statusCode, body: body) {
                                includeCacheKey = false
                                retriedWithoutCacheKey = true
                                cacheKeyRejectionStatus = http.statusCode
                                continue
                            }
                            continuation.yield(.failed(.httpStatus(
                                code: http.statusCode,
                                summary: MCPHTTPTransport.summarizeErrorBody(body)
                            )))
                            continuation.finish()
                            return
                        }

                        // The field is only proven unwelcome once a request
                        // without it is accepted. Until then the rejection may
                        // have been about the body's model, tools or size.
                        if retriedWithoutCacheKey {
                            self.cacheHintMemory.declinePromptCacheKey(
                                endpoint: self.chatURL.absoluteString,
                                model: model
                            )
                            // Metadata only, through the app's own log: the
                            // host and the HTTP status, never the request.
                            DiagLog.info(
                                "CloudCache",
                                "endpoint=\(self.chatURL.host ?? "provider") rejected prompt_cache_key status=\(cacheKeyRejectionStatus ?? 0) retry_without_it_succeeded=true latched=true"
                            )
                        }
                        // Same rule for the affinity headers, and the same
                        // reason to latch only after the retry is accepted: the
                        // default hint sends them to every provider, so a
                        // non-conforming endpoint would otherwise fail every
                        // turn for a hint nobody chose.
                        if retriedWithoutAffinity {
                            self.cacheHintMemory.declineAffinityHeaders(
                                endpoint: self.chatURL.absoluteString
                            )
                            DiagLog.info(
                                "CloudCache",
                                "endpoint=\(self.chatURL.host ?? "provider") rejected session-affinity headers status=\(affinityRejectionStatus ?? 0) retry_without_them_succeeded=true latched=true"
                            )
                        }

                        var lineBuffer = MCPLineBuffer(limit: Self.maxEventBytes)
                        var frames = MCPSSEFrameParser(limit: Self.maxEventBytes)

                        // Returns false when the stream is finished (either the
                        // `[DONE]` sentinel or a decode failure was reported).
                        func dispatch(_ event: (event: String?, data: String)?) -> Bool {
                            guard let event else { return true }
                            let payload = event.data.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !payload.isEmpty else { return true }
                            if payload.hasPrefix("[DONE]") { return false }
                            do {
                                for parsed in try parser.consume(dataPayload: payload) {
                                    continuation.yield(parsed)
                                }
                            } catch let failure as CloudTurnFailure {
                                sawFailure = true
                                continuation.yield(.failed(failure))
                                return false
                            } catch {
                                sawFailure = true
                                continuation.yield(.failed(.invalidResponse(
                                    "The provider sent a response chunk DynaMoE could not decode."
                                )))
                                return false
                            }
                            return true
                        }

                        var finishedBySentinel = false
                        // Coalesce bytes into one buffer before handing them to
                        // the line splitter: a 1-byte `Data` per byte allocated
                        // once per token. The batch stays small enough that the
                        // line buffer's limit still bounds a hostile endpoint.
                        var batch = Data()
                        batch.reserveCapacity(8192)
                        streamLoop: for try await byte in bytes {
                            batch.append(byte)
                            if batch.count >= 8192 {
                                for line in lineBuffer.append(batch) {
                                    if !dispatch(frames.consume(line: line)) {
                                        finishedBySentinel = true
                                        break streamLoop
                                    }
                                }
                                batch.removeAll(keepingCapacity: true)
                                if lineBuffer.overflowed || frames.overflowed { break }
                            }
                        }
                        if !batch.isEmpty && !finishedBySentinel
                            && !lineBuffer.overflowed && !frames.overflowed {
                            for line in lineBuffer.append(batch) {
                                if !dispatch(frames.consume(line: line)) {
                                    finishedBySentinel = true
                                    break
                                }
                            }
                        }

                        // A stream left before EOF — `[DONE]`, a decode failure,
                        // an oversized frame, or Stop — still owns a running data
                        // task, which keeps the connection claimed until the
                        // server closes by itself. Cancelling settles both
                        // immediately; the bytes already parsed are unaffected.
                        if Task.isCancelled || finishedBySentinel || sawFailure
                            || lineBuffer.overflowed || frames.overflowed {
                            bytes.task.cancel()
                        }

                        if Task.isCancelled {
                            continuation.finish()
                            return
                        }

                        if !finishedBySentinel && !sawFailure {
                            if !lineBuffer.overflowed && !frames.overflowed {
                                // The tail is where a `[DONE]` that arrived in
                                // the final batch lands, so its answer has to be
                                // kept: discarding `false` here left
                                // `finishedBySentinel` unset and reported a
                                // complete answer as a connection that closed
                                // early (a spurious "cut off" row, and a
                                // compaction that refused its own summary).
                                if let trailing = lineBuffer.flush() {
                                    if !dispatch(frames.consume(line: trailing)) {
                                        finishedBySentinel = true
                                    }
                                }
                                if !finishedBySentinel && !sawFailure {
                                    if !dispatch(frames.finish()) {
                                        finishedBySentinel = true
                                    }
                                }
                            }
                        }

                        if !sawFailure {
                            if lineBuffer.overflowed || frames.overflowed {
                                continuation.yield(.failed(.invalidResponse(
                                    "The provider sent a stream event larger than \(Self.maxEventBytes) bytes."
                                )))
                            } else if parser.decodedChunkCount == 0 {
                                continuation.yield(.failed(.invalidResponse(
                                    "The provider closed the stream without sending any content."
                                )))
                            } else {
                                // A clean EOF is not a finished answer. The
                                // protocol marks the end with `[DONE]` or with
                                // a `finish_reason`, and a connection that
                                // closes mid-reply looks exactly like this; the
                                // flag travels with the result so the chat
                                // surface can say the reply was cut off and a
                                // compaction can refuse to commit it.
                                continuation.yield(.finished(parser.result(
                                    terminated: finishedBySentinel || parser.finishReason != nil
                                )))
                            }
                        }
                        continuation.finish()
                        return
                    }
                } catch {
                    // A cancelled turn is the Stop button, not an error: the
                    // caller keeps whatever partial text already streamed.
                    if !Task.isCancelled {
                        if let failure = error as? CloudTurnFailure {
                            continuation.yield(.failed(failure))
                        } else {
                            continuation.yield(.failed(.transport(error.localizedDescription)))
                        }
                    }
                }
                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Whether a provider's rejection is the optional usage extension rather
    /// than a real problem with the turn.
    nonisolated static func shouldRetryWithoutStreamOptions(status: Int, body: String) -> Bool {
        guard status == 400 || status == 422 else { return false }
        let lower = body.lowercased()
        return lower.contains("stream_options") || lower.contains("include_usage")
    }

    /// Whether a rejection is specifically about the session-affinity HEADERS.
    ///
    /// Affinity is the default hint and is sent for every provider, including
    /// ones configured before the setting existed, so an endpoint that is not
    /// HTTP-conforming used to have no recovery: every turn failed on a hint
    /// nobody chose. The retry is earned only by a body that NAMES one of the
    /// headers, which keeps it a targeted fix rather than a blind re-upload —
    /// the same discipline `shouldRetryWithoutCacheKey` applies to the body
    /// field, and the reason a bare "invalid request" is not evidence enough
    /// here either.
    nonisolated static func shouldRetryWithoutAffinity(status: Int, body: String) -> Bool {
        guard status == 400 || status == 422 else { return false }
        let lower = body.lowercased()
        return lower.contains("x-session-affinity") || lower.contains("x-session-id")
    }

    /// Whether a rejection is worth retrying without `prompt_cache_key`.
    ///
    /// Deliberately not keyed on the body naming the field: a provider
    /// rejecting an unknown body field usually words it vaguely ("invalid
    /// request", "unexpected property"), and a rejected request is not billed,
    /// so one blind retry costs nothing but the upload.
    ///
    /// The upload is the cost, and it is the whole prompt. A body that names
    /// another part of the request — the model, the tool schemas, the messages
    /// — is about that part, and re-sending the same megabytes would fail
    /// identically, so those are refused here. The two content refusals the
    /// chat path recovers from on its own (a context overflow, an image
    /// rejection) are refused for the same reason, and were already.
    nonisolated static func shouldRetryWithoutCacheKey(status: Int, body: String) -> Bool {
        guard status == 400 || status == 422 else { return false }
        let lower = body.lowercased()
        if CloudTurnFailure.indicatesContextOverflow(lower) { return false }
        if CloudTurnFailure.indicatesImageRejection(lower) { return false }
        // Words that name a different part of the request. "function" is here
        // for the tool-calling refusals some endpoints answer with, not for the
        // field itself; a body that mentions it is not about the cache hint.
        let otherParts = [
            "invalid model", "unknown model", "model not found", "does not exist",
            "tool", "function", "schema", "messages"
        ]
        return !otherParts.contains { lower.contains($0) }
    }

    // MARK: Models

    /// Decodes a provider's `GET /models` payload. The OpenAI shape
    /// (`{"data":[{"id":…}]}`) is authoritative; the others are tolerated
    /// because a provider added in Settings may publish a bare array, a
    /// string array, or a `models` key, and failing a connection test over an
    /// envelope difference helps nobody.
    nonisolated static func decodeModelIds(from data: Data) -> [String] {
        guard let object = try? JSONSerialization.jsonObject(with: data) else { return [] }

        var ids: [String] = []
        func append(_ value: Any?) {
            if let id = value as? String {
                ids.append(id)
            } else if let entry = value as? [String: Any], let id = entry["id"] as? String {
                ids.append(id)
            }
        }

        if let array = object as? [Any] {
            array.forEach(append)
        } else if let dictionary = object as? [String: Any] {
            if let list = dictionary["data"] as? [Any] {
                list.forEach(append)
            } else if let list = dictionary["models"] as? [Any] {
                list.forEach(append)
            }
        }

        var seen = Set<String>()
        return ids
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    func fetchModelIds() async throws -> [String] {
        var request = URLRequest(url: modelsURL)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")

        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw CloudTurnFailure.invalidResponse("The provider returned a non-HTTP response.")
            }
            guard (200..<300).contains(http.statusCode) else {
                let body = String(decoding: data.prefix(4096), as: UTF8.self)
                throw CloudTurnFailure.httpStatus(
                    code: http.statusCode,
                    summary: MCPHTTPTransport.summarizeErrorBody(body)
                )
            }
            return Self.decodeModelIds(from: data)
        } catch let failure as CloudTurnFailure {
            throw failure
        } catch {
            throw CloudTurnFailure.transport(error.localizedDescription)
        }
    }

    /// Reads an error body, keeping at most `limit` bytes.
    ///
    /// The sequence is consumed past the kept prefix instead of abandoned at
    /// it: a partially-read response stays claimed by the shared session (its
    /// data task alive, its connection unreusable) until the server closes on
    /// its own. Error bodies are short, so reading to EOF costs nothing and
    /// returns the connection to the pool. `hardCap` stops an endpoint that
    /// answers with an endless body; hitting it cancels the underlying task so
    /// nothing is left running.
    private static func drain(_ bytes: URLSession.AsyncBytes, limit: Int) async -> String {
        let hardCap = max(limit, 256 * 1024)
        var data = Data()
        var consumed = 0
        do {
            for try await byte in bytes {
                consumed += 1
                if data.count < limit { data.append(byte) }
                if consumed >= hardCap {
                    bytes.task.cancel()
                    break
                }
            }
        } catch {
            return String(decoding: data, as: UTF8.self)
        }
        return String(decoding: data, as: UTF8.self)
    }
}

// MARK: - Anthropic-Compatible Service (stub)

/// Placeholder for Synthetic's `/anthropic` surface. Declared against the
/// same seam as the OpenAI service so switching flavors later is a matter of
/// implementing this class, never rewiring call sites. Fails explicitly
/// rather than silently pretending to stream.
nonisolated final class AnthropicCloudChatService: CloudChatServicing, @unchecked Sendable {
    func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]?
    ) -> AsyncStream<CloudTurnEvent> {
        AsyncStream { continuation in
            continuation.yield(.failed(.unsupportedFlavor(flavor: CloudApiFlavor.anthropic.rawValue)))
            continuation.finish()
        }
    }

    func fetchModelIds() async throws -> [String] {
        throw CloudTurnFailure.unsupportedFlavor(flavor: CloudApiFlavor.anthropic.rawValue)
    }
}

// MARK: - Client Facade

/// One turn's worth of provider client. Construct per turn (or per
/// connection test): the underlying URLSession is shared per endpoint, so a
/// multi-step agent run reuses connections, while a per-turn client means a
/// rotated key is picked up without restarting anything.
public nonisolated final class CloudChatClient: @unchecked Sendable {
    /// Idle (between-bytes) timeout for a chat turn. Generous on purpose: a
    /// large prompt or a reasoning model can take longer than half a minute to
    /// first token, and the timeout only has to catch a wedged connection.
    public static let defaultRequestTimeout: TimeInterval = 120

    public let provider: CloudProvider
    public let flavor: CloudApiFlavor
    private let service: CloudChatServicing

    public init(
        provider: CloudProvider,
        flavor: CloudApiFlavor = .openAI,
        apiKey: String,
        requestTimeout: TimeInterval = CloudChatClient.defaultRequestTimeout
    ) {
        self.provider = provider
        self.flavor = flavor
        switch flavor {
        case .openAI:
            self.service = OpenAICloudChatService(
                chatURL: provider.chatCompletionsURL,
                modelsURL: provider.modelsURL,
                apiKey: apiKey,
                requestTimeout: requestTimeout,
                cacheHints: provider.resolvedCacheHints
            )
        case .anthropic:
            self.service = AnthropicCloudChatService()
        }
    }

    /// The same facade over an injected transport, so a test can drive the
    /// terminal-event rules of `completeText` without a network.
    init(provider: CloudProvider, flavor: CloudApiFlavor = .openAI, service: CloudChatServicing) {
        self.provider = provider
        self.flavor = flavor
        self.service = service
    }

    /// Builds a client from the Keychain, or nil when the provider has no
    /// stored key — the send path turns that into an actionable chat error.
    public static func makeStored(
        provider: CloudProvider,
        flavor: CloudApiFlavor = .openAI
    ) -> CloudChatClient? {
        try? makeStoredResult(provider: provider, flavor: flavor).get()
    }

    /// The same lookup with the reason kept. A missing key, a refusal to read
    /// one, and a key that is still in the older keychain are three different
    /// things for the person reading the message.
    public static func makeStoredResult(
        provider: CloudProvider,
        flavor: CloudApiFlavor = .openAI
    ) -> Result<CloudChatClient, CloudTurnFailure> {
        let store = CloudSecretStore.shared
        switch storedKeyResult(
            providerName: provider.displayName,
            read: store.read(providerId: provider.id),
            // Evaluated only when the read established nothing: the existence
            // check is attributes-only (so it cannot prompt) but it is still a
            // keychain round trip per token, and a readable key settles it.
            existsInEitherKeychain: { store.exists(providerId: provider.id) }
        ) {
        case .success(let key):
            return .success(CloudChatClient(provider: provider, flavor: flavor, apiKey: key))
        case .failure(let failure):
            return .failure(failure)
        }
    }

    /// The lookup rule as a pure function of the two keychain answers, so each
    /// outcome can be exercised without a keychain.
    static func storedKeyResult(
        providerName: String,
        read: SecretAccessResult,
        existsInEitherKeychain: () -> Bool
    ) -> Result<String, CloudTurnFailure> {
        switch read {
        case .found(let key) where !key.isEmpty:
            return .success(key)
        case .found:
            // An empty stored item is not a credential. Asking `exists` here
            // would find that same empty item and report "still in the older
            // keychain, import it" about a key that could never work.
            return .failure(.missingAPIKey(providerName: providerName))
        case .missing:
            return existsInEitherKeychain()
                ? .failure(.credentialNeedsImport(providerName: providerName))
                : .failure(.missingAPIKey(providerName: providerName))
        case .accessDenied, .unavailable:
            return .failure(.credentialAccessDenied(providerName: providerName))
        }
    }

    public func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]? = nil
    ) -> AsyncStream<CloudTurnEvent> {
        service.streamTurn(model: model, messages: messages, params: params, tools: tools)
    }

    /// One plain completion, collected with the usage it was billed at.
    ///
    /// This exists for compaction: the summary has to come from the model the
    /// conversation is actually using, because the on-device system model is
    /// not guaranteed to exist and a user's cloud-only Mac had no working
    /// `/compact` at all. No tools, no agent loop, temperature 0.
    ///
    /// The result carries the provider's usage block as well as the text: a
    /// summarizer pass is a billed request like any other, and dropping its
    /// usage made the session's spend understate the invoice.
    ///
    /// It also refuses to report a fragment as a completion. A `for await` over
    /// an `AsyncStream` ends the moment its task is cancelled, with no terminal
    /// event, so a Stop mid-summary used to return the characters collected so
    /// far as if the provider had finished them. Cancellation now throws, and a
    /// stream that ends without a terminal event throws instead of returning:
    /// a compaction that deletes the messages behind a half-written summary
    /// cannot be undone.
    ///
    /// "Terminal" is the protocol's own marker, not a clean EOF. A provider that
    /// closes the connection without `[DONE]` and without a `finish_reason` has
    /// written a prefix of an answer, and committing that would delete every
    /// message it never covered — which is exactly why the flag is checked here
    /// as well as in `CloudCompactionGate.refusal(for:)`.
    public func completeText(
        model: String,
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int,
        sessionKey: String?
    ) async throws -> CloudTextCompletion {
        var messages: [CloudChatMessage] = []
        if !systemPrompt.isEmpty {
            messages.append(CloudChatMessage(role: "system", content: systemPrompt))
        }
        messages.append(CloudChatMessage(role: "user", content: userPrompt))
        let params = CloudChatGenerationParams(
            temperature: 0,
            maxTokens: maxTokens,
            sessionKey: sessionKey
        )
        var text = ""
        var usage: CloudUsage?
        var result: CloudTurnResult?
        var failure: CloudTurnFailure?
        for await event in streamTurn(model: model, messages: messages, params: params, tools: nil) {
            switch event {
            case .started, .thinkingDelta:
                break
            case .contentDelta(let delta):
                text += delta
            case .finished(let finished):
                // The parser's accumulated text and the result's text are the
                // same string; prefer the result because it is what the
                // transport certified as complete.
                if !finished.text.isEmpty { text = finished.text }
                usage = finished.usage
                result = finished
            case .failed(let reported):
                failure = reported
            }
        }
        if let failure { throw failure }
        // A cancelled consumer leaves the loop with neither an outcome nor an
        // error; whatever text it collected was abandoned mid-sentence.
        try Task.checkCancellation()
        guard let result else {
            throw CloudTurnFailure.invalidResponse(
                "The provider's stream ended before the completion was finished."
            )
        }
        guard result.terminated else {
            throw CloudTurnFailure.invalidResponse(
                "The provider's stream closed before the completion was finished, so its text is only a fragment."
            )
        }
        return CloudTextCompletion(
            text: text.trimmingCharacters(in: .whitespacesAndNewlines),
            usage: usage,
            finishReason: result.finishReason
        )
    }

    public func testConnection() async -> CloudConnectionTestResult {
        do {
            let ids = try await service.fetchModelIds()
            return .success(modelCount: ids.count, sampleModelIds: Array(ids.prefix(5)))
        } catch let failure as CloudTurnFailure {
            return .failure(failure)
        } catch {
            return .failure(.transport(error.localizedDescription))
        }
    }

    /// The provider's live model catalog, used by the Settings "Sync Models"
    /// action to register the provider's chat models (callers filter the
    /// non-chat entries).
    public func fetchModelIds() async -> Result<[String], CloudTurnFailure> {
        do {
            return .success(try await service.fetchModelIds())
        } catch let failure as CloudTurnFailure {
            return .failure(failure)
        } catch {
            return .failure(.transport(error.localizedDescription))
        }
    }
}
