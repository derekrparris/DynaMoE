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

/// Remembers, per endpoint, that an optional cache field was refused.
///
/// A provider that rejects `prompt_cache_key` rejects it on every request, and
/// paying a rejected round trip per agent step would cost a turn more than the
/// hint could ever save. One entry per endpoint, process lifetime, and a lock
/// only on the miss path. A rejection is only recorded once a request WITHOUT
/// the field has been accepted, because an unrelated 400 looks identical (see
/// `streamTurn`).
private nonisolated final class CloudCacheHintMemory: @unchecked Sendable {
    static let shared = CloudCacheHintMemory()

    private let lock = NSLock()
    private var declined: Set<String> = []

    func acceptsPromptCacheKey(endpoint: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return !declined.contains(endpoint)
    }

    func declinePromptCacheKey(endpoint: String) {
        lock.lock()
        declined.insert(endpoint)
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

    init(
        chatURL: URL,
        modelsURL: URL,
        apiKey: String,
        requestTimeout: TimeInterval,
        session: URLSession? = nil,
        cacheHints: CloudCacheHints = .affinity
    ) {
        self.chatURL = chatURL
        self.modelsURL = modelsURL
        self.apiKey = apiKey
        self.cacheHints = cacheHints

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
    /// HTTP contract, so they are sent unless the provider opts out.
    ///
    /// An empty key sends nothing: an empty header value is not "no affinity",
    /// it is a value routers are free to treat as a real (and therefore
    /// shared) session, and `prompt_cache_key` is already withheld on the same
    /// condition.
    private func applyCacheAffinity(to request: inout URLRequest, params: CloudChatGenerationParams) {
        guard cacheHints.sendsAffinityHeaders,
              let sessionKey = params.sessionKey,
              !sessionKey.isEmpty else { return }
        request.setValue(sessionKey, forHTTPHeaderField: "x-session-id")
        request.setValue(sessionKey, forHTTPHeaderField: "x-session-affinity")
    }

    /// Internal so a test can pin the request shape without a live endpoint.
    func makeRequest(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]? = nil,
        includeUsage: Bool = true,
        includeCacheKey: Bool = true
    ) throws -> URLRequest {
        let wantsCacheKey = includeCacheKey
            && cacheHints.sendsPromptCacheKey
            && !(params.sessionKey ?? "").isEmpty
            && CloudCacheHintMemory.shared.acceptsPromptCacheKey(endpoint: chatURL.absoluteString)
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
        applyCacheAffinity(to: &request, params: params)
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
                    // Remembered only when the retry WITHOUT the field
                    // succeeds. A 400 is not proof the field was the reason:
                    // an unknown model name or a malformed tools schema is
                    // answered the same way, and latching on one of those
                    // would switch the hint off for every later turn on this
                    // provider. The retry is free; the memory is not.
                    var retriedWithoutCacheKey = false
                    var cacheKeyRejectionStatus: Int?
                    let asksForCacheKey = self.cacheHints.sendsPromptCacheKey
                        && !(params.sessionKey ?? "").isEmpty
                    for _ in 0..<3 {
                        let request = try self.makeRequest(
                            model: model,
                            messages: messages,
                            params: params,
                            tools: tools,
                            includeUsage: includeUsage,
                            includeCacheKey: includeCacheKey
                        )
                        let (bytes, response) = try await self.session.bytes(for: request)
                        guard let http = response as? HTTPURLResponse else {
                            continuation.yield(.failed(.invalidResponse("The provider returned a non-HTTP response.")))
                            continuation.finish()
                            return
                        }
                        guard (200..<300).contains(http.statusCode) else {
                            let body = await Self.drain(bytes, limit: 4096)
                            if includeUsage,
                               Self.shouldRetryWithoutStreamOptions(status: http.statusCode, body: body) {
                                includeUsage = false
                                continue
                            }
                            if includeCacheKey, asksForCacheKey,
                               Self.shouldRetryWithoutCacheKey(status: http.statusCode) {
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
                            CloudCacheHintMemory.shared.declinePromptCacheKey(
                                endpoint: self.chatURL.absoluteString
                            )
                            print("☁️ [CACHE] \(self.chatURL.host ?? "provider") rejected prompt_cache_key (HTTP \(cacheKeyRejectionStatus ?? 0)); the retry without it succeeded, so it will not be sent again.")
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
                                if let trailing = lineBuffer.flush() {
                                    _ = dispatch(frames.consume(line: trailing))
                                }
                                _ = dispatch(frames.finish())
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
                                continuation.yield(.finished(parser.result()))
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

    /// Whether a rejection is worth retrying without `prompt_cache_key`.
    ///
    /// Deliberately not keyed on the body: a provider rejecting an unknown body
    /// field usually names the field vaguely ("invalid request", "unexpected
    /// property"), and a rejected request is not billed, so one blind retry
    /// costs nothing. It is bounded to 4xx because a 5xx or a 429 is not about
    /// the field.
    nonisolated static func shouldRetryWithoutCacheKey(status: Int) -> Bool {
        status == 400 || status == 422
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

    /// One plain completion, collected into a string.
    ///
    /// This exists for compaction: the summary has to come from the model the
    /// conversation is actually using, because the on-device system model is
    /// not guaranteed to exist and a user's cloud-only Mac had no working
    /// `/compact` at all. No tools, no agent loop, temperature 0.
    public func completeText(
        model: String,
        systemPrompt: String,
        userPrompt: String,
        maxTokens: Int,
        sessionKey: String?
    ) async throws -> String {
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
        var failure: CloudTurnFailure?
        for await event in streamTurn(model: model, messages: messages, params: params, tools: nil) {
            switch event {
            case .started, .thinkingDelta:
                break
            case .contentDelta(let delta):
                text += delta
            case .finished(let result):
                // The parser's accumulated text and the result's text are the
                // same string; prefer the result because it is what the
                // transport certified as complete.
                if !result.text.isEmpty { text = result.text }
            case .failed(let reported):
                failure = reported
            }
        }
        if let failure { throw failure }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
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
