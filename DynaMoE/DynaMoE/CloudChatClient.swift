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

// MARK: - OpenAI-Compatible Service

nonisolated final class OpenAICloudChatService: CloudChatServicing, @unchecked Sendable {
    /// Stream frames are small; bounding them tightly keeps a hostile or
    /// broken endpoint from assembling an unbounded event in memory.
    static let maxEventBytes = 4 * 1024 * 1024

    private let chatURL: URL
    private let modelsURL: URL
    private let apiKey: String
    private let session: URLSession

    init(
        chatURL: URL,
        modelsURL: URL,
        apiKey: String,
        requestTimeout: TimeInterval,
        session: URLSession? = nil
    ) {
        self.chatURL = chatURL
        self.modelsURL = modelsURL
        self.apiKey = apiKey

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

        enum CodingKeys: String, CodingKey {
            case model, messages, stream, temperature, tools
            case streamOptions = "stream_options"
            case topP = "top_p"
            case presencePenalty = "presence_penalty"
            case maxTokens = "max_tokens"
            case toolChoice = "tool_choice"
        }
    }

    /// Internal so a test can pin the request shape without a live endpoint.
    func makeRequest(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]? = nil,
        includeUsage: Bool = true
    ) throws -> URLRequest {
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
            toolChoice: (tools?.isEmpty == false) ? "auto" : nil
        )
        var request = URLRequest(url: chatURL)
        request.httpMethod = "POST"
        request.httpBody = try JSONEncoder().encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
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
                    var includeUsage = true
                    for attempt in 0..<2 {
                        let request = try self.makeRequest(
                            model: model,
                            messages: messages,
                            params: params,
                            tools: tools,
                            includeUsage: includeUsage
                        )
                        let (bytes, response) = try await self.session.bytes(for: request)
                        guard let http = response as? HTTPURLResponse else {
                            continuation.yield(.failed(.invalidResponse("The provider returned a non-HTTP response.")))
                            continuation.finish()
                            return
                        }
                        guard (200..<300).contains(http.statusCode) else {
                            let body = await Self.drain(bytes, limit: 4096)
                            if attempt == 0, includeUsage,
                               Self.shouldRetryWithoutStreamOptions(status: http.statusCode, body: body) {
                                includeUsage = false
                                continue
                            }
                            continuation.yield(.failed(.httpStatus(
                                code: http.statusCode,
                                summary: MCPHTTPTransport.summarizeErrorBody(body)
                            )))
                            continuation.finish()
                            return
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

    // MARK: Models

    private struct ModelsResponse: Decodable {
        struct Entry: Decodable {
            let id: String?
        }
        let data: [Entry]?
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
            let decoded = try JSONDecoder().decode(ModelsResponse.self, from: data)
            return (decoded.data ?? []).compactMap(\.id)
        } catch let failure as CloudTurnFailure {
            throw failure
        } catch {
            throw CloudTurnFailure.transport(error.localizedDescription)
        }
    }

    private static func drain(_ bytes: URLSession.AsyncBytes, limit: Int) async -> String {
        var data = Data()
        do {
            for try await byte in bytes {
                data.append(byte)
                if data.count >= limit { break }
            }
        } catch {
            return ""
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
                requestTimeout: requestTimeout
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
        guard let key = CloudSecretStore.shared.get(providerId: provider.id) else { return nil }
        return CloudChatClient(provider: provider, flavor: flavor, apiKey: key)
    }

    public func streamTurn(
        model: String,
        messages: [CloudChatMessage],
        params: CloudChatGenerationParams,
        tools: [ToolDefinition]? = nil
    ) -> AsyncStream<CloudTurnEvent> {
        service.streamTurn(model: model, messages: messages, params: params, tools: tools)
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
