//
//  OpenAIAdapter.swift
//  DynaMoE
//
//  OpenAI-compatible chat completion protocol: Codable request/response
//  types plus the pure prompt-assembly logic that maps an OpenAI message
//  list onto the engine's per-family ChatPromptFormatter templating.
//  Minimal pure state — all networking lives in ServerController.
//
//  Protocol notes:
//  - `top_k` / `min_p` / `repetition_penalty` / `presence_penalty` follow the
//    llama.cpp server extras so local clients (Crush, opencode, Continue) can
//    reach the engine's full sampler.
//  - `enable_thinking` follows the llama.cpp chat-template-kwargs convention.
//

import Foundation
import CryptoKit
import Hummingbird

// MARK: - Request types

struct OpenAIChatRequest: Decodable {
    struct Message: Decodable {
        let role: String
        /// String form, or the concatenated text parts when the client sends
        /// the multi-part content array. Nil for assistant tool-call turns.
        let contentText: String?
        /// True when the client sent an image part (`image_url`, `input_image`,
        /// `image`). Whether the turn can actually read it is decided at plan
        /// time from the loaded checkpoint's tower, not here.
        let hasImagePart: Bool
        /// The image parts this message carried, in the order the client sent
        /// them, already decoded to the bytes the vision tower's decoder reads.
        let imageParts: [ServerImagePart]
        /// True when the client sent a content part the engine cannot process at
        /// all (`file`, `input_audio`, an unknown type). Refused explicitly, so a
        /// caller is never answered about a document nobody read.
        let hasUnsupportedPart: Bool
        /// Reasoning a client echoed back (the `reasoning_content` extension
        /// this server emits). Clients that keep it let the next turn
        /// reproduce the pinned generation instead of only its visible tail.
        let reasoningContent: String?
        let name: String?
        let toolCalls: [AssistantToolCallIn]?
        let toolCallId: String?

        enum CodingKeys: String, CodingKey {
            case role, content, name
            case reasoningContent = "reasoning_content"
            case toolCalls = "tool_calls"
            case toolCallId = "tool_call_id"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.role = try container.decode(String.self, forKey: .role)
            self.name = try container.decodeIfPresent(String.self, forKey: .name)
            self.toolCalls = try container.decodeIfPresent([AssistantToolCallIn].self, forKey: .toolCalls)
            self.toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
            self.reasoningContent = try container.decodeIfPresent(String.self, forKey: .reasoningContent)
            if let s = try? container.decode(String.self, forKey: .content) {
                self.contentText = s
                self.imageParts = []
                self.hasImagePart = false
                self.hasUnsupportedPart = false
            } else if let parts = try? container.decode([ContentPart].self, forKey: .content) {
                let joined = parts.compactMap(\.text).filter { !$0.isEmpty }.joined(separator: "\n")
                self.contentText = joined.isEmpty ? nil : joined
                self.imageParts = parts.filter(\.isImage).map { part in
                    ServerImageParts.fromOpenAIURL(part.imageURL ?? "")
                }
                self.hasImagePart = !self.imageParts.isEmpty
                self.hasUnsupportedPart = parts.contains { part in
                    let type = (part.type ?? "").lowercased()
                    return !type.isEmpty && type != "text" && !part.isImage
                }
            } else {
                self.contentText = nil
                self.imageParts = []
                self.hasImagePart = false
                self.hasUnsupportedPart = false
            }
        }

        /// Convenience constructor for tests and internal assembly.
        init(role: String, contentText: String?, name: String? = nil, toolCalls: [AssistantToolCallIn]? = nil, toolCallId: String? = nil, reasoningContent: String? = nil) {
            self.role = role
            self.contentText = contentText
            self.reasoningContent = reasoningContent
            self.name = name
            self.toolCalls = toolCalls
            self.toolCallId = toolCallId
            self.imageParts = []
            self.hasImagePart = false
            self.hasUnsupportedPart = false
        }
    }

    struct ContentPart: Decodable {
        let type: String?
        let text: String?
        /// The `image_url` value. Chat Completions spells it as an object
        /// (`{"url": "..."}`) and the Responses API as a bare string, so both
        /// shapes decode here rather than failing the whole body.
        let imageURL: String?

        enum CodingKeys: String, CodingKey {
            case type, text
            case imageURL = "image_url"
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.type = try container.decodeIfPresent(String.self, forKey: .type)
            self.text = try container.decodeIfPresent(String.self, forKey: .text)
            if let bare = try? container.decode(String.self, forKey: .imageURL) {
                self.imageURL = bare
            } else if let object = try? container.decode(ImageURLObject.self, forKey: .imageURL) {
                self.imageURL = object.url
            } else {
                self.imageURL = nil
            }
        }

        struct ImageURLObject: Decodable {
            let url: String?
        }

        /// The content-part types that name a picture: `image_url` is the Chat
        /// Completions spelling, `input_image` the Responses one, `image` the
        /// shorthand several clients send.
        static let imageTypes: Set<String> = ["image_url", "input_image", "image"]

        var isImage: Bool {
            if Self.imageTypes.contains((type ?? "").lowercased()) { return true }
            return imageURL != nil
        }
    }

    struct AssistantToolCallIn: Decodable {
        struct Fn: Decodable {
            let name: String
            let arguments: String?
        }
        let id: String?
        let type: String?
        let function: Fn
    }

    struct Tool: Decodable {
        struct Function: Decodable {
            let name: String
            let description: String?
            /// Raw JSON schema object; preserved as pre-encoded JSON text so we
            /// can embed it verbatim in the tool-declaration prompt block.
            let parametersData: Data?
            enum CodingKeys: String, CodingKey { case name, description, parameters }
            init(from decoder: Decoder) throws {
                let container = try decoder.container(keyedBy: CodingKeys.self)
                self.name = try container.decode(String.self, forKey: .name)
                self.description = try container.decodeIfPresent(String.self, forKey: .description)
                if let params = try? container.decode(AnyCodable.self, forKey: .parameters) {
                    self.parametersData = try? JSONEncoder().encode(params)
                } else {
                    self.parametersData = nil
                }
            }
        }
        let type: String?
        let function: Function
    }

    /// `tool_choice` accepts either a bare string (`"none"` / `"auto"` /
    /// `"required"`) or a `{"type":"function","function":{"name":"..."}}`
    /// object. Both forms are decoded so a valid named-function request is not
    /// rejected as an invalid body; the mode is honored (or explicitly
    /// rejected) in `OpenAIAdapter.toolUseEnabled`.
    enum ToolChoice: Decodable, Equatable {
        case auto
        case none
        case required
        case function(name: String)

        private enum CodingKeys: String, CodingKey { case type, function }
        private enum FunctionKeys: String, CodingKey { case name }

        init(from decoder: Decoder) throws {
            if let container = try? decoder.singleValueContainer(),
               let raw = try? container.decode(String.self) {
                switch raw {
                case "none": self = .none
                case "required": self = .required
                default: self = .auto
                }
                return
            }
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let type = try container.decodeIfPresent(String.self, forKey: .type)
            if type == "function",
               let fn = try? container.nestedContainer(keyedBy: FunctionKeys.self, forKey: .function),
               let name = try? fn.decode(String.self, forKey: .name) {
                self = .function(name: name)
            } else {
                self = .auto
            }
        }
    }

    let model: String?
    let messages: [Message]
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let minP: Float?
    let repetitionPenalty: Float?
    let presencePenalty: Float?
    /// OpenAI's `frequency_penalty`, charged per OCCURRENCE of a recent token
    /// where `presence_penalty` is charged once per distinct one.
    let frequencyPenalty: Float?
    /// OpenAI's `logit_bias` in the wire's own string-keyed form. Converted and
    /// range-checked by `logitBias(from:)`, which the planner calls before any
    /// generation work.
    let logitBias: [String: Float]?
    /// OpenAI's current reply-budget field. Decoded because leaving it out of the
    /// decoder answered a request for a 512-token answer with the app default budget
    /// (up to 8192): `replyBudget(maxTokens:maxCompletionTokens:)` resolves it against
    /// the deprecated `max_tokens`.
    let maxCompletionTokens: Int?
    /// OpenAI's older spelling of the reply budget, still sent by most clients.
    let maxTokens: Int?
    /// OpenAI's `parallel_tool_calls`: decoded and deliberately not honoured (see the
    /// decoder note).
    let parallelToolCalls: Bool?
    let stream: Bool?
    let stop: [String]?
    let tools: [Tool]?
    let toolChoice: ToolChoice?
    let enableThinking: Bool?
    let reasoningEffort: String?
    /// OpenAI's structured-output request. Only `type` is decoded: this server
    /// refuses the field outright rather than honouring part of it, so parsing the
    /// schema would only be a way to accept a promise it cannot keep.
    let responseFormat: ResponseFormat?
    /// How many completions the client asked for. OpenAI defaults to 1, which is all
    /// this engine can produce per request.
    let n: Int?
    /// Token log-probabilities. Presence of either field is a request this server
    /// cannot answer, so they are decoded rather than ignored.
    let logprobs: Bool?
    let topLogprobs: Int?
    /// `stream_options.include_usage` decides whether the trailing usage chunk is
    /// emitted, which is the one sub-field of it this server can honour.
    let streamOptions: StreamOptions?
    /// OpenAI's `store`: whether the request may be retained server-side for
    /// distillation or evals. Only `false` is actionable here, and it is honoured
    /// per request — `withholdBodiesForClientRetention()` keeps this turn's prompt
    /// and answer out of the request log even when the operator's Store Request
    /// Content switch is on, because "do not retain me" is a promise the client is
    /// entitled to have kept. `true` cannot force capture: retention on this Mac is
    /// the operator's own decision, and a client may not overrule it.
    let store: Bool?
    /// The output modalities the client asked for: `["text"]` is the protocol's
    /// default, `["text","audio"]` asks for a spoken reply as well. Decoded so
    /// `refuseUnsupportedResponseShape` can refuse a non-text one by name — this
    /// server has no text-to-speech, and a dropped field answered a request for
    /// audio with text and no `message.audio` for the client's parser to find.
    let modalities: [String]?
    /// The settings for that spoken reply (`{voice, format, speed?}`). Decoded for
    /// the same reason as `modalities`, and its two named fields so the refusal can
    /// say which voice and format the client asked for.
    let audio: AudioSettings?
    /// The client's own key/value labels for this request (OpenAI allows 16 pairs
    /// of strings). Decoded because nothing else carries them: this protocol has no
    /// response field to echo them into, so the request log's row is the only place a
    /// caller's labels can survive, and left out of the decoder they survived
    /// nowhere, which made them the last thing in this request a local operator could
    /// never see. They can never move an answer, so they are recorded rather than
    /// refused or obeyed.
    let metadata: [String: String]?

    struct ResponseFormat: Decodable {
        let type: String?
    }

    /// `audio`: everything a client needs to describe a spoken reply this server
    /// cannot produce. Every member is optional because the refusal is keyed on the
    /// object's presence, not on its contents.
    struct AudioSettings: Decodable {
        let voice: String?
        let format: String?
        let speed: Float?
    }

    struct StreamOptions: Decodable {
        let includeUsage: Bool?

        enum CodingKeys: String, CodingKey {
            case includeUsage = "include_usage"
        }
    }

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, stop, tools
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case logitBias = "logit_bias"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case parallelToolCalls = "parallel_tool_calls"
        case toolChoice = "tool_choice"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case responseFormat = "response_format"
        case n, logprobs
        case topLogprobs = "top_logprobs"
        case streamOptions = "stream_options"
        case store
        case modalities, audio
        case metadata
        case seed, user
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.messages = try container.decode([Message].self, forKey: .messages)
        self.temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.minP = try container.decodeIfPresent(Float.self, forKey: .minP)
        self.repetitionPenalty = try container.decodeIfPresent(Float.self, forKey: .repetitionPenalty)
        self.presencePenalty = try container.decodeIfPresent(Float.self, forKey: .presencePenalty)
        self.frequencyPenalty = try container.decodeIfPresent(Float.self, forKey: .frequencyPenalty)
        self.logitBias = try container.decodeIfPresent([String: Float].self, forKey: .logitBias)
        // `seed` and `user` are decoded only to be documented as ignored, the same
        // way the legacy route treats them: upstream calls `seed` a best-effort
        // determinism hint (this engine is already greedy at temperature 0) and
        // `user` an abuse-monitoring label with no local meaning. Neither can move
        // an answer, and a 400 for one would break clients that always send it.
        _ = try container.decodeIfPresent(Int.self, forKey: .seed)
        _ = try container.decodeIfPresent(String.self, forKey: .user)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.maxCompletionTokens = try container.decodeIfPresent(Int.self, forKey: .maxCompletionTokens)
        // `parallel_tool_calls` is decoded to be documented as ignored rather than to
        // be obeyed: honouring `false` means emitting at most one call, which is a
        // grammar constraint the sampler does not have, and refusing it would put a
        // new 400 on the path clients drive daily. The turn may therefore emit several
        // calls, which every client recovers from.
        self.parallelToolCalls = try container.decodeIfPresent(Bool.self, forKey: .parallelToolCalls)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        // OpenAI allows `stop` as a bare string or an array.
        if let single = try? container.decode(String.self, forKey: .stop) {
            self.stop = [single]
        } else {
            self.stop = try container.decodeIfPresent([String].self, forKey: .stop)
        }
        self.tools = try container.decodeIfPresent([Tool].self, forKey: .tools)
        // OpenAI allows a bare string or an object-valued tool_choice; decoding
        // only the string form rejected valid named-function requests as
        // invalid bodies before generation.
        self.toolChoice = try container.decodeIfPresent(ToolChoice.self, forKey: .toolChoice)
        self.enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        // These four are decoded so the planner can refuse them by name. Left out of
        // the decoder entirely they were ignored by Swift's unknown-key tolerance,
        // which answered a request for JSON, for two choices or for log-probabilities
        // exactly as if the client had asked for none of them.
        self.responseFormat = try container.decodeIfPresent(ResponseFormat.self, forKey: .responseFormat)
        self.n = try container.decodeIfPresent(Int.self, forKey: .n)
        self.logprobs = try container.decodeIfPresent(Bool.self, forKey: .logprobs)
        self.topLogprobs = try container.decodeIfPresent(Int.self, forKey: .topLogprobs)
        self.streamOptions = try container.decodeIfPresent(StreamOptions.self, forKey: .streamOptions)
        // Decoded because it is the one field in this protocol whose whole meaning
        // is a promise about the server's own behaviour rather than about the
        // answer: dropped, `store: false` was answered exactly as if the client had
        // asked for the request to be kept.
        self.store = try container.decodeIfPresent(Bool.self, forKey: .store)
        // The two output-modality fields join that list for the same reason: a
        // request asking for a spoken reply was answered as text with no error and
        // no `message.audio`, so the client waited for a field nothing sent.
        self.modalities = try container.decodeIfPresent([String].self, forKey: .modalities)
        self.audio = try container.decodeIfPresent(AudioSettings.self, forKey: .audio)
        // The client's labels, decoded only so the request log can record them: they
        // are the caller's own correlation data, which unknown-key tolerance was
        // dropping into silence one request at a time. Decoded leniently: a map whose
        // values are not strings yields no labels instead of a 400, because the field
        // was ignored outright before this and a client that puts a number in it must
        // not start failing. A malformed map costs the labels for that one request, not
        // the answer.
        let labels = try? container.decodeIfPresent([String: String].self, forKey: .metadata)
        self.metadata = labels ?? nil
    }

    /// True when any message carries an image content part. The turn planner
    /// accepts these only when the loaded checkpoint publishes a vision tower and
    /// refuses them by name otherwise, because dropping them would answer a
    /// question about a picture the model never saw.
    var hasImageContent: Bool {
        messages.contains(where: { $0.hasImagePart })
    }

    /// True when any message carries a content part the engine cannot process at
    /// all (a file attachment or audio). Refused rather than stripped, for the
    /// same reason as `hasImageContent`.
    var hasUnsupportedContent: Bool {
        messages.contains(where: { $0.hasUnsupportedPart })
    }
}

/// The legacy OpenAI text-completions request (`POST /v1/completions`).
///
/// This protocol is prompt-in/text-out: there is no message list, no role, and no
/// tool vocabulary, so it is its own type rather than a shim over
/// `OpenAIChatRequest`. Its `prompt` reaches the same renderer the chat route
/// uses, which is what makes both halves of the endpoint work: a prompt that
/// already carries the checkpoint's turn markers passes through byte for byte (a
/// true continuation), and a plain prompt is rendered as one user turn with the
/// checkpoint's own begin-of-sentence marker and assistant prefix, because a chat
/// model answering after its own prefix is the shape it was trained on.
///
/// Every field here is decoded, including the ones this server refuses, so a
/// request for insertion mode or for several choices is answered with a named
/// error instead of being silently narrowed to "one completion, extras ignored".
nonisolated struct OpenAICompletionsRequest: Decodable {
    /// The resolved prompt text, or nil when the client sent token ids (which
    /// this server refuses by name: it has a tokenizer but no detokenizer, so a
    /// token-id prompt could be run but never reported back to the client).
    let promptText: String?
    /// How many prompts the client sent. The protocol allows a batch; this server
    /// answers one, so more than one is refused rather than silently truncated.
    let promptCount: Int
    let promptIsTokenIDs: Bool
    let model: String?
    let suffix: String?
    let maxTokens: Int?
    let temperature: Float?
    let topP: Float?
    let topK: Int?
    let minP: Float?
    let repetitionPenalty: Float?
    let presencePenalty: Float?
    let frequencyPenalty: Float?
    let n: Int?
    let stream: Bool?
    /// Upstream spells this as an integer (how many alternatives to return per
    /// token) rather than the chat route's boolean. Decoded as a flag because
    /// either spelling means "log-probabilities", and neither is available.
    let logprobsRequested: Bool
    let echo: Bool?
    let stop: [String]?
    let bestOf: Int?
    /// OpenAI's `logit_bias` in the wire's own string-keyed form; converted and
    /// range-checked by `OpenAIAdapter.logitBias(from:)`.
    let logitBias: [String: Float]?
    /// This server's own thinking extension, which this route reads in the opposite
    /// direction from the chat route: see `completionsThinkingRequested`.
    let enableThinking: Bool?
    let reasoningEffort: String?
    let streamOptions: OpenAIChatRequest.StreamOptions?
    /// OpenAI's `store`, which this route accepts too because both routes take the
    /// same retention promise: see `OpenAIChatRequest.store`.
    let store: Bool?
    /// This route carries `metadata` too, so it takes the client's labels into the
    /// same row field: see `OpenAIChatRequest.metadata`.
    let metadata: [String: String]?

    enum CodingKeys: String, CodingKey {
        case model, prompt, suffix, n, stream, echo, stop, seed, user
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case logprobs
        case bestOf = "best_of"
        case logitBias = "logit_bias"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case streamOptions = "stream_options"
        case store
        case metadata
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        // `prompt` is a string, an array of strings, an array of token ids, or an
        // array of token-id arrays. Each shape is tried in turn so the refusal
        // below can name which one was sent.
        if let single = try? container.decode(String.self, forKey: .prompt) {
            self.promptText = single
            self.promptCount = 1
            self.promptIsTokenIDs = false
        } else if let batch = try? container.decode([String].self, forKey: .prompt) {
            self.promptText = batch.first
            self.promptCount = batch.count
            self.promptIsTokenIDs = false
        } else if (try? container.decode([Int].self, forKey: .prompt)) != nil
            || (try? container.decode([[Int]].self, forKey: .prompt)) != nil {
            self.promptText = nil
            self.promptCount = 1
            self.promptIsTokenIDs = true
        } else {
            throw DecodingError.dataCorruptedError(
                forKey: .prompt,
                in: container,
                debugDescription: "prompt must be a string or an array of strings"
            )
        }
        self.model = try container.decodeIfPresent(String.self, forKey: .model)
        self.suffix = try container.decodeIfPresent(String.self, forKey: .suffix)
        self.maxTokens = try container.decodeIfPresent(Int.self, forKey: .maxTokens)
        self.temperature = try container.decodeIfPresent(Float.self, forKey: .temperature)
        self.topP = try container.decodeIfPresent(Float.self, forKey: .topP)
        self.topK = try container.decodeIfPresent(Int.self, forKey: .topK)
        self.minP = try container.decodeIfPresent(Float.self, forKey: .minP)
        self.repetitionPenalty = try container.decodeIfPresent(Float.self, forKey: .repetitionPenalty)
        self.presencePenalty = try container.decodeIfPresent(Float.self, forKey: .presencePenalty)
        self.frequencyPenalty = try container.decodeIfPresent(Float.self, forKey: .frequencyPenalty)
        self.n = try container.decodeIfPresent(Int.self, forKey: .n)
        self.stream = try container.decodeIfPresent(Bool.self, forKey: .stream)
        // `logprobs` is an Int on this route and a Bool on the chat route; either
        // one is a request for log-probabilities, which the engine does not emit.
        if let count = try? container.decode(Int.self, forKey: .logprobs) {
            self.logprobsRequested = count > 0
        } else if let flag = try? container.decode(Bool.self, forKey: .logprobs) {
            self.logprobsRequested = flag
        } else {
            self.logprobsRequested = false
        }
        self.echo = try container.decodeIfPresent(Bool.self, forKey: .echo)
        if let single = try? container.decode(String.self, forKey: .stop) {
            self.stop = [single]
        } else {
            self.stop = try container.decodeIfPresent([String].self, forKey: .stop)
        }
        self.bestOf = try container.decodeIfPresent(Int.self, forKey: .bestOf)
        self.logitBias = try container.decodeIfPresent([String: Float].self, forKey: .logitBias)
        self.enableThinking = try container.decodeIfPresent(Bool.self, forKey: .enableThinking)
        self.reasoningEffort = try container.decodeIfPresent(String.self, forKey: .reasoningEffort)
        self.streamOptions = try container.decodeIfPresent(OpenAIChatRequest.StreamOptions.self, forKey: .streamOptions)
        self.store = try container.decodeIfPresent(Bool.self, forKey: .store)
        // Lenient for the same reason as the chat route: a map whose values are not
        // strings costs the labels, never the answer.
        let labels = try? container.decodeIfPresent([String: String].self, forKey: .metadata)
        self.metadata = labels ?? nil
        // `seed` and `user` are decoded only to be documented as ignored: upstream
        // calls `seed` a best-effort determinism hint (this server is already greedy
        // at temperature 0) and `user` an abuse-monitoring label with no local
        // meaning. Neither can change an answer, so neither is refused.
        _ = try container.decodeIfPresent(Int.self, forKey: .seed)
        _ = try container.decodeIfPresent(String.self, forKey: .user)
    }
}

// MARK: - Response types

struct OpenAIUsage: Encodable {
    let promptTokens: Int
    let completionTokens: Int
    var totalTokens: Int { promptTokens + completionTokens }

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(promptTokens, forKey: .promptTokens)
        try container.encode(completionTokens, forKey: .completionTokens)
        try container.encode(totalTokens, forKey: .totalTokens)
    }
}

struct OpenAIToolCallOut: Encodable {
    struct Fn: Encodable {
        let name: String
        let arguments: String
    }
    /// Streaming-only index (nil in the full-response object form).
    let index: Int?
    let id: String
    let type: String
    let function: Fn
}

struct OpenAIResponseMessage: Encodable {
    let role: String = "assistant"
    let content: String?
    let reasoningContent: String?
    let toolCalls: [OpenAIToolCallOut]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

struct OpenAIChatResponse: ResponseEncodable {
    struct Choice: Encodable {
        let index: Int
        let message: OpenAIResponseMessage
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case index, message
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: OpenAIUsage
}

struct OpenAIStreamDelta: Encodable {
    let role: String?
    let content: String?
    let reasoningContent: String?
    let toolCalls: [OpenAIToolCallOut]?

    enum CodingKeys: String, CodingKey {
        case role, content
        case toolCalls = "tool_calls"
        case reasoningContent = "reasoning_content"
    }
}

struct OpenAIChatChunk: Encodable {
    struct Choice: Encodable {
        let index: Int
        let delta: OpenAIStreamDelta
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index, delta
            case finishReason = "finish_reason"
        }
    }

    let id: String
    let object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
}

/// One completed legacy text completion.
///
/// The object is `text_completion` and a choice is `text`/`index`/`logprobs`/
/// `finish_reason`, with no message and no role: this protocol predates both. Its
/// `logprobs` key is written as an explicit null rather than omitted, because that
/// is what upstream sends and a strict decoder reads the key as always-present.
nonisolated struct OpenAICompletionsResponse: ResponseEncodable {
    struct Choice: Encodable {
        let text: String
        let index: Int
        let finishReason: String

        enum CodingKeys: String, CodingKey {
            case text, index, logprobs
            case finishReason = "finish_reason"
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(text, forKey: .text)
            try container.encode(index, forKey: .index)
            try container.encodeNil(forKey: .logprobs)
            try container.encode(finishReason, forKey: .finishReason)
        }
    }

    let id: String
    let object: String = "text_completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: OpenAIUsage

    enum CodingKeys: String, CodingKey {
        case id, object, created, model, choices, usage
    }
}

/// One streamed legacy text completion delta.
nonisolated struct OpenAICompletionsChunk: Encodable {
    struct Choice: Encodable {
        let text: String
        let index: Int
        let finishReason: String?

        enum CodingKeys: String, CodingKey {
            case text, index, logprobs
            case finishReason = "finish_reason"
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(text, forKey: .text)
            try container.encode(index, forKey: .index)
            try container.encodeNil(forKey: .logprobs)
            if let finishReason {
                try container.encode(finishReason, forKey: .finishReason)
            } else {
                try container.encodeNil(forKey: .finishReason)
            }
        }
    }

    let id: String
    let object: String = "text_completion"
    let created: Int
    let model: String
    let choices: [Choice]
}

struct OpenAIModelList: ResponseEncodable {
    /// What a client may ask this model to do.
    ///
    /// Every field is copied from the fact the turn planner reads, never written
    /// by hand, so what is advertised cannot drift from what a request is refused
    /// for: `vision` decides whether an attachment is read or answered with a 400,
    /// and `thinking`/`thinkingLevels` decide whether a reasoning request is
    /// honoured or quietly ignored. A client that reads them can predict its own
    /// rejection instead of discovering it. The object is additive to the OpenAI
    /// schema on purpose; clients ignore keys they do not know.
    struct Capabilities: Encodable {
        var vision: Bool
        var thinking: Bool
        /// True when the family renders a graded effort level (High/Low/Off) into
        /// its system block, rather than a plain on/off switch.
        var thinkingLevels: Bool
        /// The `reasoning_effort` values this server accepts, low to high.
        ///
        /// `thinkingLevels` says a graded level is honoured and nothing about which
        /// words name one, which left a client to guess. A client that guessed this
        /// engine's own two ("low", "high") was served, while one that sent the
        /// protocol's `medium` met a 400 it could not have predicted from here. The
        /// list is the adapter's own accepted vocabulary, never a second copy of it,
        /// and it is nil beside `thinkingLevels: false` for the reason `visionCost`
        /// is nil beside `vision: false`.
        var reasoningEffortValues: [String]?
        /// What one attachment is expected to cost, when this family reads them.
        ///
        /// `vision` alone says a picture is accepted and nothing about whether that
        /// picture is 184 prompt positions and a third of a second or 994 and seven.
        /// A client picking a request timeout, or deciding whether a near-full
        /// prompt has room, has to know which. Nil beside `vision: false`, and
        /// rounded because it is a model of a machine class rather than a promise.
        var visionCost: VisionCost?

        enum CodingKeys: String, CodingKey {
            case vision, thinking
            case thinkingLevels = "thinking_levels"
            case reasoningEffortValues = "reasoning_effort_values"
            case visionCost = "vision_cost"
        }

        struct VisionCost: Encodable {
            var minPromptPositions: Int
            var maxPromptPositions: Int
            var minPatchTokens: Int
            var maxPatchTokens: Int
            var minEncodeSeconds: Double
            var maxEncodeSeconds: Double

            enum CodingKeys: String, CodingKey {
                case minPromptPositions = "min_prompt_positions"
                case maxPromptPositions = "max_prompt_positions"
                case minPatchTokens = "min_patch_tokens"
                case maxPatchTokens = "max_patch_tokens"
                case minEncodeSeconds = "min_encode_seconds"
                case maxEncodeSeconds = "max_encode_seconds"
            }

            init(_ bounds: DeepSeekV41VisionPricing.Bounds) {
                minPromptPositions = bounds.minSpanPositions
                maxPromptPositions = bounds.maxSpanPositions
                minPatchTokens = bounds.minPatchTokens
                maxPatchTokens = bounds.maxPatchTokens
                minEncodeSeconds = VisionCost.rounded(bounds.minSeconds)
                maxEncodeSeconds = VisionCost.rounded(bounds.maxSeconds)
            }

            /// Two decimals, so a client sees an estimate rather than a fabricated
            /// microsecond. The composer's own chip ("about 2.9 s") is the same
            /// posture.
            private static func rounded(_ seconds: Double) -> Double {
                (seconds * 100).rounded() / 100
            }
        }
    }

    struct Model: Encodable {
        let id: String
        let object: String = "model"
        let created: Int
        let ownedBy: String = "dynamoe"
        /// The checkpoint's own architectural ceiling (`max_position_embeddings`),
        /// not what this machine's KV budget affords. A client asking "how much
        /// fits" wants the weights' answer; the server's own refusal threshold
        /// travels separately as `max_prompt_tokens`.
        let contextLength: Int?
        /// The prompt length this server refuses outright
        /// (`dynamoe_server_max_prompt_tokens`). Nil when the server has no cap.
        let maxPromptTokens: Int?
        let capabilities: Capabilities?

        enum CodingKeys: String, CodingKey {
            case id, object, created, capabilities
            case ownedBy = "owned_by"
            case contextLength = "context_length"
            case maxPromptTokens = "max_prompt_tokens"
        }
    }

    /// The one entry this server advertises for a loaded checkpoint.
    ///
    /// Kept a function of the facts so the mapping is testable without a loaded
    /// model: a swapped flag here is silent on the wire and would tell a client
    /// the opposite of what the planner does. The id is the checkpoint's served id
    /// rather than its display name, because a client resolves it from here and
    /// then addresses the retrieve route with it as a path segment.
    static func entry(
        for info: CompletionEngine.LoadedModelInfo,
        contextLength: Int?,
        maxPromptTokens: Int?,
        created: Int
    ) -> Model {
        Model(
            id: info.servedID,
            created: created,
            contextLength: contextLength,
            maxPromptTokens: maxPromptTokens,
            capabilities: Capabilities(
                vision: info.supportsVision,
                thinking: info.supportsThinking,
                thinkingLevels: info.supportsThinkingLevels,
                reasoningEffortValues: info.supportsThinkingLevels
                    ? OpenAIAdapter.reasoningEffortValues
                    : nil,
                visionCost: info.supportsVision
                    ? info.visionCost.map { Capabilities.VisionCost($0) }
                    : nil
            )
        )
    }

    let object: String = "list"
    let data: [Model]
}

struct OpenAIErrorEnvelope: Encodable {
    struct Body: Encodable {
        let message: String
        let type: String
        let code: String?
    }
    let error: Body
}

// MARK: - Pure protocol mapping

/// Pure protocol-mapping logic for the OpenAI adapter. Every function is a
/// free static so it is unit-testable without a live engine or socket.
enum OpenAIAdapter {

    /// Whether tools may be offered for this request. OpenAI's forced choices
    /// (`"required"`, or a named-function object) demand a generation
    /// constraint this server cannot enforce; silently treating them as `auto`
    /// would let the model answer with a different tool or no tool at all.
    /// Reject them explicitly instead.
    static func toolUseEnabled(_ choice: OpenAIChatRequest.ToolChoice?) throws -> Bool {
        switch choice {
        case nil, .some(.auto):
            return true
        case .some(.none):
            return false
        case .some(.required):
            throw ServerAPIError.badRequest("tool_choice \"required\" is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".")
        case .some(.function(let name)):
            throw ServerAPIError.badRequest(
                "tool_choice for a specific function (\"\(name)\") is not supported: this server cannot force a tool call. Use \"auto\" or \"none\".",
                logSafeMessage: "tool_choice for a specific function is not supported: this server cannot force a tool call. Use \"auto\" or \"none\"."
            )
        }
    }

    /// Converts OpenAI's `logit_bias` object into the additive bias the sampler
    /// applies, refusing by name what it could not honour.
    ///
    /// The wire keys token ids as decimal strings and upstream bounds a bias to
    /// -100...100. A key that is not a token id, or a bias outside that range, is
    /// refused here rather than clamped: a clamped value answers a request the
    /// client did not send, which is the silent lie this posture exists to remove.
    /// An id beyond the LOADED checkpoint's vocabulary is deliberately NOT refused,
    /// because the vocabulary belongs to the model rather than to the protocol: such
    /// an id has no candidate logit to bias, so the sampler skips it.
    nonisolated static func logitBias(from raw: [String: Float]?) throws -> [Int32: Float] {
        guard let raw, !raw.isEmpty else { return [:] }
        var resolved: [Int32: Float] = [:]
        resolved.reserveCapacity(raw.count)
        for (key, bias) in raw {
            guard let id = Int32(key.trimmingCharacters(in: .whitespaces)), id >= 0 else {
                throw ServerAPIError.badRequest(
                    "logit_bias carries the key \"\(key)\", which is not a token id. Send token ids "
                        + "as decimal strings, e.g. {\"50256\": -100}.",
                    logSafeMessage: "logit_bias carries a key that is not a token id"
                )
            }
            guard bias >= -100, bias <= 100 else {
                throw ServerAPIError.badRequest(
                    "logit_bias for token \(id) is \(bias): this server honours OpenAI's range of "
                        + "-100 to 100. Clamp the value or omit the entry."
                )
            }
            // A zero bias changes nothing, and carrying it would only widen the
            // sampler's touched set.
            if abs(bias) > 0.0001 { resolved[id] = bias }
        }
        return resolved
    }

    /// OpenAI's `frequency_penalty`, in the range the sampler can honour.
    ///
    /// Negative values reward repetition and positive values punish it, exactly as
    /// upstream; the value rides `TurnSampling` and is charged per OCCURRENCE of a
    /// recent token, which is what separates it from `presence_penalty`.
    nonisolated static func frequencyPenalty(_ value: Float?) throws -> Float {
        guard let value else { return 0 }
        guard value >= -2.0, value <= 2.0 else {
            throw ServerAPIError.badRequest(
                "frequency_penalty is \(value): this server honours OpenAI's range of -2.0 to 2.0. "
                    + "Clamp the value or omit frequency_penalty."
            )
        }
        return abs(value) > 0.001 ? value : 0
    }

    /// The reply budget a request asks for, resolved across OpenAI's two spellings.
    ///
    /// `max_completion_tokens` is the field OpenAI kept (and the only one its newer
    /// models accept); `max_tokens` is the deprecated spelling most clients still
    /// send. Neither was a lie while ONLY `max_tokens` was decoded, but leaving
    /// `max_completion_tokens` out of the decoder entirely meant a client that sent
    /// `{"max_completion_tokens": 512}` and no `max_tokens` was answered with the app
    /// default budget of up to 8192 tokens — the same class of silent substitution
    /// `#88` removed for `response_format` and `n`.
    ///
    /// A request that carries BOTH fields with different values is refused rather
    /// than resolved by preference: whichever value the server picked, the answer
    /// would be a budget the client did not ask for, and the two fields being present
    /// together means the client itself is unsure which one its SDK sends.
    nonisolated static func replyBudget(
        maxTokens: Int?,
        maxCompletionTokens: Int?
    ) throws -> Int? {
        switch (maxTokens, maxCompletionTokens) {
        case let (old?, new?) where old != new:
            throw ServerAPIError.badRequest(
                "max_tokens = \(old) and max_completion_tokens = \(new) disagree: send one of them. "
                    + "This server honours whichever arrives, they are the same budget, and it will "
                    + "not pick a value the client did not mean."
            )
        case let (_, new?):
            return new
        case let (old?, _):
            return old
        case (nil, nil):
            return nil
        }
    }

    /// Refuse the request fields this server cannot honour, by name.
    ///
    /// Swift's `Decodable` ignores the keys it does not know, which is what made
    /// these fields a silent lie: a client that asked for `response_format:
    /// json_object` was answered in prose, one that asked for `n: 2` got one choice,
    /// and one that asked for `logprobs` got a response carrying none — each with a
    /// 200 and nothing on the wire to tell it apart from the request it meant to
    /// send. Forced `tool_choice` already established this codebase's rule for the
    /// class: refuse it explicitly, because a client that planned around an answer it
    /// did not receive fails somewhere further from the cause than this 400.
    ///
    /// `stream_options.include_usage` is deliberately NOT on this list: it asks for
    /// something this server can do, so the streaming writer honours it instead.
    ///
    /// The two sampler fields (`frequency_penalty`, `logit_bias`) are neither
    /// refused nor ignored: the sampler charges a per-occurrence penalty and biases
    /// a named token, so they are VALIDATED here and resolved into the turn's
    /// sampling. Validating rather than refusing keeps a new 400 off the path
    /// clients already drive daily.
    nonisolated static func refuseUnsupportedResponseShape(_ request: OpenAIChatRequest) throws {
        // Output modalities first: they describe what the RESPONSE should be, so a
        // request that asks for audio is answered before any other field is weighed.
        // `modalities: ["text"]` is the protocol's default and is served, because a
        // client naming the default is not asking for anything this server lacks.
        if let audio = request.audio {
            throw ServerAPIError.badRequest(
                "audio output is not supported: this server runs a text model and has no "
                    + "text-to-speech, so the spoken reply your client would wait for is never produced"
                    + (audio.format.map { " (it asked for \($0))" } ?? "")
                    + ". Omit audio and modalities, and synthesise speech on your side if you need it.",
                logSafeMessage: "audio output requested; this server has no text-to-speech"
            )
        }
        let unservable = unservableModalities(request)
        if !unservable.isEmpty {
            throw ServerAPIError.badRequest(
                "modalities \(unservable.map { "\"\($0)\"" }.joined(separator: ", ")) not supported: "
                    + "this server produces text only, so the reply would arrive without the output the "
                    + "request asked for. Send {\"modalities\": [\"text\"]} or omit modalities.",
                logSafeMessage: "modalities names an output this server cannot produce"
            )
        }
        switch request.responseFormat?.type ?? "text" {
        case "text":
            break
        case "json_object":
            throw ServerAPIError.badRequest(
                "response_format \"json_object\" is not supported: this server cannot constrain "
                    + "decoding to JSON, so a turn that expects JSON would receive prose. Ask for the "
                    + "JSON shape in the prompt and validate it on your side, or omit response_format."
            )
        case "json_schema":
            throw ServerAPIError.badRequest(
                "response_format \"json_schema\" is not supported: this server cannot enforce a "
                    + "schema on the model's output. Ask for the JSON shape in the prompt and validate "
                    + "it on your side, or omit response_format."
            )
        case let other:
            throw ServerAPIError.badRequest(
                "response_format \"\(other)\" is not a type this server knows.",
                logSafeMessage: "response_format carries a type this server does not know"
            )
        }
        if let n = request.n, n != 1 {
            throw ServerAPIError.badRequest(
                "n = \(n) is not supported: this server generates one completion per request. Send "
                    + "one request per completion (they queue), or omit n."
            )
        }
        if request.logprobs == true || request.topLogprobs != nil {
            throw ServerAPIError.badRequest(
                "logprobs is not supported: this server's responses carry no token log-probabilities, "
                    + "so a client scoring its own output would silently receive none. Omit logprobs "
                    + "and top_logprobs."
            )
        }
        // Resolved rather than refused (see the note above), and validated before the
        // model gate like every other refusal so the ordering a test proves holds.
        _ = try frequencyPenalty(request.frequencyPenalty)
        _ = try logitBias(from: request.logitBias)
        // The reply budget is resolved from either spelling, so two disagreeing
        // spellings are a 400 here rather than a choice the server made for the client.
        _ = try replyBudget(
            maxTokens: request.maxTokens,
            maxCompletionTokens: request.maxCompletionTokens
        )
    }

    /// The requested output modalities this server cannot produce, in the order the
    /// client named them. `text` is the one modality the engine has, and it is also
    /// the protocol's DEFAULT (SDKs send `["text"]` explicitly), so naming it is not
    /// a request for anything and never refuses; every other value is a promise the
    /// response cannot keep.
    nonisolated static func unservableModalities(_ request: OpenAIChatRequest) -> [String] {
        (request.modalities ?? []).filter { $0.lowercased() != "text" }
    }

    /// The same posture for the legacy text-completions route, over that route's
    /// own field set.
    ///
    /// It is a separate function rather than a widening of the chat refusal
    /// because the two routes carry different fields and this one is new: nothing
    /// that works against the chat route today starts failing because of a rule
    /// added here. The sampler fields (`frequency_penalty`, `logit_bias`) used to be
    /// named here as unsupported because the chat route still ignored them; now that
    /// both routes resolve them into the sampler, this route validates them the same
    /// way instead of refusing them.
    nonisolated static func refuseUnsupportedCompletionsShape(_ request: OpenAICompletionsRequest) throws {
        if request.promptIsTokenIDs {
            throw ServerAPIError.badRequest(
                "prompt as token ids is not supported: this server tokenizes but does not detokenize, "
                    + "so it could run an id prompt but never report what it completed. Send prompt as text."
            )
        }
        if request.promptCount != 1 {
            throw ServerAPIError.badRequest(
                "prompt carries \(request.promptCount) prompts: this server completes one prompt per "
                    + "request. Send one request per prompt (they queue)."
            )
        }
        guard let prompt = request.promptText, !prompt.isEmpty else {
            throw ServerAPIError.badRequest("prompt is empty: send the text to continue.")
        }
        if request.suffix != nil {
            throw ServerAPIError.badRequest(
                "suffix is not supported: this server cannot do insertion mode, so the text it returned "
                    + "would not fit the gap the suffix describes. Omit suffix."
            )
        }
        if let n = request.n, n != 1 {
            throw ServerAPIError.badRequest(
                "n = \(n) is not supported: this server generates one completion per request. Send "
                    + "one request per completion (they queue), or omit n."
            )
        }
        if request.logprobsRequested {
            throw ServerAPIError.badRequest(
                "logprobs is not supported: this server's responses carry no token log-probabilities, "
                    + "so a client scoring its own output would silently receive none. Omit logprobs."
            )
        }
        if let bestOf = request.bestOf, bestOf != 1 {
            throw ServerAPIError.badRequest(
                "best_of = \(bestOf) is not supported: this server runs one candidate per request, so "
                    + "best_of would change nothing about the completion it returned. Omit best_of."
            )
        }
        if request.logitBias != nil || request.frequencyPenalty != nil {
            _ = try frequencyPenalty(request.frequencyPenalty)
            _ = try logitBias(from: request.logitBias)
        }
    }

    /// Whether the legacy route renders the reasoning channel into its prompt.
    ///
    /// OFF unless the client asks, which is the opposite of the chat route's default
    /// and deliberate: `text_completion` has no field to carry reasoning, so a turn
    /// that reasons first spends its whole budget there and returns a trace instead of
    /// the continuation the caller asked for (measured on the real checkpoint: eight
    /// tokens of "The user asks a simple factual question</think>" and no answer). A
    /// client that wants the reasoning can say so with this server's own
    /// `enable_thinking`, and then the fold in `completionsText` carries it in `text`.
    nonisolated static func completionsThinkingRequested(_ request: OpenAICompletionsRequest) -> Bool {
        request.enableThinking ?? false
    }

    /// The legacy route's prompt as the one shape the turn planner reads.
    ///
    /// There is no system field on this protocol, so the system prompt is empty and
    /// the checkpoint's own default system block is what the renderer writes — the
    /// same thing the chat route produces for a request that sent no system
    /// message. The prompt is a single user turn; the renderer decides from the
    /// text itself whether it is a plain question or an already-templated
    /// continuation.
    nonisolated static func completionsSplit(_ request: OpenAICompletionsRequest) -> ServerPromptSplit {
        ServerPromptSplit(systemPrompt: "", turns: [.init(role: .user, text: request.promptText ?? "")])
    }

    /// OpenAI's documented `reasoning_effort` vocabulary, low to high, and the
    /// engine level each value selects.
    ///
    /// The protocol names four graded values and the OpenAI SDK sends `medium` by
    /// default for reasoning models, so a decoder that knew only this engine's own
    /// two words refused half the vocabulary: a default-configured client met an HTTP
    /// 400 saying the supported values were "low and high" for sending the value its
    /// own protocol documents. The vocabulary is mapped onto the two levels the
    /// engine exposes instead, `minimal` and `low` onto the short budget and
    /// `medium` and `high` onto the full one.
    ///
    /// `medium` landing on high is the rule the app already applies to its own
    /// stored level (`ThinkingEffort.clamped`), so a client and the chat UI cannot
    /// read the same word two ways. This table is the only place the vocabulary
    /// lives: the mapper below and the `/v1/models` advertisement both read it, so
    /// what is accepted and what is announced cannot drift apart.
    nonisolated static let reasoningEffortVocabulary: [(value: String, level: ThinkingEffort)] = [
        ("minimal", .low),
        ("low", .low),
        ("medium", .high),
        ("high", .high)
    ]

    /// The values a client may send, for the models route to advertise.
    nonisolated static var reasoningEffortValues: [String] {
        reasoningEffortVocabulary.map(\.value)
    }

    /// The level a client's `reasoning_effort` selects, or nil when the string is
    /// not one of the protocol's own values. Case and surrounding whitespace are
    /// ignored, because a hand-written request is not required to be canonical.
    nonisolated static func thinkingEffortLevel(forProtocolValue raw: String?) -> ThinkingEffort? {
        guard let raw else { return nil }
        let normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return reasoningEffortVocabulary.first { $0.value == normalized }?.level
    }

    /// Maps the client's `reasoning_effort` onto a graded level. Returns nil
    /// when the field is absent or thinking is explicitly off. A value outside
    /// the protocol's vocabulary, or a level the loaded model cannot honor, is
    /// rejected rather than silently dropped.
    nonisolated static func thinkingEffort(
        from raw: String?,
        levelCapable: Bool,
        thinkingRequested: Bool
    ) throws -> ThinkingEffort? {
        guard let raw, !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        guard thinkingRequested else { return nil }
        guard levelCapable else {
            throw ServerAPIError.badRequest(
                "reasoning_effort is not supported by the loaded model; use enable_thinking for on/off thinking.",
                logSafeMessage: "reasoning_effort is not supported by the loaded model"
            )
        }
        guard let effort = thinkingEffortLevel(forProtocolValue: raw) else {
            throw ServerAPIError.badRequest(
                "Unsupported reasoning_effort value. Supported values are "
                    + reasoningEffortValues.joined(separator: ", ") + ".",
                logSafeMessage: "Unsupported reasoning_effort value"
            )
        }
        return effort
    }

    /// Maps the engine's terminal reasons onto OpenAI finish_reason values.
    static func finishReasonString(_ reason: TurnFinishReason, hasToolCalls: Bool) -> String {
        switch reason {
        case .lengthCapped:
            return "length"
        case .toolFreeze:
            return hasToolCalls ? "tool_calls" : "stop"
        case .eos, .endTag, .degenerateCycle, .stoppedBySequence, .stopped:
            return hasToolCalls ? "tool_calls" : "stop"
        }
    }

    /// Renders the OpenAI tool schemas as a compact declaration block that is
    /// appended to the effective system prompt. Mirrors the shape the in-app
    /// agent system prompt uses (a JSON array of function definitions) so
    /// grammar-masked tool-call models recognize the vocabulary.
    ///
    /// The instruction teaches the engine's canonical tagged call markup, not
    /// bare JSON: the bare-JSON fallback only recognizes built-in tool names,
    /// so a client-defined tool emitted as bare JSON would be returned as
    /// ordinary text. The tagged markup parses for any tool name and is the
    /// exact shape the grammar masks enforce, so constrained sampling stays
    /// compatible.
    static func toolDeclarationText(_ tools: [OpenAIChatRequest.Tool]) -> String? {
        let fns = tools.filter { ($0.type ?? "function") == "function" }
        guard !fns.isEmpty else { return nil }
        var definitions: [[String: String]] = []
        for tool in fns {
            var def: [String: String] = ["name": tool.function.name]
            if let desc = tool.function.description, !desc.isEmpty {
                def["description"] = desc
            }
            if let params = tool.function.parametersData,
               let paramsJSON = String(data: params, encoding: .utf8) {
                def["parameters"] = paramsJSON
            }
            definitions.append(def)
        }
        guard let data = try? JSONSerialization.data(withJSONObject: definitions, options: [.sortedKeys]),
              let json = String(data: data, encoding: .utf8) else { return nil }
        let open = StreamingToolParser.qwenToolCallOpen
        let close = StreamingToolParser.qwenToolCallClose
        let fnOpen = StreamingToolParser.qwenFunctionOpen
        let fnClose = StreamingToolParser.qwenFunctionClose
        let paramOpen = "<parameter="
        let paramClose = "</parameter>"
        return """
        You have access to the following functions. When you need to use one, emit a single call in the tagged form below, then stop; the client will execute the call and send the result back before your next turn. Do not fabricate results.

        \(open)\(fnOpen)FUNCTION_NAME>\(paramOpen)ARG_NAME>ARG_VALUE\(paramClose)\(fnClose)\(close)

        Wrap each argument as \(paramOpen)ARG_NAME>ARG_VALUE\(paramClose); omit the parameter markup entirely for a call with no arguments. Emit the call alone, with no extra prose after the closing tag.

        FUNCTIONS: \(json)
        """
    }

    /// Splits the message list into the system prompt and role-tagged turns.
    ///
    /// History stays role-structured (rather than being flattened into one
    /// user block with "User:"/"Assistant:" labels) so the server can render it
    /// with the model family's native role boundaries. A flattened transcript
    /// re-tokenizes the whole conversation every turn, so each turn's prompt
    /// can no longer extend the previous turn's pinned prompt+generation and
    /// multi-turn requests fully re-prefill.
    static func splitMessages(_ messages: [OpenAIChatRequest.Message]) throws -> ServerPromptSplit {
        let systemTexts = messages
            .filter { $0.role == "system" || $0.role == "developer" }
            .compactMap(\.contentText)
            .filter { !$0.isEmpty }
        let systemPrompt = systemTexts.joined(separator: "\n\n")

        let conversation = messages.filter { $0.role != "system" && $0.role != "developer" }
        guard !conversation.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one non-system message")
        }

        var split = ServerPromptSplit(systemPrompt: systemPrompt)
        for message in conversation {
            // A picture belongs to the turn it was sent with. Only a user turn can
            // carry one: an image on an assistant or tool message has no position
            // in the rendered template, so it is refused rather than moved.
            if message.hasImagePart, message.role != "user" {
                throw ServerAPIError.unsupportedContent(
                    "An image part can only be attached to a user message, not to a \"\(message.role)\" message."
                )
            }
            switch message.role {
            case "user":
                if message.hasImagePart { split.images[split.turns.count] = message.imageParts }
                split.turns.append(.init(role: .user, text: message.contentText ?? ""))
            case "assistant":
                var lines: [String] = []
                // Reasoning a client echoed back is re-rendered ahead of the
                // answer, closed with the canonical tag, so the turn reproduces
                // what was generated rather than only its visible tail.
                if let reasoning = message.reasoningContent?.trimmingCharacters(in: .whitespacesAndNewlines),
                   !reasoning.isEmpty {
                    lines.append(reasoning + TurnTextSplitter.canonicalReasoningCloseTag)
                }
                if let content = message.contentText, !content.isEmpty {
                    lines.append("\(content)")
                }
                if let calls = message.toolCalls, !calls.isEmpty {
                    let rendered = calls.map { call in
                        var obj: [String: String] = ["name": call.function.name]
                        // Keep the call id: the tool-result turns below
                        // reference it via tool_call_id, and with several calls
                        // (or repeated calls to the same function) the id is the
                        // only thing tying a result back to its arguments.
                        if let id = call.id, !id.isEmpty {
                            obj["id"] = id
                        }
                        if let args = call.function.arguments, !args.isEmpty {
                            obj["arguments"] = args
                        }
                        if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys]),
                           let json = String(data: data, encoding: .utf8) {
                            return json
                        }
                        return "{\"name\": \"\(call.function.name)\"}"
                    }
                    .joined(separator: "\n")
                    lines.append(ChatPromptFormatter.openAIToolCallEchoPrefix + "\n" + rendered)
                }
                let body = lines.joined(separator: "\n\n")
                split.turns.append(.init(role: .assistant, text: body.isEmpty ? "(no content)" : body))
            case "tool", "function":
                let label = message.name ?? message.toolCallId ?? "tool"
                let args = message.contentText ?? "(no result)"
                split.turns.append(.init(role: .user, text: "Function result (\(label)):\n\(args)"))
            default:
                if let content = message.contentText, !content.isEmpty {
                    split.turns.append(.init(role: .user, text: "\(message.role.capitalized): \(content)"))
                }
            }
        }
        guard !split.turns.isEmpty else {
            throw ServerAPIError.badRequest("messages must contain at least one non-system message")
        }
        return split
    }

    /// Stable prefix-cache session key. The OpenAI protocol is stateless, so
    /// identity is anchored on the conversation's FIRST USER message plus the
    /// model: stateless clients resend the full history, that message never
    /// changes as a conversation grows, and — unlike the leading system
    /// message, which every conversation a client opens usually shares — it is
    /// what distinguishes unrelated conversations. Deterministic UUID (FNV-1a
    /// over two lanes); pure and testable. Two conversations that open
    /// identically still collide, so reuse is additionally gated on exact
    /// token-prefix equality inside PrefixCacheManager.findCommonPrefix and on
    /// `ServerTurnMemory` re-validating the rendered history.
    ///
    /// A picture is part of that identity. Two requests can carry the same TEXT
    /// and different images, and both render to the same list of image token ids,
    /// so neither the anchor's text nor token-prefix equality can separate them:
    /// without the digest folded in here, the second picture's turn would splice
    /// the first one's recorded generation and reuse its pinned prefix.
    static func sessionKey(for messages: [OpenAIChatRequest.Message], model: String?) -> UUID {
        var lane0: UInt64 = 0xcbf29ce484222325
        var lane1: UInt64 = 0x9e3779b97f4a7c15
        func mix(_ byte: UInt8) {
            lane0 = (lane0 ^ UInt64(byte)) &* 0x100000001b3
            lane1 = (lane1 ^ UInt64(byte)) &* 0x00000100000001b3
            lane1 = (lane1 << 7) | (lane1 >> 57)
        }
        func mixString(_ s: String) {
            for byte in s.utf8 { mix(byte) }
            mix(0x1f)
        }
        mixString(model ?? "")
        if let anchor = messages.first(where: { $0.role == "user" }) ?? messages.first {
            mixString(anchor.role)
            mixString(anchor.contentText ?? "")
            for call in anchor.toolCalls ?? [] {
                mixString("call:" + call.function.name)
                mixString(call.function.arguments ?? "")
            }
            mixString(anchor.name ?? "")
            if !anchor.imageParts.isEmpty {
                mixString("images:" + ServerImageParts.digest(anchor.imageParts))
            }
        }

        return uuid(fromLanes: (lane0, lane1))
    }

    /// Stable prefix-cache session key for the legacy text-completions route.
    ///
    /// That protocol has no message list, so identity is the prompt itself: two
    /// requests whose prompts are byte-identical are the same continuation, and any
    /// difference (including one that is already templated) is a different one. The
    /// model is mixed in for the reason the chat route mixes it: a model switch must
    /// not inherit the previous model's pinned prefix. `"completions"` is mixed in
    /// first so a completion prompt and a chat message with the same text cannot
    /// collide on one session.
    nonisolated static func sessionKey(forPrompt prompt: String, model: String?) -> UUID {
        var lane0: UInt64 = 0xcbf29ce484222325
        var lane1: UInt64 = 0x9e3779b97f4a7c15
        func mix(_ byte: UInt8) {
            lane0 = (lane0 ^ UInt64(byte)) &* 0x100000001b3
            lane1 = (lane1 ^ UInt64(byte)) &* 0x00000100000001b3
            lane1 = (lane1 << 7) | (lane1 >> 57)
        }
        func mixString(_ s: String) {
            for byte in s.utf8 { mix(byte) }
            mix(0x1f)
        }
        mixString("completions")
        mixString(model ?? "")
        mixString(prompt)
        return uuid(fromLanes: (lane0, lane1))
    }

    /// The one place the two lanes become a UUID, so both key derivations agree on
    /// the bit layout (including the RFC 4122 v4 bits downstream consumers parse).
    nonisolated private static func uuid(fromLanes lanes: (UInt64, UInt64)) -> UUID {
        var lane0 = lanes.0
        var lane1 = lanes.1
        var bytes = [UInt8](repeating: 0, count: 16)
        withUnsafeMutableBytes(of: &lane0) { bytes.replaceSubrange(0..<8, with: $0) }
        withUnsafeMutableBytes(of: &lane1) { bytes.replaceSubrange(8..<16, with: $0) }
        // RFC 4122 v4 bits so downstream UUID consumers parse it cleanly.
        bytes[6] = (bytes[6] & 0x0f) | 0x40
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        return UUID(uuid: (bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7], bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]))
    }
}
