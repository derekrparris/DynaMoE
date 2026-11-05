//
//  ServerImageParts.swift
//  DynaMoE
//
//  Phase: server image content parts (slice #77)
//
//  Both wire protocols describe a picture differently. OpenAI nests a URL inside
//  an `image_url` object (a bare string in the Responses spelling), and Anthropic
//  carries a `source` object whose `type` is `base64` or `url`. The turn planner
//  must not care: this is the one form both adapters decode into, so there is one
//  thing to price, one thing to refuse by name, and one thing to hand the engine.
//
//  The refusal cases are first-class rather than thrown at decode time, because a
//  model with no tower should answer "this checkpoint cannot see images" before
//  the request is rejected for its base64.
//

import CryptoKit
import Foundation

nonisolated struct ServerImagePart: Sendable, Equatable {
    nonisolated enum Payload: Sendable, Equatable {
        /// The decoded base64 payload: exactly the bytes the client sent, which
        /// is what `DeepSeekV41ImagePixels.decode` expects.
        case inline(Data, mediaType: String?)
        /// A reference this server will not dereference. Fetching a URL from
        /// inside a request handler is an outbound connection the caller chose,
        /// which is both a privacy surface and an SSRF one, so the turn is
        /// refused with the URL named instead.
        case remoteURL(String)
        /// A part the client shaped in a way base64 cannot express.
        case malformed(reason: String)
    }

    var payload: Payload

    init(payload: Payload) {
        self.payload = payload
    }

    /// The media type the client declared. Carried for error text only; the
    /// decoder sniffs the format itself, so a wrong media type is not fatal.
    var mediaType: String? {
        if case let .inline(_, mediaType) = payload { return mediaType }
        return nil
    }

    var bytes: Data? {
        if case let .inline(data, _) = payload { return data }
        return nil
    }

    var isInline: Bool { bytes != nil }
}

nonisolated enum ServerImageParts {

    /// Where a picture belongs inside a turn's text, before the loaded
    /// checkpoint's own image marker is known.
    ///
    /// Both protocols describe content as an ORDERED list of parts, and the turn
    /// renderer needs that order: a picture between two paragraphs asks a different
    /// question from the same two pictures placed after all of it. Collecting the
    /// pictures beside the joined text loses the association, and nothing is left
    /// afterwards to restore it from. So the decoders keep the order as a sentinel
    /// in the turn text and `ServerController` replaces each one with the
    /// checkpoint's marker. U+FFFC is the Unicode object replacement character,
    /// which is what local runners conventionally use for exactly this. It never
    /// reaches the model.
    static let imagePlaceholder = "\u{FFFC}"

    /// One element of a message's content, in the order the client wrote it.
    nonisolated enum Piece: Sendable {
        case text(String)
        case image(ServerImagePart)
    }

    /// Assembles a message's pieces into one turn string, with the placeholder
    /// exactly where each picture sits.
    ///
    /// Consecutive text pieces keep the newline join the adapters always used;
    /// a picture ends the run, because the marker substitution supplies its own
    /// trailing newline. Nil when the content carries no text and no picture.
    static func interleave(_ pieces: [Piece]) -> String? {
        var out = ""
        var separatorPending = false
        for piece in pieces {
            switch piece {
            case .image:
                out += imagePlaceholder
                // The substitution writes the marker AND its newline, so a text
                // piece that follows starts on the next line already.
                separatorPending = false
            case let .text(text):
                guard !text.isEmpty else { continue }
                if separatorPending { out += "\n" }
                out += text
                separatorPending = true
            }
        }
        return out.isEmpty ? nil : out
    }

    /// The images a message carries, in the order the client wrote them.
    static func images(in pieces: [Piece]) -> [ServerImagePart] {
        pieces.compactMap { piece in
            if case let .image(part) = piece { return part }
            return nil
        }
    }

    /// Decodes an OpenAI `image_url` value.
    ///
    /// Three shapes reach here: a `data:` URL carrying base64 (what every SDK
    /// sends), a bare base64 payload (llama.cpp and several hand-rolled clients),
    /// and a plain URL. The last is kept as a reference rather than fetched.
    static func fromOpenAIURL(_ raw: String) -> ServerImagePart {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return ServerImagePart(payload: .malformed(reason: "the image_url was empty"))
        }
        if trimmed.lowercased().hasPrefix("data:") {
            return fromDataURL(trimmed)
        }
        return fromBase64OrReference(trimmed, mediaType: nil)
    }

    /// Decodes an Anthropic `source` object.
    ///
    /// `base64` is the only source this server can read; `url` is a reference to
    /// something it must not fetch, and any other type (`file`, a future kind) is
    /// named rather than dropped.
    static func fromAnthropicSource(
        type rawType: String?,
        mediaType: String?,
        data: String?,
        url: String?
    ) -> ServerImagePart {
        let type = (rawType ?? "").lowercased()
        switch type {
        case "base64":
            guard let data, !data.isEmpty else {
                return ServerImagePart(payload: .malformed(reason: "the base64 source carried no data"))
            }
            guard let decoded = decodeBase64(data) else {
                return ServerImagePart(payload: .malformed(reason: "the base64 source was not valid base64"))
            }
            return ServerImagePart(payload: .inline(decoded, mediaType: mediaType))
        case "url":
            if let url, !url.isEmpty {
                return ServerImagePart(payload: .remoteURL(url))
            }
            return ServerImagePart(payload: .malformed(reason: "the url source carried no url"))
        case "":
            // A block with no declared source type: fall back to whichever
            // field it actually populated, so a lenient client still works.
            if let data, !data.isEmpty { return fromAnthropicSource(type: "base64", mediaType: mediaType, data: data, url: url) }
            if let url, !url.isEmpty { return ServerImagePart(payload: .remoteURL(url)) }
            return ServerImagePart(payload: .malformed(reason: "the image block carried no source"))
        default:
            return ServerImagePart(payload: .malformed(reason: "the image source type \"\(type)\" is not supported"))
        }
    }

    /// A stable digest over the parts, folded into the prefix-cache session key.
    ///
    /// Two requests can carry identical TEXT and different pictures, and the
    /// rendered prompt of each is the same list of image token ids, so neither the
    /// session key's text nor token-prefix equality can tell them apart. Without
    /// this a follow-up question about a second picture would splice the first
    /// one's recorded turn and reuse its pinned KV prefix.
    static func digest(_ parts: [ServerImagePart]) -> String {
        guard !parts.isEmpty else { return "" }
        var hasher = SHA256()
        for part in parts {
            switch part.payload {
            case let .inline(data, mediaType):
                hasher.update(data: Data("inline:".utf8))
                hasher.update(data: Data((mediaType ?? "").utf8))
                hasher.update(data: data)
            case let .remoteURL(url):
                hasher.update(data: Data("url:\(url)".utf8))
            case let .malformed(reason):
                hasher.update(data: Data("bad:\(reason)".utf8))
            }
            hasher.update(data: Data([0x1e]))
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// One digest per turn that carries pictures, over that turn's images in the
    /// order the client sent them.
    ///
    /// A conversation's pictures are not all in its opening message: a follow-up
    /// can replace the image on turn four while every turn's TEXT stays identical,
    /// and two pictures of the same size render to the same run of image token ids.
    /// Identity therefore has to cover every turn the prompt was rendered from, not
    /// just the anchor — otherwise the second picture's turn splices the first
    /// picture's recorded generation.
    static func turnDigests(_ images: [Int: [ServerImagePart]]) -> [Int: String] {
        var digests: [Int: String] = [:]
        for (turn, parts) in images where !parts.isEmpty {
            digests[turn] = digest(parts)
        }
        return digests
    }

    /// The marker a turn's prompt must carry for this model, or nil when the
    /// loaded checkpoint cannot read pixels at all.
    ///
    /// The gate is the tower's own files rather than the model row's advertised
    /// flag: a half-finished download publishes `config.json` and not the tower,
    /// and expanding a placeholder there would bind no rows.
    static func visionMarker(modelPath: String?, config: ModelConfig?) -> String? {
        guard let modelPath, !modelPath.isEmpty,
              config?.isDeepSeekV41Model == true,
              ModelConfig.supportsVision(modelPath: modelPath) else { return nil }
        return DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
    }

    // MARK: - Decoding helpers

    private static func fromDataURL(_ raw: String) -> ServerImagePart {
        guard let comma = raw.firstIndex(of: ",") else {
            return ServerImagePart(payload: .malformed(reason: "the data URL carried no payload"))
        }
        let header = raw[raw.startIndex..<comma].lowercased()
        guard header.contains(";base64") else {
            // A non-base64 data URL is percent-encoded bytes; nothing in the
            // vision path reads that, so name it instead of guessing.
            return ServerImagePart(payload: .malformed(reason: "the data URL was not base64-encoded"))
        }
        var mediaType: String?
        if header.hasPrefix("data:"), let semicolon = header.firstIndex(of: ";") {
            let declared = String(header[header.index(header.startIndex, offsetBy: 5)..<semicolon])
            mediaType = declared.isEmpty ? nil : declared
        }
        let payload = String(raw[raw.index(after: comma)...])
        guard let decoded = decodeBase64(payload) else {
            return ServerImagePart(payload: .malformed(reason: "the data URL payload was not valid base64"))
        }
        return ServerImagePart(payload: .inline(decoded, mediaType: mediaType))
    }

    private static func fromBase64OrReference(_ raw: String, mediaType: String?) -> ServerImagePart {
        // A scheme is checked before base64 is attempted: most of a URL's
        // characters are legal base64, so a lenient decode would turn
        // `https://host/a.png` into plausible-looking garbage bytes.
        let lowered = raw.lowercased()
        if lowered.hasPrefix("http://") || lowered.hasPrefix("https://")
            || lowered.hasPrefix("file://") || lowered.hasPrefix("/") {
            return ServerImagePart(payload: .remoteURL(raw))
        }
        guard let decoded = decodeBase64(raw) else {
            return ServerImagePart(payload: .malformed(reason: "the image payload was neither base64 nor a URL"))
        }
        return ServerImagePart(payload: .inline(decoded, mediaType: mediaType))
    }

    /// Strict base64, tolerating the line wrapping clients insert.
    private static func decodeBase64(_ raw: String) -> Data? {
        let compact = raw.filter { !$0.isWhitespace }
        guard !compact.isEmpty else { return nil }
        return Data(base64Encoded: compact)
    }
}

/// A decoded request's prompt inputs, in the one shape the turn planner reads.
///
/// Both adapters used to return `(systemPrompt, turns)`; a picture belongs to one
/// message, so it has to travel beside the turns rather than inside them (the turn
/// text is what gets rendered, and a raw picture is not text).
nonisolated struct ServerPromptSplit: Sendable {
    var systemPrompt: String = ""
    var turns: [ChatPromptFormatter.ConversationTurn] = []
    /// Images attached to each turn, keyed by that turn's index in `turns`, in the
    /// order the client sent them inside its message.
    var images: [Int: [ServerImagePart]] = [:]

    var hasImages: Bool { images.values.contains { !$0.isEmpty } }

    /// Every image in prompt order, so a cost or a digest can be taken over the
    /// whole turn at once.
    var orderedImages: [ServerImagePart] {
        images.keys.sorted().flatMap { images[$0] ?? [] }
    }
}
