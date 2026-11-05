//
//  ServerImagePartsTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the local server's image content parts (slice #77):
//  both wire spellings decoded into one form, the refusals named rather than
//  guessed, the session identity that keeps two different pictures apart, and the
//  marker placement the turn planner writes.
//
//  Note on tag-bearing literals: the image marker is read from the checkpoint's
//  own constant, never retyped from a rendered view (AGENTS.md).
//

import XCTest
@testable import DynaMoE

final class ServerImagePartsTests: XCTestCase {

    /// A 2x2 solid PNG, which is the smallest thing ImageIO will decode.
    private func png(side: Int = 2, red: UInt8 = 200) throws -> Data {
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(
            data: nil,
            width: side,
            height: side,
            bitsPerComponent: 8,
            bytesPerRow: side * 4,
            space: space,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        context.setFillColor(red: CGFloat(red) / 255, green: 0.2, blue: 0.1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: side, height: side))
        let image = try XCTUnwrap(context.makeImage())
        return try XCTUnwrap(NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]))
    }

    // MARK: - OpenAI spellings

    func testADataURLDecodesToItsPayloadAndMediaType() throws {
        let payload = try png()
        let base64 = payload.base64EncodedString()
        let part = ServerImageParts.fromOpenAIURL("data:image/png;base64,\(base64)")
        XCTAssertEqual(part.bytes, payload)
        XCTAssertEqual(part.mediaType, "image/png")
        XCTAssertTrue(part.isInline)
    }

    func testABareBase64PayloadDecodes() throws {
        let payload = try png()
        let part = ServerImageParts.fromOpenAIURL(payload.base64EncodedString())
        XCTAssertEqual(part.bytes, payload)
        XCTAssertNil(part.mediaType)
    }

    func testLineWrappedBase64StillDecodes() throws {
        let payload = try png()
        let wrapped = payload.base64EncodedString()
            .enumerated()
            .reduce(into: "") { out, pair in
                out.append(pair.element)
                if pair.offset % 8 == 7 { out.append("\n") }
            }
        XCTAssertEqual(ServerImageParts.fromOpenAIURL(wrapped).bytes, payload)
    }

    /// A URL is a reference, not a payload. Every character of `https://host/a.png`
    /// except `:` and `/` is legal base64, so a lenient decode would turn a plain
    /// URL into plausible-looking garbage and the tower would read noise.
    func testAnHTTPURLIsAReferenceRatherThanBase64Garbage() throws {
        let part = ServerImageParts.fromOpenAIURL("https://example.com/cat.png")
        guard case let .remoteURL(url) = part.payload else {
            return XCTFail("an http URL must stay a reference, got \(part.payload)")
        }
        XCTAssertEqual(url, "https://example.com/cat.png")
        XCTAssertNil(part.bytes)
    }

    func testAnUndecodableStringIsNamedRatherThanDropped() {
        guard case let .malformed(reason) = ServerImageParts.fromOpenAIURL("not base64 at all!").payload else {
            return XCTFail("a non-base64 payload must be reported")
        }
        XCTAssertTrue(reason.contains("base64"), reason)
    }

    func testANonBase64DataURLIsNamed() {
        guard case let .malformed(reason) = ServerImageParts.fromOpenAIURL("data:image/png,abc%20def").payload else {
            return XCTFail("a percent-encoded data URL must be reported")
        }
        XCTAssertTrue(reason.contains("base64"), reason)
    }

    func testAnEmptyImageURLIsNamed() {
        guard case .malformed = ServerImageParts.fromOpenAIURL("   ").payload else {
            return XCTFail("an empty image_url must be reported")
        }
    }

    // MARK: - Anthropic spellings

    func testAnAnthropicBase64SourceDecodes() throws {
        let payload = try png()
        let part = ServerImageParts.fromAnthropicSource(
            type: "base64",
            mediaType: "image/png",
            data: payload.base64EncodedString(),
            url: nil
        )
        XCTAssertEqual(part.bytes, payload)
        XCTAssertEqual(part.mediaType, "image/png")
    }

    func testAnAnthropicURLSourceIsAReference() {
        let part = ServerImageParts.fromAnthropicSource(
            type: "url", mediaType: nil, data: nil, url: "https://example.com/a.jpg"
        )
        guard case let .remoteURL(url) = part.payload else {
            return XCTFail("a url source must stay a reference")
        }
        XCTAssertEqual(url, "https://example.com/a.jpg")
    }

    func testAnUnknownAnthropicSourceTypeIsNamed() {
        guard case let .malformed(reason) = ServerImageParts.fromAnthropicSource(
            type: "file", mediaType: nil, data: nil, url: nil
        ).payload else {
            return XCTFail("an unsupported source type must be reported")
        }
        XCTAssertTrue(reason.contains("file"), reason)
    }

    func testAnAnthropicBase64SourceWithNoDataIsNamed() {
        guard case .malformed = ServerImageParts.fromAnthropicSource(
            type: "base64", mediaType: "image/png", data: nil, url: nil
        ).payload else {
            return XCTFail("a base64 source with no data must be reported")
        }
    }

    // MARK: - Session identity

    /// Two requests can carry the same TEXT and different pictures, and both render
    /// to the same list of image token ids, so the digest is the only thing that can
    /// keep the second picture's turn from splicing the first one's generation.
    func testTheDigestSeparatesTwoPicturesAndSurvivesARepeat() throws {
        let a = ServerImagePart(payload: .inline(try png(red: 10), mediaType: "image/png"))
        let b = ServerImagePart(payload: .inline(try png(red: 240), mediaType: "image/png"))
        XCTAssertEqual(ServerImageParts.digest([a]), ServerImageParts.digest([a]))
        XCTAssertNotEqual(ServerImageParts.digest([a]), ServerImageParts.digest([b]))
        XCTAssertNotEqual(ServerImageParts.digest([a, b]), ServerImageParts.digest([b, a]))
        XCTAssertEqual(ServerImageParts.digest([]), "")
    }

    func testTheOpenAISessionKeySeparatesTwoPicturesWithTheSameText() throws {
        let a = try png(red: 10).base64EncodedString()
        let b = try png(red: 240).base64EncodedString()
        func key(_ base64: String) throws -> UUID {
            let body = """
            {"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"what is this"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}}
            ]}]}
            """
            let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
            return OpenAIAdapter.sessionKey(for: request.messages, model: "m")
        }
        XCTAssertNotEqual(try key(a), try key(b), "a different picture is a different conversation")
        XCTAssertEqual(try key(a), try key(a))
    }

    func testTheAnthropicSessionKeySeparatesTwoPicturesWithTheSameText() throws {
        let a = try png(red: 10).base64EncodedString()
        let b = try png(red: 240).base64EncodedString()
        func key(_ base64: String) throws -> UUID {
            let body = """
            {"max_tokens":10,"model":"m","messages":[{"role":"user","content":[
                {"type":"text","text":"what is this"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(base64)"}}
            ]}]}
            """
            let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(body.utf8))
            return AnthropicAdapter.sessionKey(for: request.messages, model: "m")
        }
        XCTAssertNotEqual(try key(a), try key(b), "a different picture is a different conversation")
        XCTAssertEqual(try key(a), try key(a))
    }

    // MARK: - The tower gate

    /// The anchor key cannot see a picture on a LATER turn, so the history digest
    /// has to. Two pictures of the same size render to the same run of image token
    /// ids and the marker renders as the same text, so without the per-turn digest
    /// a request that replaced turn 2's image would keep every identity and splice
    /// the old picture's generation into the new conversation.
    func testTheHistoryDigestCoversEveryTurnsPictures() throws {
        let a = ServerImagePart(payload: .inline(try png(red: 10), mediaType: "image/png"))
        let b = ServerImagePart(payload: .inline(try png(red: 240), mediaType: "image/png"))
        let turns = [
            ChatPromptFormatter.ConversationTurn(role: .user, text: "first"),
            ChatPromptFormatter.ConversationTurn(role: .assistant, text: "answer"),
            ChatPromptFormatter.ConversationTurn(role: .user, text: "second"),
        ]
        let withA = ServerImageParts.turnDigests([2: [a]])
        let withB = ServerImageParts.turnDigests([2: [b]])
        XCTAssertEqual(withA[2], ServerImageParts.digest([a]))
        XCTAssertNil(withA[0], "a turn with no picture contributes no digest")

        let digestA = ServerTurnMemory.historyDigest(
            systemPrompt: "sys", turns: turns[...], imageDigests: withA
        )
        let digestB = ServerTurnMemory.historyDigest(
            systemPrompt: "sys", turns: turns[...], imageDigests: withB
        )
        let digestNone = ServerTurnMemory.historyDigest(systemPrompt: "sys", turns: turns[...])
        XCTAssertNotEqual(digestA, digestB, "a replaced picture is a different history")
        XCTAssertNotEqual(digestA, digestNone, "a pictured turn is not a text-only one")
        // Stable across a repeat, which is what makes it usable as a record.
        XCTAssertEqual(
            digestA,
            ServerTurnMemory.historyDigest(
                systemPrompt: "sys", turns: turns[...], imageDigests: withA
            )
        )
        XCTAssertTrue(ServerTurnMemory.historyMatches(
            systemPrompt: "sys",
            conversation: turns,
            imageDigests: withA,
            turnCount: 3,
            digest: digestA
        ))
        XCTAssertFalse(ServerTurnMemory.historyMatches(
            systemPrompt: "sys",
            conversation: turns,
            imageDigests: withB,
            turnCount: 3,
            digest: digestA
        ))
    }

    func testNoLoadedPathMeansNoMarker() {
        XCTAssertNil(ServerImageParts.visionMarker(modelPath: nil, config: nil))
        XCTAssertNil(ServerImageParts.visionMarker(modelPath: "", config: nil))
        XCTAssertNil(ServerImageParts.visionMarker(modelPath: "/tmp/not-a-model", config: nil))
    }

    // MARK: - Marker placement

    private func split(_ json: String) throws -> ServerPromptSplit {
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
        return try OpenAIAdapter.splitMessages(request.messages)
    }

    func testAnImagePartReachesItsOwnTurnAndNothingElse() throws {
        let base64 = try png().base64EncodedString()
        let split = try split("""
        {"model":"m","messages":[
            {"role":"user","content":"look at this"},
            {"role":"assistant","content":"a red square"},
            {"role":"user","content":[
                {"type":"text","text":"and this one"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}}
            ]}
        ]}
        """)
        XCTAssertTrue(split.hasImages)
        XCTAssertEqual(split.images.count, 1)
        XCTAssertEqual(split.images[2]?.count, 2, "both pictures belong to the third turn")
        XCTAssertEqual(split.orderedImages.count, 2)
        XCTAssertFalse(split.turns[0].text.contains(ServerImageParts.imagePlaceholder))
        // Both pictures followed the text in the client's content, so both sentinels
        // follow it here; `attachImageMarkers` turns each into the model's marker.
        XCTAssertEqual(
            split.turns[2].text,
            "and this one" + String(repeating: ServerImageParts.imagePlaceholder, count: 2)
        )
    }

    func testAnImageOnAnAssistantTurnIsRefused() throws {
        let base64 = try png().base64EncodedString()
        XCTAssertThrowsError(try split("""
        {"model":"m","messages":[
            {"role":"user","content":"hi"},
            {"role":"assistant","content":[
                {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}}
            ]}
        ]}
        """)) { error in
            XCTAssertTrue((error as? ServerAPIError)?.message.contains("user message") == true)
        }
    }

    /// The Responses API spells `image_url` as a bare string where Chat Completions
    /// uses an object. Both decode, because a client that uses the other one must not
    /// have the whole body rejected.
    func testTheResponsesSpellingOfImageURLDecodes() throws {
        let base64 = try png().base64EncodedString()
        let split = try split("""
        {"model":"m","messages":[
            {"role":"user","content":[
                {"type":"input_image","image_url":"data:image/png;base64,\(base64)"}
            ]}
        ]}
        """)
        XCTAssertEqual(split.images[0]?.count, 1)
        // An image-only turn's text IS its placeholder: the planner replaces it with
        // the checkpoint's marker, so an empty string would drop the picture.
        XCTAssertEqual(split.turns[0].text, ServerImageParts.imagePlaceholder)
    }

    func testAnImageInsideAnAnthropicToolResultIsRefused() throws {
        let base64 = try png().base64EncodedString()
        let body = """
        {"max_tokens":10,"model":"m","messages":[
            {"role":"user","content":"go"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"shot","input":{}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(base64)"}}
            ]}]}
        ]}
        """
        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(body.utf8))
        XCTAssertThrowsError(try AnthropicAdapter.splitMessages(request.messages)) { error in
            XCTAssertTrue((error as? ServerAPIError)?.message.contains("tool_result") == true)
        }
    }

    func testAnAnthropicImageBlockReachesItsTurn() throws {
        let base64 = try png().base64EncodedString()
        let body = """
        {"max_tokens":10,"model":"m","messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":"hi"},
            {"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(base64)"}},
                {"type":"text","text":"what is this"}
            ]}
        ]}
        """
        let request = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(body.utf8))
        let split = try AnthropicAdapter.splitMessages(request.messages)
        XCTAssertEqual(split.images.count, 1)
        XCTAssertEqual(split.images[2]?.count, 1)
        // The image block came FIRST in the client's content, so its sentinel does
        // too. (`attachImageMarkers` turns it into the checkpoint's marker.)
        XCTAssertEqual(
            split.turns[2].text,
            ServerImageParts.imagePlaceholder + "what is this"
        )
    }

    /// Content blocks are ordered and the order is meaning. Collecting every
    /// picture beside the joined text silently rewrites
    /// `text → image → text → image` into "both pictures, then both paragraphs",
    /// which is a different question about a different relation between them.
    func testAnInterleavedImageAndTextKeepsItsOrder() throws {
        let base64 = try png().base64EncodedString()
        let marker = DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder

        // OpenAI: text, image, text, image, text.
        let openAIBody = """
        {"messages":[{"role":"user","content":[
          {"type":"text","text":"before"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}},
          {"type":"text","text":"between"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}},
          {"type":"text","text":"after"}
        ]}]}
        """
        let openAI = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: Data(openAIBody.utf8)
        )
        let openAISplit = try OpenAIAdapter.splitMessages(openAI.messages)
        XCTAssertEqual(openAISplit.images[0]?.count, 2)
        let openAIPlaced = try ServerController.attachImageMarkers(
            turns: openAISplit.turns, images: openAISplit.images, marker: marker
        )
        XCTAssertEqual(
            openAIPlaced.turns[0].text,
            "before" + marker + "\nbetween" + marker + "\nafter"
        )

        // Anthropic: the same order, spelled with blocks.
        let anthropicBody = """
        {"max_tokens":10,"model":"m","messages":[{"role":"user","content":[
          {"type":"text","text":"before"},
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(base64)"}},
          {"type":"text","text":"between"},
          {"type":"image","source":{"type":"base64","media_type":"image/png","data":"\(base64)"}},
          {"type":"text","text":"after"}
        ]}]}
        """
        let anthropic = try JSONDecoder().decode(
            AnthropicMessagesRequest.self, from: Data(anthropicBody.utf8)
        )
        let anthropicSplit = try AnthropicAdapter.splitMessages(anthropic.messages)
        let anthropicPlaced = try ServerController.attachImageMarkers(
            turns: anthropicSplit.turns, images: anthropicSplit.images, marker: marker
        )
        XCTAssertEqual(anthropicPlaced.turns[0].text, openAIPlaced.turns[0].text)
        XCTAssertEqual(anthropicPlaced.images.count, 2)
    }

    /// An image written after the text stays after the text.
    func testAnImageWrittenAfterTheTextStaysAfterIt() throws {
        let base64 = try png().base64EncodedString()
        let marker = DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        let body = """
        {"messages":[{"role":"user","content":[
          {"type":"text","text":"describe this then"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}}
        ]}]}
        """
        let request = try JSONDecoder().decode(
            OpenAIChatRequest.self, from: Data(body.utf8)
        )
        let split = try OpenAIAdapter.splitMessages(request.messages)
        let placed = try ServerController.attachImageMarkers(
            turns: split.turns, images: split.images, marker: marker
        )
        XCTAssertEqual(placed.turns[0].text, "describe this then" + marker + "\n")
    }

    /// A message that positions fewer sentinels than it carries pictures cannot be
    /// bound faithfully, so it is refused rather than guessed at: either choice
    /// puts a row set on the wrong span.
    func testAMismatchedPlaceholderCountIsRefused() throws {
        let payload = try png()
        let turns = [
            ChatPromptFormatter.ConversationTurn(
                role: .user, text: "one place " + ServerImageParts.imagePlaceholder + " only"
            )
        ]
        XCTAssertThrowsError(try ServerController.attachImageMarkers(
            turns: turns,
            images: [0: [
                ServerImagePart(payload: .inline(payload, mediaType: "image/png")),
                ServerImagePart(payload: .inline(payload, mediaType: "image/png")),
            ]],
            marker: DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        )) { error in
            XCTAssertTrue(
                (error as? ServerAPIError)?.message.contains("cannot be bound") == true
            )
        }
    }

    func testTheMarkerLandsAtTheHeadOfItsOwnTurnOnly() throws {
        let payload = try png()
        let marker = DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        let turns = [
            ChatPromptFormatter.ConversationTurn(role: .user, text: "first"),
            ChatPromptFormatter.ConversationTurn(role: .assistant, text: "answer"),
            ChatPromptFormatter.ConversationTurn(role: .user, text: "third"),
        ]
        let images = [2: [ServerImagePart(payload: .inline(payload, mediaType: "image/png"))]]
        let resolved = try ServerController.attachImageMarkers(turns: turns, images: images, marker: marker)
        XCTAssertEqual(resolved.turns[0].text, "first")
        XCTAssertEqual(resolved.turns[1].text, "answer")
        XCTAssertEqual(resolved.turns[2].text, marker + "\nthird")
        XCTAssertEqual(resolved.images.count, 1)
        XCTAssertEqual(resolved.images[0].width, 2)
        XCTAssertEqual(resolved.images[0].height, 2)
        XCTAssertFalse(resolved.images[0].sha256.isEmpty)
    }

    func testTwoPicturesGetOneMarkerEachInOrder() throws {
        let a = try png(red: 10)
        let b = try png(red: 240)
        let marker = DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        let turns = [ChatPromptFormatter.ConversationTurn(role: .user, text: "both")]
        let resolved = try ServerController.attachImageMarkers(
            turns: turns,
            images: [0: [
                ServerImagePart(payload: .inline(a, mediaType: "image/png")),
                ServerImagePart(payload: .inline(b, mediaType: "image/png")),
            ]],
            marker: marker
        )
        XCTAssertEqual(resolved.turns[0].text, marker + "\n" + marker + "\nboth")
        XCTAssertEqual(resolved.images.count, 2)
        XCTAssertNotEqual(resolved.images[0].sha256, resolved.images[1].sha256)
    }

    func testARemoteURLIsRefusedRatherThanFetched() throws {
        let turns = [ChatPromptFormatter.ConversationTurn(role: .user, text: "look")]
        XCTAssertThrowsError(try ServerController.attachImageMarkers(
            turns: turns,
            images: [0: [ServerImagePart(payload: .remoteURL("https://example.com/a.png"))]],
            marker: DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        )) { error in
            XCTAssertTrue((error as? ServerAPIError)?.message.contains("will not fetch") == true)
        }
    }

    func testAnUndecodableAttachmentIsRefusedRatherThanSkipped() throws {
        let turns = [ChatPromptFormatter.ConversationTurn(role: .user, text: "look")]
        XCTAssertThrowsError(try ServerController.attachImageMarkers(
            turns: turns,
            images: [0: [ServerImagePart(payload: .inline(Data([0, 1, 2, 3]), mediaType: "image/png"))]],
            marker: DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        )) { error in
            XCTAssertTrue((error as? ServerAPIError)?.message.contains("could not be decoded") == true)
        }
    }

    func testAnImageOnATurnThatDoesNotExistIsRefused() throws {
        let turns = [ChatPromptFormatter.ConversationTurn(role: .user, text: "look")]
        XCTAssertThrowsError(try ServerController.attachImageMarkers(
            turns: turns,
            images: [7: [ServerImagePart(payload: .inline(try png(), mediaType: "image/png"))]],
            marker: DeepSeekV41VisionMarkers.familyDefault.imagePlaceholder
        ))
    }

    func testNoImagesLeavesTheConversationUntouched() throws {
        let turns = [ChatPromptFormatter.ConversationTurn(role: .user, text: "plain")]
        let resolved = try ServerController.attachImageMarkers(
            turns: turns, images: [:], marker: "MARK"
        )
        XCTAssertEqual(resolved.turns, turns)
        XCTAssertTrue(resolved.images.isEmpty)
    }
}
