//
//  VisionAttachmentLogicTests.swift
//  DynaMoETests
//
//  Pure-logic coverage for cloud image input: preparation policy (downscale,
//  format choice, limits, deterministic bytes), the asset store's file layout
//  and pruning, the OpenAI `image_url` wire shape, the assembler's re-send and
//  omission rules, request-log redaction, and per-model vision registration.
//

import XCTest
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import DynaMoE

// MARK: - Helpers

private enum TestImageFactory {
    static func data(width: Int, height: Int, type: UTType, color: (CGFloat, CGFloat, CGFloat) = (0.2, 0.4, 0.9)) -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return Data() }
        if let fill = CGColor(colorSpace: colorSpace, components: [color.0, color.1, color.2, 1]) {
            context.setFillColor(fill)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        }
        guard let image = context.makeImage() else { return Data() }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            output as CFMutableData,
            type.identifier as CFString,
            1,
            nil
        ) else { return Data() }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return Data() }
        return output as Data
    }

    static func prepared(width: Int = 40, height: Int = 30, mimeType: String = "image/png", seed: UInt8 = 1) -> PreparedImage {
        let bytes = Data([seed, seed, seed, seed, seed, seed])
        return PreparedImage(
            data: bytes,
            mimeType: mimeType,
            fileName: "shot-\(seed).png",
            pixelWidth: width,
            pixelHeight: height,
            sha256: "sha-\(seed)"
        )
    }
}

// MARK: - Preparer

final class ImageAttachmentPreparerTests: XCTestCase {

    func testDownscalesLongestEdgeAndKeepsAspectRatio() throws {
        let source = TestImageFactory.data(width: 2000, height: 1000, type: .png)
        let prepared = try ImageAttachmentPreparer.prepare(data: source, fileName: "wide.png")

        XCTAssertLessThanOrEqual(max(prepared.pixelWidth, prepared.pixelHeight), ImageAttachmentPreparer.maxLongEdge)
        XCTAssertEqual(prepared.pixelWidth, ImageAttachmentPreparer.maxLongEdge)
        XCTAssertEqual(prepared.pixelHeight, ImageAttachmentPreparer.maxLongEdge / 2)
        XCTAssertLessThanOrEqual(prepared.data.count, ImageAttachmentPreparer.maxBytesPerImage)
    }

    func testPNGSourceStaysPNGForTextLegibility() throws {
        let source = TestImageFactory.data(width: 300, height: 200, type: .png)
        let prepared = try ImageAttachmentPreparer.prepare(data: source, fileName: "screenshot.png")
        XCTAssertEqual(prepared.mimeType, "image/png")
    }

    func testJPEGSourceUsesJPEG() throws {
        let source = TestImageFactory.data(width: 300, height: 200, type: .jpeg)
        let prepared = try ImageAttachmentPreparer.prepare(data: source, fileName: "photo.jpg")
        XCTAssertEqual(prepared.mimeType, "image/jpeg")
    }

    func testTIFFSourceStaysPNGForScreenshots() throws {
        // macOS puts a screenshot on the clipboard as TIFF; that is the text case
        // the PNG policy exists for, so it must not be re-encoded as JPEG.
        let source = TestImageFactory.data(width: 300, height: 200, type: .tiff)
        let prepared = try ImageAttachmentPreparer.prepare(data: source, fileName: "screenshot.tiff")
        XCTAssertEqual(prepared.mimeType, "image/png")
    }

    func testEncodingIsDeterministic() throws {
        let source = TestImageFactory.data(width: 640, height: 480, type: .png)
        let first = try ImageAttachmentPreparer.prepare(data: source, fileName: "a.png")
        let second = try ImageAttachmentPreparer.prepare(data: source, fileName: "a.png")
        XCTAssertEqual(first.sha256, second.sha256)
        XCTAssertEqual(first.data, second.data)
    }

    func testNonImageDataIsRejected() {
        XCTAssertThrowsError(
            try ImageAttachmentPreparer.prepare(data: Data("not an image".utf8), fileName: "notes.txt")
        ) { error in
            guard case ImageAttachmentError.unreadableFile = error else {
                return XCTFail("expected unreadableFile, got \(error)")
            }
        }
    }

    func testOversizedSourceIsRejectedBeforeDecoding() {
        let oversized = Data(repeating: 0, count: ImageAttachmentPreparer.maxSourceBytes + 1)
        XCTAssertThrowsError(
            try ImageAttachmentPreparer.prepare(data: oversized, fileName: "huge.png")
        ) { error in
            guard case ImageAttachmentError.sourceTooLarge = error else {
                return XCTFail("expected sourceTooLarge, got \(error)")
            }
        }
    }

    func testDataURLShape() {
        let prepared = TestImageFactory.prepared()
        let url = PreparedImage.dataURL(mimeType: prepared.mimeType, data: prepared.data)
        XCTAssertTrue(url.hasPrefix("data:image/png;base64,"))
    }

    func testTypedPathCandidates() {
        let text = "look at /Users/me/Desktop/shot.png and /tmp/nope.txt plus ~/relative.jpg"
        let candidates = ImageAttachmentPreparer.imageFilePathCandidates(in: text)
        XCTAssertEqual(candidates, ["/Users/me/Desktop/shot.png"])
    }

    func testAttachableFileRequiresExistingRegularImage() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let imageURL = directory.appendingPathComponent("shot.png")
        try TestImageFactory.data(width: 20, height: 20, type: .png).write(to: imageURL)
        XCTAssertTrue(ImageAttachmentPreparer.isAttachableImageFile(atPath: imageURL.path))

        let textURL = directory.appendingPathComponent("notes.txt")
        try Data("hi".utf8).write(to: textURL)
        XCTAssertFalse(ImageAttachmentPreparer.isAttachableImageFile(atPath: textURL.path))
        XCTAssertFalse(ImageAttachmentPreparer.isAttachableImageFile(atPath: directory.path))
        XCTAssertFalse(ImageAttachmentPreparer.isAttachableImageFile(atPath: "relative/shot.png"))
    }
}

// MARK: - Store

final class ImageAttachmentStoreTests: XCTestCase {

    private func makeStore() -> (store: ImageAttachmentStore, root: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dynamoe-attachments-\(UUID().uuidString)", isDirectory: true)
        return (ImageAttachmentStore(root: root), root)
    }

    func testWriteReadsBackAndResolvesWithinTheStoreRoot() throws {
        let (store, root) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let sessionId = UUID()

        let image = TestImageFactory.prepared(seed: 7)
        let ref = try XCTUnwrap(store.write(image, sessionId: sessionId))

        XCTAssertEqual(ref.byteCount, image.data.count)
        XCTAssertEqual(ref.sha256, image.sha256)
        XCTAssertTrue(ref.relativePath.hasPrefix(sessionId.uuidString.lowercased() + "/"))
        XCTAssertEqual(store.data(for: ref), image.data)
        XCTAssertEqual(store.preparedImage(for: ref)?.sha256, image.sha256)

        let url = try XCTUnwrap(store.absoluteURL(for: ref))
        XCTAssertTrue(url.path.hasPrefix(root.path))
    }

    func testTraversalRelativePathIsRefused() {
        let (store, root) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let escape = ImageAttachmentRef(
            fileName: "x.png",
            pixelWidth: 1,
            pixelHeight: 1,
            byteCount: 1,
            sha256: "x",
            relativePath: "../outside.png",
            mimeType: "image/png"
        )
        XCTAssertNil(store.absoluteURL(for: escape))
        XCTAssertNil(store.data(for: escape))
    }

    func testRemoveDropsOnlyTheGivenAssets() throws {
        let (store, root) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let sessionId = UUID()
        let first = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 1), sessionId: sessionId))
        let second = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 2), sessionId: sessionId))

        store.remove([first])
        store.flushPendingIO()

        XCTAssertFalse(store.exists(first))
        XCTAssertTrue(store.exists(second))
    }

    func testPruneRemovesDeletedSessionsAndUnreferencedFiles() throws {
        let (store, root) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let liveSession = UUID()
        let deadSession = UUID()

        let liveKept = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 3), sessionId: liveSession))
        let liveOrphan = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 4), sessionId: liveSession))
        let deadAsset = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 5), sessionId: deadSession))

        store.prune(
            liveSessionIds: [liveSession],
            liveRelativePaths: [liveKept.relativePath]
        )
        store.flushPendingIO()

        XCTAssertTrue(store.exists(liveKept))
        XCTAssertFalse(store.exists(liveOrphan))
        XCTAssertFalse(store.exists(deadAsset))
    }

    func testDeleteRemovesTheSessionDirectory() throws {
        let (store, root) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let sessionId = UUID()
        let ref = try XCTUnwrap(store.write(TestImageFactory.prepared(seed: 9), sessionId: sessionId))
        store.delete(sessionIds: [sessionId])
        store.flushPendingIO()
        XCTAssertFalse(store.exists(ref))
    }
}

// MARK: - Wire format

final class CloudVisionWireFormatTests: XCTestCase {

    private func service() -> OpenAICloudChatService {
        OpenAICloudChatService(
            chatURL: CloudProviderRegistry.synthetic.chatCompletionsURL,
            modelsURL: CloudProviderRegistry.synthetic.modelsURL,
            apiKey: "sk-test-key",
            requestTimeout: 5
        )
    }

    func testImagePartsEncodeAsContentArrayWithDataURL() throws {
        let message = CloudChatMessage(
            role: "user",
            content: "what is wrong with this UI?",
            imageParts: ["data:image/png;base64,AAAA"],
            imageDescriptions: ["[image: shot.png 1024×768 412 KB]"]
        )
        let request = try service().makeRequest(
            model: "hf:deepseek-ai/DeepSeek-V4.1-Flash-1017",
            messages: [message],
            params: CloudChatGenerationParams()
        )
        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try XCTUnwrap(json["messages"] as? [[String: Any]])
        let parts = try XCTUnwrap(messages.first?["content"] as? [[String: Any]])

        XCTAssertEqual(parts.count, 2)
        XCTAssertEqual(parts[0]["type"] as? String, "text")
        XCTAssertEqual(parts[0]["text"] as? String, "what is wrong with this UI?")
        XCTAssertEqual(parts[1]["type"] as? String, "image_url")
        let imageURL = try XCTUnwrap(parts[1]["image_url"] as? [String: Any])
        XCTAssertEqual(imageURL["url"] as? String, "data:image/png;base64,AAAA")
        // The metadata label is ours, not the provider's.
        XCTAssertFalse(String(data: body, encoding: .utf8)?.contains("412 KB") ?? true)
    }

    func testTextOnlyMessageKeepsLegacyBareStringContent() throws {
        let request = try service().makeRequest(
            model: "hf:org/model",
            messages: [CloudChatMessage(role: "user", content: "Hello")],
            params: CloudChatGenerationParams()
        )
        let body = try XCTUnwrap(request.httpBody)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: body) as? [String: Any])
        let messages = try XCTUnwrap(json["messages"] as? [[String: Any]])
        XCTAssertEqual(messages.first?["content"] as? String, "Hello")
    }
}

// MARK: - Assembler

final class CloudVisionAssemblerTests: XCTestCase {

    private func userMessage(text: String = "see this", attachments: [ImageAttachmentRef]) -> ChatMessage {
        ChatMessage(role: .user, content: text, attachments: attachments)
    }

    private func ref(id: UUID = UUID(), name: String = "shot.png") -> ImageAttachmentRef {
        ImageAttachmentRef(
            id: id,
            fileName: name,
            pixelWidth: 1024,
            pixelHeight: 768,
            byteCount: 421_888,
            sha256: "sha-\(name)",
            relativePath: "session/\(name).png",
            mimeType: "image/png"
        )
    }

    func testImagesRideEveryAssembledUserTurn() {
        let attachment = ref()
        let message = userMessage(attachments: [attachment])
        let assembled = CloudMessageAssembler.assemble(
            from: [message],
            imageDataURLs: [message.id: ["data:image/png;base64,AAAA"]]
        )
        XCTAssertEqual(assembled.count, 1)
        XCTAssertEqual(assembled[0].imageParts, ["data:image/png;base64,AAAA"])
        XCTAssertEqual(assembled[0].imageDescriptions, [attachment.logDescription])
        XCTAssertEqual(assembled[0].content, "see this")
    }

    func testImageOnlyTurnIsNotDropped() {
        let message = userMessage(text: "", attachments: [ref()])
        let assembled = CloudMessageAssembler.assemble(
            from: [message],
            imageDataURLs: [message.id: ["data:image/png;base64,AAAA"]]
        )
        XCTAssertEqual(assembled.count, 1)
        XCTAssertNil(assembled[0].content)
        XCTAssertEqual(assembled[0].imageParts?.count, 1)
    }

    func testUnsupportedModelGetsMetadataOnlyNotice() {
        let message = userMessage(attachments: [ref(), ref(name: "second.png")])
        let assembled = CloudMessageAssembler.assemble(
            from: [message],
            imageDataURLs: [message.id: ["data:image/png;base64,AAAA"]],
            imagesSupported: false
        )
        XCTAssertNil(assembled[0].imageParts)
        XCTAssertEqual(assembled[0].content, "see this\n\n[2 images omitted: model has no vision support]")
    }

    func testMissingAssetBytesAreReportedNotSilent() {
        let message = userMessage(attachments: [ref()])
        let assembled = CloudMessageAssembler.assemble(from: [message], imageDataURLs: [:])
        XCTAssertNil(assembled[0].imageParts)
        XCTAssertEqual(assembled[0].content, "see this\n\n[1 image omitted: image data unavailable]")
    }

    func testPartiallyMissingAssetsReportOnlyTheMissingCount() {
        let message = userMessage(attachments: [ref(), ref(name: "second.png")])
        let assembled = CloudMessageAssembler.assemble(
            from: [message],
            imageDataURLs: [message.id: ["data:image/png;base64,AAAA"]]
        )
        XCTAssertEqual(assembled[0].imageParts?.count, 1)
        XCTAssertEqual(assembled[0].content, "see this\n\n[1 image omitted: image data unavailable]")
        XCTAssertEqual(assembled[0].imageDescriptions?.count, 2)
    }

    func testTokenEstimatePricesImagesFlatAndLogNeverCarriesBase64() {
        let message = userMessage(attachments: [ref()])
        let assembled = CloudMessageAssembler.assemble(
            from: [message],
            imageDataURLs: [message.id: ["data:image/png;base64,\(String(repeating: "A", count: 4096))"]]
        )
        let tokens = CloudMessageAssembler.approximatePromptTokens(system: nil, messages: assembled)
        XCTAssertGreaterThanOrEqual(tokens, ImageAttachmentPreparer.estimatedTokensPerImage)

        let rendered = CloudMessageAssembler.renderedPromptForLog(system: nil, messages: assembled)
        XCTAssertTrue(rendered.contains("[image: shot.png 1024×768 412 KB]"))
        XCTAssertFalse(rendered.contains("base64"))
        XCTAssertFalse(rendered.contains("AAAA"))
    }

    func testImageRejectionMarkers() {
        XCTAssertTrue(CloudTurnFailure.indicatesImageRejection("This model does not support image input."))
        XCTAssertTrue(CloudTurnFailure.indicatesImageRejection("invalid image_url content part"))
        XCTAssertTrue(CloudTurnFailure.indicatesImageRejection("This model does not support vision."))
        XCTAssertFalse(CloudTurnFailure.indicatesImageRejection("maximum context length exceeded"))
        // Broad words used to match unrelated 400s and persistently disable a
        // working model; they must not be markers on their own.
        XCTAssertFalse(CloudTurnFailure.indicatesImageRejection("reasoning is unsupported for this model"))
        XCTAssertFalse(CloudTurnFailure.indicatesImageRejection("unknown modality in request"))
        XCTAssertFalse(CloudTurnFailure.indicatesImageRejection("tool schemas are not supported"))
        XCTAssertFalse(CloudTurnFailure.indicatesContextOverflow("This model does not support image input."))
    }
}

// MARK: - Vision registration

final class CloudVisionRegistrationTests: XCTestCase {

    private func scratchDefaults() -> (UserDefaults, String) {
        let suite = "dynamoe-vision-tests-\(UUID().uuidString)"
        return (UserDefaults(suiteName: suite)!, suite)
    }

    func testVisionSupportIsPerModelAndDefaultsOff() throws {
        let (defaults, suite) = scratchDefaults()
        defer { defaults.removePersistentDomain(forName: suite) }
        let provider = CloudProviderRegistry.synthetic
        CloudModelRegistry.setRegisteredModelIds(["model-text", "model-vision"], for: provider, defaults: defaults)

        var specs = CloudModelRegistry.registeredSpecs(for: provider, defaults: defaults)
        XCTAssertEqual(specs.filter(\.supportsVision).count, 0, "vision must be opt-in")

        CloudModelRegistry.setVisionSupport(
            true,
            forModel: "model-vision",
            provider: provider,
            defaults: defaults
        )
        specs = CloudModelRegistry.registeredSpecs(for: provider, defaults: defaults)
        let visionSpec = try XCTUnwrap(specs.first { $0.providerModelId == "model-vision" })
        let textSpec = try XCTUnwrap(specs.first { $0.providerModelId == "model-text" })
        XCTAssertTrue(visionSpec.supportsVision)
        XCTAssertFalse(textSpec.supportsVision)

        let discovered = CloudModelRegistry.makeDiscoveredModel(for: visionSpec)
        XCTAssertTrue(discovered.supportsVision)
        XCTAssertTrue(discovered.isCloudModel)
    }

    func testTurningVisionOffRemovesOnlyThatModel() throws {
        let (defaults, suite) = scratchDefaults()
        defer { defaults.removePersistentDomain(forName: suite) }
        let provider = CloudProviderRegistry.synthetic
        CloudModelRegistry.setRegisteredModelIds(["a", "b"], for: provider, defaults: defaults)
        CloudModelRegistry.setVisionSupport(true, forModel: "a", provider: provider, defaults: defaults)
        CloudModelRegistry.setVisionSupport(true, forModel: "b", provider: provider, defaults: defaults)

        CloudModelRegistry.setVisionSupport(false, forModel: "a", provider: provider, defaults: defaults)
        let ids = CloudModelRegistry.modelIdsWithVisionSupport(for: provider, defaults: defaults)
        XCTAssertEqual(ids, ["b"])
    }
}

// MARK: - Server refusal

final class ServerImagePartRefusalTests: XCTestCase {

    private func decodeOpenAI(_ json: String) throws -> OpenAIChatRequest {
        try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(json.utf8))
    }

    func testOpenAIImagePartIsDetected() throws {
        let request = try decodeOpenAI("""
        {"messages":[{"role":"user","content":[
          {"type":"text","text":"what is this"},
          {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
        ]}]}
        """)
        XCTAssertTrue(request.hasImageContent)
        XCTAssertTrue(request.hasUnsupportedContent)
        XCTAssertEqual(request.messages.first?.contentText, "what is this")
    }

    func testOpenAIFileAndAudioPartsAreFlagged() throws {
        let file = try decodeOpenAI("""
        {"messages":[{"role":"user","content":[{"type":"file","file":{"file_id":"f1"}}]}]}
        """)
        XCTAssertFalse(file.hasImageContent)
        XCTAssertTrue(file.hasUnsupportedContent)

        let audio = try decodeOpenAI("""
        {"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA"}}]}]}
        """)
        XCTAssertTrue(audio.hasUnsupportedContent)
    }

    func testTextOnlyRequestIsNotFlagged() throws {
        let request = try decodeOpenAI(#"{"messages":[{"role":"user","content":"hi"}]}"#)
        XCTAssertFalse(request.hasImageContent)
        XCTAssertFalse(request.hasUnsupportedContent)
    }

    func testAnthropicImageAndToolResultImageAreDetected() throws {
        let direct = try JSONDecoder().decode(
            AnthropicMessagesRequest.self,
            from: Data("""
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"look"},
              {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
            ]}],"max_tokens":16}
            """.utf8)
        )
        XCTAssertTrue(direct.hasImageContent)

        let inToolResult = try JSONDecoder().decode(
            AnthropicMessagesRequest.self,
            from: Data("""
            {"messages":[{"role":"user","content":[
              {"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"text","text":"here"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}}
              ]}
            ]}],"max_tokens":16}
            """.utf8)
        )
        XCTAssertTrue(inToolResult.hasImageContent)
    }

    func testTextOnlyAnthropicRequestIsNotFlagged() throws {
        let request = try JSONDecoder().decode(
            AnthropicMessagesRequest.self,
            from: Data(#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":16}"#.utf8)
        )
        XCTAssertFalse(request.hasImageContent)
        XCTAssertFalse(request.hasUnsupportedContent)
    }

    func testAnthropicDocumentBlockIsFlagged() throws {
        let request = try JSONDecoder().decode(
            AnthropicMessagesRequest.self,
            from: Data("""
            {"messages":[{"role":"user","content":[
              {"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"AAAA"}}
            ]}],"max_tokens":16}
            """.utf8)
        )
        XCTAssertFalse(request.hasImageContent)
        XCTAssertTrue(request.hasUnsupportedContent)
    }
}
