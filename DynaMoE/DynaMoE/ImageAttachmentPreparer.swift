//
//  ImageAttachmentPreparer.swift
//  DynaMoE
//
//  Turns a user-supplied image (file, pasteboard payload, or dragged file) into
//  the exact bytes that travel to a vision model.
//
//  Every decision here is a policy, not an accident:
//
//  - **Downscale to 1024 px on the long edge.** Providers bill per tile and cap
//    their own resolution; sending an 8 MP screenshot wastes tokens and body
//    size without adding detail the model can use.
//  - **PNG for screenshots, JPEG otherwise.** Text and UI edges are what image
//    input is mostly used for in an app like this, and JPEG artifacts around
//    glyphs measurably hurt transcription. A PNG screenshot compresses well;
//    photographic input goes to JPEG so it does not blow the byte budget.
//  - **Strip metadata and honour EXIF orientation.** Orientation is baked into
//    the pixels (so a sideways phone photo is not sideways to the model) and
//    GPS/device tags are dropped rather than sent to a remote provider.
//  - **Bounds are hard.** 6 images, 4 MB each, 32 MB source cap; anything else
//    is refused with a message the composer can show, never a silent drop.
//
//  The type is `nonisolated` and free of UI state so the heavy work can run in a
//  detached task and the pure policy is unit-testable.
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CryptoKit

nonisolated struct PreparedImage: Equatable, Sendable {
    var data: Data
    var mimeType: String
    var fileName: String
    var pixelWidth: Int
    var pixelHeight: Int
    var sha256: String

    var byteCount: Int { data.count }

    /// OpenAI-compatible `image_url` payload. A static helper (not an instance
    /// property) so the turn assembler builds the same string from a ref + bytes
    /// without materializing a `PreparedImage`.
    static func dataURL(mimeType: String, data: Data) -> String {
        "data:\(mimeType);base64,\(data.base64EncodedString())"
    }
}

nonisolated enum ImageAttachmentError: LocalizedError, Equatable {
    case unreadableFile(String)
    case unsupportedType(String)
    case sourceTooLarge(byteCount: Int, limit: Int)
    case encodingFailed
    case tooLargeAfterResize(limit: Int)
    case tooManyImages(limit: Int)

    var errorDescription: String? {
        switch self {
        case .unreadableFile(let name):
            return "Could not read \(name) as an image."
        case .unsupportedType(let name):
            return "\(name) is not an image file DynaMoE can attach."
        case .sourceTooLarge(let byteCount, let limit):
            let size = ByteCountFormatter.string(fromByteCount: Int64(byteCount), countStyle: .file)
            let cap = ByteCountFormatter.string(fromByteCount: Int64(limit), countStyle: .file)
            return "That image is \(size); the limit is \(cap)."
        case .encodingFailed:
            return "The image could not be re-encoded for upload."
        case .tooLargeAfterResize(let limit):
            let cap = ByteCountFormatter.string(fromByteCount: Int64(limit), countStyle: .file)
            return "The image is still over \(cap) after downscaling."
        case .tooManyImages(let limit):
            return "A message can carry at most \(limit) images."
        }
    }
}

nonisolated enum ImageAttachmentPreparer {
    /// 4096 is the "low detail" tile of most vision APIs; 1024 keeps text legible
    /// while staying inside one tile's worth of tokens.
    static let maxLongEdge = 1024
    static let maxBytesPerImage = 4 * 1024 * 1024
    static let maxSourceBytes = 32 * 1024 * 1024
    static let maxImagesPerMessage = 6

    /// Fallback pre-turn input estimate. The provider's reported `usage` is
    /// authoritative once a turn finishes; this only keeps the odometer and the
    /// request log honest before the first delta arrives.
    static let estimatedTokensPerImage = 1200

    private static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "gif", "webp", "bmp"
    ]

    // MARK: - Preparation

    static func prepare(fileURL url: URL) throws -> PreparedImage {
        let name = url.lastPathComponent
        guard let type = UTType(filenameExtension: url.pathExtension.lowercased()),
              type.conforms(to: .image) else {
            throw ImageAttachmentError.unsupportedType(name)
        }
        let resourceValues = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        if resourceValues?.isRegularFile == false {
            throw ImageAttachmentError.unsupportedType(name)
        }
        if let fileSize = resourceValues?.fileSize, fileSize > maxSourceBytes {
            throw ImageAttachmentError.sourceTooLarge(byteCount: fileSize, limit: maxSourceBytes)
        }
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            throw ImageAttachmentError.unreadableFile(name)
        }
        return try prepare(source: source, fileName: name, sourceType: type)
    }

    static func prepare(data: Data, fileName: String) throws -> PreparedImage {
        guard data.count <= maxSourceBytes else {
            throw ImageAttachmentError.sourceTooLarge(byteCount: data.count, limit: maxSourceBytes)
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let sourceType = CGImageSourceGetType(source).flatMap({ UTType($0 as String) }),
              sourceType.conforms(to: .image) else {
            throw ImageAttachmentError.unreadableFile(fileName)
        }
        return try prepare(source: source, fileName: fileName, sourceType: sourceType)
    }

    /// Pure path detection for the composer's "typing a filepath attaches it"
    /// affordance. Existence is *not* checked here so the parser stays testable;
    /// the caller validates the file before preparing it.
    static func imageFilePathCandidates(in text: String) -> [String] {
        guard let regex = try? NSRegularExpression(
            pattern: #"(?<![\w./~-])(/[^\s<>"'`]{1,1024}?\.(?:png|jpe?g|heic|heif|tiff?|gif|webp|bmp))(?=$|[\s,.;:!?)\]])"#,
            options: [.caseInsensitive]
        ) else { return [] }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, options: [], range: range).compactMap { match in
            guard let matchRange = Range(match.range(at: 1), in: text) else { return nil }
            return String(text[matchRange])
        }
    }

    /// Whether a concrete path should be offered as an attachment. The file must
    /// exist, be a regular file, and carry a supported image extension.
    static func isAttachableImageFile(atPath path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/") else { return false }
        let extensionName = (trimmed as NSString).pathExtension.lowercased()
        guard imageExtensions.contains(extensionName) else { return false }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: trimmed, isDirectory: &isDirectory),
              !isDirectory.boolValue else { return false }
        return true
    }

    // MARK: - Core pipeline

    private static func prepare(
        source: CGImageSource,
        fileName: String,
        sourceType: UTType
    ) throws -> PreparedImage {
        let targetEdge = maxLongEdge
        var edge = targetEdge
        var attempts = 0

        while attempts < 5 {
            guard let image = thumbnail(from: source, maxPixelSize: edge) else {
                throw ImageAttachmentError.unreadableFile(fileName)
            }
            // Screenshots arrive as PNG (drag) or TIFF (macOS clipboard) and are
            // the text-legibility case the policy favors. Everything else goes to
            // JPEG so photographic input cannot blow the byte budget.
            let preferPNG = sourceType.conforms(to: .png) || sourceType.conforms(to: .tiff)
            if let encoded = encode(image, preferPNG: preferPNG), encoded.data.count <= maxBytesPerImage {
                return PreparedImage(
                    data: encoded.data,
                    mimeType: encoded.mimeType,
                    fileName: fileName,
                    pixelWidth: image.width,
                    pixelHeight: image.height,
                    sha256: SHA256.hash(data: encoded.data).map { String(format: "%02x", $0) }.joined()
                )
            }
            attempts += 1
            let nextEdge = Int(Double(edge) * 0.75)
            guard nextEdge >= 256 else { break }
            edge = nextEdge
        }
        throw ImageAttachmentError.tooLargeAfterResize(limit: maxBytesPerImage)
    }

    /// Orients and downscales in one ImageIO pass. `kCGImageSourceCreateThumbnailWithTransform`
    /// is what bakes EXIF orientation into the pixels.
    private static func thumbnail(from source: CGImageSource, maxPixelSize: Int) -> CGImage? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceCreateThumbnailFromImageIfAbsent: true,
            kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
            kCGImageSourceShouldCacheImmediately: true
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    private static func encode(_ image: CGImage, preferPNG: Bool) -> (data: Data, mimeType: String)? {
        if preferPNG, let png = encode(image, type: .png) {
            return (png, "image/png")
        }
        for quality in [0.9, 0.8] {
            guard let jpeg = encode(image, type: .jpeg, quality: quality) else { continue }
            if jpeg.count <= maxBytesPerImage {
                return (jpeg, "image/jpeg")
            }
        }
        if let last = encode(image, type: .jpeg, quality: 0.7) {
            return (last, "image/jpeg")
        }
        return nil
    }

    private static func encode(_ image: CGImage, type: UTType, quality: Double? = nil) -> Data? {
        let drawable = type == .jpeg ? flattenedOnWhite(image) ?? image : image
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData,
            type.identifier as CFString,
            1,
            nil
        ) else { return nil }
        var properties: [CFString: Any] = [:]
        if let quality {
            properties[kCGImageDestinationLossyCompressionQuality] = quality
        }
        CGImageDestinationAddImage(destination, drawable, properties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }

    /// JPEG has no alpha. Without flattening, a screenshot's transparent corner
    /// renders black and the model sees a black wedge instead of the UI.
    private static func flattenedOnWhite(_ image: CGImage) -> CGImage? {
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let white = CGColor(colorSpace: colorSpace, components: [1, 1, 1, 1]),
              let context = CGContext(
                data: nil,
                width: image.width,
                height: image.height,
                bitsPerComponent: 8,
                bytesPerRow: 0,
                space: colorSpace,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
              ) else { return nil }
        context.setFillColor(white)
        context.fill(CGRect(x: 0, y: 0, width: image.width, height: image.height))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return context.makeImage()
    }
}
