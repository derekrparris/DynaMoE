//
//  DeepSeekV41ImagePixels.swift
//  DynaMoE
//
//  Phase: DeepSeek V4.1 vision, the composer's image path (slice #38)
//
//  Encoded image bytes to the row-major RGB buffer the tower reads. The tower
//  samples that buffer itself (`DeepSeekV41VisionPreprocess.patches`), so the
//  only job here is decode, orient and cap: no resize, because the fit that
//  decides the patch grid already happens inside the tower and rescaling twice
//  would put its nearest-neighbour sampling on already-resampled pixels.
//

import CoreGraphics
import Foundation
import ImageIO

nonisolated public enum DeepSeekV41ImagePixelsError: LocalizedError, Equatable {
    case undecodable
    case tooLarge(pixels: Int, limit: Int)

    public var errorDescription: String? {
        switch self {
        case .undecodable:
            return "That image could not be decoded."
        case .tooLarge(let pixels, let limit):
            return "That image is \(pixels) pixels; the limit is \(limit)."
        }
    }
}

/// Row-major 8-bit RGB, `width * height * 3` bytes, top row first.
nonisolated public struct DeepSeekV41ImagePixels: Sendable, Equatable {
    public var rgb: [UInt8]
    public var width: Int
    public var height: Int

    /// The largest source this decodes.
    ///
    /// The tower's own budget is the token cap: 1,024 aligner cells is a
    /// 1,344 x 1,344 pixel box, so 4,096 on the long side keeps a comfortable
    /// margin while refusing the 100-megapixel photo that would cost 300 MB of
    /// RGB to end up as 31 x 31 cells. Bounded here rather than in the tower
    /// because this is where the allocation happens.
    public static let maximumSourceSide = 4_096

    public var isWellFormed: Bool {
        width > 0 && height > 0 && rgb.count == width * height * 3
    }

    /// Decodes `data` (any format ImageIO reads) into upright RGB.
    ///
    /// Orientation is applied through the thumbnail pass, so a phone photo whose
    /// pixels are stored sideways arrives the way the user sees it. The pass also
    /// caps the source; a smaller image is returned unmodified rather than
    /// upscaled.
    public static func decode(_ data: Data) throws -> DeepSeekV41ImagePixels {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(source) > 0 else {
            throw DeepSeekV41ImagePixelsError.undecodable
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumSourceSide,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw DeepSeekV41ImagePixelsError.undecodable
        }
        return try decode(image)
    }

    public static func decode(_ image: CGImage) throws -> DeepSeekV41ImagePixels {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { throw DeepSeekV41ImagePixelsError.undecodable }
        let pixels = width * height
        guard pixels <= maximumSourceSide * maximumSourceSide else {
            throw DeepSeekV41ImagePixelsError.tooLarge(
                pixels: pixels, limit: maximumSourceSide * maximumSourceSide
            )
        }
        // A 24-bit RGB bitmap context is the natural fit and is NOT supported here
        // (`CGContext` returns nil for 8bpc/24bpp in sRGB on macOS), so the draw
        // goes through 32-bit RGBA and is compacted. The measured cost of that is a
        // copy of one image, next to a tower encode that reads 970 MB of weights.
        var rgba = [UInt8](repeating: 0, count: pixels * 4)
        let drawn: Bool = rgba.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(
                      data: base,
                      width: width,
                      height: height,
                      bitsPerComponent: 8,
                      bytesPerRow: width * 4,
                      space: space,
                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
                  ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw DeepSeekV41ImagePixelsError.undecodable }
        var rgb = [UInt8](repeating: 0, count: pixels * 3)
        for index in 0..<pixels {
            let source = index * 4
            let target = index * 3
            rgb[target] = rgba[source]
            rgb[target + 1] = rgba[source + 1]
            rgb[target + 2] = rgba[source + 2]
        }
        return DeepSeekV41ImagePixels(rgb: rgb, width: width, height: height)
    }
}
