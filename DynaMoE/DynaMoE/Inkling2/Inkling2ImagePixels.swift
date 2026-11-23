//
//  Inkling2ImagePixels.swift
//  DynaMoE
//
//  Encoded image bytes to the buffer the vision tower reads.
//
//  This family's processor does NOT resize a picture. It keeps the source at
//  its own resolution, pads it up to a whole number of 40-pixel cells, and
//  normalizes `(x / 255 - 0.5) / 0.5`. Two consequences are easy to get wrong
//  and both are silent:
//
//    * the PAD VALUE is written in the 0-255 space (`-1.0`) BEFORE the rescale,
//      so a padded pixel normalizes to -1.0078 rather than -1.0;
//    * a picture with transparency is composited onto a 204/153 checkerboard in
//      8-pixel cells, because the processor converts RGBA to RGB rather than
//      dropping the alpha channel.
//
//  Cell count is what the prompt pays for (`Inkling2VisionSpec.languagePositions`),
//  so it is reported here rather than recomputed by a caller.
//

import CoreGraphics
import Foundation
import ImageIO

nonisolated public enum Inkling2ImagePixelsError: LocalizedError, Equatable {
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

nonisolated public enum Inkling2ImagePixels {
    /// One language-token cell, matching the image processor's fixed size.
    public static let cellSide = 40
    /// The padded region's value in the 0-255 space, before normalization. It
    /// is NEGATIVE on purpose: the processor writes it before the rescale, so a
    /// padded pixel normalizes to -1.0078, not to `0`.
    public static let padValue = -1
    /// A source larger than this is refused: past it the cell count alone makes
    /// the prompt unservable, and the RGB allocation is the thing that hurts.
    public static let maximumSourceSide = 4_096

    /// Row-major 8-bit RGB, `width * height * 3` bytes, top row first. Alpha has
    /// already been composited onto the checkerboard.
    public struct RGB: Sendable, Equatable {
        public var rgb: [UInt8]
        public var width: Int
        public var height: Int

        public var isWellFormed: Bool {
            width > 0 && height > 0 && rgb.count == width * height * 3
        }
    }

    /// The tower's input for one picture: how many cells it occupies and the
    /// normalized `[cell][40][40][3]` values, cell order row-major.
    public struct Cells: Sendable, Equatable {
        public var values: [Float]
        public var cellRows: Int
        public var cellColumns: Int

        public var languagePositions: Int { cellRows * cellColumns }
    }

    /// Decodes `data` (any format ImageIO reads) into upright RGB, applying EXIF
    /// orientation and the alpha composite.
    public static func decode(_ data: Data) throws -> RGB {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(source) > 0 else {
            throw Inkling2ImagePixelsError.undecodable
        }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumSourceSide,
            kCGImageSourceShouldCacheImmediately: true
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            ?? CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw Inkling2ImagePixelsError.undecodable
        }
        return try decode(image)
    }

    public static func decode(_ image: CGImage) throws -> RGB {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { throw Inkling2ImagePixelsError.undecodable }
        let pixels = width * height
        guard pixels <= maximumSourceSide * maximumSourceSide else {
            throw Inkling2ImagePixelsError.tooLarge(
                pixels: pixels, limit: maximumSourceSide * maximumSourceSide
            )
        }
        // A 24-bit sRGB context is refused on this OS, so the draw goes through
        // 32-bit and is compacted -- one extra copy next to a tower encode.
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
                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                  ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw Inkling2ImagePixelsError.undecodable }
        return composite(rgba: rgba, width: width, height: height)
    }

    /// RGBA (premultiplied) to the processor's checkerboard composite. In
    /// premultiplied space the reference's
    /// `(rgb * alpha + background * (255 - alpha) + 127) / 255` is
    /// `premultiplied + background * (255 - alpha) / 255`, so the alpha
    /// multiply is already done by the draw.
    public static func composite(rgba: [UInt8], width: Int, height: Int) -> RGB {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for row in 0..<height {
            for column in 0..<width {
                let index = row * width + column
                let source = index * 4
                let alpha = Int(rgba[source + 3])
                let background = ((row / 8) + (column / 8)) % 2 == 0 ? 204 : 153
                let contribution = (background * (255 - alpha) + 127) / 255
                let target = index * 3
                for channel in 0..<3 {
                    let value = Int(rgba[source + channel]) + contribution
                    rgb[target + channel] = UInt8(min(255, max(0, value)))
                }
            }
        }
        return RGB(rgb: rgb, width: width, height: height)
    }

    /// Pads to whole cells and normalizes, in the processor's order: the pad is
    /// written on the 0-255 scale and then normalized with everything else.
    public static func cells(_ pixels: RGB) throws -> Cells {
        guard pixels.isWellFormed else { throw Inkling2ImagePixelsError.undecodable }
        let rows = (pixels.height + cellSide - 1) / cellSide
        let columns = (pixels.width + cellSide - 1) / cellSide
        let paddedSide = cellSide
        var values = [Float](repeating: 0, count: rows * columns * paddedSide * paddedSide * 3)
        for cellRow in 0..<rows {
            for cellColumn in 0..<columns {
                let cell = cellRow * columns + cellColumn
                for row in 0..<cellSide {
                    for column in 0..<cellSide {
                        let sourceRow = cellRow * cellSide + row
                        let sourceColumn = cellColumn * cellSide + column
                        let target = ((cell * cellSide + row) * cellSide + column) * 3
                        for channel in 0..<3 {
                            let raw: Int
                            if sourceRow < pixels.height && sourceColumn < pixels.width {
                                let index = (sourceRow * pixels.width + sourceColumn) * 3 + channel
                                raw = Int(pixels.rgb[index])
                            } else {
                                raw = padValue
                            }
                            // (x / 255 - 0.5) / 0.5
                            values[target + channel] = (Float(raw) / 255 - 0.5) / 0.5
                        }
                    }
                }
            }
        }
        return Cells(values: values, cellRows: rows, cellColumns: columns)
    }

    /// Bytes to the tower's input in one call, which is the path a turn takes.
    public static func cells(from data: Data) throws -> Cells {
        try cells(decode(data))
    }
}
