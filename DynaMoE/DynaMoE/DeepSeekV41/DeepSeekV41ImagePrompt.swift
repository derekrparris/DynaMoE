//
//  DeepSeekV41ImagePrompt.swift
//  DynaMoE
//
//  Where an image lands in a rendered prompt, and which residual row each of its
//  positions takes.
//
//  The checkpoint writes `image_token_id` at EVERY position of an image's span and
//  keeps the start / image / newline / end distinction in a parallel type array, so
//  the prompt's token stream carries one id per span position and nothing more. That
//  means an image enters a prompt as a token-id expansion, not by pasting text: the
//  caller tokenizes the prompt with one placeholder token where the picture belongs,
//  and this expands that token to `rows * (columns + 1) + 2` copies, each bound to
//  its own aligner row.
//
//  Two mismatches are possible and both are reported rather than dropped: an image
//  with no placeholder (its rows would have nowhere to go) and a placeholder with no
//  image (its span length is unknown, so it stays one token). Dropping either one
//  silently shifts every position after it, which is the one failure mode that reads
//  as "the model answered about something else".
//

import Foundation

/// Pixels an attachment would send, as the tower wants them: row-major 8-bit RGB.
nonisolated public struct DeepSeekV41ImageInput: Sendable, Equatable {
    public let rgb: [UInt8]
    public let width: Int
    public let height: Int
    /// Content digest from the asset store, when the caller has one, so an encoded
    /// span can be cached instead of re-running the tower for a picture already in
    /// the conversation.
    public let sha256: String

    public init(rgb: [UInt8], width: Int, height: Int, sha256: String = "") {
        self.rgb = rgb
        self.width = width
        self.height = height
        self.sha256 = sha256
    }

    public var isWellFormed: Bool {
        width > 0 && height > 0 && rgb.count == width * height * 3
    }
}

nonisolated public enum DeepSeekV41ImagePromptError: Error, Equatable, CustomStringConvertible {
    case spanLengthMismatch(image: Int, expected: Int, actual: Int)
    case rowWidthMismatch(image: Int, position: Int, expected: Int, actual: Int)

    public var description: String {
        switch self {
        case let .spanLengthMismatch(image, expected, actual):
            return "image \(image) has \(actual) span rows where its grid needs \(expected)"
        case let .rowWidthMismatch(image, position, expected, actual):
            return "image \(image) row \(position) is \(actual) wide, expected \(expected)"
        }
    }
}

nonisolated public enum DeepSeekV41ImagePrompt {

    /// One image's contribution: the aligner rows in reading order, plus the grid
    /// they came from so the span length can be checked instead of trusted.
    public struct ImageSpan: Sendable, Equatable {
        public let rows: [[Float]]
        public let columns: Int
        public let gridRows: Int

        public init(rows: [[Float]], columns: Int, gridRows: Int) {
            self.rows = rows
            self.columns = columns
            self.gridRows = gridRows
        }

        /// `num_image_tokens`: the grid plus one newline position per row and two
        /// delimiters.
        public var tokenCount: Int {
            DeepSeekV41VisionLayout.imageTokenCount(columns: columns, rows: gridRows)
        }
    }

    /// A prompt with its images expanded, and the row each image position takes.
    public struct Plan: Sendable, Equatable {
        /// The prompt's tokens with every placeholder replaced by its span.
        public let tokenIds: [UInt32]
        /// Expanded position -> residual row. Empty for a text-only prompt.
        public let rows: [Int: [Float]]
        /// The expanded range each image occupies, in image order.
        public let spans: [Range<Int>]
        public let report: DeepSeekV41VisionLayout.ExpansionReport

        public var promptTokens: Int { tokenIds.count }
        public var imagePositionCount: Int { rows.count }
    }

    /// Expands every `imageTokenID` in `tokenIds` into the next image's span.
    ///
    /// - Parameters:
    ///   - tokenIds: the tokenized prompt, with one placeholder token per image.
    ///   - spans: the images' spans, in prompt order.
    ///   - imageTokenID: the checkpoint's `image_token_id`.
    ///   - rowWidth: the residual width a row must have (the text hidden size).
    /// - Throws: when a span's length disagrees with its grid, or a row is not
    ///   `rowWidth` wide. Both are prompt-assembly bugs that would otherwise show up
    ///   as a misaligned conversation.
    public static func expand(
        tokenIds: [UInt32],
        spans: [ImageSpan],
        imageTokenID: Int,
        rowWidth: Int
    ) throws -> Plan {
        for (index, span) in spans.enumerated() {
            guard span.rows.count == span.tokenCount else {
                throw DeepSeekV41ImagePromptError.spanLengthMismatch(
                    image: index, expected: span.tokenCount, actual: span.rows.count
                )
            }
            for (position, row) in span.rows.enumerated() where row.count != rowWidth {
                throw DeepSeekV41ImagePromptError.rowWidthMismatch(
                    image: index, position: position, expected: rowWidth, actual: row.count
                )
            }
        }

        let marker = UInt32(clamping: imageTokenID)
        var out: [UInt32] = []
        out.reserveCapacity(tokenIds.count + spans.reduce(0) { $0 + max(0, $1.tokenCount - 1) })
        var rows: [Int: [Float]] = [:]
        var ranges: [Range<Int>] = []
        var next = 0
        var unfilled = 0

        for id in tokenIds {
            guard id == marker else {
                out.append(id)
                continue
            }
            guard next < spans.count else {
                // No image left: the placeholder is passed through verbatim, because
                // deleting it would move every position after it.
                unfilled += 1
                out.append(id)
                continue
            }
            let span = spans[next]
            next += 1
            let start = out.count
            out.append(contentsOf: Array(repeating: marker, count: span.tokenCount))
            for (offset, row) in span.rows.enumerated() {
                rows[start + offset] = row
            }
            ranges.append(start..<out.count)
        }

        return Plan(
            tokenIds: out,
            rows: rows,
            spans: ranges,
            report: DeepSeekV41VisionLayout.ExpansionReport(
                unusedImageCount: spans.count - next,
                unfilledPlaceholderCount: unfilled
            )
        )
    }
}
