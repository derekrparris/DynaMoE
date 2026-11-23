//
//  Inkling2ImagePrompt.swift
//  DynaMoE
//
//  Where a picture lands in a rendered prompt, and which residual row each of
//  its positions takes.
//
//  The checkpoint's template writes ONE `<|unused_200054|>` per image part. The
//  released processor then replaces that single placeholder with
//  `rows * columns` copies, where the grid is `ceil(height / 40)` by
//  `ceil(width / 40)` -- one placeholder per CELL, because the tower emits one
//  `textHiddenSize` row per cell. That replacement happens here as a token-id
//  expansion rather than a string edit: the rendered prompt stays exactly what
//  the template writes, so its goldens and its digest do not move, and the
//  expanded ids are a separate value the cache and the sampler see.
//
//  Counts are REFUSED rather than reconciled. The reference raises when the
//  number of placeholders in the text and the number of images disagree, and
//  the reason is the one failure this file exists to prevent: dropping either
//  side silently shifts every position after it, which reads as the model
//  answering about a different conversation.
//

import Foundation

// MARK: - Errors

nonisolated public enum Inkling2ImagePromptError: Error, Equatable, CustomStringConvertible {
    case invalidImageTokenID(Int)
    case placeholderCountMismatch(placeholders: Int, images: Int)
    case invalidGrid(image: Int, rows: Int, columns: Int)
    case spanLengthMismatch(image: Int, expected: Int, actual: Int)
    case rowWidthMismatch(image: Int, position: Int, expected: Int, actual: Int)
    case spanCountMismatch(spans: Int, ranges: Int)
    case spanRangeInvalid(image: Int, range: Range<Int>, promptTokens: Int)
    case spanRangeOutOfOrder(image: Int)
    case spanTokenMismatch(image: Int, position: Int, expected: Int, actual: UInt32)

    public var description: String {
        switch self {
        case .invalidImageTokenID(let id):
            return "image token id \(id) is not a token id"
        case .placeholderCountMismatch(let placeholders, let images):
            return "the prompt has \(placeholders) image placeholder(s) but \(images) image(s) were supplied"
        case .invalidGrid(let image, let rows, let columns):
            return "image \(image) has a \(rows)x\(columns) cell grid, which has no language positions"
        case .spanLengthMismatch(let image, let expected, let actual):
            return "image \(image) has \(actual) span rows where its grid needs \(expected)"
        case .rowWidthMismatch(let image, let position, let expected, let actual):
            return "image \(image) row \(position) is \(actual) wide, expected \(expected)"
        case .spanCountMismatch(let spans, let ranges):
            return "\(spans) image span(s) were supplied for \(ranges) recorded range(s)"
        case .spanRangeInvalid(let image, let range, let promptTokens):
            return "image \(image)'s recorded span \(range) is not inside a \(promptTokens)-token prompt"
        case .spanRangeOutOfOrder(let image):
            return "image \(image)'s recorded span is not after the previous image's"
        case .spanTokenMismatch(let image, let position, let expected, let actual):
            return "image \(image)'s span holds token \(actual) at position \(position), expected \(expected)"
        }
    }
}

// MARK: - Expansion

nonisolated public enum Inkling2ImagePrompt {

    /// One image's contribution: one `textHiddenSize` row per cell, in cell
    /// order, plus the grid they came from so the span length can be checked
    /// instead of trusted.
    public struct ImageSpan: Sendable, Equatable {
        public let rows: [[Float]]
        public let grid: Inkling2ImageGrid

        public init(rows: [[Float]], grid: Inkling2ImageGrid) {
            self.rows = rows
            self.grid = grid
        }

        /// The markers this image occupies: one per cell.
        public var tokenCount: Int { grid.languagePositions }
    }

    /// A prompt with its images expanded, and the row each image position takes.
    public struct Plan: Sendable, Equatable {
        /// The prompt's tokens with every placeholder replaced by its span.
        public let tokenIds: [UInt32]
        /// Expanded position -> residual row. Empty for a text-only prompt.
        public let rows: [Int: [Float]]
        /// The expanded range each image occupies, in image order.
        public let spans: [Range<Int>]

        public var promptTokens: Int { tokenIds.count }
        public var imagePositionCount: Int { rows.count }
    }

    /// Expands every `imageTokenID` in `tokenIds` into the next image's span.
    ///
    /// - Parameters:
    ///   - tokenIds: the tokenized prompt, with one placeholder token per image.
    ///   - spans: the images' encoded rows, in prompt order.
    ///   - imageTokenID: the checkpoint's `image_token_id` (200054 here).
    ///   - rowWidth: the residual width a row must have (the text hidden size).
    /// - Throws: when the placeholder and image counts disagree, when a span's
    ///   length disagrees with its grid, or when a row is not `rowWidth` wide.
    ///   Every one of these is a prompt-assembly bug that would otherwise show
    ///   up as a misaligned conversation.
    public static func expand(
        tokenIds: [UInt32],
        spans: [ImageSpan],
        imageTokenID: Int,
        rowWidth: Int
    ) throws -> Plan {
        guard imageTokenID >= 0, imageTokenID <= Int(UInt32.max) else {
            throw Inkling2ImagePromptError.invalidImageTokenID(imageTokenID)
        }
        let marker = UInt32(imageTokenID)
        let placeholderCount = tokenIds.reduce(0) { $0 + ($1 == marker ? 1 : 0) }
        guard placeholderCount == spans.count else {
            throw Inkling2ImagePromptError.placeholderCountMismatch(
                placeholders: placeholderCount, images: spans.count
            )
        }
        try validate(spans, rowWidth: rowWidth)

        var output: [UInt32] = []
        output.reserveCapacity(tokenIds.count + spans.reduce(0) { $0 + max(0, $1.tokenCount - 1) })
        var rows: [Int: [Float]] = [:]
        var ranges: [Range<Int>] = []
        var next = 0

        for id in tokenIds {
            guard id == marker else {
                output.append(id)
                continue
            }
            // Safe: the counts were proved equal above, so every placeholder
            // has a span and every span is consumed once, in prompt order.
            let span = spans[next]
            next += 1
            let start = output.count
            output.append(contentsOf: repeatElement(marker, count: span.tokenCount))
            for (offset, row) in span.rows.enumerated() {
                rows[start + offset] = row
            }
            ranges.append(start..<output.count)
        }

        return Plan(tokenIds: output, rows: rows, spans: ranges)
    }

    /// Puts already-encoded rows back at the spans a previous turn recorded.
    ///
    /// A continuation splices the previous turn's prompt forward VERBATIM, so
    /// it arrives already expanded: there is no single placeholder left to
    /// expand, and counting the markers would read one picture as hundreds of
    /// placeholders. The recorded layout is what says where each picture sits,
    /// and every part of it is checked rather than trusted -- a range of the
    /// wrong length, a token that is not the marker, an out-of-order or
    /// overlapping range, or a grid whose row count disagrees all mean the
    /// spliced prompt and the recorded layout are describing different
    /// conversations, and binding rows there would show the model one picture
    /// where its prompt says another.
    public static func place(
        tokenIds: [UInt32],
        spans: [ImageSpan],
        at ranges: [Range<Int>],
        imageTokenID: Int,
        rowWidth: Int
    ) throws -> Plan {
        guard imageTokenID >= 0, imageTokenID <= Int(UInt32.max) else {
            throw Inkling2ImagePromptError.invalidImageTokenID(imageTokenID)
        }
        guard spans.count == ranges.count else {
            throw Inkling2ImagePromptError.spanCountMismatch(spans: spans.count, ranges: ranges.count)
        }
        try validate(spans, rowWidth: rowWidth)
        let marker = UInt32(imageTokenID)
        var rows: [Int: [Float]] = [:]
        var previousEnd = 0
        for (index, range) in ranges.enumerated() {
            guard range.lowerBound >= 0, range.lowerBound < range.upperBound,
                  range.upperBound <= tokenIds.count else {
                throw Inkling2ImagePromptError.spanRangeInvalid(
                    image: index, range: range, promptTokens: tokenIds.count
                )
            }
            guard range.lowerBound >= previousEnd else {
                throw Inkling2ImagePromptError.spanRangeOutOfOrder(image: index)
            }
            let span = spans[index]
            guard range.count == span.tokenCount else {
                throw Inkling2ImagePromptError.spanLengthMismatch(
                    image: index, expected: span.tokenCount, actual: range.count
                )
            }
            for (offset, position) in range.enumerated() {
                guard tokenIds[position] == marker else {
                    throw Inkling2ImagePromptError.spanTokenMismatch(
                        image: index, position: position, expected: imageTokenID,
                        actual: tokenIds[position]
                    )
                }
                rows[position] = span.rows[offset]
            }
            previousEnd = range.upperBound
        }
        return Plan(tokenIds: tokenIds, rows: rows, spans: ranges)
    }

    /// The span invariants both entry points share: a real grid, one row per
    /// cell and every row in the text hidden size's width.
    private static func validate(_ spans: [ImageSpan], rowWidth: Int) throws {
        for (index, span) in spans.enumerated() {
            guard span.grid.isValid else {
                throw Inkling2ImagePromptError.invalidGrid(
                    image: index, rows: span.grid.cellRows, columns: span.grid.cellColumns
                )
            }
            guard span.rows.count == span.tokenCount else {
                throw Inkling2ImagePromptError.spanLengthMismatch(
                    image: index, expected: span.tokenCount, actual: span.rows.count
                )
            }
            for (position, row) in span.rows.enumerated() where row.count != rowWidth {
                throw Inkling2ImagePromptError.rowWidthMismatch(
                    image: index, position: position, expected: rowWidth, actual: row.count
                )
            }
        }
    }
}
