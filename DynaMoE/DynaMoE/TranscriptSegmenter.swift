//
//  TranscriptSegmenter.swift
//  DynaMoE
//
//  Splits a message into transcript rows small enough to paint.
//
//  The transcript's `LazyVStack` emits one child per message, so a single
//  assistant turn of 50–60k tokens laid out to 15,000–20,000 points in ONE row.
//  That crosses the compositor's maximum layer size on Apple Silicon (~16,384
//  pt): the row is realized but cannot be painted, and the transcript goes
//  blank. Bounding the text views inside the row (`LongTextChunker`,
//  `ChunkedText`) never fixed it, because the row that contains them stayed
//  unbounded.
//
//  This type bounds the ROW. A message that fits the budget produces exactly
//  one row, so its rendering is identical to the unsegmented transcript; a
//  message that would lay out too tall is split into several rows, each small
//  enough to draw, and laziness becomes meaningful again.
//
//  Pure and view-free on purpose: the segmentation rules are the part that can
//  be tested without a window, a model, or a snapshot.
//

import Foundation

/// One child of the transcript's `LazyVStack`: a slice of one message.
///
/// The transcript emits one of these per segment rather than one per message,
/// which is what bounds the height of any single lazy child. A message that
/// fits the budget yields exactly one row, so nothing about its rendering
/// changes.
nonisolated struct TranscriptRow: Identifiable, Equatable {
    /// What the row draws. A compaction digest and a plan card are their own
    /// kinds because they replace `ChatMessageView` entirely; everything else
    /// is a slice of an ordinary message.
    enum Kind: Equatable {
        case compactionSummary
        case planCard
        case message
    }

    let messageId: UUID
    let kind: Kind
    let segmentIndex: Int
    let segmentCount: Int
    let segment: TranscriptSegment

    var id: String { "\(messageId.uuidString)#\(segmentIndex)" }
    var isFirst: Bool { segmentIndex == 0 }
    var isLast: Bool { segmentIndex == segmentCount - 1 }

    /// The id a `scrollTo` must target for a message. A message's own id no
    /// longer names a row once it is split, and scrolling to a row that does
    /// not exist silently does nothing — so every call site goes through here.
    static func firstSegmentId(for messageId: UUID) -> String {
        "\(messageId.uuidString)#0"
    }
}

/// One row's worth of a message, ready to render.
nonisolated struct TranscriptSegment: Equatable {
    var blocks: [MarkdownBlock]
    /// Set for a user turn, whose bubble draws its text verbatim rather than
    /// through the markdown engine. Splitting it into blocks would change how
    /// it renders (a pasted `#` line would become a heading), so the slice is
    /// carried as text instead.
    var plainText: String? = nil

    var isEmpty: Bool { blocks.isEmpty && plainText == nil }
}

/// A row plus the message it came from, so the transcript's `ForEach` never
/// has to search the message array per row (which would be O(n²) on a long
/// conversation, on every body pass).
nonisolated struct TranscriptEntry: Identifiable {
    let row: TranscriptRow
    let message: ChatMessage

    var id: String { row.id }
}

nonisolated enum TranscriptSegmenter {
    /// Characters one row may carry. 12,000 characters is ~150 lines of 14 pt
    /// type, i.e. ~2,500 pt laid out — comfortably under the ceiling, with room
    /// for a code block's padding and a table's row height.
    static let maxRowCharacters = 12_000

    /// Lines one row may carry, so a message of many short lines (a diff, a
    /// list) is bounded by height rather than by characters alone.
    static let maxRowLines = 150

    /// Blocks one row may carry. Each block is its own view with its own
    /// padding, so this bounds the row's layer count as well as its height.
    static let maxRowBlocks = 24

    /// Hard ceiling on the rows one table block may carry, on top of the row's
    /// line budget. A table's height is driven by its rows rather than by its
    /// character count and a cell wraps per column, so the line budget usually
    /// splits it first; this bounds the block count too.
    static let maxTableRows = 60

    /// The compositor's layer ceiling this segmentation exists to stay under,
    /// in points. Reported by the diagnostic so a regression is visible.
    static let compositorHeightCeiling: CGFloat = 16_384

    /// Height above which the diagnostic warns about a single row.
    static let rowHeightWarning: CGFloat = 8_000

    /// Height cap for a region of INSPECTION content that rides a row and cannot
    /// be segmented: the reasoning accordion, a plan document, a compaction
    /// summary, the tool-call card list.
    ///
    /// These are the last unbounded things a transcript row can carry. A
    /// reasoning trace of 60k characters laid out inline is ~16,000 pt of views
    /// inside ONE lazy child, which is exactly what the compositor refuses to
    /// paint, so each such region is drawn inside a scroll area no taller than
    /// this instead. The content stays reachable; the row stays bounded.
    static let inspectionHeightCeiling: CGFloat = 420

    /// Chrome a row adds around its content: avatar, header, working indicator,
    /// telemetry footer, attachment strip.
    static let rowChromeHeight: CGFloat = 420

    /// A message's first row can carry two capped sections before its own
    /// blocks: the reasoning accordion and the tool-call card list.
    static var rowRiderHeightCeiling: CGFloat { inspectionHeightCeiling * 2 }

    /// Tallest a row can get: its segmented blocks plus everything that rides
    /// the row and cannot be split, plus the row's own chrome. Segmentation is
    /// only doing its job while this stays under the compositor's ceiling.
    static func worstCaseRowHeight(of segment: TranscriptSegment) -> CGFloat {
        estimatedHeight(of: segment) + rowRiderHeightCeiling + rowChromeHeight
    }

    /// True when `text` would lay out taller than an inspection region may be,
    /// so it has to be drawn inside a capped scroll area rather than inline.
    static func exceedsInspectionHeight(_ text: String) -> Bool {
        LongTextChunker.isOversized(text, maxLines: 24, maxCharacters: 2_400)
    }

    // MARK: - Rows

    /// One row per segment, always at least one: a tool-only assistant turn has
    /// no blocks at all, and its row still has to exist so the working
    /// indicator and the telemetry footer render.
    static func rows(
        for messageId: UUID,
        kind: TranscriptRow.Kind,
        segments: [TranscriptSegment]
    ) -> [TranscriptRow] {
        let resolved = segments.isEmpty ? [TranscriptSegment(blocks: [])] : segments
        return resolved.enumerated().map { index, segment in
            TranscriptRow(
                messageId: messageId,
                kind: kind,
                segmentIndex: index,
                segmentCount: resolved.count,
                segment: segment
            )
        }
    }

    /// Segments for a user turn: its text split into bounded slices, drawn
    /// verbatim. A message that fits the budget yields one segment holding the
    /// whole string, so its bubble is unchanged.
    static func plainTextSegments(for text: String) -> [TranscriptSegment] {
        let pieces = LongTextChunker.chunks(
            of: text,
            maxLines: maxRowLines,
            maxCharacters: maxRowCharacters
        )
        guard !pieces.isEmpty else { return [TranscriptSegment(blocks: [])] }
        return pieces.map { TranscriptSegment(blocks: [], plainText: $0) }
    }

    // MARK: - Segmentation

    /// Groups blocks into rows, splitting any single block that is larger than
    /// a row's budget.
    ///
    /// Always returns at least one segment, which may be empty.
    static func segments(for blocks: [MarkdownBlock]) -> [TranscriptSegment] {
        let expanded = blocks.flatMap(splitOversized)
        guard !expanded.isEmpty else { return [TranscriptSegment(blocks: [])] }

        var segments: [TranscriptSegment] = []
        var current: [MarkdownBlock] = []
        var currentCharacters = 0
        var currentLines = 0

        func flush() {
            guard !current.isEmpty else { return }
            segments.append(TranscriptSegment(blocks: current))
            current.removeAll(keepingCapacity: true)
            currentCharacters = 0
            currentLines = 0
        }

        for block in expanded {
            let size = characterCount(of: block)
            let lines = laidOutLineCount(of: block)
            if !current.isEmpty,
               currentCharacters + size > maxRowCharacters
                || current.count >= maxRowBlocks
                || currentLines + lines > maxRowLines {
                flush()
            }
            current.append(block)
            currentCharacters += size
            currentLines += lines
        }
        flush()
        return segments.isEmpty ? [TranscriptSegment(blocks: [])] : segments
    }

    /// A block's contribution to a row's budget. A table counts its cells, so
    /// a wide table is not treated as one short line.
    static func characterCount(of block: MarkdownBlock) -> Int {
        switch block {
        case .heading(_, let text): return text.count
        case .paragraph(let text): return text.count
        case .blockquote(let text): return text.count
        case .listItem(_, let text): return text.count
        case .codeBlock(_, let code): return code.count
        case .horizontalRule: return 0
        case .table(let table):
            return table.headers.reduce(0) { $0 + $1.count }
                + table.rows.reduce(0) { total, row in
                    total + row.reduce(0) { $0 + $1.count }
                }
        }
    }

    /// Lines a block occupies once laid out, long lines counted wrapped.
    ///
    /// A row has to be bounded by HEIGHT, and height is line count, not
    /// character count: 1,200 lines of ten characters each is only 12,000
    /// characters but would lay out over 20,000 points. The character budget
    /// alone let exactly that row through.
    static func laidOutLineCount(of block: MarkdownBlock) -> Int {
        switch block {
        case .heading(_, let text), .paragraph(let text),
             .blockquote(let text), .listItem(_, let text):
            return wrappedLineCount(text)
        case .codeBlock(_, let code):
            return wrappedLineCount(code)
        case .horizontalRule:
            return 1
        case .table(let table):
            // A table's height is line count too, and its cells WRAP: the
            // transcript dropped the horizontal scroll view, so a wide cell
            // occupies as many lines as its column is narrow. Counting one line
            // per row let a 60-row table of wide cells lay out several times
            // taller than the budget predicted.
            let columns = max(1, table.headers.count)
            // The header plus the divider grid row beneath it.
            var lines = tableRowLineCount(table.headers, columns: columns) + 1
            for row in table.rows {
                lines += tableRowLineCount(row, columns: columns)
            }
            return lines
        }
    }

    /// Characters a full-width line holds at the transcript's type scale: 14 pt
    /// type sets roughly 80.
    static let wrappedLineWidth = 80

    /// Lines one table row occupies once its cells wrap inside their columns.
    ///
    /// A cell's available width is the full line split across the columns, so a
    /// two-column table gives each cell half a line's characters. This is an
    /// estimate on the same terms as `wrappedLineCount`, kept in characters
    /// rather than points on purpose: the segmenter runs without a window and
    /// cannot measure text.
    static func tableRowLineCount(_ cells: [String], columns: Int) -> Int {
        let perLine = tableColumnCharacters(columns: columns)
        var lines = 1
        for cell in cells {
            lines = max(lines, wrappedLineCount(cell, charactersPerLine: perLine))
        }
        return lines
    }

    /// Characters one table cell fits per line, given a table's column count.
    private static func tableColumnCharacters(columns: Int) -> Int {
        guard columns > 0 else { return wrappedLineWidth }
        return max(8, wrappedLineWidth / columns)
    }

    /// Lines `text` occupies at the transcript's width, where a long line wraps
    /// rather than scrolling.
    private static func wrappedLineCount(
        _ text: String,
        charactersPerLine: Int = TranscriptSegmenter.wrappedLineWidth
    ) -> Int {
        let perLine = max(1, charactersPerLine)
        var total = 0
        var current = 0
        for character in text {
            if character == "\n" {
                total += max(1, (current + perLine - 1) / perLine)
                current = 0
            } else {
                current += 1
            }
        }
        total += max(1, (current + perLine - 1) / perLine)
        return total
    }

    /// Rough laid-out height of one segment, used by the diagnostic that makes
    /// the whiteout threshold falsifiable.
    static func estimatedHeight(of segment: TranscriptSegment) -> CGFloat {
        var points: CGFloat = 0
        for block in segment.blocks {
            // 14 pt type occupies ~17 pt per line; each block adds its own
            // padding and the stack's spacing.
            points += CGFloat(laidOutLineCount(of: block)) * 17 + 24
        }
        return points
    }

    // MARK: - Splitting one oversized block

    /// A single block larger than a row's budget cannot be bounded by grouping,
    /// so it is split into sibling blocks of the same kind. Splitting happens on
    /// line boundaries, so concatenating the pieces reproduces the original text.
    ///
    /// The pieces always land in different rows: a piece is at most `maxRowLines`
    /// lines and a row accepts at most that many, so nothing is left to be
    /// marked as "this continues the block before it". Rows of one message are
    /// drawn flush instead (see the transcript's `LazyVStack`), which is what
    /// makes a split block read continuously.
    private static func splitOversized(_ block: MarkdownBlock) -> [MarkdownBlock] {
        switch block {
        case .heading(let level, let text):
            return splitText(text).map { .heading(level: level, text: $0) }
        case .paragraph(let text):
            return splitText(text).map { .paragraph(text: $0) }
        case .blockquote(let text):
            return splitText(text).map { .blockquote(text: $0) }
        case .listItem(let number, let text):
            return splitText(text).map { .listItem(number: number, text: $0) }
        case .codeBlock(let language, let code):
            // Each piece becomes its own card, so a 200 KB block shows a
            // repeated language header every 12,000 characters. That is the
            // price of bounding it: one card of that size is never painted.
            return splitText(code).map { .codeBlock(language: language, code: $0) }
        case .table(let table):
            return splitTable(table).map { .table($0) }
        case .horizontalRule:
            return [block]
        }
    }

    private static func splitText(_ text: String) -> [String] {
        // Both budgets, not just characters: a block of 1,200 ten-character
        // lines is under the character budget and would lay out over 20,000
        // points, so gating on length alone left it unbounded.
        guard LongTextChunker.isOversized(
            text,
            maxLines: maxRowLines,
            maxCharacters: maxRowCharacters
        ) else { return [text] }
        let pieces = LongTextChunker.chunks(
            of: text,
            maxLines: maxRowLines,
            maxCharacters: maxRowCharacters
        )
        return pieces.isEmpty ? [text] : pieces
    }

    private static func splitTable(_ table: MarkdownTableData) -> [MarkdownTableData] {
        let columns = max(1, table.headers.count)
        // Every piece repeats the header and its divider, so both come out of
        // the row's line budget before any data row is placed.
        let headerLines = tableRowLineCount(table.headers, columns: columns) + 1
        let rowLineBudget = max(1, maxRowLines - headerLines)

        var totalLines = headerLines
        for row in table.rows {
            totalLines += tableRowLineCount(row, columns: columns)
        }
        guard table.rows.count > maxTableRows || totalLines > maxRowLines else {
            return [table]
        }

        var pieces: [MarkdownTableData] = []
        var start = 0
        while start < table.rows.count {
            var end = start
            var pieceLines = 0
            while end < table.rows.count, end - start < maxTableRows {
                let rowLines = tableRowLineCount(table.rows[end], columns: columns)
                // A single row taller than the whole budget still needs a piece
                // of its own, or the split would never terminate.
                if end > start, pieceLines + rowLines > rowLineBudget { break }
                pieceLines += rowLines
                end += 1
            }
            if end == start { end = start + 1 }
            pieces.append(
                MarkdownTableData(
                    headers: table.headers,
                    alignments: table.alignments,
                    rows: Array(table.rows[start..<end]),
                    id: "\(table.id)-part\(pieces.count)"
                )
            )
            start = end
        }
        return pieces.isEmpty ? [table] : pieces
    }
}
