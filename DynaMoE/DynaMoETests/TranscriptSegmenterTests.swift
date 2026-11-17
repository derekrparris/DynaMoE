//
//  TranscriptSegmenterTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the transcript's row segmentation, the presentation
//  cache that feeds it, and the ready-plan search the transcript memoizes.
//
//  The defect this exists for: the transcript emitted one `LazyVStack` child per
//  message, so a single 50-60k-token assistant turn laid out to 15,000-20,000 pt
//  in ONE row, crossed the compositor's maximum layer size (~16,384 pt on Apple
//  Silicon), and painted blank. A row that is realized but not painted has no
//  error, no exception and no failing assertion anywhere, so these tests pin the
//  only property that can be checked without a window: no row exceeds the budget
//  the segmentation promises, in characters, blocks, lines or laid-out height.
//

import XCTest
import SwiftUI
@testable import DynaMoE

@MainActor
final class TranscriptSegmenterTests: XCTestCase {

    // MARK: - Fixtures

    /// A paragraph of `lines` short lines. Every line is over a character, so
    /// the character budget binds before the line budget does.
    private func longParagraph(lines: Int, lineLength: Int = 50) -> String {
        (0..<lines)
            .map { index -> String in
                let head = "line \(index) "
                return head + String(repeating: "x", count: max(0, lineLength - head.count))
            }
            .joined(separator: "\n")
    }

    /// A paragraph of `lines` lines of `width` characters each: many short
    /// lines, which is a diff or a long list, and the shape the character
    /// budget alone used to let through.
    private func manyShortLines(lines: Int, width: Int = 10) -> String {
        let line = String(repeating: "q", count: width)
        return Array(repeating: line, count: lines).joined(separator: "\n")
    }

    private func text(of block: MarkdownBlock) -> String {
        switch block {
        case .heading(_, let text): return text
        case .paragraph(let text): return text
        case .blockquote(let text): return text
        case .listItem(_, let text): return text
        case .codeBlock(_, let code): return code
        case .table: return ""
        case .horizontalRule: return ""
        }
    }

    /// The two invariants the segmenter guarantees for every row, whatever the
    /// message contains: a bounded layer count and a laid-out height under the
    /// compositor's ceiling. The per-row character and line budgets are asserted
    /// inside the tests whose fixtures make them meaningful, because a single
    /// split piece (a wide table row, say) can exceed a raw budget while still
    /// laying out safely.
    private func assertWithinBudget(
        _ segments: [TranscriptSegment],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for segment in segments {
            XCTAssertLessThanOrEqual(
                segment.blocks.count,
                TranscriptSegmenter.maxRowBlocks,
                "a row exceeded the block budget",
                file: file,
                line: line
            )
            XCTAssertLessThanOrEqual(
                TranscriptSegmenter.estimatedHeight(of: segment),
                TranscriptSegmenter.compositorHeightCeiling,
                "a row would lay out over the compositor's ceiling",
                file: file,
                line: line
            )
        }
    }

    /// Character and line budgets, for fixtures whose blocks all split by those
    /// budgets (every line wraps to one line, no table is involved).
    private func assertWithinTextBudgets(
        _ segments: [TranscriptSegment],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for segment in segments {
            XCTAssertLessThanOrEqual(
                segment.blocks.reduce(0) { $0 + TranscriptSegmenter.characterCount(of: $1) },
                TranscriptSegmenter.maxRowCharacters,
                "a row exceeded the character budget",
                file: file,
                line: line
            )
            XCTAssertLessThanOrEqual(
                segment.blocks.reduce(0) { $0 + TranscriptSegmenter.laidOutLineCount(of: $1) },
                TranscriptSegmenter.maxRowLines,
                "a row exceeded the line budget",
                file: file,
                line: line
            )
        }
    }

    // MARK: - Short messages are untouched

    func testShortMessageYieldsOneRow() {
        let text = "A short answer.\n\nWith a second paragraph."
        let blocks = parseMarkdownBlocks(text)

        let segments = TranscriptSegmenter.segments(for: blocks)
        XCTAssertEqual(segments.count, 1)
        // The single row carries the whole parse, so a message that fits the
        // budget renders exactly as it did before segmentation existed.
        XCTAssertEqual(segments[0].blocks, blocks)
        XCTAssertNil(segments[0].plainText)

        let rows = TranscriptSegmenter.rows(
            for: UUID(),
            kind: .message,
            segments: segments
        )
        XCTAssertEqual(rows.count, 1)
        XCTAssertTrue(rows[0].isFirst)
        XCTAssertTrue(rows[0].isLast)
        XCTAssertEqual(rows[0].segmentCount, 1)
    }

    func testEmptyMessageYieldsOneRow() {
        // A tool-only assistant turn has no blocks at all and still has to draw
        // its working indicator and its telemetry footer, so the row must exist.
        let segments = TranscriptSegmenter.segments(for: [])
        XCTAssertEqual(segments.count, 1)
        XCTAssertTrue(segments[0].blocks.isEmpty)
        XCTAssertTrue(segments[0].isEmpty)

        let rows = TranscriptSegmenter.rows(for: UUID(), kind: .message, segments: segments)
        XCTAssertEqual(rows.count, 1)
    }

    func testRowsReturnsAtLeastOneRowForEmptySegments() {
        let rows = TranscriptSegmenter.rows(for: UUID(), kind: .message, segments: [])
        XCTAssertEqual(rows.count, 1)
        XCTAssertEqual(rows[0].segmentCount, 1)
    }

    // MARK: - Rows that need no parse

    func testAMessageThatCannotSplitIsOneUnparsedRow() {
        // A whole-message segment carries no blocks on purpose: the row parses
        // when the lazy stack realizes it. `isEmpty` decides whether anything is
        // drawn at all, so it must not read "not parsed yet" as "nothing to
        // draw".
        let segment = TranscriptSegment(isWholeMessage: true)
        XCTAssertTrue(segment.blocks.isEmpty)
        XCTAssertNil(segment.plainText)
        XCTAssertFalse(segment.isEmpty)
    }

    func testTheCheapCeilingOnlyAcceptsMessagesThatFitOneRow() {
        XCTAssertTrue(TranscriptSegmenter.isCertainlyOneRow("A short answer."))
        XCTAssertTrue(TranscriptSegmenter.isCertainlyOneRow(longParagraph(lines: 40)))
        // Over the line ceiling: a diff or a long list is parsed so its rows can
        // be bounded.
        XCTAssertFalse(TranscriptSegmenter.isCertainlyOneRow(manyShortLines(lines: 200)))
        // Over the character ceiling on a single line.
        XCTAssertFalse(
            TranscriptSegmenter.isCertainlyOneRow(String(repeating: "x", count: 20_000))
        )
        // Both ceilings are far under the row budget, so what the lazy path
        // accepts can never be a row that blanks the transcript.
        XCTAssertLessThan(
            TranscriptSegmenter.singleRowCharacterCeiling,
            TranscriptSegmenter.maxRowCharacters
        )
        XCTAssertLessThan(
            TranscriptSegmenter.singleRowLineCeiling,
            TranscriptSegmenter.maxRowLines
        )
    }

    func testTheWorstMessageTheLazyPathAcceptsStillFitsOneRow() {
        // The boundary the cheap test admits: every line near its ceiling AND the
        // character ceiling reached. The row riders and the chrome are included,
        // because this row is drawn whole and unsplit.
        let boundary = manyShortLines(
            lines: TranscriptSegmenter.singleRowLineCeiling,
            width: 99
        )
        XCTAssertTrue(TranscriptSegmenter.isCertainlyOneRow(boundary))
        XCTAssertLessThanOrEqual(
            TranscriptSegmenter.worstCaseRowHeight(
                of: TranscriptSegment(blocks: parseMarkdownBlocks(boundary))
            ),
            TranscriptSegmenter.compositorHeightCeiling,
            "a message the lazy path accepts must still fit one row"
        )
    }

    func testDecidingARowNeedsNoParse() {
        // The row loop's decision for a message that fits one row is
        // `isPlanCard` + `isCertainlyOneRow` and nothing else. Parsing every
        // message on the first body pass is what made a long conversation take
        // seconds to open, so neither call may populate the parse cache — even
        // across a whole conversation.
        let cache = TranscriptPresentationCache()
        var entries: [TranscriptEntry] = []
        for index in 0..<100 {
            let message = makeMessage(.assistant, "Answer \(index).")
            XCTAssertFalse(cache.isPlanCard(for: message))
            XCTAssertTrue(TranscriptSegmenter.isCertainlyOneRow(message.content))
            entries.append(contentsOf: TranscriptSegmenter.rows(
                for: message.id,
                kind: .message,
                segments: [TranscriptSegment(isWholeMessage: true)]
            ).map { TranscriptEntry(row: $0, message: message, displayContent: nil) })
        }
        XCTAssertEqual(cache.cachedMessageCount, 0, "no message on the lazy path was parsed")
        XCTAssertEqual(entries.count, 100)
        XCTAssertEqual(Set(entries.map(\.id)).count, 100)
        XCTAssertTrue(entries.allSatisfy { $0.row.segmentCount == 1 && $0.row.isFirst })
        XCTAssertTrue(entries.allSatisfy { $0.displayContent == nil })
    }

    // MARK: - Long messages are bounded

    func testLongMessageSplitsIntoRowsUnderTheBudget() {
        let text = longParagraph(lines: 800)
        XCTAssertGreaterThan(text.count, 30_000)

        let segments = TranscriptSegmenter.segments(for: parseMarkdownBlocks(text))
        XCTAssertGreaterThan(segments.count, 1)
        assertWithinBudget(segments)
        assertWithinTextBudgets(segments)
    }

    func testManyShortLinesAreBoundedByHeightNotCharacters() {
        // 1,200 lines of ten characters is 13,199 characters and would lay out
        // at over 20,000 pt. Bounding rows by characters alone let it through,
        // which is the whole defect this segmentation exists to close.
        let text = manyShortLines(lines: 1_200)
        let segments = TranscriptSegmenter.segments(for: parseMarkdownBlocks(text))
        XCTAssertGreaterThan(segments.count, 1)
        assertWithinBudget(segments)
        assertWithinTextBudgets(segments)

        let totalLines = segments.reduce(0) { total, segment in
            total + segment.blocks.reduce(0) { $0 + TranscriptSegmenter.laidOutLineCount(of: $1) }
        }
        XCTAssertEqual(totalLines, 1_200, "splitting must not lose or invent lines")
    }

    func testTheUnsegmentedRowIsWhatCrossesTheCompositorCeiling() {
        // The falsifiable version of the whiteout hypothesis, and the reason
        // segmentation is the fix rather than a tidier layout: as ONE row, an
        // assistant turn of this length lays out past the compositor's maximum
        // layer size, so it is realized and never painted. Segmented, no row
        // comes close.
        let text = longParagraph(lines: 1_200)
        let blocks = parseMarkdownBlocks(text)
        XCTAssertEqual(blocks.count, 1, "fixture assumes one paragraph")

        let asOneRow = TranscriptSegment(blocks: blocks)
        XCTAssertGreaterThan(
            TranscriptSegmenter.estimatedHeight(of: asOneRow),
            TranscriptSegmenter.compositorHeightCeiling
        )

        let segments = TranscriptSegmenter.segments(for: blocks)
        XCTAssertGreaterThan(segments.count, 1)
        assertWithinBudget(segments)
    }

    func testSplitRowsReproduceTheOriginalText() {        // Every piece splits at a line boundary, so the pieces of the message
        // join back to exactly what the model wrote: selection and the copy
        // action cannot tell that the answer was drawn as several rows.
        let text = longParagraph(lines: 800)

        let segments = TranscriptSegmenter.segments(for: parseMarkdownBlocks(text))
        let joined = segments
            .flatMap { $0.blocks.map(text(of:)) }
            .joined(separator: "\n")
        XCTAssertEqual(joined, text)
    }

    func testManyShortBlocksSplitByBlockCount() {
        // Each block is tiny, so only the per-row block budget can bound this
        // row: every block is its own view with its own padding.
        let text = (0..<100).map { "paragraph \($0)" }.joined(separator: "\n\n")
        let blocks = parseMarkdownBlocks(text)
        XCTAssertEqual(blocks.count, 100, "fixture assumes one block per paragraph")

        let segments = TranscriptSegmenter.segments(for: blocks)
        XCTAssertGreaterThan(segments.count, 1)
        assertWithinBudget(segments)
        assertWithinTextBudgets(segments)
    }

    func testSingleOversizedBlockIsSplitIntoSiblingsOfTheSameKind() {
        // One 40k-character paragraph cannot be bounded by grouping, so it is
        // split into sibling paragraphs.
        let text = longParagraph(lines: 900)
        let segments = TranscriptSegmenter.segments(for: [.paragraph(text: text)])
        XCTAssertGreaterThan(segments.count, 1)
        for segment in segments {
            XCTAssertEqual(segment.blocks.count, 1)
            guard case .paragraph = segment.blocks[0] else {
                return XCTFail("a split paragraph must stay a paragraph")
            }
        }
        XCTAssertEqual(segments.flatMap { $0.blocks.map(text(of:)) }.joined(separator: "\n"), text)
    }

    func testSplitCodeBlockKeepsItsLanguageAndItsCode() {
        let code = (0..<900).map { "let value\($0) = \($0)" }.joined(separator: "\n")
        let segments = TranscriptSegmenter.segments(for: [.codeBlock(language: "swift", code: code)])
        XCTAssertGreaterThan(segments.count, 1)
        assertWithinBudget(segments)
        assertWithinTextBudgets(segments)

        var rejoined = ""
        for segment in segments {
            guard case .codeBlock(let language, let piece) = segment.blocks[0] else {
                return XCTFail("a split code block must stay a code block")
            }
            XCTAssertEqual(language, "swift")
            rejoined += (rejoined.isEmpty ? "" : "\n") + piece
        }
        XCTAssertEqual(rejoined, code)
    }

    func testTableIsSplitByRowCount() {
        let table = MarkdownTableData(
            headers: ["name", "value"],
            alignments: [.leading, .leading],
            rows: (0..<130).map { ["name\($0)", "value\($0)"] }
        )

        let segments = TranscriptSegmenter.segments(for: [.table(table)])
        let tables = segments.flatMap(\.blocks).compactMap { block -> MarkdownTableData? in
            if case .table(let value) = block { return value }
            return nil
        }

        XCTAssertEqual(tables.count, 3)
        XCTAssertEqual(tables.map(\.rows.count), [60, 60, 10])
        // Every piece carries the header and the alignments, so each renders as
        // a usable table rather than a headerless grid.
        for piece in tables {
            XCTAssertEqual(piece.headers, table.headers)
            XCTAssertEqual(piece.alignments, table.alignments)
        }
        // Distinct ids, or the pieces collide in the render list.
        XCTAssertEqual(Set(tables.map(\.id)).count, tables.count)
        XCTAssertEqual(tables.flatMap(\.rows), table.rows)
    }

    func testTableWithWideCellsSplitsByWrappedLinesNotRowCount() {
        // A cell wraps, and the table keeps its own horizontal scroll view, so
        // the estimate assumes less width than a cell may actually get: one line
        // per row was right only for narrow cells, and a 60-row table of wide
        // ones lays out several times taller than that, which is the row the
        // compositor refuses to paint.
        let wideCell = String(repeating: "x", count: 240)
        let table = MarkdownTableData(
            headers: ["a", "b", "c"],
            alignments: [.leading, .leading, .leading],
            rows: (0..<60).map { _ in [wideCell, wideCell, wideCell] }
        )

        // A 3-column table gives each cell a third of a line's characters, so one
        // row is many lines and the whole table is far past the line budget.
        XCTAssertGreaterThan(TranscriptSegmenter.laidOutLineCount(of: .table(table)), 150)

        let segments = TranscriptSegmenter.segments(for: [.table(table)])
        let tables = segments.flatMap(\.blocks).compactMap { block -> MarkdownTableData? in
            if case .table(let value) = block { return value }
            return nil
        }
        XCTAssertGreaterThan(tables.count, 1)
        // Every piece fits the budget it was split for, which is what bounds the
        // row rather than the block count.
        for piece in tables {
            XCTAssertLessThanOrEqual(TranscriptSegmenter.laidOutLineCount(of: .table(piece)), 150)
            XCTAssertEqual(piece.headers, table.headers)
        }
        XCTAssertEqual(tables.flatMap(\.rows), table.rows)
    }

    func testATableCellTooLargeToDrawIsCappedInsteadOfBlowingTheRow() {
        // A table is split by ROW and a cell is atomic, so one tool result
        // pasted into a cell produced a single 51,000 pt segment that no guard
        // in the segmenter could see. The cell is therefore capped where it is
        // DRAWN, and the estimate has to price the same cap or the reported
        // height and the drawn height disagree.
        let hugeCell = String(repeating: "x", count: 33_000)
        let table = MarkdownTableData(
            headers: ["a", "b", "c"],
            alignments: [.leading, .leading, .leading],
            rows: [["k", hugeCell, "v"]]
        )

        XCTAssertTrue(TranscriptSegmenter.tableCellNeedsCapping(hugeCell))
        XCTAssertFalse(TranscriptSegmenter.tableCellNeedsCapping("value"))
        XCTAssertLessThan(
            TranscriptSegmenter.laidOutLineCount(of: .table(table)),
            TranscriptSegmenter.maxRowLines,
            "the capped cell must be priced as drawn, not at its natural height"
        )

        let segments = TranscriptSegmenter.segments(for: [.table(table)])
        XCTAssertEqual(segments.count, 1, "a single row cannot be split")
        assertWithinBudget(segments)
    }

    func testAShortTableStaysOneBlock() {
        // The estimate must not split what it does not have to: a narrow, short
        // table is one block and one row, exactly as before.
        let table = MarkdownTableData(
            headers: ["name", "value"],
            alignments: [.leading, .leading],
            rows: (0..<10).map { ["name\($0)", "value\($0)"] }
        )
        XCTAssertEqual(TranscriptSegmenter.laidOutLineCount(of: .table(table)), 12)
        let segments = TranscriptSegmenter.segments(for: [.table(table)])
        XCTAssertEqual(segments.count, 1)
        XCTAssertEqual(segments[0].blocks.count, 1)
    }

    // MARK: - What rides a row

    func testARowWithEverythingThatRidesItStaysUnderTheCeiling() {
        // A row is not only its blocks. The reasoning accordion and the tool-call
        // card list ride the message's first row and cannot be split, so they are
        // capped; the worst case has to fit anyway, or segmentation alone would
        // still leave a row the compositor refuses to paint.
        let segment = TranscriptSegment(blocks: parseMarkdownBlocks(longParagraph(lines: 150)))
        XCTAssertLessThanOrEqual(
            TranscriptSegmenter.worstCaseRowHeight(of: segment),
            TranscriptSegmenter.compositorHeightCeiling
        )
        XCTAssertLessThan(
            TranscriptSegmenter.rowRiderHeightCeiling + TranscriptSegmenter.rowChromeHeight,
            TranscriptSegmenter.compositorHeightCeiling
        )
    }

    func testInspectionHeightRuleMatchesTheCap() {
        // Short reading stays inline; anything that would lay out past the cap
        // goes inside the scroll area. The threshold is below the cap so a region
        // is never taller than the space it is given.
        XCTAssertFalse(TranscriptSegmenter.exceedsInspectionHeight("a short thought"))
        XCTAssertTrue(TranscriptSegmenter.exceedsInspectionHeight(longParagraph(lines: 1_200)))
        XCTAssertTrue(TranscriptSegmenter.exceedsInspectionHeight(String(repeating: "x", count: 2_401)))
        XCTAssertFalse(TranscriptSegmenter.exceedsInspectionHeight(String(repeating: "x", count: 2_400)))

        // The cap has to be small enough to be a real bound on a row.
        XCTAssertLessThan(
            TranscriptSegmenter.inspectionHeightCeiling,
            TranscriptSegmenter.compositorHeightCeiling / 8
        )
    }

    func testLaidOutLineCountWrapsLongLines() {
        XCTAssertEqual(TranscriptSegmenter.laidOutLineCount(of: .paragraph(text: "short")), 1)
        XCTAssertEqual(
            TranscriptSegmenter.laidOutLineCount(of: .paragraph(text: String(repeating: "x", count: 160))),
            2
        )
        XCTAssertEqual(TranscriptSegmenter.laidOutLineCount(of: .paragraph(text: "a\nb\nc")), 3)
        // Trailing newline: an empty last line still occupies a line.
        XCTAssertEqual(TranscriptSegmenter.laidOutLineCount(of: .paragraph(text: "a\n")), 2)
    }

    // MARK: - User turns

    func testPlainTextSegmentsJoinToTheOriginal() {
        let text = longParagraph(lines: 900)
        let segments = TranscriptSegmenter.plainTextSegments(for: text)
        XCTAssertGreaterThan(segments.count, 1)

        for segment in segments {
            XCTAssertTrue(segment.blocks.isEmpty)
            XCTAssertLessThanOrEqual(
                segment.plainText?.count ?? Int.max,
                TranscriptSegmenter.maxRowCharacters
            )
        }
        XCTAssertEqual(segments.compactMap(\.plainText).joined(separator: "\n"), text)
    }

    func testShortUserTurnStaysOneVerbatimSlice() {
        let text = "# not a heading, just what the user typed"
        let segments = TranscriptSegmenter.plainTextSegments(for: text)
        XCTAssertEqual(segments.count, 1)
        // A user turn is carried as text, not as blocks: re-parsing it would
        // turn a pasted `#` line into a heading.
        XCTAssertEqual(segments[0].plainText, text)
        XCTAssertTrue(segments[0].blocks.isEmpty)
    }

    func testEmptyUserTurnYieldsOneEmptySlice() {
        let segments = TranscriptSegmenter.plainTextSegments(for: "")
        XCTAssertEqual(segments.count, 1)
        XCTAssertNil(segments[0].plainText)
    }

    // MARK: - Row identity

    func testFirstSegmentIdNamesTheFirstRow() {
        let messageId = UUID()
        let rows = TranscriptSegmenter.rows(
            for: messageId,
            kind: .message,
            segments: TranscriptSegmenter.segments(for: parseMarkdownBlocks(longParagraph(lines: 800)))
        )
        XCTAssertGreaterThan(rows.count, 1)
        // Scroll targets go through this, because a message's own id stops
        // naming a row the moment it is split.
        XCTAssertEqual(rows[0].id, TranscriptRow.firstSegmentId(for: messageId))
        XCTAssertEqual(Set(rows.map(\.id)).count, rows.count)
        XCTAssertEqual(rows.map(\.segmentIndex), Array(0..<rows.count))
        XCTAssertFalse(rows[0].isLast)
        XCTAssertTrue(rows[rows.count - 1].isLast)
    }

    // MARK: - Presentation cache

    private func makeMessage(
        _ role: MessageRole,
        _ content: String,
        thinking: String? = nil
    ) -> ChatMessage {
        ChatMessage(role: role, content: content, thinkingContent: thinking)
    }

    func testPresentationCacheReusesTheParseAndFollowsTheContent() {
        let cache = TranscriptPresentationCache()
        let message = makeMessage(.assistant, "First paragraph.\n\nSecond paragraph.")

        let first = cache.presentation(for: message, isGemmaDialect: false, isStreaming: false)
        let second = cache.presentation(for: message, isGemmaDialect: false, isStreaming: false)
        XCTAssertEqual(first.displayContent, second.displayContent)
        XCTAssertEqual(first.segments, second.segments)

        var grown = message
        grown.content += "\n\nThird paragraph."
        let updated = cache.presentation(for: grown, isGemmaDialect: false, isStreaming: false)
        XCTAssertNotEqual(updated.segments, first.segments)
        XCTAssertEqual(updated.segments.count, 1)
        XCTAssertEqual(updated.segments[0].blocks.count, 3)
    }

    func testPruneDropsEntriesFromASessionOfTheSameLength() {
        // A switch to a session with the SAME number of messages used to skip the
        // prune entirely: the `entries.count > ids.count` short-circuit compared
        // sizes, and every size matched while every id belonged to the other
        // conversation. The stale parses then stayed resident until some later
        // turn happened to change the counts.
        let cache = TranscriptPresentationCache()
        let firstSession = [
            makeMessage(.user, "one"),
            makeMessage(.assistant, "two"),
        ]
        for message in firstSession {
            _ = cache.presentation(for: message, isGemmaDialect: false, isStreaming: false)
        }
        XCTAssertEqual(cache.cachedMessageCount, 2)

        let secondSession = [
            makeMessage(.user, "three"),
            makeMessage(.assistant, "four"),
        ]
        cache.prune(keeping: Set(secondSession.map(\.id)))
        XCTAssertEqual(cache.cachedMessageCount, 0, "a same-length session still prunes")

        for message in secondSession {
            _ = cache.presentation(for: message, isGemmaDialect: false, isStreaming: false)
        }
        XCTAssertEqual(cache.cachedMessageCount, 2)
        cache.prune(keeping: Set(secondSession.map(\.id)))
        XCTAssertEqual(cache.cachedMessageCount, 2, "the live session's parses are kept")
    }

    func testUserTurnChunkingIsMemoizedRatherThanRepeated() {
        // A user turn draws its text verbatim, so there is no parse to share —
        // but the chunking is O(characters) and the row loop asks for it on
        // every body pass. A conversation with a few pasted turns was
        // re-chunking all of them per streamed chunk.
        let cache = TranscriptPresentationCache()
        let message = makeMessage(.user, longParagraph(lines: 400))

        let first = cache.plainTextSegments(for: message)
        XCTAssertGreaterThan(first.count, 1)
        XCTAssertEqual(cache.cachedPlainTextCount, 1)
        XCTAssertEqual(cache.plainTextSegments(for: message), first)
        XCTAssertEqual(
            cache.cachedMessageCount,
            0,
            "a user turn must never go through the markdown parse"
        )

        var grown = message
        grown.content += "\n" + longParagraph(lines: 10)
        XCTAssertNotEqual(cache.plainTextSegments(for: grown), first)

        cache.prune(keeping: [])
        XCTAssertEqual(cache.cachedPlainTextCount, 0)
    }

    func testPlanClassificationIsAnsweredWithoutParsing() {
        let cache = TranscriptPresentationCache()
        let plan = makeMessage(.assistant, planBody)

        XCTAssertTrue(cache.isPlanCard(for: plan))
        XCTAssertEqual(cache.cachedMessageCount, 0, "classifying a plan must not parse it")
        XCTAssertEqual(cache.cachedPlanClassificationCount, 1)
        XCTAssertTrue(cache.isPlanCard(for: plan))
        XCTAssertFalse(cache.isPlanCard(for: makeMessage(.assistant, "just prose")))
        // A user turn is never a plan card, whatever it contains.
        XCTAssertFalse(cache.isPlanCard(for: makeMessage(.user, planBody)))

        cache.prune(keeping: [])
        XCTAssertEqual(cache.cachedPlanClassificationCount, 0)
    }

    func testALongAnswerNeverRendersAsAnEmptyRow() {
        // The failure state this guards: a non-empty answer that ended up with
        // zero renderable blocks drew a blank row, which is indistinguishable
        // from the whiteout the segmentation exists to fix.
        let cache = TranscriptPresentationCache()
        let cases = [
            "\u{200B}",
            "   \n\t ",
            "\u{FFFC}\n",
            "a",
            "# Heading only",
            "<tool_call><function=shell_run><parameter=command>ls",
            "- item\n- item",
        ]
        for raw in cases {
            let presentation = cache.presentation(
                for: makeMessage(.assistant, raw),
                isGemmaDialect: false,
                isStreaming: false
            )
            XCTAssertEqual(presentation.segments.count, 1, "for \(raw.debugDescription)")
            let hasVisibleText = !presentation.displayContent
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            XCTAssertEqual(
                presentation.segments[0].isEmpty,
                !hasVisibleText,
                "a row is empty only when there is nothing to draw: \(raw.debugDescription)"
            )
        }
    }

    func testToolOnlyAnswerRendersAsOneEmptyRow() {
        let cache = TranscriptPresentationCache()
        let message = makeMessage(.assistant, "<tool_call><function=shell_run><parameter=command>ls")

        let presentation = cache.presentation(for: message, isGemmaDialect: false, isStreaming: false)
        XCTAssertEqual(presentation.displayContent, "")
        XCTAssertEqual(presentation.segments.count, 1)
        XCTAssertTrue(presentation.segments[0].blocks.isEmpty)
    }

    func testAnEmptyAnswerRendersOneEmptyRow() {
        let cache = TranscriptPresentationCache()
        let presentation = cache.presentation(
            for: makeMessage(.assistant, ""),
            isGemmaDialect: false,
            isStreaming: true
        )
        XCTAssertEqual(presentation.displayContent, "")
        XCTAssertEqual(presentation.segments.count, 1)
    }

    func testPlanMarkerClassificationRidesTheCacheRatherThanTheRowLoop() {
        // The row loop runs on every body pass, i.e. on every streamed chunk. The
        // probe reads every character of the answer (two substring scans, plus a
        // line split when they hit), so the classification is cached with the
        // parse instead. A plan card also skips the parse: it would be thrown
        // away, and a plan is the longest message in a conversation.
        let cache = TranscriptPresentationCache()
        let plan = makeMessage(.assistant, planBody)

        let first = cache.presentation(for: plan, isGemmaDialect: false, isStreaming: true)
        XCTAssertTrue(first.isPlanCard)
        XCTAssertTrue(first.segments.isEmpty, "a plan card renders its own text, not blocks")

        // A cache hit still answers the same thing, without re-scanning.
        let second = cache.presentation(for: plan, isGemmaDialect: false, isStreaming: true)
        XCTAssertTrue(second.isPlanCard)

        // An ordinary answer is not a card.
        let answer = cache.presentation(
            for: makeMessage(.assistant, "just prose"),
            isGemmaDialect: false,
            isStreaming: false
        )
        XCTAssertFalse(answer.isPlanCard)
        XCTAssertEqual(answer.segments.count, 1)

        // A middle-of-prose mention is not a plan: the fence-aware confirm still
        // decides, exactly as it did when the row loop asked.
        let mention = cache.presentation(
            for: makeMessage(.assistant, "the marker is `\(PlanMarkers.start)` in prose"),
            isGemmaDialect: false,
            isStreaming: false
        )
        XCTAssertFalse(mention.isPlanCard)

        // A user turn is never a card even when it quotes the markers.
        let user = cache.presentation(for: makeMessage(.user, planBody), isGemmaDialect: false, isStreaming: false)
        XCTAssertFalse(user.isPlanCard)
    }

    func testASplitListItemKeepsItsOwnNumber() {
        // Splitting a single item's text into pieces continues that ONE item, so
        // every piece carries the item's number rather than counting up. A list
        // of many items is many blocks and is grouped, never split, so its
        // numbering is untouched.
        let item = (0..<400).map { "line \($0)" }.joined(separator: "\n")
        let segments = TranscriptSegmenter.segments(for: [.listItem(number: 7, text: item)])
        let numbers = segments.flatMap(\.blocks).compactMap { block -> Int? in
            if case .listItem(let number, _) = block { return number }
            return nil
        }
        XCTAssertGreaterThan(numbers.count, 1)
        XCTAssertTrue(numbers.allSatisfy { $0 == 7 })
    }

    // MARK: - Block slices

    func testARowSliceBuildsIdsFromItsOwnBlocks() {
        // Each row derives its own render list from its own slice, so the ids are
        // local to the row: a slice always starts at index 0, which is what makes
        // the first block of a row render flush against the row above it.
        let blocks = Array(parseMarkdownBlocks("intro\n\n```swift\nlet x = 1\n```\n\nclosing").prefix(3))
        XCTAssertEqual(blocks.count, 3)

        let derived = MarkdownMessageView.identifiableBlocks(from: blocks, isStreaming: false)
        XCTAssertEqual(derived.map(\.block), blocks)
        XCTAssertEqual(derived.map(\.id), ["p-0", "code-1-swift", "p-2"])
        XCTAssertTrue(derived.allSatisfy { !$0.isLast })
    }

    // MARK: - Ready-plan search

    private var planBody: String {
        "\(PlanMarkers.start)\n# Plan\ndo the thing\n\(PlanMarkers.ready)"
    }

    func testReadyPlanSearchFindsTheNewestPlan() {
        let older = makeMessage(.assistant, planBody)
        let newer = makeMessage(.assistant, planBody.replacingOccurrences(of: "do the thing", with: "do it better"))
        let messages = [makeMessage(.user, "build it"), older, makeMessage(.user, "try again"), newer]

        let found = ChatDetailView.ReadyPlanMemo.search(messages: messages, dismissedPlanMessageId: nil)
        XCTAssertEqual(found?.id, newer.id)
    }

    func testReadyPlanSearchIgnoresAPlanTheConversationMovedPast() {
        // "Implement the plan." would otherwise point at a plan the conversation
        // has already moved past.
        let messages = [makeMessage(.user, "build it"), makeMessage(.assistant, planBody), makeMessage(.user, "next")]
        XCTAssertNil(ChatDetailView.ReadyPlanMemo.search(messages: messages, dismissedPlanMessageId: nil))
    }

    func testReadyPlanSearchHonoursDismissal() {
        let plan = makeMessage(.assistant, planBody)
        let messages = [makeMessage(.user, "build it"), plan]
        XCTAssertNil(
            ChatDetailView.ReadyPlanMemo.search(messages: messages, dismissedPlanMessageId: plan.id)
        )
        XCTAssertEqual(
            ChatDetailView.ReadyPlanMemo.search(messages: messages, dismissedPlanMessageId: UUID())?.id,
            plan.id
        )
    }

    func testReadyPlanSearchNeedsTheReadyMarker() {
        let unfinished = makeMessage(.assistant, "\(PlanMarkers.start)\n# Plan\nstill going")
        let messages = [makeMessage(.user, "build it"), unfinished]
        XCTAssertNil(ChatDetailView.ReadyPlanMemo.search(messages: messages, dismissedPlanMessageId: nil))
    }

    func testReadyPlanMemoFollowsNewMessagesAndDismissal() {
        let memo = ChatDetailView.ReadyPlanMemo()
        let plan = makeMessage(.assistant, planBody)
        let messages = [makeMessage(.user, "build it"), plan]
        XCTAssertEqual(memo.value(messages: messages, dismissedPlanMessageId: nil)?.id, plan.id)

        // Same messages, same answer: the memo key is unchanged.
        XCTAssertEqual(memo.value(messages: messages, dismissedPlanMessageId: nil)?.id, plan.id)

        // A follow-up invalidates it, exactly as the uncached search would.
        var withFollowUp = messages
        withFollowUp.append(makeMessage(.user, "implement the plan"))
        XCTAssertNil(memo.value(messages: withFollowUp, dismissedPlanMessageId: nil))
    }

    // MARK: - Diagnostics

    func testANewTurnReArmsTheOneShotReport() {
        // The report keys on the message count, which repeats across conversations
        // of one length. Without clearing the key when a turn begins, a later turn
        // that lands on the same one reports nothing at all, so the instrumentation
        // goes quiet in exactly the case it exists to describe.
        let diagnostics = TranscriptDiagnostics()
        let messages = [makeMessage(.user, "hi")]

        diagnostics.reportIfNeeded(turnKey: "abc-2", messages: messages, segmentsByMessage: [:])
        XCTAssertEqual(diagnostics.lastReportedTurnKey, "abc-2")

        diagnostics.beginTurn()
        XCTAssertNil(diagnostics.lastReportedTurnKey)

        diagnostics.reportIfNeeded(turnKey: "abc-2", messages: messages, segmentsByMessage: [:])
        XCTAssertEqual(diagnostics.lastReportedTurnKey, "abc-2")
    }

    func testANonFiniteHeightCannotCrashTheReport() {
        // `Int(CGFloat)` traps on `.infinity`, and an unlaid-out scroll view
        // publishes exactly that: the instrumentation for an intermittent
        // rendering bug killed the process when geometry handed it one.
        let diagnostics = TranscriptDiagnostics()
        diagnostics.record(rowId: "row#0/1", label: "message#0/1", height: .infinity)
        diagnostics.record(contentHeight: .infinity, viewportHeight: .nan)

        diagnostics.reportIfNeeded(
            turnKey: "abc-2",
            messages: [makeMessage(.user, "hi")],
            segmentsByMessage: [:]
        )
        XCTAssertEqual(diagnostics.lastReportedTurnKey, "abc-2")
        // An unbounded row is still a row worth warning about.
        XCTAssertEqual(diagnostics.warnedTurnKey, "abc-2")
    }

    func testTheRowWarningReArmsForARestoredConversation() {
        // `beginTurn()` only runs on the generating transition, so a conversation
        // restored from disk reports without one. A bare warn-once flag meant the
        // FIRST long conversation warned and every one opened afterwards logged
        // silently — which is the reproduction this instrumentation exists for.
        let diagnostics = TranscriptDiagnostics()
        let tall = TranscriptSegmenter.rowHeightWarning + 100
        diagnostics.record(rowId: "row#0/1", label: "message#0/1", height: tall)

        diagnostics.reportIfNeeded(turnKey: "abc-restored", messages: [], segmentsByMessage: [:])
        XCTAssertEqual(diagnostics.warnedTurnKey, "abc-restored")

        diagnostics.record(rowId: "row2#0/1", label: "message#0/1", height: tall)
        diagnostics.reportIfNeeded(turnKey: "def-restored", messages: [], segmentsByMessage: [:])
        XCTAssertEqual(
            diagnostics.warnedTurnKey,
            "def-restored",
            "a second restored conversation warns too"
        )
    }

    // MARK: - Pin / follow policy

    private func snapshot(
        offset: CGFloat,
        contentHeight: CGFloat,
        containerHeight: CGFloat = 600
    ) -> TranscriptFollowPolicy.Snapshot {
        TranscriptFollowPolicy.Snapshot(
            offset: offset,
            contentHeight: contentHeight,
            containerHeight: containerHeight
        )
    }

    private func decide(
        _ old: TranscriptFollowPolicy.Snapshot,
        _ new: TranscriptFollowPolicy.Snapshot,
        isPinned: Bool = true,
        isReaderScrolling: Bool = false,
        isFollowInFlight: Bool = false
    ) -> TranscriptFollowPolicy.Outcome {
        TranscriptFollowPolicy.decide(
            from: old,
            to: new,
            isPinned: isPinned,
            isReaderScrolling: isReaderScrolling,
            isFollowInFlight: isFollowInFlight
        )
    }

    func testContentGrowthUnderAPinnedReaderIsFollowed() {
        // A tool card landing after a capped reasoning trace: the transcript is
        // a card taller, no text streamed, and the pin has to reach it.
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 1_400, contentHeight: 2_400)

        let outcome = decide(old, new)
        XCTAssertEqual(outcome.follow, true)
        XCTAssertNil(outcome.pin, "growth never moves the pin")
    }

    func testGrowthBelowOneFollowStepIsNotFollowed() {
        // The follow is throttled by height steps so a streamed chunk does not
        // buy a layout pass each; the character bucket drives that case.
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 1_400, contentHeight: 2_020)

        XCTAssertEqual(decide(old, new).follow, false)
    }

    func testGrowthDoesNotFollowAReaderWhoLeftTheTail() {
        // The reader scrolled up: growth is theirs to catch up on, not the
        // view's to yank them out of.
        let old = snapshot(offset: 400, contentHeight: 2_000)
        let new = snapshot(offset: 400, contentHeight: 2_400)

        let outcome = decide(old, new, isPinned: false)
        XCTAssertEqual(outcome.follow, false)
        XCTAssertNil(outcome.pin)
    }

    func testOnlyAReaderDrivenScrollPhaseCanUnpin() {
        // The pin rule is only ever as good as the phase it is handed: if a
        // scroll the reader performed reports a phase this denies, the reader
        // cannot leave the tail mid-turn at all. Momentum counts as the reader
        // (a wheel or trackpad flick after the finger left), while a
        // programmatic follow reports `.animating` and must never unpin.
        XCTAssertTrue(TranscriptFollowPolicy.isReaderDriven(.tracking))
        XCTAssertTrue(TranscriptFollowPolicy.isReaderDriven(.interacting))
        XCTAssertTrue(TranscriptFollowPolicy.isReaderDriven(.decelerating))
        XCTAssertFalse(TranscriptFollowPolicy.isReaderDriven(.animating))
        XCTAssertFalse(TranscriptFollowPolicy.isReaderDriven(.idle))
    }

    func testAQuietOffsetShiftDoesNotUnpin() {
        // The defect: a lazy stack re-estimating an unrealized row, or a follow
        // landing a point short, slides the offset up with nobody scrolling.
        // Reading that as a drag dropped the pin for the rest of the turn.
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 1_180, contentHeight: 2_000)

        let outcome = decide(old, new)
        XCTAssertNil(outcome.pin)
        XCTAssertEqual(outcome.follow, false)
    }

    func testContentShrinkWhilePinnedKeepsThePin() {
        // A shrink clamps the offset to the new maximum: the scroll view moved
        // the tail into view, it did not leave it.
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 1_100, contentHeight: 1_700)

        let outcome = decide(old, new)
        XCTAssertNil(outcome.pin, "the view is still at the tail")
        XCTAssertEqual(outcome.follow, false)
    }

    func testOnlyTheReaderUnpins() {
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 1_100, contentHeight: 2_000)

        let outcome = decide(old, new, isReaderScrolling: true)
        XCTAssertEqual(outcome.pin, false)
        XCTAssertEqual(outcome.follow, false)
    }

    func testTheReaderReturningToTheTailReattachesThePin() {
        let old = snapshot(offset: 1_100, contentHeight: 2_000)
        let new = snapshot(offset: 1_395, contentHeight: 2_000)

        let outcome = decide(old, new, isPinned: false, isReaderScrolling: true)
        XCTAssertEqual(outcome.pin, true)
    }

    func testAFollowInFlightIsNeverReadAsADrag() {
        let old = snapshot(offset: 1_400, contentHeight: 2_000)
        let new = snapshot(offset: 900, contentHeight: 2_000)

        let outcome = decide(old, new, isReaderScrolling: true, isFollowInFlight: true)
        XCTAssertNil(outcome.pin)
        XCTAssertEqual(outcome.follow, false)
    }

    func testAReattachedReaderAlsoFollowsGrowthInTheSameUpdate() {
        // The pin and the follow are decided together: answering "near the
        // bottom, pin again" without following the growth that arrives in the
        // same update would leave the view pinned and a card short.
        let old = snapshot(offset: 1_000, contentHeight: 2_300)
        let new = snapshot(offset: 1_760, contentHeight: 2_360)

        let outcome = decide(old, new, isPinned: false, isReaderScrolling: true)
        XCTAssertEqual(outcome.pin, true)
        XCTAssertEqual(outcome.follow, true)
    }
}
