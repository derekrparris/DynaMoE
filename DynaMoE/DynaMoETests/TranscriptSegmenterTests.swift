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

    func testNestedScrollViewCountCountsCodeBlocksAndTables() {
        let table = MarkdownTableData(headers: ["h"], alignments: [.leading], rows: [["c"]])
        let segment = TranscriptSegment(blocks: [
            .paragraph(text: "intro"),
            .codeBlock(language: "swift", code: "let x = 1"),
            .codeBlock(language: "rust", code: "let x = 1;"),
            .table(table),
            .heading(level: 2, text: "done"),
        ])
        XCTAssertEqual(TranscriptSegmenter.nestedScrollViewCount(of: segment), 3)
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
}
