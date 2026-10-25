//
//  LongTextChunkerTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the bounded-height text splitter the transcript uses
//  to keep any single Text view small enough to paint: chunks must bound both
//  line count and character count while reproducing the input exactly.
//

import XCTest
@testable import DynaMoE

final class LongTextChunkerTests: XCTestCase {

    func testEmptyTextProducesNoChunks() {
        XCTAssertEqual(LongTextChunker.chunks(of: ""), [])
    }

    func testShortTextStaysOneChunk() {
        let text = "short\nreasoning block"
        XCTAssertEqual(LongTextChunker.chunks(of: text), [text])
    }

    func testLongTextIsSplitIntoBoundedLineChunks() {
        let lines = (0..<300).map { "line \($0) of the reasoning block" }
        let text = lines.joined(separator: "\n")

        let chunks = LongTextChunker.chunks(of: text)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.split(separator: "\n", omittingEmptySubsequences: false).count, 120)
        }
        XCTAssertEqual(chunks.joined(separator: "\n"), text)
    }

    func testCharacterBudgetSplitsFewButHugeLines() {
        let line = String(repeating: "x", count: 25_000)
        let chunks = LongTextChunker.chunks(of: line)

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, 8000)
        }
        XCTAssertEqual(chunks.joined(), line)
    }

    func testTrailingNewlineSurvivesChunking() {
        let lines = (0..<200).map { "line \($0)" }
        let text = lines.joined(separator: "\n") + "\n"

        let chunks = LongTextChunker.chunks(of: text)
        XCTAssertEqual(chunks.joined(separator: "\n"), text)
    }

    func testChunksRespectDefaultCharacterBudget() {
        let lines = (0..<400).map { "line \($0): " + String(repeating: "y", count: 200) }
        let text = lines.joined(separator: "\n")

        let chunks = LongTextChunker.chunks(of: text)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.count, 8000)
        }
        XCTAssertEqual(chunks.joined(separator: "\n"), text)
    }

    // MARK: - Parser integration

    func testLongParagraphStaysOneBlockAndIsChunkedAtRenderTime() {
        let text = (0..<300).map { "line \($0)" }.joined(separator: "\n")

        // The renderer chunks the text inside one block; the parser must not
        // split it into sibling blocks, which would add an inter-block gap at
        // every boundary and read as a new paragraph.
        let blocks = parseMarkdownBlocks(text)
        XCTAssertEqual(blocks.count, 1)
        guard case .paragraph(let value) = blocks.first else {
            return XCTFail("expected a paragraph block")
        }
        XCTAssertEqual(value, text)
        XCTAssertTrue(LongTextChunker.isOversized(value))
        XCTAssertGreaterThan(LongTextChunker.chunks(of: value).count, 1)
        XCTAssertEqual(LongTextChunker.chunks(of: value).joined(separator: "\n"), text)
    }

    func testLongBlockquoteStaysOneBlock() {
        let text = (0..<200).map { "> line \($0)" }.joined(separator: "\n")

        let blocks = parseMarkdownBlocks(text)
        XCTAssertEqual(blocks.count, 1)
        guard case .blockquote(let value) = blocks.first else {
            return XCTFail("expected a blockquote block")
        }
        XCTAssertTrue(LongTextChunker.isOversized(value))
        XCTAssertEqual(LongTextChunker.chunks(of: value).joined(separator: "\n"), value)
    }

    func testExactlyAtTheLimitsStaysOneChunk() {
        let lines = (0..<120).map { "line \($0)" }.joined(separator: "\n")
        XCTAssertEqual(LongTextChunker.chunks(of: lines), [lines])
        XCTAssertFalse(LongTextChunker.isOversized(lines))

        let characters = String(repeating: "x", count: 8000)
        XCTAssertEqual(LongTextChunker.chunks(of: characters), [characters])
        XCTAssertFalse(LongTextChunker.isOversized(characters))
    }

    func testOversizedAgreesWithChunkCount() {
        let cases = [
            "",
            "short",
            (0..<120).map { "line \($0)" }.joined(separator: "\n"),
            (0..<121).map { "line \($0)" }.joined(separator: "\n"),
            String(repeating: "x", count: 8000),
            String(repeating: "x", count: 8001),
            String(repeating: "x", count: 40_001),
        ]
        for text in cases {
            XCTAssertEqual(
                LongTextChunker.chunks(of: text).count > 1,
                LongTextChunker.isOversized(text),
                "mismatch for \(text.prefix(20))"
            )
        }
    }

    func testCodeBlockChunkingCoversTheHighlightLimitBoundary() {
        let under = String(repeating: "a", count: 40_000)
        let over = String(repeating: "a", count: 40_001)

        // The highlight limit only decides whether to highlight; the chunk
        // split is already bounded far below it, on both sides of the boundary.
        for text in [under, over] {
            let chunks = LongTextChunker.chunks(of: text)
            XCTAssertGreaterThan(chunks.count, 1)
            XCTAssertTrue(chunks.allSatisfy { $0.count <= 8000 })
            XCTAssertEqual(chunks.joined(), text)
        }
    }

    func testShortParagraphStaysOneBlock() {
        let text = "a single\nshort paragraph"

        let blocks = parseMarkdownBlocks(text)
        XCTAssertEqual(blocks.count, 1)
        guard case .paragraph(let value) = blocks.first else {
            return XCTFail("expected a paragraph block")
        }
        XCTAssertEqual(value, text)
    }

    func testRepairPassFixesGluedHeadingWhileStreamingSkipsIt() {
        let raw = "intro text. #### 1. Heading\nbody"

        let settled = parseMarkdownBlocks(raw)
        XCTAssertTrue(settled.contains {
            if case .heading(let level, let text) = $0 { return level == 4 && text.contains("Heading") }
            return false
        })

        let streaming = parseMarkdownBlocks(raw, repairs: false)
        XCTAssertFalse(streaming.contains {
            if case .heading = $0 { return true }
            return false
        })
    }

    func testNonASCIIDigitDoesNotStartANumberedList() {
        let blocks = parseMarkdownBlocks("². superscript item")
        XCTAssertFalse(blocks.contains {
            if case .listItem = $0 { return true }
            return false
        })
    }
}

// MARK: - Normalizer trigger-guard equivalence

/// The normalizer skips a regex pass when its trigger character is absent.
/// That is only safe when the trigger is a necessary condition for a match, so
/// every sample is normalized both with the guards and with every pass forced
/// to run; the two results must be identical.
final class MarkdownNormalizerGuardTests: XCTestCase {

    private let corpus: [String] = [
        "",
        "plain text with no markdown at all",
        "a sentence. And another one.",
        "a < b and c > d",
        "#not a heading",
        "1.2.3 version numbers",
        "just - a dash",
        "intro text. #### 1. Heading\nbody",
        "## UsingBF16 kernels",
        "formats: - Total 16 bits",
        "computing. 1. Technical details",
        "quantization.This",
        "Title | a | b\n| --- | --- |",
        "| a | b |\n| --- | ---",
        "| a | b |\n|---|---|\n|---|---|",
        "-- Section Name -- |",
        "---## Heading",
        "text• Bullet",
        "a\n\n\n\nb",
        "a || b",
        "<br>line",
        "<p>a</p><div>b</div>",
        "<b>bold</b> and <i>italic</i> and <code>code</code>",
        "<ul><li>one</li><li>two</li></ul>",
        "<span>span</span><sup>2</sup><sub>2</sub>",
        "<script>alert(1)</script>",
    ]

    func testTriggerGuardsDoNotChangeNormalization() {
        for sample in corpus {
            for repairs in [true, false] {
                XCTAssertEqual(
                    normalizeMarkdownText(sample, repairs: repairs),
                    normalizeMarkdownText(sample, repairs: repairs, triggerGuards: false),
                    "guard short-circuit changed output for \(sample.debugDescription) (repairs: \(repairs))"
                )
            }
        }
    }
}
