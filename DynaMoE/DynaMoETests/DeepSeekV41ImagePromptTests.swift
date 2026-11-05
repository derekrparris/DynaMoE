//
//  DeepSeekV41ImagePromptTests.swift
//  DynaMoETests
//
//  Where an image lands in a token stream, and which row each of its positions
//  takes. Pure logic: no tower, no GPU, no checkpoint.
//

import XCTest
@testable import DynaMoE

final class DeepSeekV41ImagePromptTests: XCTestCase {

    /// The checkpoint's own id. Any value works for this test, but using the real one
    /// means a fixture cannot pass while the prompt code has the wrong token.
    private let imageTokenID = 129_264

    /// A 1x1 aligner grid: `start`, one image row, one newline, `end`.
    private func singleCell(width: Int = 4) -> DeepSeekV41ImagePrompt.ImageSpan {
        DeepSeekV41ImagePrompt.ImageSpan(
            rows: (0..<4).map { index in
                [Float](repeating: Float(index) + 0.5, count: width)
            },
            columns: 1,
            gridRows: 1
        )
    }

    func testTheSpanLengthIsTheGridPlusItsDelimiters() {
        let span = singleCell()
        XCTAssertEqual(span.tokenCount, 4)
        XCTAssertEqual(
            span.tokenCount,
            DeepSeekV41VisionLayout.imageTokenCount(columns: 1, rows: 1)
        )
        XCTAssertEqual(
            DeepSeekV41VisionLayout.imageTokenCount(columns: 3, rows: 2),
            2 * (3 + 1) + 2
        )
    }

    /// One placeholder becomes one token per span position, and every one of those
    /// positions is bound to the row the tower produced for it.
    func testAPlaceholderExpandsToItsSpanAndBindsEveryRow() throws {
        let marker = UInt32(imageTokenID)
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [7, marker, 8],
            spans: [singleCell()],
            imageTokenID: imageTokenID,
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [7, marker, marker, marker, marker, 8])
        XCTAssertEqual(plan.spans, [1..<5])
        XCTAssertEqual(plan.rows.keys.sorted(), [1, 2, 3, 4])
        for (offset, position) in [1, 2, 3, 4].enumerated() {
            XCTAssertEqual(plan.rows[position]?.first, Float(offset) + 0.5)
        }
        XCTAssertEqual(plan.report, DeepSeekV41VisionLayout.ExpansionReport(
            unusedImageCount: 0, unfilledPlaceholderCount: 0
        ))
        XCTAssertEqual(plan.imagePositionCount, 4)
    }

    /// Two pictures in one prompt expand in reading order and do not borrow each
    /// other's rows.
    func testTwoImagesTakeTheirPositionsInOrder() throws {
        let marker = UInt32(imageTokenID)
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [marker, 5, marker],
            spans: [
                singleCell(width: 2),
                DeepSeekV41ImagePrompt.ImageSpan(
                    rows: (0..<5).map { _ in [9, 9] },
                    columns: 2,
                    gridRows: 1
                )
            ],
            imageTokenID: imageTokenID,
            rowWidth: 2
        )
        // A 1x1 grid is 4 tokens; a 2x1 grid is 1 * (2 + 1) + 2 = 5: start, two
        // image positions, one newline, end.
        XCTAssertEqual(plan.tokenIds.count, 4 + 1 + 5)
        XCTAssertEqual(plan.spans, [0..<4, 5..<10])
        XCTAssertEqual(plan.rows[0], [0.5, 0.5])
        XCTAssertEqual(plan.rows[3], [3.5, 3.5])
        XCTAssertEqual(plan.rows[5], [9, 9], "the second image starts after the text token")
        XCTAssertEqual(plan.rows[9], [9, 9])
        XCTAssertNil(plan.rows[4], "the text position between the images has no row")
    }

    /// A placeholder with no image is passed through rather than deleted, because
    /// deleting it would move every position after it. The mismatch is reported.
    func testAPlaceholderWithNoImageSurvivesAndIsReported() throws {
        let marker = UInt32(imageTokenID)
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [1, marker, 2],
            spans: [],
            imageTokenID: imageTokenID,
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [1, marker, 2])
        XCTAssertEqual(plan.rows, [:])
        XCTAssertEqual(plan.spans, [])
        XCTAssertEqual(plan.report.unfilledPlaceholderCount, 1)
        XCTAssertEqual(plan.report.unusedImageCount, 0)
    }

    /// An image with no placeholder has nowhere to go. Appending its rows at the end
    /// would put a picture where the caller never asked for one.
    func testAnImageWithNoPlaceholderIsReportedRatherThanAppended() throws {
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [1, 2, 3],
            spans: [singleCell()],
            imageTokenID: imageTokenID,
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [1, 2, 3])
        XCTAssertEqual(plan.rows, [:])
        XCTAssertEqual(plan.report.unusedImageCount, 1)
        XCTAssertEqual(plan.report.unfilledPlaceholderCount, 0)
    }

    /// A span whose row count disagrees with its grid would insert a picture one
    /// position short, which misaligns everything after it.
    func testASpanThatDoesNotMatchItsGridIsRefused() {
        let short = DeepSeekV41ImagePrompt.ImageSpan(
            rows: [[0, 0], [1, 1], [2, 2]],
            columns: 1,
            gridRows: 1
        )
        XCTAssertThrowsError(
            try DeepSeekV41ImagePrompt.expand(
                tokenIds: [UInt32(imageTokenID)],
                spans: [short],
                imageTokenID: imageTokenID,
                rowWidth: 2
            )
        ) { error in
            guard case DeepSeekV41ImagePromptError.spanLengthMismatch(
                let image, let expected, let actual
            ) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(image, 0)
            XCTAssertEqual(expected, 4)
            XCTAssertEqual(actual, 3)
        }
    }

    /// A row narrower than the model is a tower bug, not a prompt with fewer
    /// dimensions: the residual width is fixed by the backbone.
    func testARowThatIsNotTheModelWidthIsRefused() {
        let narrow = DeepSeekV41ImagePrompt.ImageSpan(
            rows: (0..<4).map { _ in [0, 0] },
            columns: 1,
            gridRows: 1
        )
        XCTAssertThrowsError(
            try DeepSeekV41ImagePrompt.expand(
                tokenIds: [UInt32(imageTokenID)],
                spans: [narrow],
                imageTokenID: imageTokenID,
                rowWidth: 4
            )
        ) { error in
            guard case DeepSeekV41ImagePromptError.rowWidthMismatch(
                let image, let position, let expected, let actual
            ) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(image, 0)
            XCTAssertEqual(position, 0)
            XCTAssertEqual(expected, 4)
            XCTAssertEqual(actual, 2)
        }
    }

    /// A text-only prompt is returned unchanged, with no rows: the expansion must be
    /// free for the turns that have no image at all.
    func testATextOnlyPromptIsUnchanged() throws {
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [1, 2, 3, 4],
            spans: [],
            imageTokenID: imageTokenID,
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [1, 2, 3, 4])
        XCTAssertEqual(plan.rows, [:])
        XCTAssertTrue(plan.report.isBalanced)
    }

    /// The published `vision_config`, as the checkpoint ships it, and what it makes
    /// the SMALLEST admissible image cost.
    ///
    /// `Benchmarks/PERF_FINDINGS.md` "DSV4.1 #35" quotes these numbers for why the
    /// host tower cannot serve: 544x544 is the floor `min_pixels` imposes, 1,521 is
    /// the patch count a 14-pixel patch grid reads at that size, and 184 is the span
    /// the prompt and the KV pricing both count. They are pinned here so a config
    /// change cannot quietly falsify the journal.
    func testThePublishedSpecsSmallestImageIsAPricingFact() {
        let spec = DeepSeekV41VisionSpec(
            patchSize: 14,
            downsampleRatio: 3,
            minPixels: 295_936,
            maxImageTokens: 1_024,
            numHiddenLayers: 32,
            hiddenSize: 1_024,
            numAttentionHeads: 16,
            intermediateSize: 2_816,
            imageTokenID: imageTokenID
        )
        let fitted = DeepSeekV41VisionLayout.fit(width: 1, height: 1, spec: spec)
        XCTAssertGreaterThanOrEqual(fitted.width * fitted.height, spec.minPixels)
        XCTAssertEqual(fitted.width % spec.patchSize, 0)
        XCTAssertEqual(fitted.height % spec.patchSize, 0)
        XCTAssertEqual(fitted.width, 546, "1x1 is scaled up to the pixel floor, then to patches")
        XCTAssertEqual(fitted.height, 546)
        XCTAssertEqual(fitted.columns, 13, "546 / (14 * 3) = 13 aligner cells per side")
        XCTAssertEqual(fitted.rows, 13)
        XCTAssertEqual(fitted.tokens, 184, "13 * (13 + 1) + 2")

        // The patch count the tower reads at that size, which is what its cost scales
        // with: 39 x 39 patches, not the 13 x 13 aligner grid.
        let placement = DeepSeekV41VisionPreprocess.placement(
            width: fitted.width, height: fitted.height, spec: spec
        )
        let patches = placement.patchGrid(patchSize: spec.patchSize)
        XCTAssertEqual(patches.columns, 39)
        XCTAssertEqual(patches.rows, 39)
        XCTAssertEqual(patches.columns * patches.rows, 1_521)
    }
}
