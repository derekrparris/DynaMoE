//
//  ComposerLayoutLogicTests.swift
//  DynaMoE
//
//  Pure-logic coverage for the composer editor's height policy: the visible
//  editor must never collapse below one line, never grow past the cap it
//  scrolls within, and must not force TextKit to lay out a paste that could not
//  fit any cap anyway.
//

import XCTest
@testable import DynaMoE

final class ComposerLayoutLogicTests: XCTestCase {

    func testOneLineContentClampsToTheFloor() {
        // A single line's used rect is shorter than the font's line box plus the
        // text view's insets, which is exactly why the floor exists: a composer
        // collapsed to the text height would clip the first line's descenders.
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 12,
                minHeight: 21,
                maxHeight: 200
            ),
            21
        )
    }

    func testContentBetweenFloorAndCapIsExactAndRoundedUp() {
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 63,
                minHeight: 21,
                maxHeight: 200
            ),
            63,
            "content that already fits is used as-is"
        )
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 63.2,
                minHeight: 21,
                maxHeight: 200
            ),
            64,
            "a partial point would leave the last line's descent clipped"
        )
    }

    func testContentAboveTheCapClampsToTheCap() {
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 4_000,
                minHeight: 21,
                maxHeight: 200
            ),
            200,
            "past the cap the editor keeps its height and scrolls internally"
        )
    }

    func testCapAtOrBelowTheFloorYieldsTheFloor() {
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 500,
                minHeight: 21,
                maxHeight: 21
            ),
            21
        )
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 500,
                minHeight: 21,
                maxHeight: 10
            ),
            21,
            "a cap smaller than the floor is impossible to satisfy; the floor wins"
        )
    }

    func testPastesThatStayMeasurableAndPastesThatCannotFit() {
        XCTAssertFalse(ComposerLayoutPolicy.isCertainlyOversized(""))
        XCTAssertFalse(
            ComposerLayoutPolicy.isCertainlyOversized(
                String(repeating: "x", count: ComposerLayoutPolicy.measurementByteLimit)
            )
        )
        XCTAssertTrue(
            ComposerLayoutPolicy.isCertainlyOversized(
                String(repeating: "x", count: ComposerLayoutPolicy.measurementByteLimit + 1)
            )
        )

        let paste = (0..<500).map { "line \($0) of a pasted prompt" }.joined(separator: "\n")
        XCTAssertTrue(ComposerLayoutPolicy.isCertainlyOversized(paste))

        // The limit is counted in UTF-8 bytes, so four-byte scalars reach it
        // sooner: 2,001 of them is 8,004 bytes and cannot fit any composer cap.
        XCTAssertTrue(
            ComposerLayoutPolicy.isCertainlyOversized(String(repeating: "😀", count: 2_001))
        )
    }

    func testOversizedTextResolvesToTheCapWithoutMeasurement() {
        // What the coordinator does for a paste past the measurement limit: the
        // policy's own clamp applied to the cap is the cap.
        XCTAssertEqual(
            ComposerLayoutPolicy.editorHeight(
                contentHeight: 200,
                minHeight: 21,
                maxHeight: 200
            ),
            200
        )
    }
}
