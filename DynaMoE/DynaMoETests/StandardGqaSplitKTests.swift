//
//  StandardGqaSplitKTests.swift
//  DynaMoETests
//
//  The rule that decides when a standard-GQA decode step takes the split-K
//  (flash-decoding) kernels, and the chunk geometry it hands them.
//
//  This is pure arithmetic, and it needs to be tested without a checkpoint:
//  the rule engages ABOVE 128 stored KV rows, and every prompt-bound case in
//  this suite tops out around 36, so nothing that actually runs a model ever
//  reaches it. A wrong chunk count here is a memory question rather than a
//  numeric one (`attnPartialM`/`L`/`Acc` are sized `numQHeads * 256`), which is
//  the kind of defect no "did it answer" test can see.
//

import XCTest
@testable import DynaMoE

final class StandardGqaSplitKTests: XCTestCase {

    /// The streaming KV kernels carry a delta prefix as `0x80000000 | prefix`,
    /// meaning "the walk is `prefix` rows plus this one". The split-K kernels
    /// count rows, so the marker has to be decoded rather than compared.
    func testTheDeltaMarkerIsDecodedIntoAStoredRowCount() {
        XCTAssertEqual(StandardGqaSplitK.curLen(0), 1, "an empty cache still stores this token")
        XCTAssertEqual(StandardGqaSplitK.curLen(1), 1)
        XCTAssertEqual(StandardGqaSplitK.curLen(4_000), 4_000)
        XCTAssertEqual(StandardGqaSplitK.curLen(0x80000000), 1, "a zero-row prefix plus this one")
        XCTAssertEqual(StandardGqaSplitK.curLen(0x80000000 | 4_000), 4_001)
        // The marker bit is the TOP bit, so the largest representable prefix must
        // not be read as a plain length.
        XCTAssertEqual(StandardGqaSplitK.curLen(0x80000000 | 0x7FFF_FFFF), 0x8000_0000)
    }

    /// Below and AT the threshold the plain kernel keeps one dispatch; past it the
    /// walk is split. The boundary is written out because "engages at 128" and
    /// "engages past 128" are one character apart and only one is what ships.
    func testTheSplitEngagesOnlyPastTheMinimumContext() {
        let threshold = StandardGqaSplitK.minimumContext
        XCTAssertFalse(StandardGqaSplitK.curLen(threshold) > threshold, "at the threshold the plain kernel runs")
        XCTAssertFalse(StandardGqaSplitK.curLen(threshold - 1) > threshold)
        XCTAssertTrue(StandardGqaSplitK.curLen(threshold + 1) > threshold, "one row past it the split runs")
    }

    /// A chunk count of one is the plain kernel's shape and must never be what a
    /// split dispatch asks for, and the count must stay inside what the partial
    /// buffers were allocated for.
    func testTheChunkCountStaysInsideTheAllocatedPartials() {
        for len in [UInt32(1), 64, 129, 4_000, 16_384, 16_385, 100_000, 0x7FFF_FFFF] {
            let chunks = StandardGqaSplitK.chunkCount(len)
            XCTAssertGreaterThanOrEqual(chunks, 1)
            XCTAssertLessThanOrEqual(
                chunks,
                StandardGqaSplitK.maximumChunks,
                "len \(len): \(chunks) chunks exceeds the attnPartial capacity"
            )
        }
        XCTAssertEqual(StandardGqaSplitK.chunkCount(1), 1)
        XCTAssertEqual(StandardGqaSplitK.chunkCount(StandardGqaSplitK.chunkRows), 1)
        XCTAssertEqual(StandardGqaSplitK.chunkCount(StandardGqaSplitK.chunkRows + 1), 2)
        XCTAssertEqual(
            StandardGqaSplitK.chunkCount(StandardGqaSplitK.chunkRows * StandardGqaSplitK.maximumChunks),
            StandardGqaSplitK.maximumChunks
        )
    }

    /// The split is only correct if the chunks COVER the walk: the dispatch hands
    /// the kernels `chunkCount` and a chunk size computed as
    /// `ceil(len / chunkCount)`. A short final chunk is fine (the kernels bound
    /// each row by the walk length), but an under-covered walk would leave rows
    /// unvisited while the combine still normalises over their partials.
    func testTheChunksCoverTheWholeWalk() {
        for len in [UInt32(129), 200, 4_000, 16_000, 16_384, 16_385, 40_000, 100_000, 262_144] {
            let chunks = StandardGqaSplitK.chunkCount(len)
            let chunkSize = (len + chunks - 1) / chunks
            XCTAssertGreaterThanOrEqual(
                chunks * chunkSize,
                len,
                "len \(len): \(chunks) x \(chunkSize) does not cover the walk"
            )
            if len > StandardGqaSplitK.chunkRows * StandardGqaSplitK.maximumChunks {
                XCTAssertEqual(chunks, StandardGqaSplitK.maximumChunks, "past the cap the count saturates")
                XCTAssertGreaterThan(chunkSize, StandardGqaSplitK.chunkRows, "a saturated count widens each chunk")
            }
        }
    }
}
