//
//  Inkling2VisionTests.swift
//  DynaMoETests
//
//  The still-image tower against the released implementation.
//
//  The goldens come from `Tools/inkling2_vision_goldens.py`, which runs the
//  checkpoint's OWN `Inkling2VisionModel` in FP32 over a 3x2 cell picture. The
//  dump is keyed and ordered exactly as that tower captures it, so this file
//  compares activations rather than a summary: a wrong patch order, a bicubic
//  resample where the reference does bilinear, a rotate-half RoPE where it does
//  interleaved pairs, or a class token that attends when it should not all
//  survive a shape check and none of them survive this.
//

import Metal
import XCTest
import AppKit
@testable import DynaMoE

final class Inkling2VisionTests: XCTestCase {

    static let dumpPath = "/tmp/inkling2_vision_goldens.json"

    /// The engine owns the shard mappings and the zero-copy buffers point into
    /// them, so letting it go out of scope unmaps the weights under the source.
    /// Internal so an extension in another file can hold it across a turn.
    var engine: DynaMoeEngine?

    // MARK: Goldens

    private func dump() throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: Self.dumpPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: Self.dumpPath)),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("no vision dump at \(Self.dumpPath): run Tools/inkling2_vision_goldens.py")
        }
        return object
    }

    private func snapshot() throws -> URL {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(path.path)")
        }
        return path
    }

    private func makeForward() throws -> Inkling2VisionForward {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            guard shard.length <= UInt64(device.maxBufferLength) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
            ) {
                buffers[shard.index] = buffer
            }
        }
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: buffers)
        return try Inkling2VisionForward(config: config, weights: source)
    }

    /// The picture both sides build, so the input cannot drift unnoticed.
    static func pixel(cell: Int, row: Int, column: Int, channel: Int) -> Float {
        let seed = ((cell * 131 + row) * 137 + column * 149 + channel * 151) % 251
        return Float(Double(seed) / 125.5 - 1.0)
    }

    static func pixels(layout: Inkling2VisionLayout) -> [Float] {
        var values = [Float]()
        values.reserveCapacity(layout.pixelCount)
        for cell in 0..<layout.cellCount {
            for row in 0..<layout.cellSize {
                for column in 0..<layout.cellSize {
                    for channel in 0..<layout.channels {
                        values.append(pixel(cell: cell, row: row, column: column, channel: channel))
                    }
                }
            }
        }
        return values
    }

    private func floats(_ value: Any?) -> [Float] {
        (value as? [Any])?.compactMap { ($0 as? NSNumber)?.floatValue } ?? []
    }

    private func rows(_ value: Any?) -> [[Float]] {
        (value as? [Any])?.map { floats($0) } ?? []
    }

    private func relativeError(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !expected.isEmpty else { return .infinity }
        var numerator: Float = 0
        var denominator: Float = 0
        for index in expected.indices {
            numerator = max(numerator, abs(actual[index] - expected[index]))
            denominator = max(denominator, abs(expected[index]))
        }
        return numerator / max(denominator, 1e-6)
    }

    // MARK: Pure logic

    func testTheCellGridArithmetic() throws {
        let config = try Inkling2Config.decode(
            contentsOf: try snapshot().appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let layout = Inkling2VisionLayout(cellRows: 3, cellColumns: 2, spec: spec)
        XCTAssertEqual(layout.cellSize, spec.adapterMergeSize * spec.nativePatchSize)
        XCTAssertEqual(layout.patchRows, 6, "3 cells tall, 2 native patches each")
        XCTAssertEqual(layout.patchColumns, 4, "2 cells wide")
        XCTAssertEqual(layout.patchCount, 24)
        XCTAssertEqual(layout.sequenceLength, 25, "the class token is one more row")
        XCTAssertEqual(layout.pixelCount, 6 * 40 * 40 * 3)
        // Patch order is cell-major then (row-block, col-block), and positions
        // are 1-based for patches. A transposed order still has the right shape.
        XCTAssertEqual(layout.position(ofPatch: 0).row, 1)
        XCTAssertEqual(layout.position(ofPatch: 0).column, 1)
        XCTAssertEqual(layout.position(ofPatch: 1).row, 1)
        XCTAssertEqual(layout.position(ofPatch: 1).column, 2)
        XCTAssertEqual(layout.position(ofPatch: 2).row, 2)
        XCTAssertEqual(layout.position(ofPatch: 2).column, 1)
        XCTAssertEqual(layout.position(ofPatch: 4).row, 1, "cell 1 starts a new column pair")
        XCTAssertEqual(layout.position(ofPatch: 4).column, 3)
        XCTAssertEqual(layout.position(ofPatch: 24 - 1).row, 6, "the last patch is the far corner")
        XCTAssertEqual(layout.position(ofPatch: 24 - 1).column, 4)
    }

    func testTheResampleTapsFollowHalfPixelCentres() {
        // Patch (0, 0) of a 6x4 grid: source 2.1667 down and 3.5 across, so the
        // row pair is weighted 5/6 and 1/6 and the column pair half and half.
        let taps = Inkling2VisionMath.interpolationTaps(
            row: 0, column: 0, rows: 6, columns: 4, side: 32
        )
        XCTAssertEqual(taps.indices, [2 * 32 + 3, 2 * 32 + 4, 3 * 32 + 3, 3 * 32 + 4])
        XCTAssertEqual(taps.weights[0], 5.0 / 6.0 * 0.5, accuracy: 1e-5)
        XCTAssertEqual(taps.weights[1], 5.0 / 6.0 * 0.5, accuracy: 1e-5)
        XCTAssertEqual(taps.weights[2], 1.0 / 6.0 * 0.5, accuracy: 1e-5)
        XCTAssertEqual(taps.weights[3], 1.0 / 6.0 * 0.5, accuracy: 1e-5)

        // Border padding clamps an out-of-range tap and keeps its weight,
        // which is what `F.interpolate` does and `padding="zeros"` would not.
        let edge = Inkling2VisionMath.interpolationTaps(
            row: 0, column: 0, rows: 64, columns: 64, side: 32
        )
        XCTAssertTrue(edge.indices.allSatisfy { $0 >= 0 && $0 < 32 * 32 })
        XCTAssertEqual(edge.weights.reduce(0, +), 1, accuracy: 1e-5)
    }

    func testTheAxialRopeLayoutIsInterleavedNotRotateHalf() {
        let angles = Inkling2VisionMath.ropeAngles(row: 3, column: 5, headDim: 64, theta: 10_000)
        // Pairs 0..<16 carry the COLUMN coordinate, 16..<32 the row; each pair
        // repeats one frequency across its two channels.
        let first = pow(Float(10_000), 0) * 5
        XCTAssertEqual(angles.cos[0], cos(first), accuracy: 1e-6)
        XCTAssertEqual(angles.cos[1], angles.cos[0], "a pair shares one angle")
        XCTAssertEqual(angles.sin[1], angles.sin[0], "a pair shares one angle")
        let vertical = pow(Float(10_000), 0) * 3
        XCTAssertEqual(angles.cos[32], cos(vertical), accuracy: 1e-6)
        XCTAssertEqual(angles.sin[32], sin(vertical), accuracy: 1e-6)

        var values: [Float] = [1, 2]
        Inkling2VisionMath.rotateInPlace(&values, head: 0, headDim: 2, cos: [0, 0], sin: [1, 1])
        XCTAssertEqual(values, [-2, 1], "rotate_pairwise is (-x2, x1)")
    }

    // MARK: Pixels

    func testThePicturePadsToWholeCellsBeforeItNormalizes() throws {
        // 41 x 81 pixels: two cell columns (the second one 39 pixels of padding)
        // and three cell rows (the third one 39 rows of padding).
        let width = 41
        let height = 81
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in rgb.indices { rgb[index] = UInt8(index % 256) }
        let pixels = Inkling2ImagePixels.RGB(rgb: rgb, width: width, height: height)
        let cells = try Inkling2ImagePixels.cells(pixels)
        XCTAssertEqual(cells.cellRows, 3)
        XCTAssertEqual(cells.cellColumns, 2)
        XCTAssertEqual(cells.languagePositions, 6, "the cells are what the prompt pays for")
        XCTAssertEqual(cells.values.count, 6 * 40 * 40 * 3)

        // A real pixel normalizes as (x / 255 - 0.5) / 0.5.
        let expected = (Float(rgb[0]) / 255 - 0.5) / 0.5
        XCTAssertEqual(cells.values[0], expected, accuracy: 1e-6)

        // A padded one is the processor's -1.0 written on the 0-255 scale BEFORE
        // the rescale, so it lands just below -1 rather than at 0.
        // Cell (2, 0), row 1, column 0: source row 81 is past the picture.
        let cell = (2 * 2 + 0) * 40 * 40 * 3 + (1 * 40 + 0) * 3
        XCTAssertEqual(cells.values[cell], (Float(-1) / 255 - 0.5) / 0.5, accuracy: 1e-6)
        XCTAssertLessThan(cells.values[cell], -1)

        // Cell index 1 (row 0, column 1), row 0, column 2: source column 42 is
        // past the picture.
        let padded = ((1 * 40 + 0) * 40 + 2) * 3
        XCTAssertEqual(cells.values[padded], (Float(-1) / 255 - 0.5) / 0.5, accuracy: 1e-6)
    }

    func testTransparencyCompositesOntoTheCheckerboard() {
        // A fully transparent pixel shows the processor's checkerboard: 204 in an
        // even 8-pixel cell, 153 in an odd one.
        var rgba = [UInt8](repeating: 0, count: 10 * 10 * 4)
        rgba[(0 * 10 + 0) * 4 + 3] = 0
        rgba[(9 * 10 + 0) * 4 + 3] = 0
        // One opaque pixel passes its own colour through.
        let opaque = (0 * 10 + 5) * 4
        rgba[opaque] = 10
        rgba[opaque + 1] = 20
        rgba[opaque + 2] = 30
        rgba[opaque + 3] = 255
        let rgb = Inkling2ImagePixels.composite(rgba: rgba, width: 10, height: 10)
        XCTAssertEqual(rgb.rgb[0], 204, "row 0 / column 0 is an even checker cell")
        XCTAssertEqual(rgb.rgb[(9 * 10 + 0) * 3], 153, "row 9 is an odd checker row")
        XCTAssertEqual(rgb.rgb[(0 * 10 + 5) * 3], 10)
        XCTAssertEqual(rgb.rgb[(0 * 10 + 5) * 3 + 1], 20)
        XCTAssertEqual(rgb.rgb[(0 * 10 + 5) * 3 + 2], 30)
    }

    // MARK: The marker expansion

    /// Row-major RGB sized so the processor's cell arithmetic is exercised
    /// without an encoder: every byte is distinct, so a transposed or
    /// off-by-one crop is visible in the cells rather than merely plausible.
    private func picture(width: Int, height: Int) -> Inkling2ImagePixels.RGB {
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in rgb.indices { rgb[index] = UInt8(index % 256) }
        return Inkling2ImagePixels.RGB(rgb: rgb, width: width, height: height)
    }

    private func span(_ grid: Inkling2ImageGrid, rowWidth: Int = 4) -> Inkling2ImagePrompt.ImageSpan {
        let rows = (0..<grid.languagePositions).map { cell -> [Float] in
            var row = [Float](repeating: 0, count: rowWidth)
            row[0] = Float(cell)
            return row
        }
        return Inkling2ImagePrompt.ImageSpan(rows: rows, grid: grid)
    }

    func testOneAttachmentBecomesOneMarkerPerCell() throws {
        // 41 x 81 pixels pad to 3 cell rows and 2 cell columns: six language
        // positions, which is exactly what the prompt pays for.
        let cells = try Inkling2ImagePixels.cells(picture(width: 41, height: 81))
        XCTAssertEqual(cells.grid, Inkling2ImageGrid(cellRows: 3, cellColumns: 2))
        XCTAssertEqual(cells.languagePositions, 6)

        let marker: UInt32 = 200054
        let plan = try Inkling2ImagePrompt.expand(
            tokenIds: [11, marker, 12],
            spans: [span(cells.grid)],
            imageTokenID: Int(marker),
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [11] + [UInt32](repeating: marker, count: 6) + [12])
        XCTAssertEqual(plan.spans, [1..<7], "the span starts where the single placeholder was")
        XCTAssertEqual(plan.rows.keys.sorted(), Array(1...6))
        for position in 1...6 {
            XCTAssertEqual(plan.rows[position]?.first, Float(position - 1), "cell order is preserved")
        }
        XCTAssertEqual(plan.promptTokens, 8)
        XCTAssertEqual(plan.imagePositionCount, 6)

        // A text-only prompt is returned untouched.
        let text = try Inkling2ImagePrompt.expand(
            tokenIds: [1, 2, 3], spans: [], imageTokenID: Int(marker), rowWidth: 4
        )
        XCTAssertEqual(text.tokenIds, [1, 2, 3])
        XCTAssertTrue(text.rows.isEmpty)
        XCTAssertTrue(text.spans.isEmpty)
    }

    func testTwoInterleavedImagesKeepContentOrder() throws {
        let first = Inkling2ImageGrid(cellRows: 2, cellColumns: 1)
        let second = Inkling2ImageGrid(cellRows: 1, cellColumns: 3)
        let marker: UInt32 = 200054
        // Content order: picture 0, text, picture 1, text. The two grids differ
        // in size, so swapping their spans changes the output length.
        let plan = try Inkling2ImagePrompt.expand(
            tokenIds: [marker, 5, marker, 6],
            spans: [span(first), span(second)],
            imageTokenID: Int(marker),
            rowWidth: 4
        )
        XCTAssertEqual(plan.tokenIds, [marker, marker, 5, marker, marker, marker, 6])
        XCTAssertEqual(plan.spans, [0..<2, 3..<6])
        // Each image's own encoded rows land on its own span.
        XCTAssertEqual(plan.rows[0]?.first, 0)
        XCTAssertEqual(plan.rows[1]?.first, 1)
        XCTAssertEqual(plan.rows[3]?.first, 0)
        XCTAssertEqual(plan.rows[5]?.first, 2)
        XCTAssertEqual(plan.imagePositionCount, 5)
    }

    func testTheSpecPricesTheSameCellsThePixelsProduce() throws {
        let config = try Inkling2Config.decode(
            contentsOf: try snapshot().appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        for (width, height) in [(1, 1), (40, 40), (39, 39), (41, 81), (120, 80), (4_096, 1)] {
            let cells = try Inkling2ImagePixels.cells(picture(width: width, height: height))
            XCTAssertEqual(cells.grid.cellRows, (height + 39) / 40, "\(width)x\(height)")
            XCTAssertEqual(cells.grid.cellColumns, (width + 39) / 40, "\(width)x\(height)")
            XCTAssertEqual(
                cells.grid.languagePositions,
                spec.languagePositions(width: width, height: height),
                "the prompt's price and the pixel path must agree at \(width)x\(height)"
            )
        }
        // The checkpoint publishes no `image_token_id`; the decoder falls back
        // to the modular default, which is the tokenizer's own added token.
        XCTAssertEqual(config.imageTokenID, 200054)
    }

    func testAPlaceholderCountThatDisagreesWithThePicturesIsRefused() {
        let marker: UInt32 = 200054
        let one = span(Inkling2ImageGrid(cellRows: 1, cellColumns: 1))
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [marker, marker], spans: [one], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .placeholderCountMismatch(placeholders: 2, images: 1)
            )
        }
        // The other direction matters as much: an image whose marker is absent
        // would otherwise have nowhere to put its rows and be dropped.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [1, 2], spans: [one], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .placeholderCountMismatch(placeholders: 0, images: 1)
            )
        }
    }

    func testASpanThatDisagreesWithItsGridIsRefused() {
        let marker: UInt32 = 200054
        let grid = Inkling2ImageGrid(cellRows: 2, cellColumns: 2)
        // Four cells, three rows: the span would leave one marker unbound.
        let short = Inkling2ImagePrompt.ImageSpan(rows: [[0, 0], [1, 0], [2, 0]], grid: grid)
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [marker], spans: [short], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .spanLengthMismatch(image: 0, expected: 4, actual: 3)
            )
        }
        // A row that is not the text hidden size cannot become a residual row.
        let ragged = Inkling2ImagePrompt.ImageSpan(rows: [[0], [1], [2], [0]], grid: grid)
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [marker], spans: [ragged], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .rowWidthMismatch(image: 0, position: 0, expected: 4, actual: 1)
            )
        }
        // A grid with no cells is refused rather than shrinking the span.
        let empty = Inkling2ImagePrompt.ImageSpan(
            rows: [], grid: Inkling2ImageGrid(cellRows: 0, cellColumns: 3)
        )
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [marker], spans: [empty], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .invalidGrid(image: 0, rows: 0, columns: 3)
            )
        }
        // A marker id outside the token space is a config bug, not token 0.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.expand(
                tokenIds: [marker], spans: [span(grid)], imageTokenID: -1, rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(error as? Inkling2ImagePromptError, .invalidImageTokenID(-1))
        }
    }

    func testTheGridsAreCollectedInRenderOrder() throws {
        let first = Inkling2ImageGrid(cellRows: 1, cellColumns: 2)
        let second = Inkling2ImageGrid(cellRows: 3, cellColumns: 1)
        let turns: [Inkling2Turn] = [
            .system("s"),
            .user([.text("before"), .image(first), .text("middle")]),
            .assistant(reasoning: nil, content: "ok"),
            .tool(name: "shot", content: [.image(second)])
        ]
        XCTAssertEqual(Inkling2Prompt.imageGrids(in: turns), [first, second])
        XCTAssertEqual(
            Inkling2Prompt.imageGrids(in: [.user([.image(first), .image(second)])]),
            [first, second]
        )
    }

    /// A continuation splices the expanded ids forward, so re-expanding would
    /// read a span as many placeholders. `place` puts the same rows back at the
    /// recorded positions instead, and the two entry points must agree on the
    /// result.
    func testACarriedSpanBindsTheRecordedRows() throws {
        let marker: UInt32 = 200054
        let first = Inkling2ImageGrid(cellRows: 2, cellColumns: 1)
        let second = Inkling2ImageGrid(cellRows: 1, cellColumns: 2)
        let spans = [span(first), span(second)]
        let expanded = try Inkling2ImagePrompt.expand(
            tokenIds: [7, marker, 8, marker, 9],
            spans: spans,
            imageTokenID: Int(marker),
            rowWidth: 4
        )
        XCTAssertEqual(expanded.spans, [1..<3, 4..<6])
        let placed = try Inkling2ImagePrompt.place(
            tokenIds: expanded.tokenIds,
            spans: spans,
            at: expanded.spans,
            imageTokenID: Int(marker),
            rowWidth: 4
        )
        XCTAssertEqual(placed.tokenIds, expanded.tokenIds)
        XCTAssertEqual(placed.rows, expanded.rows)
        XCTAssertEqual(placed.spans, expanded.spans)
        XCTAssertEqual(placed.imagePositionCount, 4)
    }

    func testACarriedSpanThatDoesNotMatchItsPromptIsRefused() {
        let marker: UInt32 = 200054
        let one = Inkling2ImagePrompt.ImageSpan(
            rows: [[0, 0, 0, 0], [1, 0, 0, 0]], grid: Inkling2ImageGrid(cellRows: 2, cellColumns: 1)
        )
        // Position 1 holds an ordinary token where the span says a marker is.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.place(
                tokenIds: [marker, 5, marker], spans: [one], at: [1..<3],
                imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .spanTokenMismatch(image: 0, position: 1, expected: 200054, actual: 5)
            )
        }
        // A range whose length disagrees with its grid: one marker would be
        // bound twice or not at all.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.place(
                tokenIds: [marker, marker, marker], spans: [one], at: [0..<3],
                imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .spanLengthMismatch(image: 0, expected: 2, actual: 3)
            )
        }
        // A range past the prompt's end cannot be addressed at all.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.place(
                tokenIds: [marker, marker], spans: [one], at: [1..<3],
                imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .spanRangeInvalid(image: 0, range: 1..<3, promptTokens: 2)
            )
        }
        // Spans and ranges must describe the same number of pictures.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.place(
                tokenIds: [marker, marker], spans: [one, one], at: [0..<2],
                imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(
                error as? Inkling2ImagePromptError,
                .spanCountMismatch(spans: 2, ranges: 1)
            )
        }
        // Two images must keep prompt order: a second span before the first is
        // a spliced prompt the recorded layout does not describe.
        XCTAssertThrowsError(
            try Inkling2ImagePrompt.place(
                tokenIds: [marker, marker, marker, marker], spans: [one, one],
                at: [2..<4, 0..<2], imageTokenID: Int(marker), rowWidth: 4
            )
        ) { error in
            XCTAssertEqual(error as? Inkling2ImagePromptError, .spanRangeOutOfOrder(image: 1))
        }
    }

    // MARK: The turn

    /// Loads the checkpoint the way the APP loads it and installs it in the
    /// shared engine. The engine holds the shard mappings (see `engine`), so it
    /// must outlive the caller's turn.
    func configureAppOrderEngine() throws -> (
        directory: URL, tokenizer: DynaMoeTokenizer, config: Inkling2Config
    ) {
        let directory = try snapshot()
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU device") }
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: directory.appendingPathComponent("tokenizer.json").path
        )
        let modelConfig = try XCTUnwrap(ModelConfig.load(from: directory))
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        CompletionEngine.shared.configure(
            summary: summary, modelConfig: modelConfig, tokenizer: tokenizer,
            shardBuffers: buffers, activeLoadedModelPath: directory.path
        )
        return (directory, tokenizer, config)
    }

    /// A turn must actually use the Metal visit runner: a silent host fallback
    /// is correct but turns a fraction of a second per visit into seconds, and
    /// nothing in the stream says so. The status `launchTurn` records is
    /// assertable, and this is what pins it.
    func testTheEngineInstallsTheMetalVisitRunner() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_image_turn"),
            "opt-in; touch /tmp/dynamoe_inkling2_image_turn to run"
        )
        let (directory, tokenizer, _) = try configureAppOrderEngine()
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
        }
        let rendered = try Inkling2Prompt().render(
            turns: [.user([.text("Hi")])], effort: 0
        )
        let ids = try tokenizer.encode(text: rendered)
        let request = CompletionTurnRequest(
            promptTokenIds: ids,
            formattedPrompt: rendered,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                repetitionPenalty: 1, maxTokens: 1
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM, budgetMode: .unrestricted, kvPrecision: .fp32
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false
        )
        let launch = try CompletionEngine.shared.launchTurn(request)
        let status = CompletionEngine.shared.inklingDeviceHalfStatus
        print("🔎 [inkling2 device runner] \(status) for \(directory.lastPathComponent)")
        launch.handle.task.cancel()
        XCTAssertNotEqual(status, "not attempted", "launchTurn never reached the device-half decision")
        XCTAssertFalse(status.hasPrefix("failed"), status)
        XCTAssertNotEqual(status, "no Metal device, queue or library", status)
        XCTAssertNotEqual(status, "disabled by dynamoe_inkling2_device_off", status)
    }

    /// The engine's own device-half construction, with the engine's own
    /// parameters (`stagedPositions` = the provisioned context). The direct
    /// kernel tests build the executor with a DEFAULT workspace, so this is
    /// the one place the production widths are exercised before a turn: a
    /// failure here is why a turn would silently run the host graph.
    func testTheEnginesDeviceHalfBuildsAtTheProvisionedWidth() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_image_turn"),
            "opt-in; touch /tmp/dynamoe_inkling2_image_turn to run"
        )
        let directory = try snapshot()
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("No Metal GPU device") }
        guard let library = device.makeDefaultLibrary() else { throw XCTSkip("no default library") }
        guard let queue = device.makeCommandQueue() else { throw XCTSkip("no compute queue") }
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        let plan = Inkling2MemoryPlan.make(
            config: config, catalog: catalog, contextLength: config.maxPositionEmbeddings
        )
        // `unrestricted`'s RSS budget, which is what an app turn asks for.
        let contextLength = plan.affordableContextLength(budgetGB: 36.6)
        print("🔎 [inkling2 device half] provisioned \(contextLength) positions")
        let source = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        let hostForward = try Inkling2CPUForward(
            config: config, weights: source,
            options: Inkling2ForwardOptions(maxPositions: contextLength)
        )
        let weights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        do {
            _ = try Inkling2MetalForward(
                device: device, library: library, queue: queue, weights: weights, host: hostForward
            )
            print("🔎 [inkling2 device half] executor ok")
            _ = try Inkling2DeviceWorkspace(
                device: device, config: config, stagedPositions: contextLength
            )
            print("🔎 [inkling2 device half] workspace ok")
        } catch {
            XCTFail("the engine's device half cannot be built: \(error)")
        }
    }

    /// The engine seam end to end on real weights: a red rectangle pads to two
    /// 40-pixel cells, the tower emits one residual row per cell, the single
    /// template placeholder expands to two markers, and the turn answers about
    /// the colour. Opt-in because the host tower is scalar Swift.
    func testARealImageTurnAnswersAboutThePicture() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_image_turn"),
            "opt-in image turn; touch /tmp/dynamoe_inkling2_image_turn to run"
        )
        let (_, tokenizer, config) = try configureAppOrderEngine()
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
        }

        // 50 x 40 pixels: one cell row, two cell columns. Two cells is the
        // smallest grid that proves the placeholder really expands rather than
        // passing through.
        let width = 50
        let height = 40
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for index in stride(from: 0, to: rgb.count, by: 3) {
            rgb[index] = 220
            rgb[index + 1] = 20
            rgb[index + 2] = 20
        }
        let pixels = Inkling2ImagePixels.RGB(rgb: rgb, width: width, height: height)
        let grid = try Inkling2ImagePixels.cells(pixels).grid
        XCTAssertEqual(grid, Inkling2ImageGrid(cellRows: 1, cellColumns: 2))

        let rendered = try Inkling2Prompt().render(
            turns: [.user([.image(grid), .text("What colour is this rectangle? Answer with one word.")])],
            effort: 0
        )
        let promptIds = try tokenizer.encode(text: rendered)
        let marker = UInt32(config.imageTokenID)
        XCTAssertEqual(promptIds.filter { $0 == marker }.count, 1, "the template writes one marker")

        var result: TurnResult?
        let request = CompletionTurnRequest(
            promptTokenIds: promptIds,
            formattedPrompt: rendered,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                repetitionPenalty: 1, maxTokens: 64
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM, budgetMode: .unrestricted, kvPrecision: .fp32
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false,
            images: [TurnImage(
                rgb: pixels.rgb, width: pixels.width, height: pixels.height,
                sha256: "inkling2-red-rectangle"
            )]
        )
        for await event in try CompletionEngine.shared.launchTurn(request).stream {
            switch event {
            case .finished(let value):
                result = value
            case .failedTurn(let failure):
                XCTFail("the image turn failed: \(failure.message)")
            default:
                break
            }
        }
        let turn = try XCTUnwrap(result)
        print("🔎 [inkling2 image] answer: \(turn.rawDecodedText)")
        print("🔎 [inkling2 image] response: \(turn.responseText)")
        // The checkpoint opens the reasoning channel itself at effort 0, so the
        // colour lives in the content channel and not in the whole transcript.
        let contentMarker = Inkling2PromptMarkers.familyDefault.contentText
        guard let answerStart = turn.rawDecodedText.range(of: contentMarker) else {
            XCTFail("the model never opened a content block: \(turn.rawDecodedText)")
            return
        }
        let answer = turn.rawDecodedText[answerStart.upperBound...]
        XCTAssertTrue(
            answer.lowercased().contains("red"),
            "the answer did not name the colour: \(answer)"
        )
        // The prompt the cache and the sampler saw carries one marker per CELL,
        // and the span is where the picture's rows sit.
        XCTAssertEqual(turn.promptTokenIds.filter { $0 == marker }.count, grid.languagePositions)
        XCTAssertEqual(turn.promptTokenIds.count, promptIds.count + grid.languagePositions - 1)
        XCTAssertEqual(turn.imageSpans.count, 1)
        XCTAssertEqual(turn.imageSpans.first?.range.count, grid.languagePositions)
        XCTAssertEqual(turn.imageSpans.first?.sha256, "inkling2-red-rectangle")
        XCTAssertEqual(
            turn.imageSpans.first.map { Array(turn.promptTokenIds[$0.range]) },
            [UInt32](repeating: marker, count: grid.languagePositions)
        )
    }

    // MARK: The tower

    func testTheTowerMatchesTheReferenceStageByStage() throws {
        // The host tower is scalar Swift, and at -Onone the 23 layers cost ~10
        // minutes (measured 582 s): a real gate, but not one that belongs in
        // every run. `touch /tmp/dynamoe_inkling2_vision` to include it; the
        // pure-logic cases above stay in the default suite.
        guard FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_vision") else {
            throw XCTSkip("opt-in: touch /tmp/dynamoe_inkling2_vision to run the 23-layer tower")
        }
        let goldens = try dump()
        let forward = try makeForward()
        let spec = forward.spec
        let layout = Inkling2VisionLayout(
            cellRows: goldens["cellRows"] as? Int ?? 0,
            cellColumns: goldens["cellColumns"] as? Int ?? 0,
            spec: spec
        )
        XCTAssertEqual(layout.sequenceLength, goldens["sequenceLength"] as? Int)
        XCTAssertEqual(layout.patchCount, goldens["towerTokens"] as? Int)

        // The input first: a fixture that drifted would otherwise be compared
        // against itself.
        let pixels = Self.pixels(layout: layout)
        XCTAssertEqual(pixels.count, goldens["pixelCount"] as? Int)
        let checksum = Float(pixels.reduce(0.0) { $0 + Double($1) })
        XCTAssertEqual(checksum, (goldens["pixelChecksum"] as? NSNumber)?.floatValue ?? .nan, accuracy: 0.01)

        // The resample, independently of the tower.
        let referenceIndices = (goldens["interpolation"] as? [String: Any])?["indices"] as? [[Int]] ?? []
        let referenceWeights = rows((goldens["interpolation"] as? [String: Any])?["weights"])
        XCTAssertEqual(referenceIndices.count, layout.patchCount)
        for patch in 0..<layout.patchCount {
            let position = layout.position(ofPatch: patch)
            let mine = Inkling2VisionMath.interpolationTaps(
                row: position.row - 1, column: position.column - 1,
                rows: layout.patchRows, columns: layout.patchColumns, side: spec.posembGridSize
            )
            XCTAssertEqual(mine.indices, referenceIndices[patch], "patch \(patch) taps")
            XCTAssertEqual(relativeError(mine.weights, referenceWeights[patch]), 0, accuracy: 2e-6)
        }

        // Every stage the tower can name, compared as it is produced.
        let referenceLayers = (goldens["layers"] as? [Any])?.map { rows($0) } ?? []
        var stageErrors: [(String, Float)] = []
        forward.onStage = { name, actual in
            // The dump keeps the layers as a LIST; the forward names them
            // `layerN`, so the mapping is here rather than in either side's
            // storage format.
            let expected: [[Float]]
            if name.hasPrefix("layer"), let index = Int(name.dropFirst("layer".count)) {
                guard index < referenceLayers.count else { return }
                expected = referenceLayers[index]
            } else {
                expected = self.rows(goldens[name])
            }
            guard let first = expected.first, !first.isEmpty else { return }
            let error = self.relativeError(actual.flatMap { $0 }, expected.flatMap { $0 })
            stageErrors.append((name, error))
        }
        let pooler = try forward.encode(pixelValues: pixels, layout: layout)
        forward.onStage = nil

        // The pooler's own rows are compared below from its dump key; this tally
        // is the stages the forward reports by the same name.
        XCTAssertEqual(stageErrors.count, spec.numHiddenLayers + 2, "embeddings, pre-norm, every layer")
        let bar: Float = 2e-3
        var worst = (name: "", error: Float(0))
        for (name, error) in stageErrors {
            if error > worst.error { worst = (name, error) }
            XCTAssertLessThanOrEqual(
                error, bar,
                "\(name) is \(error) relative from the reference; worst so far \(worst)"
            )
        }

        // The language model's rows: one per cell, `textHiddenSize` wide.
        let reference = rows(goldens["poolerOutput"])
        XCTAssertEqual(pooler.count, layout.cellCount)
        XCTAssertEqual(pooler.first?.count, spec.textHiddenSize)
        XCTAssertEqual(relativeError(pooler.flatMap { $0 }, reference.flatMap { $0 }), 0, accuracy: bar)
        print("vision tower: \(spec.numHiddenLayers) layers, worst stage \(worst.name) \(worst.error) relative")
    }
}

// MARK: - Plumbing the picture into the prompt

/// The half of the image path that runs with no weights: how a picture becomes
/// a typed content block, how the API server hands one over, and when the
/// composer is allowed to offer an attachment at all.
///
/// These are the seams that fail SILENTLY when they drift -- a picture in the
/// wrong block, a sentinel reaching the tokenizer, a half-downloaded snapshot
/// advertising a tower -- so each is pinned here rather than at the turn.
final class Inkling2VisionPlumbingTests: XCTestCase {

    private let marker = Inkling2PromptMarkers.familyDefault
    private let oneByTwo = Inkling2ImageGrid(cellRows: 1, cellColumns: 2)
    private let twoByThree = Inkling2ImageGrid(cellRows: 2, cellColumns: 3)

    // MARK: Content parts

    /// The checkpoint gives a picture its own content block, so the renderer
    /// must be handed parts rather than a sentence with a marker inside it.
    func testAPictureBecomesItsOwnContentBlock() {
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(text: "What colour?", grids: []),
            [.text("What colour?")],
            "a turn with no picture is one text block"
        )
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(text: "What colour?", grids: [oneByTwo]),
            [.image(oneByTwo), .text("What colour?")],
            "a picture with no sentinel goes in front of the words"
        )
        // Text, picture, text, picture: the server's decoders keep a message's
        // content order as one sentinel per picture, and that order has to survive
        // as parts or the model answers about a picture in a different relation to
        // the words.
        let sentinel = ServerImageParts.imagePlaceholder
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(
                text: "Before \(sentinel) between \(sentinel) after",
                grids: [oneByTwo, twoByThree]
            ),
            [
                .text("Before "), .image(oneByTwo),
                .text(" between "), .image(twoByThree),
                .text(" after")
            ]
        )
    }

    /// A count mismatch is fail-closed: nothing is rendered, so the prompt is left
    /// without the placeholders the pictures need and the engine refuses the turn
    /// by name. Repeating or dropping a marker would shift every later position.
    func testAPictureCountThatDisagreesWithItsMarkersRendersNothing() {
        let sentinel = ServerImageParts.imagePlaceholder
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(text: "before \(sentinel) after", grids: []),
            [],
            "a sentinel with no picture has nowhere to go"
        )
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(
                text: "a \(sentinel) b \(sentinel) c", grids: [oneByTwo]
            ),
            [],
            "two sentinels and one picture cannot both be honoured"
        )
    }

    /// The parts render with the checkpoint's own markers, and the sentinel itself
    /// never reaches them.
    func testTheRenderedBlockCarriesTheCheckpointsMarkers() throws {
        let sentinel = ServerImageParts.imagePlaceholder
        let parts = ChatPromptFormatter.inklingContentParts(
            text: "Which colour? \(sentinel)", grids: [oneByTwo]
        )
        let rendered = try Inkling2Prompt().render(parts: parts, role: "user")
        XCTAssertEqual(
            rendered,
            marker.contentText + "Which colour? " + marker.endMessage
                + marker.contentImage + marker.imagePlaceholder + marker.endMessage
        )
        XCTAssertFalse(rendered.contains(sentinel), "the sentinel is a caller's, not the model's")
    }

    // MARK: Server seam

    /// The server decodes each part, records one picture per part, and leaves the
    /// turn's sentinel in place for the renderer to consume.
    func testTheServerGivesEachPictureItsOwnBlock() throws {
        let png = try inklingTestPNG(width: 50, height: 40)
        var turns = [
            ChatPromptFormatter.ConversationTurn(
                role: .user, text: "Which colour? \(ServerImageParts.imagePlaceholder)"
            )
        ]
        let resolved = try ServerController.attachImageGrids(
            turns: turns,
            images: [0: [ServerImagePart(payload: .inline(png, mediaType: "image/png"))]]
        )
        turns = resolved.turns
        XCTAssertEqual(turns[0].imageGrids, [oneByTwo], "50 x 40 pixels is one cell row of two")
        XCTAssertEqual(resolved.images.count, 1)
        XCTAssertEqual(resolved.images[0].width, 50)
        XCTAssertEqual(resolved.images[0].height, 40)
        XCTAssertTrue(
            turns[0].text.contains(ServerImageParts.imagePlaceholder),
            "the sentinel is what places the block, so the server leaves it in place"
        )
        XCTAssertEqual(
            ChatPromptFormatter.inklingContentParts(
                text: turns[0].text, grids: turns[0].imageGrids
            ),
            [.text("Which colour? "), .image(oneByTwo)]
        )
    }

    /// The count the server checks the budget against and reports as usage is
    /// the EXPANDED one, because that is the prompt the cache runs.
    func testTheServerPricesTheExpandedPicture() throws {
        let markerID = 200_054
        let ids: [UInt32] = [11, 12, UInt32(markerID), 13]
        XCTAssertEqual(
            try ServerController.inklingPromptTokenCount(
                tokenIds: ids, grids: [oneByTwo], imageTokenID: markerID
            ),
            4 - 1 + oneByTwo.languagePositions,
            "one placeholder becomes one marker per cell"
        )
        XCTAssertThrowsError(
            try ServerController.inklingPromptTokenCount(
                tokenIds: [UInt32(markerID)], grids: [oneByTwo, twoByThree], imageTokenID: markerID
            ),
            "a prompt with fewer placeholders than pictures has nowhere to put the rest"
        )
    }

    func testAPictureOnATurnThatDoesNotExistIsRefused() throws {
        let png = try inklingTestPNG(width: 40, height: 40)
        XCTAssertThrowsError(
            try ServerController.attachImageGrids(
                turns: [ChatPromptFormatter.ConversationTurn(role: .user, text: "hi")],
                images: [3: [ServerImagePart(payload: .inline(png, mediaType: "image/png"))]]
            )
        )
    }

    // MARK: The composer gate

    /// The attachment button is opened by `supportsVision`, so a snapshot that
    /// merely CLAIMS a tower must not open it, and the flag has to turn on once
    /// the files land.
    func testTheGateOpensOnlyWhenTheTowerHasShipped() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-vision-gate-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer {
            ModelConfig.invalidateVisionCache()
            try? FileManager.default.removeItem(at: root)
        }
        ModelConfig.invalidateVisionCache()

        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(Inkling2ConfigTests.smallJSON.utf8)
            ) as? [String: Any]
        )
        config["vision_config"] = [
            "model_type": "inkling2_vision",
            "text_hidden_size": 64,
            "temporal_patch_size": 2,
            "native_patch_size": 20,
            "adapter_merge_size": 2,
            "projection_hidden_dim": 32,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "mlp_ratio": 4.0,
            "posemb_grid_size": 32,
            "layer_norm_eps": 1e-05
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: root.appendingPathComponent("config.json"))

        XCTAssertFalse(
            ModelConfig.supportsVision(modelPath: root.path),
            "a config that promises a tower the snapshot has not shipped is not vision"
        )

        let anchors = [
            "model.visual.backbone.embeddings.patch_embedding.weight",
            "model.visual.backbone.embeddings.class_embedding",
            "model.visual.backbone.embeddings.position_embedding",
            "model.visual.backbone.ln_pre.weight",
            "model.visual.backbone.layers.0.self_attn.q_proj.weight",
            "model.visual.backbone.layers.0.mlp.c_fc.weight",
            "model.visual.proj_in.weight",
            "model.visual.proj_out.weight"
        ]
        let index: [String: Any] = [
            "weight_map": Dictionary(
                uniqueKeysWithValues: anchors.map { ($0, "model-00001-of-00001.safetensors") }
            )
        ]
        try JSONSerialization.data(withJSONObject: index)
            .write(to: root.appendingPathComponent("model.safetensors.index.json"))
        ModelConfig.invalidateVisionCache()
        XCTAssertFalse(
            ModelConfig.supportsVision(modelPath: root.path),
            "a name in the index proves nothing while the shard it names is missing"
        )

        try Data().write(to: root.appendingPathComponent("model-00001-of-00001.safetensors"))
        ModelConfig.invalidateVisionCache()
        XCTAssertTrue(
            ModelConfig.supportsVision(modelPath: root.path),
            "the shards have landed, so the composer may offer an attachment"
        )
    }

    /// The downloaded checkpoint is one, and that is what makes the composer's
    /// button appear for it rather than only for the family's tests.
    func testThePublishedCheckpointAdvertisesItsTower() throws {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: path.path),
            "Inkling 2 snapshot is not downloaded at \(path.path)"
        )
        XCTAssertTrue(ModelConfig.supportsVision(modelPath: path.path))
    }

    // MARK: Helpers

}

/// A real PNG of a solid red rectangle, because both the server seam and the
/// live request decode ENCODED bytes rather than the raw RGB a `TurnImage`
/// carries.
private func inklingTestPNG(width: Int, height: Int) throws -> Data {
    guard let rep = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
        bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
    ) else {
        throw XCTSkip("NSBitmapImageRep refused \(width)x\(height)")
    }
    if let base = rep.bitmapData {
        for y in 0..<height {
            let row = base + y * rep.bytesPerRow
            for x in 0..<width {
                let pixel = row + x * 3
                pixel[0] = 220
                pixel[1] = 20
                pixel[2] = 20
            }
        }
    }
    return try XCTUnwrap(rep.representation(using: .png, properties: [:]))
}

// MARK: - The same picture, over HTTP

extension Inkling2VisionTests {

    /// The API's half of the image path, on the published checkpoint.
    ///
    /// Every other case here drives the engine directly, and the two seams this
    /// slice added -- the grid-carrying turn and the expanded price -- are only
    /// PART of what a client meets: the adapter decodes the part, the server
    /// chooses the family's placement, the renderer writes the block, the
    /// planner prices the expansion and `launchTurn` binds the rows. A mistake
    /// anywhere along it is a refusal or a 500, never a wrong token, so the
    /// request is the only thing that can prove the whole path.
    ///
    /// The picture's cost is checked as a DELTA against the same question sent
    /// plain: the block is three tokens (`<|content_image|>`, the placeholder
    /// and `<|end_message|>`) and the placeholder then expands to one marker per
    /// cell, so the difference is `2 + cells` with no renderer detail assumed.
    func testAnAPIRequestWithAnImageAnswersOnThePublishedCheckpoint() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_image_turn"),
            "opt-in image turn; touch /tmp/dynamoe_inkling2_image_turn to run"
        )
        let (directory, _, _) = try configureAppOrderEngine()
        defer {
            CompletionEngine.shared.configure(
                summary: nil, modelConfig: nil, tokenizer: nil,
                shardBuffers: [:], activeLoadedModelPath: nil
            )
            withExtendedLifetime(engine) {}
        }
        XCTAssertTrue(
            ModelConfig.supportsVision(modelPath: directory.path),
            "the gate the request passes through must see this checkpoint's tower"
        )
        let png = try inklingTestPNG(width: 50, height: 40)
        let pixels = try Inkling2ImagePixels.decode(png)
        let grid = try Inkling2ImagePixels.cells(pixels).grid
        XCTAssertEqual(grid, Inkling2ImageGrid(cellRows: 1, cellColumns: 2))

        let port = 8795
        try await withLocalServer(port: port) { _ in
            let catalog = try await getModels(port: port)
            let vision = (catalog.first?["capabilities"] as? [String: Any])?["vision"] as? Bool
            XCTAssertEqual(vision, true, "the models list must advertise the tower it uses")
            XCTAssertNil(
                (catalog.first?["capabilities"] as? [String: Any])?["vision_cost"],
                "no price is published for a tower whose cost has not been measured"
            )

            let question = "What colour is this rectangle? Answer with one word."
            let plain = try await postChat(port: port, text: question, imageBase64: nil)
            let pictured = try await postChat(
                port: port, text: question, imageBase64: png.base64EncodedString()
            )
            print(
                "🔎 [inkling2 server] plain \(plain.promptTokens) / pictured "
                    + "\(pictured.promptTokens) prompt tokens, \(pictured.completionTokens) "
                    + "generated, answer \(pictured.content.debugDescription)"
            )
            XCTAssertEqual(
                pictured.promptTokens - plain.promptTokens,
                2 + grid.languagePositions,
                "the block is three tokens and its placeholder expands to one per cell"
            )
            XCTAssertTrue(
                pictured.content.lowercased().contains("red"),
                "the answer did not name the colour: \(pictured.content)"
            )
        }
    }

    private struct ServerChatTurn {
        var content: String
        var promptTokens: Int
        var completionTokens: Int
        var finishReason: String?
    }

    /// Runs the real server on a private port with every setting saved and
    /// restored, so a developer's app on 8787 is neither disturbed nor used.
    private func withLocalServer(
        port: Int, _ body: (ServerRequestLogStore) async throws -> Void
    ) async throws {
        let defaults = UserDefaults.standard
        let savedEnabled = defaults.object(forKey: ServerController.enabledKey)
        let savedPort = defaults.object(forKey: ServerController.portKey)
        let savedKey = defaults.object(forKey: ServerController.apiKeyKey)
        let savedCapture = defaults.object(forKey: ServerRequestLogRecorder.captureBodiesKey)
        let store = ServerRequestLogStore.inMemory()
        func restore() {
            if let savedEnabled {
                defaults.set(savedEnabled, forKey: ServerController.enabledKey)
            } else {
                defaults.removeObject(forKey: ServerController.enabledKey)
            }
            if let savedPort {
                defaults.set(savedPort, forKey: ServerController.portKey)
            } else {
                defaults.removeObject(forKey: ServerController.portKey)
            }
            if let savedKey {
                defaults.set(savedKey, forKey: ServerController.apiKeyKey)
            } else {
                defaults.removeObject(forKey: ServerController.apiKeyKey)
            }
            if let savedCapture {
                defaults.set(savedCapture, forKey: ServerRequestLogRecorder.captureBodiesKey)
            } else {
                defaults.removeObject(forKey: ServerRequestLogRecorder.captureBodiesKey)
            }
            ServerRequestLogRecorder.shared.useOverrideStore(nil)
        }
        defaults.set(true, forKey: ServerController.enabledKey)
        defaults.set(port, forKey: ServerController.portKey)
        // The case sends unauthenticated requests, so a saved key would turn them
        // into a 401.
        defaults.removeObject(forKey: ServerController.apiKeyKey)
        defaults.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        ServerRequestLogRecorder.shared.useOverrideStore(store)
        if ServerController.shared.isRunning {
            await ServerController.shared.stop()
        }
        ServerController.shared.start()
        do {
            let deadline = Date().addingTimeInterval(20)
            var listening = false
            while Date() < deadline {
                if let (_, response) = try? await URLSession.shared.data(
                    from: URL(string: "http://127.0.0.1:\(port)/health")!
                ), (response as? HTTPURLResponse)?.statusCode == 200 {
                    listening = true
                    break
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            try XCTSkipUnless(listening, "the local server never answered /health on port \(port)")
            try await body(store)
        } catch {
            await ServerController.shared.stop()
            restore()
            throw error
        }
        await ServerController.shared.stop()
        restore()
    }

    private func getModels(port: Int) async throws -> [[String: Any]] {
        let (data, response) = try await URLSession.shared.data(
            from: URL(string: "http://127.0.0.1:\(port)/v1/models")!
        )
        XCTAssertEqual((response as? HTTPURLResponse)?.statusCode, 200)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return json["data"] as? [[String: Any]] ?? []
    }

    private func postChat(
        port: Int, text: String, imageBase64: String?
    ) async throws -> ServerChatTurn {
        var parts = ["{\"type\":\"text\",\"text\":\"\(text)\"}"]
        if let imageBase64, !imageBase64.isEmpty {
            parts.append(
                "{\"type\":\"image_url\",\"image_url\":"
                    + "{\"url\":\"data:image/png;base64,\(imageBase64)\"}}"
            )
        }
        // The checkpoint opens its reasoning channel at effort 0, so the budget
        // has to cover the trace before the one-word answer.
        let body = """
        {"model":"any","max_tokens":96,"messages":[
            {"role":"user","content":[\(parts.joined(separator: ","))]}
        ]}
        """
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        )
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(body.utf8)
        request.timeoutInterval = 1800
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any],
            "the reply was not JSON: " + String(decoding: data, as: UTF8.self)
        )
        guard http.statusCode == 200 else {
            XCTFail("\(http.statusCode) from the local server: \(json)")
            return ServerChatTurn(content: "", promptTokens: -1, completionTokens: -1, finishReason: nil)
        }
        let choice = try XCTUnwrap((json["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        let usage = try XCTUnwrap(json["usage"] as? [String: Any])
        return ServerChatTurn(
            content: message["content"] as? String ?? "",
            promptTokens: usage["prompt_tokens"] as? Int ?? -1,
            completionTokens: usage["completion_tokens"] as? Int ?? -1,
            finishReason: choice["finish_reason"] as? String
        )
    }
}
