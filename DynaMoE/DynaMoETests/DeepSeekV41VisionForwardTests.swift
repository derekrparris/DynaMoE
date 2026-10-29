//
//  DeepSeekV41VisionForwardTests.swift
//  DynaMoETests
//
//  The vision tower has no reference vectors checked into the checkpoint, so these
//  tests are built two ways: structural properties that pin the parts a wrong port
//  gets wrong silently (2D shared rotary, channel-major unshuffle, the padding that
//  `F.unfold` needs), and one end-to-end fixture whose numbers come from an
//  independent float64 numpy mirror of `inference/vision.py`.
//
//  The fixture is deliberately tiny: 2x3 patches of 2x2 pixels, hidden 8, two
//  heads, one block, a downsample ratio of 2. That is small enough to reason about
//  and wide enough to exercise the padding (3 patch columns pad to 4) and the
//  per-head rotary.
//

import XCTest
@testable import DynaMoE

final class DeepSeekV41VisionForwardTests: XCTestCase {

    // MARK: Fixture

    private let hidden = 8
    private let heads = 2
    private let intermediate = 4
    private let ratio = 2
    private let patchSize = 2
    private let textHidden = 5

    private func spec() -> DeepSeekV41VisionSpec {
        DeepSeekV41VisionSpec(
            patchSize: patchSize,
            downsampleRatio: ratio,
            minPixels: 0,
            maxImageTokens: 1_024,
            numHiddenLayers: 1,
            hiddenSize: hidden,
            numAttentionHeads: heads,
            intermediateSize: intermediate,
            imageTokenID: 7,
            maxWidthHeightRatio: nil,
            ropeTheta: 100
        )
    }

    /// The same deterministic formula the numpy mirror uses, so the two disagree
    /// about nothing except floating-point width.
    private func entry(_ row: Int, _ column: Int) -> Float {
        Float(((row * 7 + column * 3) % 5) - 2) * 0.25
    }

    private func matrix(_ rows: Int, _ columns: Int) -> DeepSeekV41Matrix {
        var values: [Float] = []
        values.reserveCapacity(rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                values.append(entry(row, column))
            }
        }
        return DeepSeekV41Matrix(rows: rows, columns: columns, values: values)
    }

    private func vector(_ count: Int, offset: Int) -> [Float] {
        (0..<count).map { Float((($0 * 3 + offset) % 5) - 2) * 0.25 }
    }

    private func patches() -> [[Float]] {
        (0..<6).map { token in
            (0..<(3 * patchSize * patchSize)).map { dimension in
                entry(token, dimension) + Float(token) * 0.5
            }
        }
    }

    private func weights() -> DeepSeekV41DenseWeightSource {
        let source = DeepSeekV41DenseWeightSource()
        source.set(matrix(hidden, 3 * patchSize * patchSize), role: .visionPatchEmbed, layer: nil)
        source.set(vector(hidden, offset: 1), role: .visionPatchEmbedBias, layer: nil)
        source.set((0..<hidden).map { 1 + Float($0 % 4) * 0.125 }, role: .visionBlockNorm1, layer: 0)
        source.set(matrix(3 * hidden, hidden), role: .visionBlockWqkv, layer: 0)
        source.set(vector(3 * hidden, offset: 2), role: .visionBlockWqkvBias, layer: 0)
        source.set(matrix(hidden, hidden), role: .visionBlockWo, layer: 0)
        source.set(vector(hidden, offset: 3), role: .visionBlockWoBias, layer: 0)
        source.set((0..<hidden).map { 1 + Float($0 % 3) * 0.25 }, role: .visionBlockNorm2, layer: 0)
        source.set(matrix(2 * intermediate, hidden), role: .visionBlockW1, layer: 0)
        source.set(matrix(hidden, intermediate), role: .visionBlockW2, layer: 0)
        source.set((0..<hidden).map { 1 + Float($0 % 2) * 0.5 }, role: .visionFinalNorm, layer: nil)
        source.set(matrix(textHidden, hidden * ratio * ratio), role: .visionAlignerW1, layer: nil)
        source.set(vector(textHidden, offset: 4), role: .visionAlignerW1Bias, layer: nil)
        source.set(matrix(textHidden, textHidden), role: .visionAlignerW2, layer: nil)
        source.set(vector(textHidden, offset: 0), role: .visionAlignerW2Bias, layer: nil)
        source.set(vector(textHidden, offset: 1), role: .visionImageStart, layer: nil)
        source.set(vector(textHidden, offset: 2), role: .visionImageNewline, layer: nil)
        source.set(vector(textHidden, offset: 3), role: .visionImageEnd, layer: nil)
        return source
    }

    private func forward() -> DeepSeekV41VisionForward {
        DeepSeekV41VisionForward(spec: spec(), weights: weights(), textHiddenSize: textHidden)
    }

    // MARK: End to end

    /// Reference values from an independent float64 numpy mirror of `vision.py`
    /// over this fixture. Two aligner rows, five wide.
    func testTowerMatchesTheReferenceNumbers() throws {
        let output = try forward().encode(patches: patches(), rows: 2, columns: 3)
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output.first?.count, textHidden)
        let expected: [[Float]] = [
            [0.262_165_24, -0.764_425_98, -0.321_022_40, 1.665_410_14, -0.842_127_00],
            [-1.313_606_29, 2.380_801_02, -2.570_785_12, 1.061_984_97, 0.441_605_41]
        ]
        for (row, expectedRow) in expected.enumerated() {
            for (column, value) in expectedRow.enumerated() {
                XCTAssertEqual(
                    output[row][column], value, accuracy: 1e-4,
                    "row \(row) column \(column)"
                )
            }
        }
    }

    /// The span the prompt sees: a start marker, then per aligner row its cells and
    /// a newline, then an end marker. The count is what
    /// `DeepSeekV41VisionLayout` prices the image at.
    func testSpanIsTheAlignerGridWrappedByMarkers() throws {
        let span = try forward().imageSpan(patches: patches(), rows: 2, columns: 3)
        // 2x3 patches with a ratio of 2: one aligner row of two cells.
        XCTAssertEqual(span.count, DeepSeekV41VisionLayout.imageTokenCount(columns: 2, rows: 1))
        XCTAssertEqual(span.count, 5)
        XCTAssertEqual(span.first, vector(textHidden, offset: 1))
        XCTAssertEqual(span[3], vector(textHidden, offset: 2), "the newline closes the row")
        XCTAssertEqual(span.last, vector(textHidden, offset: 3))

        // The cells themselves are the aligner rows, not the markers.
        let cells = try forward().encode(patches: patches(), rows: 2, columns: 3)
        XCTAssertEqual(span[1], cells[0])
        XCTAssertEqual(span[2], cells[1])
        XCTAssertThrowsError(try forward().imageSpan(patches: [], rows: 0, columns: 0))
    }

    /// Pixels to span in one call, and the length of that span must be the count the
    /// prompt expansion and the KV budget price the image at. A disagreement here
    /// would put the aligner rows at the wrong prompt positions.
    func testPixelsToSpanLengthMatchesThePricedTokenCount() throws {
        let width = 8
        let height = 12
        let rgb = [UInt8](repeating: 128, count: width * height * 3)
        let span = try forward().imageSpan(rgb: rgb, width: width, height: height)
        let fitted = DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec())
        XCTAssertEqual(span.count, fitted.tokens)
        // 4x6 patches on a ratio of 2 is a 2x3 aligner grid.
        XCTAssertEqual(span.count, 3 * (2 + 1) + 2)
        for row in span {
            XCTAssertEqual(row.count, textHidden)
        }
    }

    // MARK: Structure

    /// The rotation is 2D and shared across heads: the height frequency lands in
    /// the first half of the angle vector and the width frequency in the second, and
    /// a token's own row and column are what select it.
    func testRotaryEncodesRowAndColumn() {
        let tables = DeepSeekV41VisionForward.rotaryTables(
            rows: 2, columns: 3, ropeDim: 4, theta: 100
        )
        XCTAssertEqual(tables.cosine.count, 6)
        // ropeDim 4 gives two inverse frequencies, so each token's vector is
        // [row*f0, row*f1, column*f0, column*f1].
        let inverse: [Float] = [1, 1 / pow(100, Float(2) / 4)]
        let angle = { (row: Int, column: Int) -> [Float] in
            [Float(row) * inverse[0], Float(row) * inverse[1],
             Float(column) * inverse[0], Float(column) * inverse[1]]
        }
        for row in 0..<2 {
            for column in 0..<3 {
                let index = row * 3 + column
                let expected = angle(row, column).map(cos)
                for axis in 0..<4 {
                    XCTAssertEqual(tables.cosine[index][axis], expected[axis], accuracy: 1e-6)
                }
            }
        }
    }

    /// Split-half rotation over the whole head: the first half rotates against the
    /// second, which is not the adjacent-pair form the text tower uses.
    func testRotaryRotatesTheHalvesNotAdjacentPairs() {
        var head: [Float] = [1, 0, 0, 1]
        // A quarter turn: (1, 0) -> (0, 1) and (0, 1) -> (-1, 0).
        DeepSeekV41VisionForward.applyRotary(&head, cosine: [0, 0], sine: [1, 1])
        XCTAssertEqual(head[0], 0, accuracy: 1e-6)
        XCTAssertEqual(head[1], -1, accuracy: 1e-6)
        XCTAssertEqual(head[2], 1, accuracy: 1e-6)
        XCTAssertEqual(head[3], 0, accuracy: 1e-6)
    }

    /// Attention is bidirectional and unmasked, so a token's output depends on the
    /// token AFTER it too. A causal port would leave the first row untouched.
    func testAttentionSeesLaterTokens() {
        let qkv: [[Float]] = [
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        ]
        // One head, head_dim 4, no rotation.
        let output = DeepSeekV41VisionForward.attention(
            qkv: qkv, heads: 1, headDim: 4, ropeDim: 2, cosine: [], sine: []
        )
        XCTAssertEqual(output.count, 2)
        // Both tokens have attention weights, so neither row is a copy of its own
        // value alone.
        XCTAssertNotEqual(output[0], Array(qkv[0][8..<12]))
        XCTAssertNotEqual(output[0], output[1])
    }

    /// `F.unfold` after padding right and bottom: a cell is channel-major and each
    /// channel's block is row-major, so a transposed port is visible immediately.
    func testPixelUnshufflePadsAndKeepsChannelsOuter() {
        // 2x3 grid of 2-wide rows; ratio 2 pads the column count to 4.
        let rows: [[Float]] = [
            [1, 10], [2, 20], [3, 30],
            [4, 40], [5, 50], [6, 60]
        ]
        let cells = DeepSeekV41VisionForward.pixelUnshuffle(
            rows, patchRows: 2, patchColumns: 3, ratio: 2, width: 2
        )
        XCTAssertEqual(cells.count, 2, "1 aligner row of 2 cells")
        // Cell 0 covers rows 0..1 and columns 0..1:
        // channel 0 -> (0,0),(0,1),(1,0),(1,1) = 1, 2, 4, 5.
        XCTAssertEqual(Array(cells[0][0..<4]), [1, 2, 4, 5])
        // channel 1 -> 10, 20, 40, 50.
        XCTAssertEqual(Array(cells[0][4..<8]), [10, 20, 40, 50])
        // Cell 1's second column pair is padding, so the zero columns stay zero.
        XCTAssertEqual(Array(cells[1][0..<4]), [3, 0, 6, 0])
    }

    /// A tower missing one weight is an error, not a zero matrix: silently binding
    /// zeros would produce plausible-looking embeddings for a broken load.
    func testMissingWeightIsReported() {
        let source = DeepSeekV41DenseWeightSource()
        source.set(matrix(hidden, 3 * patchSize * patchSize), role: .visionPatchEmbed, layer: nil)
        let forward = DeepSeekV41VisionForward(
            spec: spec(), weights: source, textHiddenSize: textHidden
        )
        XCTAssertThrowsError(try forward.encode(patches: patches(), rows: 2, columns: 3)) { error in
            XCTAssertEqual(
                error as? DeepSeekV41VisionForwardError,
                .missingTensor(.visionBlockNorm1, 0)
            )
        }
    }
}
