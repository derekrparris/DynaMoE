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

    // MARK: The GEMM forms

    /// A linear is one dot product per output row, and the tower runs the same thing
    /// as one GEMM over the whole block of tokens. The two must agree to float
    /// rounding, or the tower's numbers are no longer the definition's numbers.
    func testTheBatchedLinearMatchesItsDefinition() {
        // An odd shape on purpose: three tokens into 7 rows of 5, so a transposed
        // matrix or an `lda` that assumed one token per row cannot cancel out.
        let weight = matrix(7, 5)
        let inputs: [[Float]] = (0..<3).map { token in
            (0..<5).map { dimension in entry(token, dimension) + Float(token) * 0.25 }
        }
        let bias = vector(7, offset: 2)
        let batched = DeepSeekV41VisionForward.linearBatched(inputs, weight: weight, bias: bias)
        let scalar = DeepSeekV41VisionForward.linearRowByRow(inputs, weight: weight, bias: bias)
        XCTAssertEqual(batched.count, scalar.count)
        XCTAssertEqual(batched.first?.count, 7)
        for token in inputs.indices {
            for unit in 0..<7 {
                XCTAssertEqual(
                    batched[token][unit], scalar[token][unit], accuracy: 1e-5,
                    "token \(token) unit \(unit)"
                )
            }
        }
    }

    /// A block of ONE token is the decode shape, and it goes through the same GEMM
    /// rather than a separate path, so it has to agree with the definition too.
    func testTheBatchedLinearHandlesASingleToken() {
        let weight = matrix(6, 4)
        let inputs: [[Float]] = [[0.5, -0.25, 0.75, 0]]
        let batched = DeepSeekV41VisionForward.linearBatched(inputs, weight: weight, bias: nil)
        let scalar = DeepSeekV41VisionForward.linearRowByRow(inputs, weight: weight, bias: nil)
        XCTAssertEqual(batched.count, 1)
        for unit in 0..<6 {
            XCTAssertEqual(batched[0][unit], scalar[0][unit], accuracy: 1e-6, "unit \(unit)")
        }
    }

    /// Attention as one GEMM for the scores and one for the weighted sum. The scale
    /// rides on the score GEMM's alpha, so a missing `1/sqrt(d)` shows up here.
    func testTheBatchedAttentionMatchesItsDefinition() {
        let inputs: [[Float]] = (0..<4).map { token in
            (0..<(3 * hidden)).map { dimension in entry(token, dimension) + Float(token) * 0.1 }
        }
        let headDim = hidden / heads
        let rotary = DeepSeekV41VisionForward.rotaryTables(
            rows: 2, columns: 2, ropeDim: headDim / 2, theta: 100
        )
        let scalar = DeepSeekV41VisionForward.attention(
            qkv: inputs, heads: heads, headDim: headDim, ropeDim: headDim / 2,
            cosine: rotary.cosine, sine: rotary.sine
        )
        let batched = DeepSeekV41VisionForward.attentionBatched(
            qkv: inputs, heads: heads, headDim: headDim, ropeDim: headDim / 2,
            cosine: rotary.cosine, sine: rotary.sine
        )
        XCTAssertEqual(batched.count, scalar.count)
        for token in inputs.indices {
            for dimension in 0..<hidden {
                XCTAssertEqual(
                    batched[token][dimension], scalar[token][dimension], accuracy: 1e-5,
                    "token \(token) dimension \(dimension)"
                )
            }
        }
    }

    /// No rotary table is the "this prompt does not rotate" path, and it must not be
    /// mistaken for a rotation by zero (which would collapse every q and k pair).
    func testTheBatchedAttentionWithoutRotationMatchesItsDefinition() {
        let inputs: [[Float]] = (0..<3).map { token in
            (0..<(3 * hidden)).map { dimension in Float((token + dimension) % 4) * 0.5 - 0.5 }
        }
        let headDim = hidden / heads
        let scalar = DeepSeekV41VisionForward.attention(
            qkv: inputs, heads: heads, headDim: headDim, ropeDim: headDim / 2,
            cosine: [], sine: []
        )
        let batched = DeepSeekV41VisionForward.attentionBatched(
            qkv: inputs, heads: heads, headDim: headDim, ropeDim: headDim / 2,
            cosine: [], sine: []
        )
        XCTAssertEqual(batched.count, scalar.count)
        for token in inputs.indices {
            for dimension in 0..<hidden {
                XCTAssertEqual(
                    batched[token][dimension], scalar[token][dimension], accuracy: 1e-6,
                    "token \(token) dimension \(dimension)"
                )
            }
        }
    }

    /// A patch row that is not the projection's width would make the GEMM read past
    /// its own values, so the tower refuses it instead of zeroing that row.
    func testARaggedPatchRowIsRefused() throws {
        var ragged = patches()
        ragged[2] = Array(ragged[2].dropLast())
        XCTAssertThrowsError(try forward().encode(patches: ragged, rows: 2, columns: 3)) { error in
            guard case .shapeMismatch(let detail)? = error as? DeepSeekV41VisionForwardError else {
                return XCTFail("expected a shape mismatch, got \(error)")
            }
            XCTAssertTrue(detail.contains("input row"), detail)
        }
    }
}

/// The tower's checkpoint weights are BF16 and the widening is a bit shift, not a
/// conversion: a bfloat16 IS the high half of the float it came from. These bytes
/// are what the shards hold, so a decoder that rounds or sign-extends loses values.
final class DeepSeekV41VisionWeightDecodeTests: XCTestCase {

    private func decode(_ halves: [UInt16], dtype: String = "BF16") -> [Float] {
        let bytes = halves.withUnsafeBytes { Array($0) }
        return bytes.withUnsafeBytes {
            DeepSeekV41VisionWeights.decode($0, dtype: dtype, count: halves.count)
        }
    }

    func testBF16IsTheHighHalfOfItsFloat() {
        // 1.0 is 0x3F80, -2.0 is 0xC000, 0.5 is 0x3F00, and zero is zero.
        XCTAssertEqual(decode([0x3F80, 0xC000, 0x3F00, 0x0000]), [1, -2, 0.5, 0])
    }

    /// The truncated low half is NOT reconstructed: 1.00390625 (0x3F808000) is
    /// published as 0x3F80, and widening it must give exactly 1, not the original.
    func testTheDiscardedLowHalfIsNotReinvented() {
        XCTAssertEqual(decode([0x3F80])[0], 1)
        XCTAssertEqual(Float(bitPattern: 0x3F808000), 1 + 1 / 256)
        XCTAssertEqual(decode([0x0080])[0], Float.leastNormalMagnitude)
    }

    func testFP32PassesThroughBitForBit() {
        let words: [UInt32] = [Float(1.5).bitPattern, Float(-0.25).bitPattern]
        let bytes = words.withUnsafeBytes { Array($0) }
        let decoded = bytes.withUnsafeBytes {
            DeepSeekV41VisionWeights.decode($0, dtype: "F32", count: words.count)
        }
        XCTAssertEqual(decoded, [1.5, -0.25])
    }

    /// A short buffer is a broken read, not a licence to read past it.
    func testAShortBufferYieldsZeros() {
        let bytes = [UInt8](repeating: 0x80, count: 4)
        let decoded = bytes.withUnsafeBytes {
            DeepSeekV41VisionWeights.decode($0, dtype: "BF16", count: 8)
        }
        XCTAssertEqual(decoded.count, 8)
        XCTAssertEqual(decoded, [Float](repeating: 0, count: 8))
    }

    func testAnUnknownDtypeIsWidenedAsBF16() {
        XCTAssertEqual(DeepSeekV41VisionWeights.bytesPerValue("BF16"), 2)
        XCTAssertEqual(DeepSeekV41VisionWeights.bytesPerValue("F32"), 4)
        XCTAssertNil(DeepSeekV41VisionWeights.bytesPerValue("I8"))
    }

    /// FP16 and BF16 are both two bytes and are NOT the same encoding. `0x3C00`
    /// is 1.0 as an IEEE half and 0.0078125 as the high half of a float, so a
    /// decoder that widens an accepted FP16 tensor with a shift corrupts every
    /// projection it reads, silently and in the direction of "the model answers
    /// something plausible about the wrong picture".
    func testFP16IsDecodedAsFP16() {
        // 1.0, -2.0, 0.5, 0.0 in IEEE half bit patterns.
        let halves: [UInt16] = [0x3C00, 0xC000, 0x3800, 0x0000]
        XCTAssertEqual(decode(halves, dtype: "F16"), [1, -2, 0.5, 0])
        XCTAssertEqual(decode(halves, dtype: "FLOAT16"), [1, -2, 0.5, 0])
        // The same bytes read as bf16 are a different number, and that is the
        // whole point: the two readings must not be interchangeable.
        XCTAssertEqual(decode([0x3C00], dtype: "BF16")[0], Float(bitPattern: 0x3C000000))
        XCTAssertNotEqual(decode([0x3C00], dtype: "BF16")[0], 1)
    }

    /// Values with no exact bf16 representation are where the two decoders
    /// diverge hardest: 1/3 rounds to a different float, and the smallest
    /// subnormal half is 2^-24 where the shift reading gives 3.6e-41.
    func testFP16CarriesMantissaAndSubnormalsThatBF16WouldDrop() {
        let third = Float16(1.0 / 3.0)
        XCTAssertEqual(decode([third.bitPattern], dtype: "F16")[0], Float(third))
        XCTAssertNotEqual(
            decode([third.bitPattern], dtype: "F16")[0],
            decode([third.bitPattern], dtype: "BF16")[0]
        )
        XCTAssertEqual(decode([0x0001], dtype: "F16")[0], Float(bitPattern: 0x33800000))
        // Negative zero and both infinities survive the conversion.
        XCTAssertEqual(decode([0x8000], dtype: "F16")[0].bitPattern, Float(-0.0).bitPattern)
        XCTAssertEqual(decode([0x7C00], dtype: "F16")[0], Float.infinity)
        XCTAssertEqual(decode([0xFC00], dtype: "F16")[0], -Float.infinity)
    }

    func testTheStorageFormatClassifiesEverySpellingTheTowerPublishes() {
        XCTAssertEqual(DeepSeekV41VisionWeights.storageFormat("BF16"), .bf16)
        XCTAssertEqual(DeepSeekV41VisionWeights.storageFormat("F16"), .fp16)
        XCTAssertEqual(DeepSeekV41VisionWeights.storageFormat("FLOAT16"), .fp16)
        XCTAssertEqual(DeepSeekV41VisionWeights.storageFormat("F32"), .fp32)
        XCTAssertNil(DeepSeekV41VisionWeights.storageFormat("I8"))
        // A bf16 spelling must not be captured by the "16" half's rules: the
        // order of the two tests is what keeps `BF16` out of the FP16 branch.
        XCTAssertEqual(DeepSeekV41VisionWeights.bytesPerValue("F16"), 2)
        XCTAssertEqual(DeepSeekV41VisionWeights.bytesPerValue("BF16"), 2)
    }
}
