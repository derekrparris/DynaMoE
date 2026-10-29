//
//  DeepSeekV41VisionForward.swift
//  DynaMoE
//
//  The vision tower as a CPU reference: patches in, one language-model-width row
//  per aligner grid cell out.
//
//  Ported from the checkpoint's `inference/vision.py` (`ViT`, `Block`,
//  `Attention`, `MLP`, `Aligner`, `PatchEmbed`) and `model.py`'s
//  `merge_image_embeddings`. Three details are easy to get wrong and each one
//  changes every output row:
//
//    * the RoPE here is 2D and SHARED across heads: one frequency vector per patch
//      token, built from the row and column indices, applied split-half over the
//      whole head with the height frequencies in the first half of the vector and
//      the width frequencies in the second;
//    * attention is fully bidirectional with no mask and a scale of
//      `1/sqrt(head_dim)`, so every patch sees every other patch;
//    * the aligner unshuffles with `F.unfold` after zero-padding the grid up to a
//      multiple of the downsample ratio, so a cell's vector is channel-major with
//      each channel's block row-major inside it.
//
//  It is a REFERENCE, not the serving path: attention is O(tokens^2) per head per
//  block in scalar Swift, which is fine for the handful of tokens a test uses and
//  far too slow for a 1024-token image. The device path is the kernels; this exists
//  so the device path has something to be compared against.
//

import Foundation

nonisolated public enum DeepSeekV41VisionForwardError: Error, Equatable, CustomStringConvertible {
    case missingTensor(DeepSeekV41TensorRole, Int?)
    case shapeMismatch(String)
    case emptyImage

    public var description: String {
        switch self {
        case .missingTensor(let role, let layer):
            let where_ = layer.map { "vision block \($0) " } ?? ""
            return "vision tower is missing \(where_)\(role.rawValue)"
        case .shapeMismatch(let detail):
            return "vision tower shape mismatch: \(detail)"
        case .emptyImage:
            return "vision tower was given an image with no patches"
        }
    }
}

/// The tower's forward for one image.
nonisolated public struct DeepSeekV41VisionForward {
    public let spec: DeepSeekV41VisionSpec
    public let weights: DeepSeekV41WeightSource
    /// The language model's hidden size, which is the aligner's output width.
    public let textHiddenSize: Int

    /// `RMSNorm.__init__`'s default, which the tower's norms keep: the text tower's
    /// `norm_eps` is 1e-20 and using it here would change every normalized value.
    public static let normalizationEpsilon: Float = 1e-6

    public init(
        spec: DeepSeekV41VisionSpec,
        weights: DeepSeekV41WeightSource,
        textHiddenSize: Int
    ) {
        self.spec = spec
        self.weights = weights
        self.textHiddenSize = textHiddenSize
    }

    /// One row per aligner grid cell, `textHiddenSize` wide, in reading order.
    ///
    /// - Parameters:
    ///   - patches: `rows * columns` vectors of `3 * patchSize^2` values, in the
    ///     grid's reading order, as `DeepSeekV41VisionPreprocess.patches` emits.
    ///   - rows: ViT patch rows (`image height / patchSize`).
    ///   - columns: ViT patch columns (`image width / patchSize`).
    public func encode(
        patches: [[Float]],
        rows: Int,
        columns: Int
    ) throws -> [[Float]] {
        guard rows > 0, columns > 0, patches.count >= rows * columns else {
            throw DeepSeekV41VisionForwardError.emptyImage
        }
        let hidden = spec.hiddenSize
        let heads = max(1, spec.numAttentionHeads)
        guard hidden % heads == 0 else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "hidden size \(hidden) is not divisible by \(heads) attention heads"
            )
        }
        let headDim = hidden / heads
        let ropeDim = headDim / 2
        let tokenCount = rows * columns

        // Patch embedding: `proj(x.flatten(1))`.
        var x = try linear(
            Array(patches.prefix(tokenCount)),
            weight: matrix(.visionPatchEmbed, layer: nil),
            bias: weights.vector(.visionPatchEmbedBias, layer: nil),
            expectedInput: 3 * spec.patchSize * spec.patchSize,
            expectedOutput: hidden
        )

        // One frequency vector per patch token, shared by every head.
        let rotary = Self.rotaryTables(
            rows: rows, columns: columns, ropeDim: ropeDim, theta: Float(spec.ropeTheta)
        )

        for block in 0..<max(0, spec.numHiddenLayers) {
            // x = x + attn(norm1(x))
            let normed = Self.rmsNorm(
                x, weight: try normWeight(.visionBlockNorm1, layer: block, dimension: hidden),
                dimension: hidden
            )
            let qkv = try linear(
                normed,
                weight: matrix(.visionBlockWqkv, layer: block),
                bias: weights.vector(.visionBlockWqkvBias, layer: block),
                expectedInput: hidden,
                expectedOutput: 3 * hidden
            )
            let attended = Self.attention(
                qkv: qkv,
                heads: heads,
                headDim: headDim,
                ropeDim: ropeDim,
                cosine: rotary.cosine,
                sine: rotary.sine
            )
            let projected = try linear(
                attended,
                weight: matrix(.visionBlockWo, layer: block),
                bias: weights.vector(.visionBlockWoBias, layer: block),
                expectedInput: hidden,
                expectedOutput: hidden
            )
            for index in 0..<tokenCount {
                for dimension in 0..<hidden {
                    x[index][dimension] += projected[index][dimension]
                }
            }

            // x = x + mlp(norm2(x)): `w1` emits gate and up halves, and only the
            // gate goes through the activation.
            let normed2 = Self.rmsNorm(
                x, weight: try normWeight(.visionBlockNorm2, layer: block, dimension: hidden),
                dimension: hidden
            )
            let gateAndUp = try linear(
                normed2,
                weight: matrix(.visionBlockW1, layer: block),
                bias: nil,
                expectedInput: hidden,
                expectedOutput: 2 * spec.intermediateSize
            )
            var activated = [[Float]](repeating: [], count: tokenCount)
            for index in 0..<tokenCount {
                var value = [Float](repeating: 0, count: spec.intermediateSize)
                for unit in 0..<spec.intermediateSize {
                    let gate = gateAndUp[index][unit]
                    let up = gateAndUp[index][spec.intermediateSize + unit]
                    value[unit] = Self.silu(gate) * up
                }
                activated[index] = value
            }
            let mlp = try linear(
                activated,
                weight: matrix(.visionBlockW2, layer: block),
                bias: nil,
                expectedInput: spec.intermediateSize,
                expectedOutput: hidden
            )
            for index in 0..<tokenCount {
                for dimension in 0..<hidden {
                    x[index][dimension] += mlp[index][dimension]
                }
            }
        }

        // `ViT.forward` closes with a norm.
        let normalized = Self.rmsNorm(
            x, weight: try normWeight(.visionFinalNorm, layer: nil, dimension: hidden),
            dimension: hidden
        )

        // Aligner: pixel-unshuffle onto the downsample grid, then two linears.
        let cells = Self.pixelUnshuffle(
            normalized,
            patchRows: rows,
            patchColumns: columns,
            ratio: max(1, spec.downsampleRatio),
            width: hidden
        )
        let unshuffledWidth = hidden * max(1, spec.downsampleRatio) * max(1, spec.downsampleRatio)
        let hiddenCells = try linear(
            cells,
            weight: matrix(.visionAlignerW1, layer: nil),
            bias: weights.vector(.visionAlignerW1Bias, layer: nil),
            expectedInput: unshuffledWidth,
            expectedOutput: textHiddenSize
        )
        let activatedCells = hiddenCells.map { row in row.map(Self.gelu) }
        return try linear(
            activatedCells,
            weight: matrix(.visionAlignerW2, layer: nil),
            bias: weights.vector(.visionAlignerW2Bias, layer: nil),
            expectedInput: textHiddenSize,
            expectedOutput: textHiddenSize
        )
    }

    /// The whole span an image occupies in the residual stream: the start marker,
    /// then per grid row `columns` aligner rows followed by a newline marker, then
    /// the end marker. Length matches
    /// `DeepSeekV41VisionLayout.imageTokenCount(columns:rows:)`.
    /// An image's whole span, from the pixels: resize plan, patch grid, tower, and
    /// the delimiter markers.
    ///
    /// This is the same pipeline `image_processor.load_image` plus
    /// `Model.merge_image_embeddings` runs, and its length is what the prompt
    /// expansion and the KV pricing both count, so a mismatch here misaligns the
    /// conversation.
    public func imageSpan(
        rgb: [UInt8],
        width: Int,
        height: Int
    ) throws -> [[Float]] {
        let placement = DeepSeekV41VisionPreprocess.placement(
            width: width, height: height, spec: spec
        )
        let (columns, rows) = placement.patchGrid(patchSize: spec.patchSize)
        guard rows > 0, columns > 0 else {
            throw DeepSeekV41VisionForwardError.emptyImage
        }
        let patches = DeepSeekV41VisionPreprocess.patches(
            rgb: rgb, width: width, height: height, spec: spec
        )
        guard patches.count == rows * columns else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "patchify produced \(patches.count) rows for a \(rows)x\(columns) grid"
            )
        }
        return try imageSpan(patches: patches, rows: rows, columns: columns)
    }

    public func imageSpan(
        patches: [[Float]],
        rows: Int,
        columns: Int
    ) throws -> [[Float]] {
        let cells = try encode(patches: patches, rows: rows, columns: columns)
        let ratio = max(1, spec.downsampleRatio)
        let gridRows = (rows + ratio - 1) / ratio
        let gridColumns = (columns + ratio - 1) / ratio
        guard cells.count >= gridRows * gridColumns else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "aligner produced \(cells.count) rows for a \(gridRows)x\(gridColumns) grid"
            )
        }
        let start = try marker(.visionImageStart)
        let newline = try marker(.visionImageNewline)
        let end = try marker(.visionImageEnd)
        var span: [[Float]] = [start]
        span.reserveCapacity(gridRows * (gridColumns + 1) + 2)
        for row in 0..<gridRows {
            span.append(contentsOf: cells[(row * gridColumns)..<((row + 1) * gridColumns)])
            span.append(newline)
        }
        span.append(end)
        return span
    }

    // MARK: Weights

    private func matrix(
        _ role: DeepSeekV41TensorRole,
        layer: Int?
    ) throws -> DeepSeekV41Matrix {
        guard let matrix = weights.matrix(role, layer: layer, expert: nil) else {
            throw DeepSeekV41VisionForwardError.missingTensor(role, layer)
        }
        return matrix
    }

    /// A norm's learned gain. Required rather than defaulted to ones: the tower's
    /// norms are trained, and a silent 1.0 would leave every block mis-scaled.
    private func normWeight(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        dimension: Int
    ) throws -> [Float] {
        guard let weight = weights.vector(role, layer: layer), weight.count == dimension else {
            throw DeepSeekV41VisionForwardError.missingTensor(role, layer)
        }
        return weight
    }

    private func marker(_ role: DeepSeekV41TensorRole) throws -> [Float] {
        guard let vector = weights.vector(role, layer: nil), vector.count == textHiddenSize else {
            throw DeepSeekV41VisionForwardError.missingTensor(role, nil)
        }
        return vector
    }

    /// `Linear`: one row of `weight` per output unit, plus the optional bias.
    private func linear(
        _ inputs: [[Float]],
        weight: DeepSeekV41Matrix,
        bias: [Float]?,
        expectedInput: Int,
        expectedOutput: Int
    ) throws -> [[Float]] {
        guard weight.rows == expectedOutput, weight.columns == expectedInput else {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "expected \(expectedOutput)x\(expectedInput), got \(weight.rows)x\(weight.columns)"
            )
        }
        if let bias, bias.count != expectedOutput {
            throw DeepSeekV41VisionForwardError.shapeMismatch(
                "bias has \(bias.count) values, expected \(expectedOutput)"
            )
        }
        return inputs.map { input in
            guard input.count == expectedInput else {
                return [Float](repeating: 0, count: expectedOutput)
            }
            var output = weight.apply(input)
            if let bias, output.count == bias.count {
                for unit in 0..<bias.count { output[unit] += bias[unit] }
            }
            return output
        }
    }

    // MARK: Numerics

    /// `get_vision_cos_sin`: `dim/2` inverse frequencies, multiplied by the token's
    /// row index into the first half of the vector and its column index into the
    /// second, so one rotation encodes both axes.
    nonisolated public static func rotaryTables(
        rows: Int,
        columns: Int,
        ropeDim: Int,
        theta: Float
    ) -> (cosine: [[Float]], sine: [[Float]]) {
        let pairs = max(0, ropeDim / 2)
        var inverse: [Float] = []
        inverse.reserveCapacity(pairs)
        for index in 0..<pairs {
            // `theta ** (2 * index / dim)` with `dim == ropeDim`.
            let exponent = Float(2 * index) / Float(max(1, ropeDim))
            inverse.append(1 / pow(max(theta, 1), exponent))
        }
        var cosine: [[Float]] = []
        var sine: [[Float]] = []
        cosine.reserveCapacity(rows * columns)
        sine.reserveCapacity(rows * columns)
        for row in 0..<max(0, rows) {
            for column in 0..<max(0, columns) {
                var angles = [Float](repeating: 0, count: ropeDim)
                for index in 0..<pairs {
                    angles[index] = Float(row) * inverse[index]
                    if pairs + index < ropeDim {
                        angles[pairs + index] = Float(column) * inverse[index]
                    }
                }
                cosine.append(angles.map(cos))
                sine.append(angles.map(sin))
            }
        }
        return (cosine, sine)
    }

    /// `apply_rotary`: split-half rotation over the whole head. `x1` is the first
    /// `headDim/2` values, `x2` the rest, and both rotate against the same angles.
    nonisolated public static func applyRotary(
        _ head: inout [Float],
        cosine: [Float],
        sine: [Float]
    ) {
        let half = head.count / 2
        guard half > 0, cosine.count >= half, sine.count >= half else { return }
        var rotated = head
        for index in 0..<half {
            let first = head[index]
            let second = head[half + index]
            rotated[index] = first * cosine[index] - second * sine[index]
            rotated[half + index] = second * cosine[index] + first * sine[index]
        }
        head = rotated
    }

    /// Full bidirectional attention over every patch, per head, scale
    /// `1/sqrt(head_dim)`. `qkv` holds one `3 * hidden` row per token.
    nonisolated public static func attention(
        qkv: [[Float]],
        heads: Int,
        headDim: Int,
        ropeDim: Int,
        cosine: [[Float]],
        sine: [[Float]]
    ) -> [[Float]] {
        let tokens = qkv.count
        let hidden = heads * headDim
        guard tokens > 0, heads > 0, headDim > 0 else { return qkv }
        let scale = 1 / sqrt(Float(headDim))

        // Per head: rotate q and k, then attend over all tokens. v is untouched.
        var output = [[Float]](repeating: [Float](repeating: 0, count: hidden), count: tokens)
        for head in 0..<heads {
            var queries: [[Float]] = []
            var keys: [[Float]] = []
            var values: [[Float]] = []
            queries.reserveCapacity(tokens)
            keys.reserveCapacity(tokens)
            values.reserveCapacity(tokens)
            for token in 0..<tokens {
                let row = qkv[token]
                guard row.count >= 3 * hidden else {
                    return output
                }
                let base = head * headDim
                var query = Array(row[base..<(base + headDim)])
                var key = Array(row[(hidden + base)..<(hidden + base + headDim)])
                // The angles cover `ropeDim == headDim / 2`, exactly one half of the
                // head, which is what the split-half rotation consumes.
                if !cosine.isEmpty, !sine.isEmpty {
                    let angles = cosine[min(token, cosine.count - 1)]
                    let sines = sine[min(token, sine.count - 1)]
                    applyRotary(
                        &query, cosine: Array(angles.prefix(ropeDim)),
                        sine: Array(sines.prefix(ropeDim))
                    )
                    applyRotary(
                        &key, cosine: Array(angles.prefix(ropeDim)),
                        sine: Array(sines.prefix(ropeDim))
                    )
                }
                queries.append(query)
                keys.append(key)
                values.append(Array(row[(2 * hidden + base)..<(2 * hidden + base + headDim)]))
            }
            for token in 0..<tokens {
                var scores = [Float](repeating: 0, count: tokens)
                for other in 0..<tokens {
                    var dot: Float = 0
                    for dimension in 0..<headDim {
                        dot += queries[token][dimension] * keys[other][dimension]
                    }
                    scores[other] = dot * scale
                }
                let weights = softmax(scores)
                let base = head * headDim
                for other in 0..<tokens {
                    let weight = weights[other]
                    guard weight != 0 else { continue }
                    for dimension in 0..<headDim {
                        output[token][base + dimension] += weight * values[other][dimension]
                    }
                }
            }
        }
        return output
    }

    nonisolated public static func softmax(_ values: [Float]) -> [Float] {
        guard let maximum = values.max() else { return [] }
        var exponentials = values.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        guard total > 0 else { return [Float](repeating: 0, count: values.count) }
        for index in exponentials.indices { exponentials[index] /= total }
        return exponentials
    }

    /// `RMSNorm.forward`: normalize over the last axis with `eps`, then scale.
    nonisolated public static func rmsNorm(
        _ rows: [[Float]],
        weight: [Float]?,
        dimension: Int
    ) -> [[Float]] {
        let gain = (weight?.count == dimension) ? weight : nil
        return rows.map { row in
            guard row.count == dimension, dimension > 0 else {
                return [Float](repeating: 0, count: max(0, dimension))
            }
            var squares: Float = 0
            for value in row { squares += value * value }
            let rstd = 1 / sqrt(squares / Float(dimension) + normalizationEpsilon)
            var output = [Float](repeating: 0, count: dimension)
            for axis in 0..<dimension {
                output[axis] = row[axis] * rstd * (gain?[axis] ?? 1)
            }
            return output
        }
    }

    /// `F.unfold` after `F.pad(x, (0, -n_w % r, 0, -n_h % r))`: cells come out
    /// channel-major, and inside a channel the `ratio x ratio` block is row-major.
    nonisolated public static func pixelUnshuffle(
        _ values: [[Float]],
        patchRows: Int,
        patchColumns: Int,
        ratio: Int,
        width: Int
    ) -> [[Float]] {
        let r = max(1, ratio)
        guard patchRows > 0, patchColumns > 0, width > 0,
              values.count >= patchRows * patchColumns else { return [] }
        // `F.pad` pads right and bottom, so the grid grows upward to the next
        // multiple of the ratio rather than being cropped.
        let paddedRows = ((patchRows + r - 1) / r) * r
        let paddedColumns = ((patchColumns + r - 1) / r) * r
        let cellVolume = width * r * r
        var cells: [[Float]] = []
        cells.reserveCapacity((paddedRows / r) * (paddedColumns / r))
        for cellRow in 0..<(paddedRows / r) {
            for cellColumn in 0..<(paddedColumns / r) {
                var cell = [Float](repeating: 0, count: cellVolume)
                for channel in 0..<width {
                    for innerRow in 0..<r {
                        for innerColumn in 0..<r {
                            let sourceRow = cellRow * r + innerRow
                            let sourceColumn = cellColumn * r + innerColumn
                            guard sourceRow < patchRows, sourceColumn < patchColumns else { continue }
                            let source = values[sourceRow * patchColumns + sourceColumn]
                            guard source.count == width else { continue }
                            cell[channel * r * r + innerRow * r + innerColumn] = source[channel]
                        }
                    }
                }
                cells.append(cell)
            }
        }
        return cells
    }

    nonisolated public static func gelu(_ value: Float) -> Float {
        // `F.gelu` default is the exact erf form, not the tanh approximation.
        0.5 * value * (1 + erf(value / Float(2).squareRoot()))
    }

    nonisolated public static func silu(_ value: Float) -> Float {
        value / (1 + exp(-value))
    }
}
