//
//  Inkling2VisionForward.swift
//  DynaMoE
//
//  The still-image tower as a HOST reference graph.
//
//  Transcribed from the released `Inkling2VisionModel` rather than inferred
//  from the architecture name, because four of this tower's rules are not the
//  generic ViT's:
//
//    * the picture arrives already cut into LANGUAGE-token cells (40x40 pixels,
//      2x2 native patches of 20), and the patch order inside a cell is
//      (row-block, col-block, row-within, col-within) -- the same order the
//      merger later concatenates, so a transposed order still runs;
//    * every image starts with ONE class token that takes part in attention
//      but is dropped before the merger;
//    * the learned 32x32 position table is bilinearly resampled per patch with
//      align_corners=false, half-pixel centres and border padding -- not a
//      bicubic resize and not a crop;
//    * the 2D RoPE is axial over interleaved pairs: pairs 0..<dim/4 carry the
//      column coordinate, pairs dim/4..<dim/2 the row, each pair rotating with
//      its own frequency. It is NOT a rotate-half layout.
//
//  Everything is FP32 over the checkpoint's BF16 tensors, which is exactly the
//  reference's own FP32 load (BF16 -> FP32 is a bit shift), so this graph is
//  expected to match its goldens to rounding and is held to `1e-4` rather than
//  to the text path's BF16 band.
//

import Foundation

/// Where a picture's rows land for a given cell grid.
nonisolated public struct Inkling2VisionLayout: Sendable, Equatable {
    public let cellRows: Int
    public let cellColumns: Int
    public let merge: Int
    public let patchSize: Int
    public let channels: Int

    public init(cellRows: Int, cellColumns: Int, spec: Inkling2VisionSpec) {
        self.cellRows = max(1, cellRows)
        self.cellColumns = max(1, cellColumns)
        self.merge = max(1, spec.adapterMergeSize)
        self.patchSize = max(1, spec.nativePatchSize)
        self.channels = 3
    }

    public var cellSize: Int { merge * patchSize }
    public var cellCount: Int { cellRows * cellColumns }
    public var patchRows: Int { cellRows * merge }
    public var patchColumns: Int { cellColumns * merge }
    public var patchCount: Int { patchRows * patchColumns }
    /// One class token per image, then every patch.
    public var sequenceLength: Int { patchCount + 1 }
    public var pixelCount: Int { cellCount * cellSize * cellSize * channels }

    /// The patch grid coordinate of the patch at `index` (0-based, cell-major
    /// then row-block then col-block), with the reference's 1-based numbering
    /// for patches and `(0, 0)` for the class token.
    public func position(ofPatch index: Int) -> (row: Int, column: Int) {
        let perCell = merge * merge
        let cell = index / perCell
        let within = index % perCell
        let rowBlock = within / merge
        let columnBlock = within % merge
        return (
            (cell / cellColumns) * merge + rowBlock + 1,
            (cell % cellColumns) * merge + columnBlock + 1
        )
    }

    /// The pixel offset of one 20x20 patch's first value inside `pixelValues`.
    public func pixelOffset(cell: Int, rowBlock: Int, columnBlock: Int) -> Int {
        let perCell = cellSize * cellSize * channels
        return cell * perCell
            + rowBlock * patchSize * cellSize * channels
            + columnBlock * patchSize * channels
    }
}

nonisolated public enum Inkling2VisionError: Error, CustomStringConvertible {
    case invalidInput(String)
    case missingWeight(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .invalidInput(let reason): return "Inkling 2 vision input: \(reason)"
        case .missingWeight(let name): return "Inkling 2 vision weight: \(name)"
        case .shapeMismatch(let reason): return "Inkling 2 vision shape: \(reason)"
        }
    }
}

/// The host tower. `encode` returns the rows the language model consumes: one
/// `textHiddenSize`-wide vector per language-token cell, in cell order.
nonisolated public final class Inkling2VisionForward {
    public let config: Inkling2Config
    public let spec: Inkling2VisionSpec
    private let weights: Inkling2WeightSource

    /// Fires after each named stage with that stage's rows. Named so a test can
    /// compare one layer without keeping the whole tower's intermediates.
    public var onStage: ((String, [[Float]]) -> Void)?

    private struct Block {
        let ln1Weight: [Float]
        let ln1Bias: [Float]
        let ln2Weight: [Float]
        let ln2Bias: [Float]
        let scale1: [Float]
        let scale2: [Float]
        let qWeight: [Float]
        let qBias: [Float]
        let kWeight: [Float]
        let kBiass: [Float]
        let vWeight: [Float]
        let vBias: [Float]
        let oWeight: [Float]
        let oBias: [Float]
        let fcWeight: [Float]
        let fcBias: [Float]
        let projWeight: [Float]
        let projBias: [Float]
    }

    private let patchEmbedding: [Float]
    private let positionEmbedding: [Float]
    private let classEmbedding: [Float]
    private let clsPositionEmbedding: [Float]
    private let preNormWeight: [Float]
    private let preNormBias: [Float]
    private let blocks: [Block]
    private let inWeight: [Float]
    private let outWeight: [Float]
    private let hidden: Int
    private let heads: Int
    private let headDim: Int

    public init(config: Inkling2Config, weights: Inkling2WeightSource) throws {
        guard let spec = config.vision else {
            throw Inkling2VisionError.invalidInput("the checkpoint publishes no vision_config")
        }
        self.config = config
        self.spec = spec
        self.weights = weights
        self.hidden = spec.hiddenSize
        self.heads = spec.numAttentionHeads
        guard heads > 0, hidden % heads == 0 else {
            throw Inkling2VisionError.invalidInput("\(hidden) hidden / \(heads) heads is not a whole head")
        }
        self.headDim = hidden / heads

        func matrix(_ role: Inkling2TensorRole, _ layer: Int? = nil) throws -> [Float] {
            let shape = try weights.shape(role, layer: layer)
            let count = shape.reduce(1, *)
            let values = try weights.matrix(role, layer: layer)
            guard values.count == count else {
                throw Inkling2VisionError.shapeMismatch(
                    "\(role.rawValue)\(layer.map { " \($0)" } ?? "") reads \(values.count) of \(count) values"
                )
            }
            return values
        }

        patchEmbedding = try matrix(.visionPatchEmbedding)
        positionEmbedding = try matrix(.visionPositionEmbedding)
        classEmbedding = try matrix(.visionClassEmbedding)
        clsPositionEmbedding = try matrix(.visionClsPositionEmbedding)
        preNormWeight = try matrix(.visionNormPreWeight)
        preNormBias = try matrix(.visionNormPreBias)
        var loaded: [Block] = []
        loaded.reserveCapacity(spec.numHiddenLayers)
        for layer in 0..<spec.numHiddenLayers {
            loaded.append(Block(
                ln1Weight: try matrix(.visionBlockLn1Weight, layer),
                ln1Bias: try matrix(.visionBlockLn1Bias, layer),
                ln2Weight: try matrix(.visionBlockLn2Weight, layer),
                ln2Bias: try matrix(.visionBlockLn2Bias, layer),
                scale1: try matrix(.visionBlockScale1, layer),
                scale2: try matrix(.visionBlockScale2, layer),
                qWeight: try matrix(.visionBlockQProjWeight, layer),
                qBias: try matrix(.visionBlockQProjBias, layer),
                kWeight: try matrix(.visionBlockKProjWeight, layer),
                kBiass: try matrix(.visionBlockKProjBias, layer),
                vWeight: try matrix(.visionBlockVProjWeight, layer),
                vBias: try matrix(.visionBlockVProjBias, layer),
                oWeight: try matrix(.visionBlockOutProjWeight, layer),
                oBias: try matrix(.visionBlockOutProjBias, layer),
                fcWeight: try matrix(.visionBlockFcWeight, layer),
                fcBias: try matrix(.visionBlockFcBias, layer),
                projWeight: try matrix(.visionBlockProjWeight, layer),
                projBias: try matrix(.visionBlockProjBias, layer)
            ))
        }
        blocks = loaded
        inWeight = try matrix(.visionProjIn)
        outWeight = try matrix(.visionProjOut)
    }

    // MARK: Entry point

    /// `pixelValues` is the image processor's tensor: one 40x40x3 cell per
    /// language token, cells in raster order, rows of 40, channel last, values
    /// in `[-1, 1]` after `(x / 255 - 0.5) / 0.5`.
    public func encode(pixelValues: [Float], layout: Inkling2VisionLayout) throws -> [[Float]] {
        guard pixelValues.count == layout.pixelCount else {
            throw Inkling2VisionError.invalidInput(
                "\(pixelValues.count) pixel values for a \(layout.cellRows)x\(layout.cellColumns) cell grid "
                    + "(expected \(layout.pixelCount))"
            )
        }
        let patchCount = layout.patchCount
        guard patchCount > 0 else {
            throw Inkling2VisionError.invalidInput("a picture with no patches")
        }

        // 1. Patch projection. One patch is `patchSize x patchSize x channels`
        // read from its own row/column block, in (row-block, col-block) order.
        var patches = [[Float]](repeating: [], count: patchCount)
        let inFeatures = layout.patchSize * layout.patchSize * layout.channels
        guard patchEmbedding.count == hidden * inFeatures else {
            throw Inkling2VisionError.shapeMismatch(
                "patch embedding is \(patchEmbedding.count), expected \(hidden * inFeatures)"
            )
        }
        for cell in 0..<layout.cellCount {
            for rowBlock in 0..<layout.merge {
                for columnBlock in 0..<layout.merge {
                    let index = (cell * layout.merge + rowBlock) * layout.merge + columnBlock
                    var vector = [Float](repeating: 0, count: inFeatures)
                    let base = layout.pixelOffset(cell: cell, rowBlock: rowBlock, columnBlock: columnBlock)
                    for row in 0..<layout.patchSize {
                        for column in 0..<layout.patchSize {
                            for channel in 0..<layout.channels {
                                let source = base
                                    + row * layout.cellSize * layout.channels
                                    + column * layout.channels + channel
                                let target = (row * layout.patchSize + column) * layout.channels + channel
                                vector[target] = pixelValues[source]
                            }
                        }
                    }
                    patches[index] = Inkling2VisionMath.linear(
                        patchEmbedding, rows: hidden, columns: inFeatures, vector
                    )
                }
            }
        }

        // 2. Position embeddings: bilinear resample of the learned square table,
        // one gather per patch, weighted 2x2.
        let side = spec.posembGridSize
        for index in 0..<patchCount {
            let position = layout.position(ofPatch: index)
            let (tapIndices, tapWeights) = Inkling2VisionMath.interpolationTaps(
                row: position.row - 1, column: position.column - 1,
                rows: layout.patchRows, columns: layout.patchColumns, side: side
            )
            var embedding = [Float](repeating: 0, count: hidden)
            for tap in 0..<tapIndices.count {
                let base = tapIndices[tap] * hidden
                let weight = tapWeights[tap]
                for dimension in 0..<hidden {
                    embedding[dimension] += positionEmbedding[base + dimension] * weight
                }
            }
            for dimension in 0..<hidden { patches[index][dimension] += embedding[dimension] }
        }

        // 3. The class token in front of this image's patches.
        var sequence: [[Float]] = []
        sequence.reserveCapacity(patchCount + 1)
        var classToken = [Float](repeating: 0, count: hidden)
        for dimension in 0..<hidden {
            classToken[dimension] = classEmbedding[dimension] + clsPositionEmbedding[dimension]
        }
        sequence.append(classToken)
        sequence.append(contentsOf: patches)
        onStage?("embeddings", sequence)

        // 4. Pre-layernorm.
        var hiddenStates = sequence.map {
            Inkling2VisionMath.layerNorm($0, weight: preNormWeight, bias: preNormBias, eps: spec.layerNormEps)
        }
        onStage?("preLayernorm", hiddenStates)

        // 5. RoPE angles, one per token: the class token sits at (0, 0) and is
        // therefore unrotated.
        var positions: [(row: Int, column: Int)] = [(0, 0)]
        for index in 0..<patchCount { positions.append(layout.position(ofPatch: index)) }
        let angles = positions.map {
            Inkling2VisionMath.ropeAngles(
                row: $0.row, column: $0.column, headDim: headDim, theta: 10_000
            )
        }

        for (index, block) in blocks.enumerated() {
            hiddenStates = runBlock(block, states: hiddenStates, angles: angles)
            onStage?("layer\(index)", hiddenStates)
        }

        // 6. Merger: the patch rows only, one merged cell at a time, in the same
        // (row-block, col-block) order the patches were built in.
        let merge = layout.merge
        let mergedWidth = merge * merge * hidden
        var merged: [[Float]] = []
        merged.reserveCapacity(layout.cellCount)
        for cell in 0..<layout.cellCount {
            var row = [Float](repeating: 0, count: mergedWidth)
            for slot in 0..<(merge * merge) {
                let patch = cell * merge * merge + slot
                // Sequence index: the class token precedes every patch.
                let source = hiddenStates[patch + 1]
                let base = slot * hidden
                for dimension in 0..<hidden { row[base + dimension] = source[dimension] }
            }
            var projected = Inkling2VisionMath.linear(
                inWeight, rows: spec.projectionHiddenDim, columns: mergedWidth, row
            )
            for value in projected.indices { projected[value] = Inkling2VisionMath.gelu(projected[value]) }
            merged.append(Inkling2VisionMath.linear(
                outWeight, rows: spec.textHiddenSize, columns: spec.projectionHiddenDim, projected
            ))
        }
        onStage?("pooler", merged)
        return merged
    }

    // MARK: One layer

    private func runBlock(
        _ block: Block,
        states: [[Float]],
        angles: [(cos: [Float], sin: [Float])]
    ) -> [[Float]] {
        let count = states.count
        let scaling = 1 / Float(headDim).squareRoot()

        var query = [[Float]](repeating: [], count: count)
        var key = [[Float]](repeating: [], count: count)
        var value = [[Float]](repeating: [], count: count)
        for token in 0..<count {
            let normed = Inkling2VisionMath.layerNorm(
                states[token], weight: block.ln1Weight, bias: block.ln1Bias, eps: spec.layerNormEps
            )
            var q = Inkling2VisionMath.linear(block.qWeight, rows: hidden, columns: hidden, normed)
            var k = Inkling2VisionMath.linear(block.kWeight, rows: hidden, columns: hidden, normed)
            for index in 0..<hidden {
                q[index] += block.qBias[index]
                k[index] += block.kBiass[index]
            }
            for head in 0..<heads {
                Inkling2VisionMath.rotateInPlace(
                    &q, head: head, headDim: headDim, cos: angles[token].cos, sin: angles[token].sin
                )
                Inkling2VisionMath.rotateInPlace(
                    &k, head: head, headDim: headDim, cos: angles[token].cos, sin: angles[token].sin
                )
            }
            var v = Inkling2VisionMath.linear(block.vWeight, rows: hidden, columns: hidden, normed)
            for index in 0..<hidden { v[index] += block.vBias[index] }
            query[token] = q
            key[token] = k
            value[token] = v
        }

        // Bidirectional attention over the image's own tokens.
        var attention = [[Float]](repeating: [Float](repeating: 0, count: hidden), count: count)
        var scores = [Float](repeating: 0, count: count)
        for head in 0..<heads {
            let base = head * headDim
            for token in 0..<count {
                for other in 0..<count {
                    var dot: Float = 0
                    for dimension in 0..<headDim {
                        dot += query[token][base + dimension] * key[other][base + dimension]
                    }
                    scores[other] = dot * scaling
                }
                let maximum = scores.max() ?? 0
                var total: Float = 0
                for other in 0..<count {
                    scores[other] = exp(scores[other] - maximum)
                    total += scores[other]
                }
                for other in 0..<count {
                    let weight = scores[other] / total
                    for dimension in 0..<headDim {
                        attention[token][base + dimension] += weight * value[other][base + dimension]
                    }
                }
            }
        }

        var output = [[Float]](repeating: [Float](repeating: 0, count: hidden), count: count)
        for token in 0..<count {
            var projected = Inkling2VisionMath.linear(block.oWeight, rows: hidden, columns: hidden, attention[token])
            for index in 0..<hidden { projected[index] += block.oBias[index] }
            for index in 0..<hidden {
                projected[index] = states[token][index] + block.scale1[index] * projected[index]
            }
            var normed = Inkling2VisionMath.layerNorm(
                projected, weight: block.ln2Weight, bias: block.ln2Bias, eps: spec.layerNormEps
            )
            let feedWidth = block.fcWeight.count / hidden
            var feed = Inkling2VisionMath.linear(
                block.fcWeight, rows: feedWidth, columns: hidden, normed
            )
            for index in feed.indices {
                feed[index] = Inkling2VisionMath.gelu(feed[index] + block.fcBias[index])
            }
            var mixed = Inkling2VisionMath.linear(block.projWeight, rows: hidden, columns: feedWidth, feed)
            for index in 0..<hidden { mixed[index] += block.projBias[index] }
            for index in 0..<hidden {
                output[token][index] = projected[index] + block.scale2[index] * mixed[index]
            }
        }
        return output
    }
}

// MARK: - Scalar rules

nonisolated public enum Inkling2VisionMath {
    /// `0.5 * x * (1 + erf(x / sqrt(2)))`, the exact form `hidden_act: "gelu"`
    /// selects -- not the tanh approximation.
    public static func gelu(_ value: Float) -> Float {
        0.5 * value * (1 + erf(value / Float(2).squareRoot()))
    }

    /// LayerNorm with a learned shift, which this tower's blocks use (the text
    /// path's normalizations are all RMS-style and carry no bias).
    public static func layerNorm(
        _ values: [Float], weight: [Float], bias: [Float], eps: Float
    ) -> [Float] {
        let n = values.count
        guard n > 0 else { return values }
        var mean: Float = 0
        for value in values { mean += value }
        mean /= Float(n)
        var variance: Float = 0
        for value in values {
            let centered = value - mean
            variance += centered * centered
        }
        variance /= Float(n)
        let inverse = 1 / (variance + eps).squareRoot()
        var output = [Float](repeating: 0, count: n)
        for index in 0..<n {
            let shift = index < bias.count ? bias[index] : 0
            let scale = index < weight.count ? weight[index] : 1
            output[index] = (values[index] - mean) * inverse * scale + shift
        }
        return output
    }

    /// `y = weight @ vector`, weight `[rows][columns]` row-major.
    public static func linear(
        _ weight: [Float], rows: Int, columns: Int, _ vector: [Float]
    ) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        let count = min(columns, vector.count)
        vector.withUnsafeBufferPointer { input in
            weight.withUnsafeBufferPointer { matrix in
                for row in 0..<rows {
                    var total: Float = 0
                    let base = row * columns
                    for column in 0..<count {
                        total += matrix[base + column] * input[column]
                    }
                    output[row] = total
                }
            }
        }
        return output
    }

    /// The 2x2 bilinear taps of one axis, matching `F.interpolate`'s
    /// `align_corners=false` half-pixel centres with border padding.
    static func axisTaps(index: Int, size: Int, side: Int) -> (taps: [Int], weights: [Float]) {
        let source = (Float(index) + 0.5) * Float(side) / Float(size) - 0.5
        let base = Int(source.rounded(.down))
        var taps: [Int] = []
        var weights: [Float] = []
        for offset in 0...1 {
            let raw = base + offset
            let clamped = min(max(raw, 0), side - 1)
            let distance = abs(source - Float(base) - Float(offset))
            taps.append(clamped)
            weights.append(max(0, 1 - distance))
        }
        return (taps, weights)
    }

    /// The 2D separable resample: `(row0,col0), (row0,col1), (row1,col0),
    /// (row1,col1)`, indices into the flattened `side x side` table.
    public static func interpolationTaps(
        row: Int, column: Int, rows: Int, columns: Int, side: Int
    ) -> (indices: [Int], weights: [Float]) {
        let vertical = axisTaps(index: row, size: rows, side: side)
        let horizontal = axisTaps(index: column, size: columns, side: side)
        var indices: [Int] = []
        var weights: [Float] = []
        for verticalTap in 0..<2 {
            for horizontalTap in 0..<2 {
                indices.append(vertical.taps[verticalTap] * side + horizontal.taps[horizontalTap])
                weights.append(vertical.weights[verticalTap] * horizontal.weights[horizontalTap])
            }
        }
        return (indices, weights)
    }

    /// Axial 2D RoPE: `headDim / 2` interleaved pairs, the first half carrying
    /// the column coordinate and the second the row, each pair with its own
    /// frequency `theta^(-i / (headDim / 4))`.
    public static func ropeAngles(
        row: Int, column: Int, headDim: Int, theta: Float
    ) -> (cos: [Float], sin: [Float]) {
        let spatial = headDim / 2
        let frequencies = spatial / 2
        var cos = [Float](repeating: 1, count: headDim)
        var sin = [Float](repeating: 0, count: headDim)
        guard frequencies > 0 else { return (cos, sin) }
        for index in 0..<frequencies {
            let inverse = pow(theta, -Float(index) / Float(frequencies))
            let horizontal = Float(column) * inverse
            let vertical = Float(row) * inverse
            let pairs = [horizontal, vertical]
            for (half, angle) in pairs.enumerated() {
                let pair = half * frequencies + index
                cos[pair * 2] = Foundation.cos(angle)
                cos[pair * 2 + 1] = cos[pair * 2]
                sin[pair * 2] = Foundation.sin(angle)
                sin[pair * 2 + 1] = sin[pair * 2]
            }
        }
        return (cos, sin)
    }

    /// Rotates one head's interleaved pairs `(x1, x2) -> (-x2, x1)`.
    public static func rotateInPlace(
        _ values: inout [Float], head: Int, headDim: Int, cos: [Float], sin: [Float]
    ) {
        let base = head * headDim
        guard base + headDim <= values.count else { return }
        var dimension = 0
        while dimension + 1 < headDim {
            let first = values[base + dimension]
            let second = values[base + dimension + 1]
            let cosine = cos[dimension]
            let sine = sin[dimension]
            values[base + dimension] = first * cosine - second * sine
            values[base + dimension + 1] = first * sine + second * cosine
            dimension += 2
        }
    }
}
