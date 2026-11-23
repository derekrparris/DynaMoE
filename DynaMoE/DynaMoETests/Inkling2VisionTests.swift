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
@testable import DynaMoE

final class Inkling2VisionTests: XCTestCase {

    static let dumpPath = "/tmp/inkling2_vision_goldens.json"

    /// The engine owns the shard mappings and the zero-copy buffers point into
    /// them, so letting it go out of scope unmaps the weights under the source.
    private var engine: DynaMoeEngine?

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
