//
//  DeepSeekV41VisionEncodeDeviceTests.swift
//  DynaMoETests
//
//  The whole vision tower as one device pass. Slices #41 and #42 measured the
//  attention and the four block projections on the GPU, but both were stage
//  runners and `encode` still ran the host GEMMs: a `w1` product at the largest
//  image is 195 MB, so a host<->device round trip per projection costs more than
//  the GEMM it replaces. `DeepSeekV41VisionEncodeDevice` keeps the token matrix on
//  the GPU for all 32 blocks, so these tests exist to prove the pass is the host
//  tower's own arithmetic before any of its speed is believed.
//
//  The parity cases use a tower small enough to run both paths in Debug (hidden 8,
//  two heads, one or two blocks, a downsample ratio of 2, and a grid whose column
//  count pads). The published-geometry case is opt-in (touch
//  `/tmp/dynamoe_dsv41_vision_encode_measure`): it reads the real 970 MB tower and
//  a 93x93 host encode alone is 41.7 s at -O.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41VisionEncodeDeviceTests: XCTestCase {

    private static let measurementMarkerPath = "/tmp/dynamoe_dsv41_vision_encode_measure"

    private let hidden = 8
    private let heads = 2
    private let intermediate = 4
    private let ratio = 2
    private let patchSize = 2
    private let textHidden = 5

    // MARK: Fixture

    private func spec(blocks: Int = 1) -> DeepSeekV41VisionSpec {
        DeepSeekV41VisionSpec(
            patchSize: patchSize,
            downsampleRatio: ratio,
            minPixels: 0,
            maxImageTokens: 1_024,
            numHiddenLayers: blocks,
            hiddenSize: hidden,
            numAttentionHeads: heads,
            intermediateSize: intermediate,
            imageTokenID: 7,
            maxWidthHeightRatio: nil,
            ropeTheta: 100
        )
    }

    private func entry(_ row: Int, _ column: Int) -> Float {
        Float(((row * 7 + column * 3) % 5) - 2) * 0.25
    }

    /// A second block must not be a copy of the first, or a chaining defect cannot
    /// change a value.
    private func blockEntry(_ row: Int, _ column: Int, _ block: Int) -> Float {
        entry(row + block * 3, column + block * 5) + Float(block) * 0.125
    }

    private func matrix(_ rows: Int, _ columns: Int, block: Int? = nil) -> DeepSeekV41Matrix {
        var values: [Float] = []
        values.reserveCapacity(rows * columns)
        for row in 0..<rows {
            for column in 0..<columns {
                values.append(
                    block.map { blockEntry(row, column, $0) } ?? entry(row, column)
                )
            }
        }
        return DeepSeekV41Matrix(rows: rows, columns: columns, values: values)
    }

    private func vector(_ count: Int, offset: Int) -> [Float] {
        (0..<count).map { Float((($0 * 3 + offset) % 5) - 2) * 0.25 }
    }

    private func gain(_ count: Int, block: Int) -> [Float] {
        (0..<count).map { 1 + Float(($0 + block) % 4) * 0.125 }
    }

    private func patches(rows: Int, columns: Int) -> [[Float]] {
        let width = 3 * patchSize * patchSize
        return (0..<(rows * columns)).map { token in
            (0..<width).map { dimension in
                entry(token, dimension) + Float(token) * 0.5
            }
        }
    }

    private func weights(
        blocks: Int = 1,
        omitting omitted: DeepSeekV41TensorRole? = nil
    ) -> DeepSeekV41DenseWeightSource {
        let source = DeepSeekV41DenseWeightSource()
        if omitted != .visionPatchEmbed {
            source.set(
                matrix(hidden, 3 * patchSize * patchSize), role: .visionPatchEmbed, layer: nil
            )
        }
        if omitted != .visionPatchEmbedBias {
            source.set(vector(hidden, offset: 1), role: .visionPatchEmbedBias, layer: nil)
        }
        for block in 0..<blocks {
            if omitted != .visionBlockNorm1 {
                source.set(gain(hidden, block: block), role: .visionBlockNorm1, layer: block)
            }
            if omitted != .visionBlockWqkv {
                source.set(
                    matrix(3 * hidden, hidden, block: block), role: .visionBlockWqkv, layer: block
                )
            }
            if omitted != .visionBlockWqkvBias {
                source.set(
                    vector(3 * hidden, offset: 2 + block), role: .visionBlockWqkvBias, layer: block
                )
            }
            if omitted != .visionBlockWo {
                source.set(matrix(hidden, hidden, block: block), role: .visionBlockWo, layer: block)
            }
            if omitted != .visionBlockWoBias {
                source.set(
                    vector(hidden, offset: 3 + block), role: .visionBlockWoBias, layer: block
                )
            }
            if omitted != .visionBlockNorm2 {
                source.set(gain(hidden, block: block + 1), role: .visionBlockNorm2, layer: block)
            }
            if omitted != .visionBlockW1 {
                source.set(
                    matrix(2 * intermediate, hidden, block: block),
                    role: .visionBlockW1, layer: block
                )
            }
            if omitted != .visionBlockW2 {
                source.set(
                    matrix(hidden, intermediate, block: block),
                    role: .visionBlockW2, layer: block
                )
            }
        }
        if omitted != .visionFinalNorm {
            source.set(gain(hidden, block: 2), role: .visionFinalNorm, layer: nil)
        }
        if omitted != .visionAlignerW1 {
            source.set(
                matrix(textHidden, hidden * ratio * ratio), role: .visionAlignerW1, layer: nil
            )
        }
        if omitted != .visionAlignerW1Bias {
            source.set(vector(textHidden, offset: 4), role: .visionAlignerW1Bias, layer: nil)
        }
        if omitted != .visionAlignerW2 {
            source.set(matrix(textHidden, textHidden), role: .visionAlignerW2, layer: nil)
        }
        if omitted != .visionAlignerW2Bias {
            source.set(vector(textHidden, offset: 0), role: .visionAlignerW2Bias, layer: nil)
        }
        source.set(vector(textHidden, offset: 1), role: .visionImageStart, layer: nil)
        source.set(vector(textHidden, offset: 2), role: .visionImageNewline, layer: nil)
        source.set(vector(textHidden, offset: 3), role: .visionImageEnd, layer: nil)
        return source
    }

    private func deviceTower(
        blocks: Int = 1,
        textHiddenSize: Int? = nil,
        omitting omitted: DeepSeekV41TensorRole? = nil
    ) throws -> (DeepSeekV41VisionEncodeDevice, DeepSeekV41DenseWeightSource) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        let source = weights(blocks: blocks, omitting: omitted)
        let tower = try DeepSeekV41VisionEncodeDevice(
            spec: spec(blocks: blocks), weights: source,
            textHiddenSize: textHiddenSize ?? textHidden, device: device
        )
        return (tower, source)
    }

    private func worstRelativeDifference(_ lhs: [[Float]], _ rhs: [[Float]]) -> Float {
        var worst: Float = 0
        var scale: Float = 1e-6
        for row in 0..<min(lhs.count, rhs.count) {
            for column in 0..<min(lhs[row].count, rhs[row].count) {
                worst = max(worst, abs(lhs[row][column] - rhs[row][column]))
                scale = max(scale, abs(rhs[row][column]))
            }
        }
        return worst / scale
    }

    // MARK: Parity

    /// The whole tower, both paths: patch embed, norm, qkv, split/rotate, attention,
    /// wo, the residual, the gated MLP, the final norm, the unshuffle and the
    /// aligner. Every elementwise rule the device pass adds is covered by this one
    /// comparison, and so is every weight layout.
    func testTheDeviceEncodeMatchesTheHostOnATinyTower() throws {
        let (tower, source) = try deviceTower()
        let input = patches(rows: 2, columns: 3)
        let deviceRows = try tower.encode(patches: input, rows: 2, columns: 3)
        let host = DeepSeekV41VisionForward(
            spec: spec(), weights: source, textHiddenSize: textHidden
        )
        let hostRows = try host.encodeOnHost(patches: input, rows: 2, columns: 3)
        XCTAssertEqual(deviceRows.count, hostRows.count)
        XCTAssertEqual(deviceRows.first?.count, textHidden)
        let difference = worstRelativeDifference(deviceRows, hostRows)
        print("🔎 [dsv41 vision] encode parity: worst \(difference) relative")
        XCTAssertLessThan(difference, 2e-4, "the device encode left the host tower")
    }

    /// A grid whose column count is not a multiple of the downsample ratio pads right
    /// and bottom, and the padded cells must be zero rather than whatever was in the
    /// cell buffer before. Three-by-five patches on a ratio of 2 is that grid.
    func testTheDeviceEncodePadsTheGridLikeTheHost() throws {
        let (tower, source) = try deviceTower()
        let input = patches(rows: 3, columns: 5)
        let deviceRows = try tower.encode(patches: input, rows: 3, columns: 5)
        let host = DeepSeekV41VisionForward(
            spec: spec(), weights: source, textHiddenSize: textHidden
        )
        let hostRows = try host.encodeOnHost(patches: input, rows: 3, columns: 5)
        // 3x5 patches on a ratio of 2 is a 2x3 aligner grid.
        XCTAssertEqual(deviceRows.count, 6)
        XCTAssertLessThan(worstRelativeDifference(deviceRows, hostRows), 2e-4)
    }

    /// Two blocks with different weights: the residual chain, the second block's
    /// norms and every weight buffer's layer key are checked together. A pass that
    /// bound block 0's matrices for both blocks would pass a one-block test.
    func testTheDeviceEncodeChainsBlocks() throws {
        let (tower, source) = try deviceTower(blocks: 2)
        let input = patches(rows: 2, columns: 2)
        let deviceRows = try tower.encode(patches: input, rows: 2, columns: 2)
        let host = DeepSeekV41VisionForward(
            spec: spec(blocks: 2), weights: source, textHiddenSize: textHidden
        )
        let hostRows = try host.encodeOnHost(patches: input, rows: 2, columns: 2)
        XCTAssertLessThan(worstRelativeDifference(deviceRows, hostRows), 2e-4)
    }

    /// One tower, three encodes, two grid shapes. The workspace and the rotation
    /// tables are sized per grid; a pass that kept the first grid's buffers would
    /// read another grid's tokens, so the second and third encodes are compared too.
    func testTheDeviceEncodeReusesItsWorkspaceAcrossGrids() throws {
        let (tower, source) = try deviceTower()
        let host = DeepSeekV41VisionForward(
            spec: spec(), weights: source, textHiddenSize: textHidden
        )
        for (rows, columns) in [(2, 3), (3, 5), (2, 3)] {
            let input = patches(rows: rows, columns: columns)
            let deviceRows = try tower.encode(patches: input, rows: rows, columns: columns)
            let hostRows = try host.encodeOnHost(patches: input, rows: rows, columns: columns)
            XCTAssertLessThan(
                worstRelativeDifference(deviceRows, hostRows), 2e-4,
                "\(rows)x\(columns)"
            )
        }
    }

    // MARK: Refusals

    func testTheDeviceEncodeRefusesARaggedPatchRow() throws {
        let (tower, _) = try deviceTower()
        var input = patches(rows: 2, columns: 3)
        input[1] = Array(input[1].dropLast())
        XCTAssertThrowsError(try tower.encode(patches: input, rows: 2, columns: 3)) { error in
            switch error as? DeepSeekV41VisionForwardError {
            case .shapeMismatch:
                break
            default:
                XCTFail("expected a shape mismatch, got \(error)")
            }
        }
    }

    func testTheDeviceEncodeReportsAMissingWeight() throws {
        let (tower, _) = try deviceTower(omitting: .visionBlockNorm1)
        XCTAssertThrowsError(
            try tower.encode(patches: patches(rows: 2, columns: 3), rows: 2, columns: 3)
        ) { error in
            XCTAssertEqual(
                error as? DeepSeekV41VisionForwardError,
                .missingTensor(.visionBlockNorm1, 0)
            )
        }
    }

    /// The device tower's aligner width is the language model's hidden size, so a
    /// forward that expects a different width must refuse rather than run a tower
    /// whose rows cannot enter the residual stream.
    func testADeviceTowerMustProjectToTheForwardsTextWidth() throws {
        let (tower, source) = try deviceTower(textHiddenSize: textHidden + 2)
        let forward = DeepSeekV41VisionForward(
            spec: spec(), weights: source, textHiddenSize: textHidden, device: tower
        )
        XCTAssertThrowsError(try forward.encode(patches: patches(rows: 2, columns: 3), rows: 2, columns: 3)) {
            switch $0 as? DeepSeekV41VisionForwardError {
            case .shapeMismatch:
                break
            default:
                XCTFail("expected a shape mismatch, got \($0)")
            }
        }
    }

    // MARK: The published tower

    /// The encode on the checkpoint's own weights, and the host stage split that
    /// #37's three-point fit could only price as one per-token term.
    ///
    /// The two arms are separate measurements: the host arm runs `encodeOnHost` with
    /// its stage clock, the device arm runs one command buffer and reports its own
    /// GPU time. The device tower is warmed first (a one-token grid uploads every
    /// weight), so no upload is inside the timed arm.
    func testWhatAnEncodeCostsAtThePublishedSizes() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_VISION_ENCODE_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.measurementMarkerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in published-checkpoint measurement; touch \(Self.measurementMarkerPath) to run"
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let summary = try engine.getSummary()
        let catalog = DeepSeekV41WeightCatalog.from(
            summary: summary, layerCount: config.numHiddenLayers
        )
        let source = try XCTUnwrap(
            DeepSeekV41VisionWeights(snapshotDir: snapshot, catalog: catalog),
            "the tower's tensors could not be resolved from the published shards"
        )
        XCTAssertEqual(source.problems, [])
        let tower = try DeepSeekV41VisionEncodeDevice(
            spec: spec, weights: source, textHiddenSize: config.hiddenSize, device: device
        )
        let host = DeepSeekV41VisionForward(
            spec: spec, weights: source, textHiddenSize: config.hiddenSize
        )

        // The weight upload runs once per tower; keep it outside every timed arm.
        let warmStart = CFAbsoluteTimeGetCurrent()
        _ = try tower.encode(
            patches: publishedPatches(rows: 1, columns: 1, spec: spec),
            rows: 1, columns: 1
        )
        let warmSeconds = CFAbsoluteTimeGetCurrent() - warmStart
        print("🔎 [dsv41 vision] encode warm-up (weight upload): "
            + String(format: "%.2f s", warmSeconds))

        var devicePoints: [(tokens: Int, seconds: Double)] = []
        var hostPoints: [(tokens: Int, seconds: Double)] = []
        for (rows, columns) in [(39, 39), (63, 63), (93, 93)] {
            let input = publishedPatches(rows: rows, columns: columns, spec: spec)
            let tokenCount = rows * columns

            var stages: [String: Double] = [:]
            let hostStart = CFAbsoluteTimeGetCurrent()
            let hostRows = try host.encodeOnHost(
                patches: input, rows: rows, columns: columns
            ) { stage, seconds in
                stages[stage, default: 0] += seconds
            }
            let hostSeconds = CFAbsoluteTimeGetCurrent() - hostStart

            let deviceStart = CFAbsoluteTimeGetCurrent()
            let result = try tower.run(patches: input, rows: rows, columns: columns)
            let deviceWall = CFAbsoluteTimeGetCurrent() - deviceStart

            let difference = worstRelativeDifference(result.rows, hostRows)
            XCTAssertLessThan(difference, 2e-3, "\(rows)x\(columns)")
            devicePoints.append((tokenCount, result.gpuSeconds))
            hostPoints.append((tokenCount, hostSeconds))
            let speedup = hostSeconds / max(result.gpuSeconds, 1e-9)
            print(
                "🔎 [dsv41 vision] encode \(rows)x\(columns) (\(tokenCount) tokens): "
                    + "host " + String(format: "%.2f", hostSeconds) + " s, device "
                    + String(format: "%.2f", result.gpuSeconds) + " s GPU / "
                    + String(format: "%.2f", deviceWall) + " s wall, "
                    + String(format: "%.2fx", speedup) + ", parity "
                    + String(format: "%g", difference)
            )
            for (stage, seconds) in stages.sorted(by: { $0.value > $1.value }) {
                let share = 100 * seconds / max(hostSeconds, 1e-9)
                let perToken = 1000 * seconds / Double(tokenCount)
                print(
                    "🔎 [dsv41 vision]   \(stage): "
                        + String(format: "%.3f", seconds) + " s, "
                        + String(format: "%.1f", share) + "%, "
                        + String(format: "%.4f", perToken) + " ms/token"
                )
            }
        }

        // A three-point solve, the same form #37 fitted from wall clock: the fixed
        // term is the tower's per-encode overhead, the linear term the projections
        // and glue, and the quadratic term the attention. Solving it here is what
        // tells the cost model what to publish for the SHIPPED path.
        if let deviceModel = solveCostModel(devicePoints) {
            print(
                "🔎 [dsv41 vision] device model: "
                    + String(format: "%.3f", deviceModel.fixed) + " s + "
                    + String(format: "%.4f", deviceModel.perToken * 1000)
                    + " ms/token + "
                    + String(format: "%.4g", deviceModel.perTokenSquared)
                    + " s/token^2"
            )
        }
        if let hostModel = solveCostModel(hostPoints) {
            print(
                "🔎 [dsv41 vision] host model: "
                    + String(format: "%.3f", hostModel.fixed) + " s + "
                    + String(format: "%.4f", hostModel.perToken * 1000)
                    + " ms/token + "
                    + String(format: "%.4g", hostModel.perTokenSquared)
                    + " s/token^2"
            )
        }
    }

    /// Cramer-free solve of `s = fixed + perToken*t + perTokenSquared*t^2` from
    /// three distinct token counts.
    private func solveCostModel(
        _ points: [(tokens: Int, seconds: Double)]
    ) -> (fixed: Double, perToken: Double, perTokenSquared: Double)? {
        guard points.count == 3 else { return nil }
        let t = points.map { Double($0.tokens) }
        let s = points.map(\.seconds)
        let slope01 = (s[0] - s[1]) / (t[0] - t[1])
        let slope02 = (s[0] - s[2]) / (t[0] - t[2])
        guard abs(t[1] - t[2]) > 1e-9 else { return nil }
        let squared = (slope01 - slope02) / (t[1] - t[2])
        let linear = slope01 - squared * (t[0] + t[1])
        let fixed = s[0] - linear * t[0] - squared * t[0] * t[0]
        guard fixed.isFinite, linear.isFinite, squared.isFinite else { return nil }
        return (fixed, linear, squared)
    }

    // MARK: Published harness

    private func snapshotDirectory() throws -> URL {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots"
            )
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: nil
        ) else {
            throw XCTSkip("the DeepSeek-V4.1 hub directory is missing")
        }
        guard let snapshot = entries.first(where: {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("config.json").path
            ) && FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("model.safetensors.index.json").path
            )
        }) else {
            throw XCTSkip("the DeepSeek-V4.1 snapshot is not downloaded")
        }
        return snapshot
    }

    /// Patches of the right width, in the same order `VisionPreprocess.patches`
    /// emits them, with values that vary by token so a row mix-up is visible.
    private func publishedPatches(
        rows: Int,
        columns: Int,
        spec: DeepSeekV41VisionSpec
    ) -> [[Float]] {
        let width = 3 * spec.patchSize * spec.patchSize
        return (0..<(rows * columns)).map { token in
            (0..<width).map { dimension in
                Float(((token * 13 + dimension * 7) % 11) - 5) / 8
            }
        }
    }
}
