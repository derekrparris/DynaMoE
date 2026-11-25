//
//  Inkling2DecodeTests.swift
//  DynaMoETests
//
//  Phase 6, slice 1: the output head.
//
//  The unembedding is 200,058 x 2,304 BF16 -- 922 MB stored, 1.84 GB as the
//  `[Float]` the base rule decoded -- and it is read ONCE PER TOKEN by both the
//  shipping device path and the host fallback, because the device visit runner
//  only replaces `runVisit`. Three things are pinned here:
//
//    * the chunked host rule sums EXACTLY like the whole-table rule, at chunk
//      sizes that do and do not divide the row count, because the chunk is a
//      memory decision and must not move a single pinned host number;
//    * the device kernel agrees with the host rule on the published table, in
//      both shipped kernel forms (the choice between them stays a measurement);
//    * the head seal (`headRunner`) is a substitution of the projection and
//      nothing else -- a runner of the wrong width is refused rather than
//      silently shrinking the vocabulary the sampler sees.
//
//  The measurement case is opt-in: `touch /tmp/dynamoe_inkling2_decode_measure`,
//  and run it from an `-O` build (a Debug build of this port is ~60x slower and
//  would measure the compiler, not the graph).
//

import Metal
import XCTest
@testable import DynaMoE

// MARK: - The head rule, model-free

final class Inkling2HeadRuleTests: XCTestCase {

    /// Deterministic, wide-ranging values: every row and column distinct and of
    /// both signs, so a chunk seam read twice, a skipped row, or a dropped tail
    /// all change the sum instead of cancelling.
    private func values(_ count: Int, seed: Int = 1) -> [Float] {
        (0..<count).map { index in
            let mixed = (index &* 37 &+ seed &* 101) % 23
            let sign: Float = (index &+ seed) % 3 == 0 ? -1 : 1
            return sign * (Float(mixed) + 0.5) / 8
        }
    }

    private func stored(_ bits: [UInt16]) -> [Float] {
        bits.map { Float(bitPattern: UInt32($0) << 16) }
    }

    private func bits(_ values: [Float]) -> [UInt16] {
        values.map { UInt16($0.bitPattern >> 16) }
    }

    /// The rule being replaced: decode the table whole, then one column-ordered
    /// FP32 dot per row.
    private func expected(rows: Int, columns: Int, table: [Float], hidden: [Float]) -> [Float] {
        (0..<rows).map { row in
            var total: Float = 0
            for column in 0..<columns { total += table[row * columns + column] * hidden[column] }
            return total
        }
    }

    func testTheChunkedHeadSumsExactlyLikeTheWholeTable() throws {
        // 7 columns is not a multiple of 4, so the kernel-side `float4` loops
        // are irrelevant here but the ROW is odd, and 5 rows do not divide by
        // any chunk below 5 -- the tail path runs in every arm.
        let rows = 5
        let columns = 7
        let weights = bits(values(rows * columns))
        let hidden = values(columns, seed: 9)
        let reference = expected(
            rows: rows, columns: columns, table: stored(weights), hidden: hidden
        )
        for chunk in [1, 2, 3, 5, 8] {
            let actual: [Float] = try weights.withUnsafeBytes { raw in
                try Inkling2SnapshotWeights.headLogits(
                    from: try XCTUnwrap(raw.baseAddress),
                    dtype: "BF16", rows: rows, hiddenSize: columns,
                    hidden: hidden, chunkRows: chunk
                )
            }
            // EXACT, not approximate: the chunk changes the allocation, not the
            // order of the additions, and a published host golden that moved by
            // a rounding would look like a graph regression.
            XCTAssertEqual(actual, reference, "chunk of \(chunk) row(s)")
        }
    }

    func testTheHeadDecodesEveryStoredFormatTheSnapshotCanPublish() throws {
        let rows = 3
        let columns = 5
        let hidden = values(columns, seed: 4)

        // F32: the stored values ARE the values.
        let floatTable = values(rows * columns, seed: 2)
        let floatResult: [Float] = try floatTable.withUnsafeBytes { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "F32", rows: rows,
                hiddenSize: columns, hidden: hidden, chunkRows: 2
            )
        }
        XCTAssertEqual(
            floatResult,
            expected(rows: rows, columns: columns, table: floatTable, hidden: hidden)
        )

        // F16: the table is the half-precision value widened, so the fixture's
        // 1/8 steps survive but its exactness is not assumed. Stored as TWO
        // bytes, which is what `elementWidth("F16")` prices.
        var halfValues = values(rows * columns, seed: 3).map { Float16($0) }
        let halfResult: [Float] = try halfValues.withUnsafeMutableBytes { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "F16", rows: rows,
                hiddenSize: columns, hidden: hidden, chunkRows: 2
            )
        }
        withExtendedLifetime(halfValues) {}
        XCTAssertEqual(
            halfResult,
            expected(
                rows: rows, columns: columns,
                table: halfValues.map { Float($0) }, hidden: hidden
            )
        )

        // E4M3: the codes are decoded through the format table, which is the
        // one path with a non-linear widening.
        let codes: [UInt8] = (0..<(rows * columns)).map { UInt8(0x30 + $0) }
        var codeCopy = codes
        let codeResult: [Float] = try codeCopy.withUnsafeMutableBytes { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "F8_E4M3", rows: rows,
                hiddenSize: columns, hidden: hidden, chunkRows: 2
            )
        }
        withExtendedLifetime(codeCopy) {}
        XCTAssertEqual(
            codeResult,
            expected(
                rows: rows, columns: columns,
                table: codes.map { Inkling2Element.decodeE4M3($0) }, hidden: hidden
            )
        )
    }

    func testAnEmptyRowCountProducesNoLogits() throws {
        var byte: UInt8 = 0
        let logits: [Float] = try withUnsafeBytes(of: &byte) { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "BF16", rows: 0,
                hiddenSize: 4, hidden: [1, 2, 3, 4], chunkRows: 2
            )
        }
        XCTAssertTrue(logits.isEmpty)
    }

    func testAChunkOfZeroRowsIsRefusedByName() throws {
        var byte: UInt8 = 0
        XCTAssertThrowsError(try withUnsafeBytes(of: &byte) { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "BF16", rows: 2,
                hiddenSize: 4, hidden: [1, 2, 3, 4], chunkRows: 0
            )
        }) { error in
            guard case Inkling2ForwardError.unsupported = error else {
                return XCTFail("expected a named refusal, got \(error)")
            }
        }
    }

    func testAHiddenRowOfTheWrongWidthIsRefused() throws {
        var byte: UInt8 = 0
        XCTAssertThrowsError(try withUnsafeBytes(of: &byte) { raw in
            try Inkling2SnapshotWeights.headLogits(
                from: try XCTUnwrap(raw.baseAddress), dtype: "BF16", rows: 2,
                hiddenSize: 4, hidden: [1, 2, 3], chunkRows: 2
            )
        }) { error in
            guard case Inkling2ForwardError.shapeMismatch = error else {
                return XCTFail("expected a shape refusal, got \(error)")
            }
        }
    }
}

// MARK: - The head seam

/// A weight source with an unembedding and nothing else. `rawLogits` only ever
/// asks for `.lmHead`, so the rest of the graph is deliberately absent: a
/// refusal that names the missing role is the honest answer, and a test that
/// built the whole table would be testing the catalog instead.
private struct HeadOnlySource: Inkling2WeightSource {
    let table: [Float]
    let rows: Int
    let columns: Int

    func matrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float] {
        guard role == .lmHead else {
            throw Inkling2ForwardError.missingWeight(role.rawValue)
        }
        return table
    }

    func shape(_ role: Inkling2TensorRole, layer: Int?) throws -> [Int] {
        guard role == .lmHead else {
            throw Inkling2ForwardError.missingWeight(role.rawValue)
        }
        return [rows, columns]
    }

    func expertMatrix(
        _ role: Inkling2TensorRole, layer: Int, expert: Int
    ) throws -> Inkling2PackedExpertMatrix {
        throw Inkling2ForwardError.missingWeight(role.rawValue)
    }
}

final class Inkling2HeadSeamTests: XCTestCase {

    private func config() throws -> Inkling2Config {
        try Inkling2Config.decode(jsonData: Data(Inkling2ConfigTests.smallJSON.utf8))
    }

    /// `rawLogits` without a runner is the weight source's own rule, and the
    /// rows it asks for are the UNPADDED vocabulary -- the padded rows exist in
    /// the checkpoint's table and must never reach the sampler.
    func testTheHostRuleAsksForTheUnpaddedVocabularyOnly() throws {
        let config = try config()
        let source = HeadOnlySource(
            table: (0..<(config.unpaddedVocabSize * config.hiddenSize)).map { Float($0 % 7) - 3 },
            rows: config.vocabSize,
            columns: config.hiddenSize
        )
        let forward = try Inkling2CPUForward(config: config, weights: source)
        let hidden = (0..<config.hiddenSize).map { Float($0 % 5) - 2 }
        let logits = try forward.rawLogits(hidden)
        XCTAssertEqual(logits.count, config.unpaddedVocabSize)
        let reference = try source.logits(
            hidden, rows: config.unpaddedVocabSize, hiddenSize: config.hiddenSize
        )
        XCTAssertEqual(logits, reference)
    }

    func testTheHeadRunnerReplacesTheProjectionAndIsNotSoftcapped() throws {
        let config = try config()
        let source = HeadOnlySource(
            table: [Float](repeating: 0, count: config.vocabSize * config.hiddenSize),
            rows: config.vocabSize,
            columns: config.hiddenSize
        )
        let forward = try Inkling2CPUForward(config: config, weights: source)
        let sentinel = (0..<config.unpaddedVocabSize).map { Float($0) * 0.001 }
        var asked: [Float] = []
        forward.headRunner = { hidden in
            asked = hidden
            return sentinel
        }
        let hidden = (0..<config.hiddenSize).map { Float($0) + 1 }
        XCTAssertEqual(try forward.rawLogits(hidden), sentinel)
        XCTAssertEqual(asked, hidden, "the runner must receive the graph's own hidden row")
    }

    /// A runner that answers the wrong number of logits would otherwise shrink
    /// the vocabulary the sampler is handed, silently and with no error.
    func testAHeadRunnerOfTheWrongLengthIsRefused() throws {
        let config = try config()
        let source = HeadOnlySource(
            table: [Float](repeating: 0, count: config.vocabSize * config.hiddenSize),
            rows: config.vocabSize,
            columns: config.hiddenSize
        )
        let forward = try Inkling2CPUForward(config: config, weights: source)
        forward.headRunner = { _ in [Float](repeating: 1, count: config.unpaddedVocabSize - 1) }
        XCTAssertThrowsError(try forward.rawLogits([Float](repeating: 0, count: config.hiddenSize))) {
            guard case Inkling2ForwardError.shapeMismatch = $0 else {
                return XCTFail("expected a shape refusal, got \($0)")
            }
        }
    }
}

// MARK: - The published table

/// The device head against the host rule on the checkpoint's own unembedding.
///
/// Skipped when the snapshot is absent, like every other published case here:
/// a fixture can check the kernel's strides, but only the real table has the
/// published 2,304-wide rows and 200,058 outputs the shipping dispatch runs.
final class Inkling2PublishedHeadTests: XCTestCase {

    private var engine: DynaMoeEngine?

    private func snapshot() throws -> URL {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(path.path)")
        }
        return path
    }

    private struct Harness {
        let config: Inkling2Config
        let source: Inkling2SnapshotWeights
        let oracle: Inkling2CPUForward
        let metal: Inkling2MetalForward
        let workspace: Inkling2DeviceWorkspace
    }

    private func harness(stagedPositions: Int = 32) throws -> Harness {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let summary = try inkling2MetalReadySummary(try engine.getSummary())
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        guard let queue = device.makeCommandQueue(), let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no Metal queue or default library")
        }
        var buffers: [UInt32: MTLBuffer] = [:]
        buffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        let oracle = try Inkling2CPUForward(config: config, weights: source)
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: oracle
        )
        let workspace = try Inkling2DeviceWorkspace(
            device: device, config: config, stagedPositions: stagedPositions
        )
        return Harness(
            config: config, source: source, oracle: oracle, metal: metal, workspace: workspace
        )
    }

    /// A deterministic hidden row with both signs and a wide range: the point is
    /// the projection, so a synthetic row keeps the comparison independent of
    /// every visit (`testTheDeviceVisitMatchesTheHostGraph` covers those).
    private func hiddenRow(_ width: Int) -> [Float] {
        (0..<width).map { index in
            let mixed = (index &* 37 &+ 11) % 29
            let sign: Float = index % 4 == 0 ? -1 : 1
            return sign * (Float(mixed) + 0.5) / 16
        }
    }

    private func topK(_ values: [Float], _ k: Int) -> [Int] {
        Array(values.indices.sorted { values[$0] > values[$1] }.prefix(k))
    }

    /// Both shipped kernel forms against the host rule, on the published table.
    func testTheDeviceHeadMatchesTheHostRuleOnThePublishedCheckpoint() throws {
        let harness = try harness()
        let hidden = hiddenRow(harness.config.hiddenSize)
        let reference = try harness.oracle.rawLogits(hidden)
        XCTAssertEqual(reference.count, harness.config.unpaddedVocabSize)

        for form in Inkling2MetalForward.HeadKernelForm.allCases {
            let actual = try harness.metal.headLogits(
                hidden: hidden, workspace: harness.workspace, form: form
            )
            XCTAssertEqual(actual.count, reference.count, "\(form)")
            let scale = reference.reduce(Float(0)) { max($0, abs($1)) }
            let worst = zip(actual, reference).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            let relative = scale > 0 ? worst / scale : worst
            print("🔎 [inkling2 head] \(form.rawValue): worst \(relative) relative")
            // 1e-4: the two accumulate 2,304 products in a different order and
            // agree far closer than that (the measured figure is printed).
            XCTAssertLessThanOrEqual(relative, 1e-4, "\(form) against the host rule")
            let actualArgmax = actual.indices.max { actual[$0] < actual[$1] }
            let referenceArgmax = reference.indices.max { reference[$0] < reference[$1] }
            XCTAssertEqual(actualArgmax, referenceArgmax, "\(form) argmax")
            XCTAssertEqual(topK(actual, 8), topK(reference, 8), "\(form) top eight")
        }
    }

    /// What the head costs, and what it is a share of.
    ///
    /// Three questions, three answers: which kernel form wins (a rotated A/B,
    /// because the arm that runs first on an idle GPU wins on clock ramp alone
    /// -- `Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #53"), what the host rule costs
    /// beside it, and what share of a position the head is once it is installed
    /// the way the engine installs it.
    ///
    /// Opt-in because it reads the 922 MB table a dozen times; run it from an
    /// `-O` build, since a Debug build of this port measures the compiler.
    func testWhatTheHeadCostsOfADecodePositionOnThePublishedCheckpoint() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_decode_measure"),
            "opt-in decode measurement; touch /tmp/dynamoe_inkling2_decode_measure to run"
        )
        let harness = try harness()
        let hidden = hiddenRow(harness.config.hiddenSize)
        let forms = Inkling2MetalForward.HeadKernelForm.allCases

        // Warm the table and the clock once, then rotate the arm order every
        // round and discard round one: otherwise "the first form is slow" and
        // "the first form is fast" are both reproducible.
        _ = try harness.metal.headLogits(hidden: hidden, workspace: harness.workspace)
        var totals: [Inkling2MetalForward.HeadKernelForm: Double] = [:]
        let rounds = 6
        for round in 0..<rounds {
            let order = (0..<forms.count).map { forms[($0 + round) % forms.count] }
            for form in order {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try harness.metal.headLogits(
                    hidden: hidden, workspace: harness.workspace, form: form
                )
                let seconds = CFAbsoluteTimeGetCurrent() - start
                if round > 0 { totals[form, default: 0] += seconds }
                print(
                    "🔎 [inkling2 head] round \(round) \(form.rawValue): "
                        + "\(String(format: "%.4f", seconds)) s"
                )
            }
        }
        let mean = totals.mapValues { $0 / Double(rounds - 1) }
        let tableBytes = Double(
            harness.config.unpaddedVocabSize * harness.config.hiddenSize * 2
        )
        for form in forms {
            let seconds = mean[form] ?? 0
            print(
                "🔎 [inkling2 head] mean \(form.rawValue): "
                    + "\(String(format: "%.4f", seconds)) s, "
                    + "\(String(format: "%.0f", tableBytes / 1e9 / seconds)) GB/s"
            )
        }

        let hostStart = CFAbsoluteTimeGetCurrent()
        let hostLogits = try harness.oracle.rawLogits(hidden)
        let hostSeconds = CFAbsoluteTimeGetCurrent() - hostStart

        // One whole position with the head installed exactly as the engine
        // installs it, so the share is measured on the turn's path. The forward
        // is the executor's OWN host graph, which is what the engine does:
        // `runVisit` reads and appends that graph's visit states, so a second
        // forward here would run a valid position against the wrong sequence.
        let forward = harness.oracle
        let executor = harness.metal
        let workspace = harness.workspace
        forward.visitRunner = { layer, pass, value, blockStates, messages, position in
            try executor.runVisit(
                layer: layer, pass: pass, hidden: value, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
        }
        forward.headRunner = { value in
            try executor.headLogits(hidden: value, workspace: workspace)
        }
        let positionStart = CFAbsoluteTimeGetCurrent()
        _ = try forward.forwardToken(token: 0, position: 0)
        let positionSeconds = CFAbsoluteTimeGetCurrent() - positionStart
        let deviceSeconds = mean[Inkling2MetalForward.shippedHeadForm] ?? 0

        print(
            "🔎 [inkling2 decode] head: host \(String(format: "%.3f", hostSeconds)) s; "
                + "position \(String(format: "%.3f", positionSeconds)) s, "
                + "head \(String(format: "%.1f", 100 * deviceSeconds / positionSeconds))% "
                + "at \(Inkling2MetalForward.shippedHeadForm.rawValue)"
        )
        XCTAssertEqual(hostLogits.count, harness.config.unpaddedVocabSize)
        XCTAssertGreaterThan(positionSeconds, 0)
    }
}
