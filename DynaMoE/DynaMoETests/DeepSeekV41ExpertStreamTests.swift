//
//  DeepSeekV41ExpertStreamTests.swift
//  DynaMoE
//
//  Coverage for the streamed routed-expert path: the slot geometry the kernels
//  index, the format decision that picks the MXFP4 kernels for this checkpoint's
//  raw-byte weights, the arena placement the streamer fills, and the pread itself
//  against a real file.
//
//  None of it needs the 510 GB checkpoint. The byte counts are the checkpoint's own
//  published shapes (hidden 5120, moe_inter 2304, packed two values per byte), and
//  the read test builds a small file whose ranges are read back and compared, which
//  is the part a wrong offset would corrupt silently.
//

import XCTest
@testable import DynaMoE

// MARK: - Slot geometry

final class DeepSeekV41ExpertSlotGeometryTests: XCTestCase {

    private let geometry = DeepSeekV41ExpertSlotGeometry.make(
        hidden: 5120,
        intermediate: 2304,
        format: .mxfp4
    )

    func testTheSlotMatchesTheCheckpointsPublishedWidths() throws {
        let geometry = try XCTUnwrap(geometry)
        // gate and up are 2304 x 5120 published as [2304, 2560] packed bytes, each
        // with a [2304, 160] E8M0 companion; down is 5120 x 2304 as [5120, 1152]
        // with [5120, 72].
        XCTAssertEqual(geometry.gateWeightBytes, 2304 * 2560)
        XCTAssertEqual(geometry.gateScaleBytes, 2304 * 160)
        XCTAssertEqual(geometry.upWeightBytes, 2304 * 2560)
        XCTAssertEqual(geometry.upScaleBytes, 2304 * 160)
        XCTAssertEqual(geometry.downWeightBytes, 5120 * 1152)
        XCTAssertEqual(geometry.downScaleBytes, 5120 * 72)
    }

    func testOneSlotIsExactlyWhatTheMemoryPlanPrices() throws {
        // `DeepSeekV41MemoryPlan.bytesPerExpert` is what sized the arena, so the
        // bytes actually read per miss have to agree with it: a disagreement makes
        // the readiness report describe a machine that does not exist.
        let values = 3 * 5120 * 2304
        let planned = Int(
            (Double(values) * DeepSeekV41DtypeWidth.bytesPerValue(dtype: "MXFP4")).rounded(.up)
        )
        XCTAssertEqual(try XCTUnwrap(geometry).slotBytes, planned)
        XCTAssertEqual(planned, 18_800_640)
    }

    func testTheSixRangesAreContiguousAndNonOverlapping() throws {
        let geometry = try XCTUnwrap(geometry)
        let ranges: [(Int, Int)] = [
            (geometry.gateWeightOffset, geometry.gateWeightBytes),
            (geometry.gateScaleOffset, geometry.gateScaleBytes),
            (geometry.upWeightOffset, geometry.upWeightBytes),
            (geometry.upScaleOffset, geometry.upScaleBytes),
            (geometry.downWeightOffset, geometry.downWeightBytes),
            (geometry.downScaleOffset, geometry.downScaleBytes)
        ]
        var cursor = 0
        for (offset, length) in ranges {
            XCTAssertEqual(offset, cursor, "a slot range must start where the last one ended")
            XCTAssertGreaterThan(length, 0)
            cursor += length
        }
        XCTAssertEqual(cursor, geometry.slotBytes)
    }

    func testAnUnalignedWidthHasNoSlot() {
        // The quantisation blocks are 32 wide, so a width that does not divide cannot
        // be indexed by the kernels at all.
        XCTAssertNil(
            DeepSeekV41ExpertSlotGeometry.make(hidden: 5120, intermediate: 2305, format: .mxfp4)
        )
        XCTAssertNil(
            DeepSeekV41ExpertSlotGeometry.make(hidden: 5120, intermediate: 2304, format: .unsupported)
        )
    }

    func testBF16ExpertsCarryNoScaleCompanion() throws {
        let bf16 = try XCTUnwrap(
            DeepSeekV41ExpertSlotGeometry.make(hidden: 5120, intermediate: 2304, format: .bf16)
        )
        XCTAssertEqual(bf16.gateScaleBytes, 0)
        XCTAssertEqual(bf16.gateWeightBytes, 2304 * 5120 * 2)
        XCTAssertEqual(bf16.slotBytes, 3 * 2304 * 5120 * 2)
    }
}

// MARK: - Format decision

final class DeepSeekV41PackedExpertFormatTests: XCTestCase {

    private func tensor(_ dtype: String, rows: Int, columns: Int) -> DeepSeekV41DeviceTensor {
        DeepSeekV41DeviceTensor(
            shardIndex: 0, byteOffset: 0, rows: rows, columns: columns, dtype: dtype
        )
    }

    func testRawBytesWithAPackedRowAreMXFP4() {
        // What this checkpoint actually publishes: `layers.N.ffn.experts.E.w1.weight`
        // is I8 with 2560 columns for a 5120-wide input, and its scale is 160 wide.
        let format = DeepSeekV41DevicePlan.linearFormat(
            tensor: tensor("I8", rows: 2304, columns: 2560),
            scale: tensor("F8_E8M0", rows: 2304, columns: 160)
        )
        XCTAssertEqual(format, .mxfp4)
        XCTAssertEqual(
            DeepSeekV41DevicePlan.swigluKernel(format, swigluLimit: 10),
            DeepSeekV41Kernels.mxfp4SwiGLU
        )
    }

    func testRawBytesWithAFullWidthRowAreTheE4M3Grid() {
        let format = DeepSeekV41DevicePlan.linearFormat(
            tensor: tensor("I8", rows: 2304, columns: 5120),
            scale: tensor("F8_E8M0", rows: 2304, columns: 160)
        )
        XCTAssertEqual(format, .fp8Block32)
        XCTAssertEqual(
            DeepSeekV41DevicePlan.gemvKernel(format),
            DeepSeekV41Kernels.fp8Gemv
        )
    }

    func testRawBytesWithNoBlockScaleStayUnsupported() {
        // An int8 matrix with no block scale is a quantisation this engine has no
        // kernel for, and guessing one would multiply every row by the wrong scale.
        XCTAssertEqual(
            DeepSeekV41DevicePlan.linearFormat(
                tensor: tensor("I8", rows: 2304, columns: 2560),
                scale: nil
            ),
            .unsupported
        )
    }

    func testARowWidthThatFitsNeitherGridIsUnsupported() {
        XCTAssertEqual(
            DeepSeekV41DevicePlan.linearFormat(
                tensor: tensor("I8", rows: 2304, columns: 3000),
                scale: tensor("F8_E8M0", rows: 2304, columns: 160)
            ),
            .unsupported
        )
    }
}

// MARK: - Arena placement

final class DeepSeekV41ExpertArenaPlacementTests: XCTestCase {

    func testAHitIsReportedAsAHit() {
        let arena = DeepSeekV41ExpertArena(slotCount: 4)
        let first = arena.acquireDetailed([(layer: 3, expert: 7)])
        XCTAssertEqual(first.count, 1)
        XCTAssertFalse(first[0].isHit)

        let second = arena.acquireDetailed([(layer: 3, expert: 7)])
        XCTAssertTrue(second[0].isHit)
        XCTAssertEqual(second[0].slot, first[0].slot)
        XCTAssertEqual(arena.hits, 1)
        XCTAssertEqual(arena.evictions, 0)
    }

    func testPairsBeyondCapacityEvictTheOldest() {
        let arena = DeepSeekV41ExpertArena(slotCount: 2)
        _ = arena.acquireDetailed([(layer: 0, expert: 0), (layer: 0, expert: 1)])
        let third = arena.acquireDetailed([(layer: 0, expert: 2)])
        XCTAssertFalse(third[0].isHit)
        XCTAssertEqual(arena.evictions, 1)
        // The oldest pair is the one that left, and the newest one is resident.
        XCTAssertNil(arena.slot(forLayer: 0, expert: 0))
        XCTAssertNotNil(arena.slot(forLayer: 0, expert: 2))
    }

    func testVacatedSlotIsAMissAgain() {
        // The streaming path vacates a slot whose read failed. Without this the arena
        // would serve half-written bytes as a hit for the rest of the turn.
        let arena = DeepSeekV41ExpertArena(slotCount: 2)
        let first = arena.acquireDetailed([(layer: 1, expert: 5)])
        arena.vacate(slot: first[0].slot)
        XCTAssertNil(arena.slot(forLayer: 1, expert: 5))
        let again = arena.acquireDetailed([(layer: 1, expert: 5)])
        XCTAssertFalse(again[0].isHit)
    }

    func testAcquireKeepsItsOldShape() {
        let arena = DeepSeekV41ExpertArena(slotCount: 3)
        let slots = arena.acquire([(layer: 0, expert: 1), (layer: 0, expert: 2)])
        XCTAssertEqual(slots.count, 2)
        XCTAssertNotEqual(slots[0], slots[1])
    }
}

// MARK: - Shard reads

final class DeepSeekV41ExpertShardReaderTests: XCTestCase {

    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("dsv41-expert-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let directory, FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }

    /// A file whose byte value is its own offset, so a read at the wrong offset is
    /// visible in the assertion rather than looking like data.
    private func makeShard(bytes: Int) throws -> URL {
        let url = directory.appendingPathComponent("model-00001-of-00048.safetensors")
        var data = Data(count: bytes)
        data.withUnsafeMutableBytes { raw in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return }
            for offset in 0..<bytes { base[offset] = UInt8(offset % 251) }
        }
        try data.write(to: url)
        return url
    }

    func testDisjointRangesLandAtTheirOffsets() throws {
        let url = try makeShard(bytes: 4096)
        let reader = try DeepSeekV41ExpertShardReader(url: url)
        defer { reader.closeNow() }

        var first = [UInt8](repeating: 0, count: 16)
        var second = [UInt8](repeating: 0, count: 16)
        let read = try first.withUnsafeMutableBytes { firstRaw -> Int in
            try second.withUnsafeMutableBytes { secondRaw -> Int in
                try reader.read([
                    DeepSeekV41ExpertShardReader.Range(
                        fileOffset: 100,
                        destination: firstRaw.baseAddress!,
                        length: 16
                    ),
                    DeepSeekV41ExpertShardReader.Range(
                        fileOffset: 2000,
                        destination: secondRaw.baseAddress!,
                        length: 16
                    )
                ])
            }
        }
        XCTAssertEqual(read, 32)
        XCTAssertEqual(first, (100..<116).map { UInt8($0 % 251) })
        XCTAssertEqual(second, (2000..<2016).map { UInt8($0 % 251) })
    }

    func testARangePastTheEndIsRefusedBeforeItReads() throws {
        let url = try makeShard(bytes: 512)
        let reader = try DeepSeekV41ExpertShardReader(url: url)
        defer { reader.closeNow() }

        var buffer = [UInt8](repeating: 0, count: 64)
        XCTAssertThrowsError(
            try buffer.withUnsafeMutableBytes { raw in
                try reader.read([
                    DeepSeekV41ExpertShardReader.Range(
                        fileOffset: 500,
                        destination: raw.baseAddress!,
                        length: 64
                    )
                ])
            }
        ) { error in
            guard case DeepSeekV41ExpertStreamError.shortRead(_, let offset, let wanted, let got)
                = error else {
                return XCTFail("expected a short-read refusal, got \(error)")
            }
            XCTAssertEqual(offset, 500)
            XCTAssertEqual(wanted, 64)
            XCTAssertEqual(got, 0)
        }
    }

    func testAMissingShardIsAnOpenError() {
        let url = directory.appendingPathComponent("does-not-exist.safetensors")
        XCTAssertThrowsError(try DeepSeekV41ExpertShardReader(url: url))
    }

    func testAClosedReaderRefusesFurtherReads() throws {
        let url = try makeShard(bytes: 512)
        let reader = try DeepSeekV41ExpertShardReader(url: url)
        reader.closeNow()

        var buffer = [UInt8](repeating: 0, count: 8)
        XCTAssertThrowsError(
            try buffer.withUnsafeMutableBytes { raw in
                try reader.read([
                    DeepSeekV41ExpertShardReader.Range(
                        fileOffset: 0,
                        destination: raw.baseAddress!,
                        length: 8
                    )
                ])
            }
        )
    }
}

// MARK: - Against the real checkpoint

/// Reads one real expert through the streamer's own reader.
///
/// The unit tests above prove the arithmetic; this proves the arithmetic matches the
/// file. A slot geometry that is off by a scale block, or a reader that treats the
/// tensor offset as table-relative, still produces finite numbers from the kernels —
/// it just multiplies the wrong bytes. Skipped when the checkpoint is not downloaded,
/// like the rest of the model-bound suites.
final class DeepSeekV41ExpertStreamCheckpointTests: XCTestCase {

    private static let snapshot: URL? = {
        let root = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots")
        let directories = (try? FileManager.default.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey]
        )) ?? []
        return directories
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .first { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.safetensors.index.json").path) }
    }()

    private struct Header: Decodable {
        struct Entry: Decodable {
            let dtype: String
            let shape: [Int]
            let dataOffsets: [Int]

            enum CodingKeys: String, CodingKey {
                case dtype, shape
                case dataOffsets = "data_offsets"
            }
        }

        let entries: [String: Entry]

        init(from decoder: Decoder) throws {
            entries = try [String: Entry](from: decoder)
        }

        func encode(to encoder: Encoder) throws {}
    }

    private func requireSnapshot() throws -> URL {
        try XCTSkipIf(
            Self.snapshot == nil,
            "DeepSeek-V4.1-Flash snapshot is not downloaded"
        )
        return Self.snapshot!
    }

    /// The JSON header safetensors puts before the data section.
    private func readHeader(shard: URL) throws -> (Header, Int) {
        let handle = try FileHandle(forReadingFrom: shard)
        defer { try? handle.close() }
        let lengthData = try handle.read(upToCount: 8) ?? Data()
        XCTAssertEqual(lengthData.count, 8)
        let length = lengthData.withUnsafeBytes { $0.load(as: UInt64.self) }
        let headerData = try handle.read(upToCount: Int(length)) ?? Data()
        let header = try JSONDecoder().decode(Header.self, from: headerData)
        return (header, 8 + Int(length))
    }

    func testOneRealExpertReadsEveryRangeTheKernelsIndex() throws {
        let snapshot = try requireSnapshot()

        // Which shard holds `layers.2.ffn.experts.0`.
        let indexURL = snapshot.appendingPathComponent("model.safetensors.index.json")
        let indexJSON = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL))
        let weightMap = try XCTUnwrap(
            (indexJSON as? [String: Any])?["weight_map"] as? [String: String]
        )
        let roles: [(String, String)] = [
            ("gate", "layers.2.ffn.experts.0.w1"),
            ("up", "layers.2.ffn.experts.0.w3"),
            ("down", "layers.2.ffn.experts.0.w2")
        ]
        let shardName = try XCTUnwrap(weightMap["layers.2.ffn.experts.0.w1.weight"])
        let shardURL = snapshot.appendingPathComponent(shardName)
        let (header, dataStart) = try readHeader(shard: shardURL)

        let geometry = try XCTUnwrap(
            DeepSeekV41ExpertSlotGeometry.make(hidden: 5120, intermediate: 2304, format: .mxfp4)
        )
        let reader = try DeepSeekV41ExpertShardReader(url: shardURL)
        defer { reader.closeNow() }

        var slot = [UInt8](repeating: 0, count: geometry.slotBytes)
        let plan: [(offset: Int, length: Int, name: String)] = [
            (geometry.gateWeightOffset, geometry.gateWeightBytes, roles[0].1 + ".weight"),
            (geometry.gateScaleOffset, geometry.gateScaleBytes, roles[0].1 + ".scale"),
            (geometry.upWeightOffset, geometry.upWeightBytes, roles[1].1 + ".weight"),
            (geometry.upScaleOffset, geometry.upScaleBytes, roles[1].1 + ".scale"),
            (geometry.downWeightOffset, geometry.downWeightBytes, roles[2].1 + ".weight"),
            (geometry.downScaleOffset, geometry.downScaleBytes, roles[2].1 + ".scale")
        ]
        try slot.withUnsafeMutableBytes { raw in
            let ranges = try plan.map { entry -> DeepSeekV41ExpertShardReader.Range in
                let tensor = try XCTUnwrap(
                    header.entries[entry.name],
                    "the shard header does not publish \(entry.name)"
                )
                XCTAssertEqual(abs(tensor.shape[0] * tensor.shape[1]), entry.length,
                               "\(entry.name) is \(tensor.shape), which is not the slot range")
                return DeepSeekV41ExpertShardReader.Range(
                    fileOffset: dataStart + tensor.dataOffsets[0],
                    destination: raw.baseAddress!.advanced(by: entry.offset),
                    length: entry.length
                )
            }
            _ = try reader.read(ranges)
        }

        // Every range has to carry bytes: a wrong offset reads padding, and padding
        // is zero, which the kernels would multiply through as a dead expert.
        let checks: [(offset: Int, length: Int, name: String)] = plan.map {
            (offset: $0.offset, length: $0.length, name: $0.name)
        }
        for check in checks {
            let slice = Array(slot[check.offset..<(check.offset + check.length)])
            XCTAssertTrue(
                slice.contains { $0 != 0 },
                "\(check.name) read as all zeros from \(shardName)"
            )
        }
        // The gate and up projections of one expert are different matrices, so an
        // offset collision would show up as identical bytes.
        XCTAssertNotEqual(
            Array(slot[geometry.gateWeightOffset..<(geometry.gateWeightOffset + 4096)]),
            Array(slot[geometry.upWeightOffset..<(geometry.upWeightOffset + 4096)])
        )
    }
}
