//
//  DeepSeekV41EngramHashTests.swift
//  DynaMoETests
//
//  The Engram hash multipliers are not stored in the checkpoint: they are drawn
//  from `np.random.default_rng(10007 * layer_id)` when the model is constructed.
//  Serving from a different stream reads different rows and still produces finite
//  output, so these tests pin the generator against numbers numpy itself printed.
//
//  Reference values were produced with numpy 2.5.4:
//
//      >>> np.random.PCG64(0).random_raw(4)
//      array([11749869230777074271, 4976686463289251617, 755828109848996024,
//              304881062738325533], dtype=uint64)
//      >>> g = np.random.default_rng(10007)
//      >>> (g.integers(0, 46539438283891, size=3, dtype=np.int64) * 2 + 1)
//      array([949203457676829153, 59949125229827907, 445414481183620465])
//

import XCTest
@testable import DynaMoE

final class DeepSeekV41EngramHashTests: XCTestCase {

    /// `compute_hash_multipliers` on the published checkpoint: 99,092 compressed
    /// tokens, `max_ngram_size` 4, layers 1 and 14.
    private let publishedCompressedVocabSize = 99_092

    func testMultipliersMatchThePublishedCheckpoint() throws {
        let first = try XCTUnwrap(DeepSeekV41EngramMultipliers.multipliers(
            layerID: 1, maxNgramSize: 4, compressedVocabSize: publishedCompressedVocabSize
        ))
        XCTAssertEqual(
            first,
            [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391]
        )
        let second = try XCTUnwrap(DeepSeekV41EngramMultipliers.multipliers(
            layerID: 14, maxNgramSize: 4, compressedVocabSize: publishedCompressedVocabSize
        ))
        XCTAssertEqual(
            second,
            [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591]
        )
        // The per-layer seed is the whole point: two layers must not share a stream.
        XCTAssertNotEqual(first, second)
        XCTAssertTrue((first + second).allSatisfy { $0 % 2 == 1 })
    }

    /// The fixture vocabulary is small enough that the bound still lands above
    /// `0xFFFFFFFF`, so the same reproduction path is exercised, but the values are
    /// cheap to eyeball.
    func testMultipliersForASmallVocabulary() {
        let multipliers = DeepSeekV41EngramMultipliers.multipliers(
            layerID: 1, maxNgramSize: 3, compressedVocabSize: 8
        )
        XCTAssertNotNil(multipliers)
        XCTAssertEqual(
            multipliers,
            [949_203_457_676_829_153, 59_949_125_229_827_907, 445_414_481_183_620_465]
        )
    }

    /// `multiplier_bound = max(1, (int64.max // vocab_size) // 2)`, kept below
    /// int64.max / vocab so `token_id * multiplier` cannot overflow.
    func testMultiplierBoundStaysInsideInt64() {
        let bound = DeepSeekV41EngramMultipliers.multiplierBound(
            compressedVocabSize: publishedCompressedVocabSize
        )
        XCTAssertEqual(bound, 46_539_438_283_891)
        // The largest token times the largest allowed multiplier fits.
        let largest = (bound * 2 + 1) &* Int64(publishedCompressedVocabSize)
        XCTAssertGreaterThan(largest, 0)
    }

    func testSeedSequenceMatchesNumpy() {
        // np.random.PCG64(seed).random_raw(4)
        let expected: [UInt64: [UInt64]] = [
            0: [11_749_869_230_777_074_271, 4_976_686_463_289_251_617,
                755_828_109_848_996_024, 304_881_062_738_325_533],
            10_007: [15_187_255_322_829_266_483, 959_186_003_677_246_517,
                     7_126_631_698_937_927_467, 14_663_106_486_854_050_627]
        ]
        for (seed, raw) in expected {
            var generator = DeepSeekV41NumpyRandom(seed: seed)
            XCTAssertEqual((0..<raw.count).map { _ in generator.next() }, raw, "seed \(seed)")
        }
    }

    /// A non-power-of-two bound goes through Lemire's multiply-shift with a
    /// rejection threshold, which is exactly where a naive `raw % bound` differs.
    func testBoundedDrawDoesNotReduceModulo() {
        var generator = DeepSeekV41NumpyRandom(seed: 0)
        let drawn = generator.uniform(below: 46_539_438_283_891)
        // `raw % bound` on the first raw value would be 10_707_804_829_610.
        XCTAssertEqual(drawn, 29_643_839_136_299)
    }

    /// Below `0xFFFFFFFF` numpy switches to a 32-bit buffered stream, so the
    /// generator declines rather than returning a different sequence, and a layout
    /// whose multipliers cannot be reproduced is a load error.
    func testSmallBoundsAreRefusedRatherThanApproximated() {
        var generator = DeepSeekV41NumpyRandom(seed: 1)
        XCTAssertNil(generator.uniform(below: 2))
        XCTAssertFalse(DeepSeekV41NumpyRandom.canReproduce(bound: 2))
        XCTAssertTrue(DeepSeekV41NumpyRandom.canReproduce(bound: 0x1_0000_0000))
        XCTAssertNil(
            DeepSeekV41EngramMultipliers.multipliers(
                layerID: 1, maxNgramSize: 4, compressedVocabSize: 4_000_000_000
            )
        )
    }
}

// MARK: - Row dequantization

final class DeepSeekV41EngramDequantTests: XCTestCase {

    /// E4M3 with bias 7: exponent 0 is subnormal (2^-6), the rest are
    /// `(1 + mant/8) * 2^(exp - 7)`. Values chosen to cover a normal, a fraction,
    /// a negative, a subnormal and the maximum exponent.
    func testE4M3Decode() {
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0x38), 1.0)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0x3C), 1.5)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0xB8), -1.0)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0x00), 0)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0x01), 0.001953125)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e4m3(0x78), 256)
    }

    /// E8M0 is a bare exponent: `2^(s - 127)`, with byte 0 meaning zero.
    func testE8M0Decode() {
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e8m0(127), 1.0)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e8m0(128), 2.0)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e8m0(126), 0.5)
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.e8m0(0), 0)
    }

    /// One scale per 32 values, stored after the values.
    func testRowAppliesBlockScalesAfterTheValues() {
        // 4 values, one scale of 1.0 at index 4.
        let raw: [UInt8] = [0x38, 0x3C, 0xB8, 0x01, 127]
        XCTAssertEqual(
            DeepSeekV41EngramRowDequantizer.row(raw, headDimension: 4),
            [1.0, 1.5, -1.0, 0.001953125]
        )
        // A scale of 2.0 doubles every value in the block.
        let doubled: [UInt8] = [0x38, 0x3C, 0xB8, 0x01, 128]
        XCTAssertEqual(
            DeepSeekV41EngramRowDequantizer.row(doubled, headDimension: 4),
            [2.0, 3.0, -2.0, 0.00390625]
        )
        // A short buffer yields zeros rather than reading past the end.
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.row([0x38], headDimension: 4), [0, 0, 0, 0])
    }

    /// The published row is 256 values plus 8 scales; the unpacked row must be
    /// exactly the head dimension, which is what the projection's input width is
    /// built from.
    func testPublishedRowShape() {
        let spec = DeepSeekV41EngramSpec(
            layer: 1, rowCount: 64, maxNgramSize: 4, headCount: 8, headDimension: 256,
            padTokenID: 2, compressedVocabSize: 99_092, bucketStart: 16_000_000
        )
        XCTAssertEqual(DeepSeekV41EngramRowDequantizer.scaleCount(headDimension: 256), 8)
        XCTAssertEqual(spec.rowBytes, 264)
        // 256 values of 1.0 followed by the eight block scales that make them 1.0.
        let raw = [UInt8](repeating: 0x38, count: 256) + [UInt8](repeating: 127, count: 8)
        XCTAssertEqual(raw.count, spec.rowBytes)
        let decoded = DeepSeekV41EngramRowDequantizer.row(raw, headDimension: 256)
        XCTAssertEqual(decoded.count, 256)
        XCTAssertTrue(decoded.allSatisfy { $0 == 1 })
    }
}

// MARK: - Checkpoint token map

/// The Engram hash draws from a compressed vocabulary built from the tokenizer, so
/// the Swift side has to reach the Rust builder and get the checkpoint's own
/// number back. 99,092 is the value `engram_compressed_vocab_size` declares, and
/// nothing in the reference constrains the normalization chain except that
/// number, so reproducing it is the whole check.
final class DeepSeekV41EngramTokenMapTests: XCTestCase {

    private func snapshotDirectory() -> URL? {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots")
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: nil
        ) else { return nil }
        return entries.first { FileManager.default.fileExists(
            atPath: $0.appendingPathComponent("tokenizer.json").path
        ) }
    }

    func testCompressedMapMatchesTheCheckpoint() throws {
        guard let snapshot = snapshotDirectory() else {
            throw XCTSkip("DeepSeek-V4.1 snapshot not downloaded")
        }
        // The HF wrapper is the shape the app loads; its `text_config` carries the
        // Engram fields, and it is the one that identifies as DeepSeek V4.1.
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let lookup = try DeepSeekV41EngramStreamingLookup.make(
            snapshotDir: snapshot, config: config, readers: [:], layouts: [:]
        )
        XCTAssertEqual(lookup.tokenMap.compressedVocabSize, 99_092)
        XCTAssertEqual(lookup.tokenMap.map.count, 129_280)
        // The pad token's COMPRESSED id, which is what a blocked look-back hashes as.
        XCTAssertEqual(lookup.tokenMap.padTokenID, 2)
    }

    /// The two Engram tables the checkpoint ships, resolved through the real
    /// config. Published row counts are the second half of the layout proof.
    func testPublishedLayoutsComeFromTheRealConfig() throws {
        guard let snapshot = snapshotDirectory() else {
            throw XCTSkip("DeepSeek-V4.1 snapshot not downloaded")
        }
        // The HF wrapper is the shape the app loads; its `text_config` carries the
        // Engram fields, and it is the one that identifies as DeepSeek V4.1.
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        XCTAssertEqual(config.engramLayerIDs, [1, 14])
        let layouts = try DeepSeekV41EngramLayout.make(specs: config.engramTables)
        XCTAssertEqual(layouts[1]?.addressedRows, 384_006_168)
        XCTAssertEqual(layouts[14]?.addressedRows, 384_016_682)
        // Layer 1's multipliers are the ones the generator produced above.
        XCTAssertEqual(
            layouts[1]?.multipliers.first,
            76_632_096_046_245
        )
    }
}
