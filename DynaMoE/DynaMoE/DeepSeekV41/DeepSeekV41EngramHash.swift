//
//  DeepSeekV41EngramHash.swift
//  DynaMoE
//
//  The hash multipliers DeepSeek-V4.1's Engram tables are indexed with.
//
//  `inference/engram.py` derives them with numpy:
//
//      multiplier_bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
//      generator = np.random.default_rng(10007 * layer_id)
//      values = generator.integers(0, multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
//      multipliers = values * 2 + 1
//
//  There is no way to read those numbers off the checkpoint: they are not stored,
//  only used to index the tables at run time. So the generator is reproduced here
//  exactly. numpy's `default_rng(seed)` is `SeedSequence(seed)` feeding PCG64, and
//  `integers` on int64 draws through Lemire's multiply-shift, so this file is
//  numpy's `SeedSequence`, `PCG64`, and `bounded_lemire_uint64` in that order.
//  Each piece is pinned against values printed by numpy itself; see
//  `DeepSeekV41EngramHashTests`.
//
//  Why this matters beyond tidiness: a wrong multiplier changes every bucket a
//  token reads, which degrades generation silently. It cannot be caught by any
//  shape or range check, only by reproducing the generator.
//

import Foundation

// MARK: - numpy's SeedSequence + PCG64

/// numpy's `np.random.default_rng(seed)`, reproduced bit for bit.
///
/// Only the paths `compute_hash_multipliers` takes are implemented: an integer
/// entropy source, a 4-word pool, `generate_state` into the 128-bit PCG64 state,
/// and 64-bit bounded draws above `0xFFFFFFFF`. Below that bound numpy delegates
/// to a 32-bit buffered path with different results, which is why
/// `uniform(below:)` refuses it instead of returning a neighbouring stream.
nonisolated public struct DeepSeekV41NumpyRandom {

    // SeedSequence mixing constants (`numpy/random/bit_generator.pyx`).
    private static let poolSize = 4
    private static let initialHash: UInt32 = 0x43b0_d7e5
    private static let multiplierA: UInt32 = 0x931e_8875
    private static let initialHashB: UInt32 = 0x8b51_f9dd
    private static let multiplierB: UInt32 = 0x58f3_8ded
    private static let mixMultiplierLeft: UInt32 = 0xca01_f9dd
    private static let mixMultiplierRight: UInt32 = 0x4973_f715
    private static let mixShift: UInt32 = 16

    // PCG64's 128-bit LCG multiplier, high word first.
    private static let lcgMultiplierHigh: UInt64 = 2_549_297_995_355_413_924
    private static let lcgMultiplierLow: UInt64 = 4_865_540_595_714_422_341

    /// Below this, numpy does not use the 64-bit multiply-shift this reproduces.
    private static let lemireMinimumBound: UInt64 = 0x1_0000_0000

    private var state: UInt128
    private let increment: UInt128

    public init(seed: UInt64) {
        let words = Self.generateState(entropy: seed, wordCount: 8)
        // `PCG64.__init__` views those 8 uint32 words as 4 little-endian uint64s and
        // seeds PCG with the first two as the state and the last two as the increment.
        let halves = (0..<4).map { index -> UInt64 in
            UInt64(words[index * 2]) | (UInt64(words[index * 2 + 1]) << 32)
        }
        let seedState = (UInt128(halves[0]) << 64) | UInt128(halves[1])
        let seedStream = (UInt128(halves[2]) << 64) | UInt128(halves[3])
        let multiplier = (UInt128(Self.lcgMultiplierHigh) << 64) | UInt128(Self.lcgMultiplierLow)

        self.state = 0
        self.increment = (seedStream << 1) | 1
        self.lcgMultiplier = multiplier
        // `pcg_setseq_128_srandom_r`: two steps, with the seed added in between.
        step()
        state = state &+ seedState
        step()
    }

    private let lcgMultiplier: UInt128

    private mutating func step() {
        state = state &* lcgMultiplier &+ increment
    }

    /// `pcg_setseq_128_xsl_rr_64_random_r`: step, then XSL-RR the new state.
    public mutating func next() -> UInt64 {
        step()
        let high = UInt64(truncatingIfNeeded: state >> 64)
        let low = UInt64(truncatingIfNeeded: state)
        return Self.rotateRight(high ^ low, by: high >> 58)
    }

    private static func rotateRight(_ value: UInt64, by rotation: UInt64) -> UInt64 {
        let shift = UInt64(rotation & 63)
        if shift == 0 { return value }
        return (value >> shift) | (value << (64 - shift))
    }

    /// Whether numpy would draw this bound from the 64-bit path reproduced here.
    ///
    /// Below `2^32` numpy routes the draw through a 32-bit buffered stream whose
    /// values are different, so this generator declines rather than returning a
    /// neighbouring sequence. No real tokenizer vocabulary puts the Engram
    /// multiplier bound near that threshold (it sits at ~4.6e13 for the published
    /// checkpoint's 99k compressed vocabulary), so declining costs nothing today and
    /// keeps a future checkpoint from hashing silently wrong.
    public static func canReproduce(bound: Int64) -> Bool {
        bound >= 1 && UInt64(bound) >= lemireMinimumBound
    }

    /// A uniform value in `0..<bound` via Lemire's multiply-shift with rejection, or
    /// nil when `canReproduce(bound:)` says no.
    public mutating func uniform(below bound: Int64) -> Int64? {
        guard Self.canReproduce(bound: bound) else { return nil }
        let range = UInt64(bound)
        var product = UInt128(next()) * UInt128(range)
        var low = UInt64(truncatingIfNeeded: product)
        if low < range {
            let threshold = (UInt64(0) &- range) % range
            while low < threshold {
                product = UInt128(next()) * UInt128(range)
                low = UInt64(truncatingIfNeeded: product)
            }
        }
        return Int64(UInt64(truncatingIfNeeded: product >> 64))
    }

    // MARK: SeedSequence

    /// `SeedSequence.generate_state`: mix the entropy into a 4-word pool, then run
    /// the pool through the output hash `wordCount` times.
    private static func generateState(entropy: UInt64, wordCount: Int) -> [UInt32] {
        let entropyWords = words(of: entropy)
        var pool = [UInt32](repeating: 0, count: poolSize)
        var hashState = initialHash

        for index in 0..<poolSize {
            let value = index < entropyWords.count ? entropyWords[index] : 0
            pool[index] = hashMix(value, hashState: &hashState)
        }
        // Mix every pool word into every other, so a late entropy word can still
        // affect an earlier pool slot.
        for source in 0..<poolSize {
            for destination in 0..<poolSize where source != destination {
                pool[destination] = mix(pool[destination], hashMix(pool[source], hashState: &hashState))
            }
        }
        for source in poolSize..<max(poolSize, entropyWords.count) where source < entropyWords.count {
            for destination in 0..<poolSize {
                pool[destination] = mix(
                    pool[destination],
                    hashMix(entropyWords[source], hashState: &hashState)
                )
            }
        }

        var output = [UInt32](repeating: 0, count: wordCount)
        var outputHash = initialHashB
        for index in 0..<wordCount {
            var value = pool[index % poolSize]
            value ^= outputHash
            outputHash = outputHash &* multiplierB
            value = value &* outputHash
            value ^= value >> mixShift
            output[index] = value
        }
        return output
    }

    /// Little-endian 32-bit words of the entropy, lowest first, with a zero seed
    /// still contributing one word.
    private static func words(of entropy: UInt64) -> [UInt32] {
        if entropy == 0 { return [0] }
        var remaining = entropy
        var output: [UInt32] = []
        while remaining > 0 {
            output.append(UInt32(truncatingIfNeeded: remaining))
            remaining >>= 32
        }
        return output
    }

    private static func hashMix(_ value: UInt32, hashState: inout UInt32) -> UInt32 {
        var mixed = value ^ hashState
        hashState = hashState &* multiplierA
        mixed = mixed &* hashState
        mixed ^= mixed >> mixShift
        return mixed
    }

    private static func mix(_ left: UInt32, _ right: UInt32) -> UInt32 {
        var mixed = mixMultiplierLeft &* left &- mixMultiplierRight &* right
        mixed ^= mixed >> mixShift
        return mixed
    }
}

// MARK: - Multipliers

nonisolated public enum DeepSeekV41EngramMultipliers {

    /// One odd multiplier per look-back, for one Engram layer.
    ///
    /// The per-layer seed (`10007 * layer_id`) is why two Engram layers hash the
    /// same n-gram into different buckets; sharing one stream across layers would
    /// correlate them. Multipliers are forced odd and bounded so that
    /// `token_id * multiplier` cannot overflow int64.
    public static func multipliers(
        layerID: Int,
        maxNgramSize: Int,
        compressedVocabSize: Int
    ) -> [Int64]? {
        guard maxNgramSize > 0 else { return [] }
        let bound = multiplierBound(compressedVocabSize: compressedVocabSize)
        guard DeepSeekV41NumpyRandom.canReproduce(bound: bound) else { return nil }
        var generator = DeepSeekV41NumpyRandom(
            seed: UInt64(bitPattern: Int64(10_007 * layerID))
        )
        var output: [Int64] = []
        output.reserveCapacity(maxNgramSize)
        for _ in 0..<maxNgramSize {
            guard let value = generator.uniform(below: bound) else { return nil }
            output.append(value &* 2 &+ 1)
        }
        return output
    }

    /// `max(1, (int64.max // vocab_size) // 2)`, the exclusive bound numpy draws
    /// below.
    public static func multiplierBound(compressedVocabSize: Int) -> Int64 {
        let vocabulary = Int64(max(1, compressedVocabSize))
        return max(1, (Int64.max / vocabulary) / 2)
    }
}
