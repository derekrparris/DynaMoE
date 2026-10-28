//
//  DeepSeekV41Engram.swift
//  DynaMoE
//
//  Engram: DeepSeek-V4.1's n-gram conditional memory.
//
//  Each Engram layer owns one table of ~384M rows. A row is `engram_head_dim`
//  E4M3 values plus one E8M0 scale per 32 bytes of scales (264 bytes for a
//  256-wide head). The tables are ~100 GB each, so they are NEVER mapped: the
//  runtime preads the handful of rows one token needs, which is why the
//  address arithmetic lives here as pure, testable code.
//
//  Layout, from the reference (`inference/engram.py` and the vLLM port):
//    * A position is hashed as `max_ngram_size - 1` n-grams (2-gram .. max),
//      and each n-gram is split over `engram_n_heads` heads. Every
//      (n-gram size, head) pair owns its own PRIME-sized bucket range; the
//      primes are drawn in ascending order from `engram_vocab_size` and never
//      reused, so the ranges stay disjoint by construction.
//    * Column 0 of a layer's table is the 2-gram of head 0, column 1 the
//      2-gram of head 1, and so on. `engram_num_embeddings` is the sum of the
//      bucket sizes, i.e. exactly the number of addressable rows.
//
//  The 32-bit hash mix below is DeepSeek's `engram_hash_i32` (xor-shift /
//  0x7feb352d / xor-shift), taken from the published tile-kernel oracle. The
//  COMBINATION of the n-gram's token ids into that hash is marked unverified
//  below: it must be confirmed against the checkpoint's own `inference/engram.py`
//  by `Tools/dsv41_engram_probe.py`, which is why the runtime refuses Engram
//  until `DeepSeekV41EngramHasher.conformance` says otherwise rather than
//  silently reading the wrong rows.
//

import Foundation

// MARK: - Prime buckets

/// Bucket arithmetic for one Engram table.
nonisolated public struct DeepSeekV41EngramLayout: Sendable, Equatable {
    public let spec: DeepSeekV41EngramSpec
    /// Bucket size of each hash column, in column order.
    public let bucketSizes: [Int]
    /// First row index of each hash column. Disjoint and ascending.
    public let bucketOffsets: [Int]

    /// `max_ngram_size - 1` n-gram sizes times `engram_n_heads` heads.
    public var hashColumnCount: Int { max(0, spec.maxNgramSize - 1) * spec.headCount }

    /// Rows the bucket arithmetic addresses. Must not exceed `spec.rowCount`.
    public var addressedRows: Int { bucketSizes.reduce(0, +) }

    /// Column index for one (n-gram size, head) pair. `ngramSize` is the real
    /// n-gram length (2..max), not the zero-based offset.
    public func column(ngramSize: Int, head: Int) -> Int {
        (ngramSize - 2) * spec.headCount + head
    }

    /// Byte offset of one row inside the table file.
    public func byteOffset(row: Int) -> Int {
        row * spec.rowBytes
    }

    /// Byte offset of the bucket-th row of a column.
    public func byteOffset(column: Int, bucket: Int) -> Int {
        let row = bucketOffsets[column] + bucket
        return byteOffset(row: row)
    }

    public init(spec: DeepSeekV41EngramSpec) throws {
        guard spec.maxNgramSize >= 2 else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_max_ngram_size", "must be at least 2"
            )
        }
        guard spec.headCount > 0 else {
            throw DeepSeekV41ConfigError.invalidField("engram_n_heads", "must be positive")
        }
        guard spec.bucketStart > 0 else {
            throw DeepSeekV41ConfigError.invalidField("engram_vocab_size", "must be positive")
        }
        let columns = (spec.maxNgramSize - 1) * spec.headCount
        var sizes: [Int] = []
        var offsets: [Int] = []
        sizes.reserveCapacity(columns)
        offsets.reserveCapacity(columns)
        var cursor = spec.bucketStart
        var row = 0
        for _ in 0..<columns {
            let prime = DeepSeekV41Primes.next(atLeast: cursor)
            sizes.append(prime)
            offsets.append(row)
            row += prime
            // The next range starts after this one, so ranges can never overlap
            // even if a later prime is smaller than an earlier start.
            cursor = prime + 1
        }
        if row > spec.rowCount {
            throw DeepSeekV41ConfigError.inconsistentLayout(
                "engram table for layer \(spec.layer) needs \(row) rows for its \(columns) hash buckets but engram_num_embeddings is \(spec.rowCount)"
            )
        }
        self.spec = spec
        self.bucketSizes = sizes
        self.bucketOffsets = offsets
    }
}

/// Prime search for the Engram bucket layout. Deterministic trial division over
/// odd candidates is fast enough for the ~24 columns a table needs and avoids
/// any dependency on a big-integer primality library.
nonisolated public enum DeepSeekV41Primes {
    public static func isPrime(_ value: Int) -> Bool {
        guard value >= 2 else { return false }
        if value % 2 == 0 { return value == 2 }
        if value % 3 == 0 { return value == 3 }
        var factor = 5
        while factor * factor <= value {
            if value % factor == 0 || value % (factor + 2) == 0 { return false }
            factor += 6
        }
        return true
    }

    public static func next(atLeast value: Int) -> Int {
        var candidate = max(2, value)
        while !isPrime(candidate) { candidate += 1 }
        return candidate
    }
}

// MARK: - Hashing

/// Whether the n-gram hash combination has been matched against the
/// checkpoint's own `inference/engram.py`. The runtime refuses to serve Engram
/// while this is `.unverified`: a wrong combination reads the wrong memory rows,
/// which degrades text silently instead of failing.
nonisolated public enum DeepSeekV41EngramConformance: String, Sendable, Equatable {
    case unverified
    case verifiedAgainstReference
}

nonisolated public enum DeepSeekV41EngramHasher {

    /// Conformance gate for the token-id combination step below.
    ///
    /// Flipped to `.verifiedAgainstReference` only by a run of
    /// `Tools/dsv41_engram_probe.py` that reproduces `inference/engram.py`'s
    /// hashes for the shipped `encoding/tests/*` fixtures, or by the equivalent
    /// XCTest with the reference vectors checked in.
    public static let conformance: DeepSeekV41EngramConformance = .unverified

    /// DeepSeek's `engram_hash_i32` finalizer: xor-shift 16, multiply by
    /// 0x7feb352d, xor-shift 15. Identical to the published tile-kernel oracle
    /// (`deepseek_engram_hash_i32`), which asserts it lane by lane.
    public static func mix(_ value: UInt32) -> UInt32 {
        var h = value
        h ^= h >> 16
        h = h &* 0x7feb_352d
        h ^= h >> 15
        return h
    }

    /// Folds the compressed token ids of one n-gram into a seed.
    ///
    /// UNVERIFIED: the ordering and the multiplier are the part the reference
    /// implementation defines, not the finalizer above. One function so the
    /// probe result is a one-line change once confirmed.
    public static func seed(tokenIDs: [Int32], column: Int) -> UInt32 {
        var h: UInt32 = 0x811c_9dc5
        for id in tokenIDs {
            h = (h ^ UInt32(bitPattern: id)) &* 0x0100_0193
        }
        return (h ^ UInt32(truncatingIfNeeded: column)) &* 0x9e37_79b1
    }

    /// Bucket index of one n-gram in one column of `layout`.
    public static func bucket(
        tokenIDs: [Int32],
        column: Int,
        layout: DeepSeekV41EngramLayout
    ) -> Int {
        precondition(column >= 0 && column < layout.hashColumnCount, "engram column out of range")
        let mixed = mix(seed(tokenIDs: tokenIDs, column: column))
        return Int(mixed % UInt32(layout.bucketSizes[column]))
    }
}

// MARK: - Token map

/// Maps a checkpoint token id onto the compressed vocabulary the hash draws
/// from (`engram_compressed_vocab_size`, ~99k for the published checkpoint).
nonisolated public struct DeepSeekV41EngramTokenMap: Sendable, Equatable {
    public let compressedVocabSize: Int
    public let padTokenID: Int
    /// `map[tokenID]`, valid for `0..<rowCount`.
    public let map: [Int32]

    public func compressedToken(_ tokenID: Int) -> Int32 {
        guard tokenID >= 0, tokenID < map.count else { return Int32(padTokenID) }
        return map[tokenID]
    }

    /// Builds the map. UNVERIFIED in the same sense as the hasher: the
    /// reference builds it from the tokenizer and asserts it against
    /// `engram_compressed_vocab_size`, which is exactly what this asserts too.
    public init(vocabSize: Int, compressedVocabSize: Int, padTokenID: Int) throws {
        guard vocabSize > 0, compressedVocabSize > 0 else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_compressed_vocab_size", "must be positive with a positive vocab_size"
            )
        }
        guard compressedVocabSize <= vocabSize else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_compressed_vocab_size",
                "\(compressedVocabSize) exceeds the tokenizer vocabulary \(vocabSize)"
            )
        }
        var table = [Int32](repeating: Int32(padTokenID), count: vocabSize)
        for id in 0..<vocabSize {
            table[id] = Int32(id % compressedVocabSize)
        }
        table[padTokenID] = Int32(padTokenID)
        self.compressedVocabSize = compressedVocabSize
        self.padTokenID = padTokenID
        self.map = table
    }
}

// MARK: - Row request

/// One (layer, column, bucket) read the runtime has to issue for a token.
nonisolated public struct DeepSeekV41EngramRowRequest: Sendable, Equatable {
    public let layer: Int
    public let column: Int
    public let bucket: Int
    public let byteOffset: Int
    public let byteCount: Int

    public init(layer: Int, column: Int, bucket: Int, byteOffset: Int, byteCount: Int) {
        self.layer = layer
        self.column = column
        self.bucket = bucket
        self.byteOffset = byteOffset
        self.byteCount = byteCount
    }
}

nonisolated public enum DeepSeekV41Engram {

    /// The row requests for one token position.
    ///
    /// `history` is the token stream ending at the current position, oldest
    /// first; the last element is the token being embedded. Positions before the
    /// start of the sequence use `spec.padTokenID`, matching training.
    ///
    /// Returns one request per (n-gram size, head) column, i.e.
    /// `(max_ngram_size - 1) * n_heads` reads per Engram layer per token:
    /// 24 reads of 264 bytes on the published checkpoint.
    public static func rowRequests(
        layer: Int,
        history: [Int32],
        tokenMap: DeepSeekV41EngramTokenMap,
        layout: DeepSeekV41EngramLayout
    ) -> [DeepSeekV41EngramRowRequest] {
        guard layout.spec.layer == layer else { return [] }
        let spec = layout.spec
        var requests: [DeepSeekV41EngramRowRequest] = []
        requests.reserveCapacity(layout.hashColumnCount)
        for ngramSize in 2...max(2, spec.maxNgramSize) {
            // The last `ngramSize` tokens, padded at the front when the sequence
            // is shorter than the n-gram.
            var window = [Int32](repeating: Int32(spec.padTokenID), count: ngramSize)
            let available = min(ngramSize, history.count)
            if available > 0 {
                window.replaceSubrange(
                    (ngramSize - available)..<ngramSize,
                    with: history.suffix(available)
                )
            }
            let compressed = window.map { tokenMap.compressedToken(Int($0)) }
            for head in 0..<spec.headCount {
                let column = layout.column(ngramSize: ngramSize, head: head)
                let bucket = DeepSeekV41EngramHasher.bucket(
                    tokenIDs: compressed,
                    column: column,
                    layout: layout
                )
                requests.append(
                    DeepSeekV41EngramRowRequest(
                        layer: layer,
                        column: column,
                        bucket: bucket,
                        byteOffset: layout.byteOffset(column: column, bucket: bucket),
                        byteCount: spec.rowBytes
                    )
                )
            }
        }
        return requests
    }
}
