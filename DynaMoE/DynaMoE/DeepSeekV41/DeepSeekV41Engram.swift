//
//  DeepSeekV41Engram.swift
//  DynaMoE
//
//  Engram: DeepSeek-V4.1's n-gram conditional memory.
//
//  Each Engram layer owns one table of ~384M rows. A row is `engram_head_dim`
//  E4M3 values plus one E8M0 scale per 32 (264 bytes for a 256-wide head). The
//  tables are ~100 GB each, so they are NEVER mapped: the runtime preads the
//  handful of rows one token needs, which is why the address arithmetic lives
//  here as pure, testable code.
//
//  Layout and hashing, from the checkpoint's `inference/engram.py`
//  (`EngramLayout.from_args`, `NgramHashState.forward`):
//    * Every (n-gram size 2..max_ngram_size, head) pair owns its own PRIME-sized
//      bucket range. The primes are drawn in ONE ascending sequence that runs
//      across every Engram layer and is never reused, so the ranges stay disjoint
//      by construction. That is why `engram_num_embeddings` differs per layer
//      (384_006_168 then 384_016_682 on the published checkpoint): the second
//      table's primes are the next 24 primes after the first table's.
//    * A position is hashed as `max_ngram_size` look-backs. Look-back `i` reads
//      the token `i` positions back, compressed into the smaller Engram
//      vocabulary, multiplied by that layer's multiplier for `i`. The running
//      value after step `i` is the XOR of the first `i + 1` products, i.e. the
//      hash of the `(i + 1)`-gram, and it is reduced mod the prime that owns
//      (ngram size `i + 1`, head `h`) then offset into the table.
//    * A look-back stops at the start of the sequence and at any dead token (an
//      image span), so an n-gram never spans one; every slot past the stop hashes
//      as the pad token.
//
//  The multipliers are not stored in the checkpoint. They come from numpy's
//  `default_rng`, reproduced in `DeepSeekV41EngramHash.swift`.
//

import Foundation

// MARK: - Prime buckets

/// Bucket arithmetic and multipliers for one Engram table.
nonisolated public struct DeepSeekV41EngramLayout: Sendable, Equatable {
    public let spec: DeepSeekV41EngramSpec
    /// Bucket size of each hash column, in column order.
    public let bucketSizes: [Int]
    /// First row index of each hash column. Disjoint and ascending.
    public let bucketOffsets: [Int]
    /// One odd run-time multiplier per look-back (`maxNgramSize` of them).
    public let multipliers: [Int64]

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

    /// Byte offset of the `bucket`-th row of a column.
    public func byteOffset(column: Int, bucket: Int) -> Int {
        byteOffset(row: bucketOffsets[column] + bucket)
    }

    /// Resolves one layer's layout, given the primes already handed out to the
    /// layers before it.
    ///
    /// Private on purpose: the prime sequence is global, so a layout built in
    /// isolation would address the right number of rows and the wrong ones. Use
    /// `make(specs:)`.
    private init(
        spec: DeepSeekV41EngramSpec,
        primes: [Int],
        multipliers: [Int64]
    ) {
        var sizes: [Int] = []
        var offsets: [Int] = []
        var row = 0
        for prime in primes {
            sizes.append(prime)
            offsets.append(row)
            row += prime
        }
        self.spec = spec
        self.bucketSizes = sizes
        self.bucketOffsets = offsets
        self.multipliers = multipliers
    }

    /// Builds every Engram table's layout together, in the checkpoint's layer
    /// order, because the prime ranges are drawn once and shared out.
    ///
    /// `specs` must be in `engram_layer_ids` order: swapping two entries keeps
    /// every table the right size while moving every bucket, which no range check
    /// can see.
    public static func make(
        specs: [DeepSeekV41EngramSpec]
    ) throws -> [Int: DeepSeekV41EngramLayout] {
        var usedPrimes: Set<Int> = []
        // The last prime handed out. The first layer seeds the sequence from its own
        // `engram_vocab_size`; every later layer continues it, because the
        // checkpoint shares one `seen` set across all layers.
        var previous: Int?
        var layouts: [Int: DeepSeekV41EngramLayout] = [:]
        for spec in specs {
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
            var cursor = previous ?? (spec.bucketStart - 1)

            var primes: [Int] = []
            primes.reserveCapacity((spec.maxNgramSize - 1) * spec.headCount)
            for _ in 0..<((spec.maxNgramSize - 1) * spec.headCount) {
                var candidate = cursor + 1
                while !DeepSeekV41Primes.isPrime(candidate) || usedPrimes.contains(candidate) {
                    candidate += 1
                }
                usedPrimes.insert(candidate)
                primes.append(candidate)
                cursor = candidate
            }
            previous = cursor

            guard let multipliers = DeepSeekV41EngramMultipliers.multipliers(
                layerID: spec.layer,
                maxNgramSize: spec.maxNgramSize,
                compressedVocabSize: spec.compressedVocabSize
            ) else {
                // The hash multipliers come from numpy's generator, and numpy draws
                // small bounds from a stream this port does not reproduce. Refusing
                // is the only safe answer: a neighbouring stream rehashes the table.
                throw DeepSeekV41ConfigError.invalidField(
                    "engram_compressed_vocab_size",
                    "\(spec.compressedVocabSize) is too small for the Engram hash multipliers to be reproduced"
                )
            }
            let layout = DeepSeekV41EngramLayout(
                spec: spec,
                primes: primes,
                multipliers: multipliers
            )
            if layout.addressedRows > spec.rowCount {
                throw DeepSeekV41ConfigError.inconsistentLayout(
                    "engram table for layer \(spec.layer) needs \(layout.addressedRows) rows for its \(layout.hashColumnCount) hash buckets but engram_num_embeddings is \(spec.rowCount)"
                )
            }
            layouts[spec.layer] = layout
        }
        return layouts
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

/// Whether the n-gram hash rule has been matched against the checkpoint's own
/// `inference/engram.py`.
nonisolated public enum DeepSeekV41EngramConformance: String, Sendable, Equatable {
    case unverified
    case verifiedAgainstReference
}

nonisolated public enum DeepSeekV41EngramHasher {

    /// Conformance gate for the hash rule below.
    ///
    /// VERIFIED AGAINST the checkpoint's numbers: `DeepSeekV41EngramTokenMapTests`
    /// pins the 24 row indices `NgramHashState.forward` produces for a fixed token
    /// stream, plain and with a dead token in the middle, using the shipped
    /// tokenizer, the shipped prime sequence and the shipped multipliers. Every
    /// input to that comparison came from the checkpoint, and a wrong token
    /// combination, modulus, column offset or dead-token stop changes the ids.
    ///
    /// What it does NOT cover: the vectors came from a numpy transcription of the
    /// reference rather than a torch run (there is no torch on this machine), so a
    /// misreading shared by both transcriptions would survive. That is a much
    /// narrower gap than an unverified port, and it is why the Engram COMBINATION
    /// stays unverified while the indexing does not.
    public static let conformance: DeepSeekV41EngramConformance = .verifiedAgainstReference

    /// The hash ids of one position, one per hash column, in column order:
    /// `[2-gram head 0..H-1, 3-gram head 0..H-1, ...]`. Each is a row index into
    /// the layer's table.
    ///
    /// - Parameters:
    ///   - history: token ids ending at the position being embedded, oldest first.
    ///     The last element is the current token.
    ///   - dead: one flag per history entry; a dead token takes no part in an
    ///     n-gram (an image span), and neither does anything before it.
    ///   - tokenMap: the compressed vocabulary the n-grams hash over.
    ///   - layout: the table's bucket ranges and multipliers.
    public static func hashIDs(
        history: [Int32],
        dead: [Bool],
        tokenMap: DeepSeekV41EngramTokenMap,
        layout: DeepSeekV41EngramLayout
    ) -> [Int] {
        let columnCount = layout.hashColumnCount
        guard columnCount > 0, layout.multipliers.count >= layout.spec.maxNgramSize else {
            return []
        }
        let padID = Int64(tokenMap.padTokenID)
        // `rolling` after step `i` is the hash of the (i+1)-gram, so one running
        // value serves every n-gram size.
        var rolling: Int64 = 0
        var ids: [Int] = []
        ids.reserveCapacity(columnCount)
        var blocked = history.isEmpty
        for shift in 0..<layout.spec.maxNgramSize {
            let index = history.count - 1 - shift
            var token: Int64 = padID
            if index >= 0 {
                let isDead = index < dead.count && dead[index]
                // The stop is cumulative: once a look-back runs off the sequence or
                // hits a dead token, every longer n-gram is blocked too.
                if isDead { blocked = true }
                if !blocked {
                    token = Int64(tokenMap.compressedToken(Int(history[index])))
                }
            } else {
                blocked = true
            }

            let product = token &* layout.multipliers[shift]
            rolling = shift == 0 ? product : rolling ^ product
            // Look-back 0 is the 1-gram, which owns no column: the smallest n-gram
            // the table hashes is a 2-gram.
            guard shift >= 1 else { continue }
            let ngramSize = shift + 1
            for head in 0..<layout.spec.headCount {
                let column = layout.column(ngramSize: ngramSize, head: head)
                let modulus = Int64(layout.bucketSizes[column])
                let bucket = Int(rolling % modulus)
                ids.append(layout.bucketOffsets[column] + bucket)
            }
        }
        return ids
    }
}

// MARK: - Token map

/// Maps a checkpoint token id onto the compressed vocabulary the hash draws
/// from (`engram_compressed_vocab_size`, 99_092 on the published checkpoint).
///
/// The map is not a formula: tokens that normalize alike collapse together, so
/// it has to be built from the tokenizer by `DynaMoeTokenizer.buildEngramTokenMap`
/// (the checkpoint's `build_compressed_token_map`).
nonisolated public struct DeepSeekV41EngramTokenMap: Sendable, Equatable {
    public let compressedVocabSize: Int
    /// Compressed id of the checkpoint's pad token, which fills n-gram slots with
    /// no history.
    public let padTokenID: Int32
    /// `map[tokenID]`, valid for `0..<map.count`.
    public let map: [Int32]

    public func compressedToken(_ tokenID: Int) -> Int32 {
        guard tokenID >= 0, tokenID < map.count else { return padTokenID }
        return map[tokenID]
    }

    /// Wraps the lookup the Rust core builds from the tokenizer.
    ///
    /// The ids the core hands back are dense and in first-seen order, so the
    /// largest id plus one is the compressed vocabulary size. It is passed in as
    /// well and compared, because the hash multiplier bound derives from that size:
    /// a disagreement would silently rehash every n-gram.
    public init(lookup: [UInt32], compressedVocabSize: Int, padID: UInt32) throws {
        guard !lookup.isEmpty else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_compressed_vocab_size", "the tokenizer produced an empty Engram token map"
            )
        }
        guard Int(padID) < lookup.count else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_pad_id", "\(padID) is outside the compressed token map"
            )
        }
        let derived = lookup.reduce(0) { max($0, Int($1) + 1) }
        guard derived == compressedVocabSize else {
            throw DeepSeekV41ConfigError.inconsistentLayout(
                "Engram token map uses \(derived) compressed ids but the tokenizer reported \(compressedVocabSize)"
            )
        }
        self.compressedVocabSize = compressedVocabSize
        self.padTokenID = Int32(padID)
        self.map = lookup.map(Int32.init)
    }

    /// A stand-in map for tests and for probing the hash rule without a
    /// tokenizer: `id % compressedVocabSize`.
    ///
    /// This is NOT the checkpoint's map and must never be used to serve a turn;
    /// real n-gram hashing has to see the normalized-token collapse the checkpoint
    /// trained with.
    public static func synthetic(
        vocabSize: Int,
        compressedVocabSize: Int,
        padTokenID: Int
    ) throws -> DeepSeekV41EngramTokenMap {
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
        guard padTokenID >= 0, padTokenID < vocabSize else {
            throw DeepSeekV41ConfigError.invalidField(
                "engram_pad_token_id", "\(padTokenID) is outside 0..<\(vocabSize)"
            )
        }
        var table = [Int32](repeating: Int32(padTokenID), count: vocabSize)
        for id in 0..<vocabSize {
            table[id] = Int32(id % compressedVocabSize)
        }
        table[padTokenID] = Int32(padTokenID)
        return DeepSeekV41EngramTokenMap(
            compressedVocabSize: compressedVocabSize, padTokenID: Int32(padTokenID), map: table
        )
    }

    private init(compressedVocabSize: Int, padTokenID: Int32, map: [Int32]) {
        self.compressedVocabSize = compressedVocabSize
        self.padTokenID = padTokenID
        self.map = map
    }
}

// MARK: - Row request

/// One (layer, column, row) read the runtime has to issue for a token.
nonisolated public struct DeepSeekV41EngramRowRequest: Sendable, Equatable {
    public let layer: Int
    public let column: Int
    /// Bucket index within the column's prime-sized range.
    public let bucket: Int
    /// Row index within the layer's table, i.e. the column offset plus `bucket`.
    public let row: Int
    public let byteOffset: Int
    public let byteCount: Int

    public init(layer: Int, column: Int, bucket: Int, row: Int, byteOffset: Int, byteCount: Int) {
        self.layer = layer
        self.column = column
        self.bucket = bucket
        self.row = row
        self.byteOffset = byteOffset
        self.byteCount = byteCount
    }
}

nonisolated public enum DeepSeekV41Engram {

    /// The row requests for one token position.
    ///
    /// `history` is the token stream ending at the current position, oldest
    /// first; the last element is the token being embedded. `dead` marks
    /// positions that take no part in an n-gram (image spans).
    ///
    /// Returns one request per (n-gram size, head) column, i.e.
    /// `(max_ngram_size - 1) * n_heads` reads per Engram layer per token:
    /// 24 reads of 264 bytes on the published checkpoint, in column order.
    public static func rowRequests(
        layer: Int,
        history: [Int32],
        dead: [Bool] = [],
        tokenMap: DeepSeekV41EngramTokenMap,
        layout: DeepSeekV41EngramLayout
    ) -> [DeepSeekV41EngramRowRequest] {
        guard layout.spec.layer == layer else { return [] }
        let ids = DeepSeekV41EngramHasher.hashIDs(
            history: history, dead: dead, tokenMap: tokenMap, layout: layout
        )
        return ids.enumerated().map { column, row in
            DeepSeekV41EngramRowRequest(
                layer: layer,
                column: column,
                bucket: row - layout.bucketOffsets[column],
                row: row,
                byteOffset: layout.byteOffset(row: row),
                byteCount: layout.spec.rowBytes
            )
        }
    }
}

// MARK: - Row dequantization

/// The CPU form of `dsv41_engram_dequant_rows`: E4M3 values with one E8M0 block
/// scale per 32, turned into a Float row.
///
/// Kept beside the addressing code so the reference forward and the kernel are
/// checked against the same decode. `DeepSeekV41EngramDequantTests` pins both
/// against hand-computed bytes.
nonisolated public enum DeepSeekV41EngramRowDequantizer {

    /// Number of E4M3 blocks in a row of `headDimension` values.
    public static func scaleCount(headDimension: Int) -> Int {
        (headDimension + 31) / 32
    }

    /// Decodes one packed row. Returns zeros when the row is short of
    /// `rowBytes(totalValues:)`.
    public static func row(_ raw: [UInt8], headDimension: Int) -> [Float] {
        let scales = scaleCount(headDimension: headDimension)
        guard headDimension > 0, raw.count >= headDimension + scales else {
            return [Float](repeating: 0, count: max(0, headDimension))
        }
        var output = [Float](repeating: 0, count: headDimension)
        for dimension in 0..<headDimension {
            let scale = e8m0(raw[headDimension + dimension / 32])
            output[dimension] = e4m3(raw[dimension]) * scale
        }
        return output
    }

    /// `unpack_e4m3`: sign, 4-bit exponent with bias 7, 3-bit mantissa. Exponent 0
    /// is subnormal, scaled by 2^-6.
    public static func e4m3(_ byte: UInt8) -> Float {
        let sign: Float = (byte & 0x80) != 0 ? -1 : 1
        let exponent = Int((byte >> 3) & 0x0F)
        let mantissa = Float(byte & 0x07)
        if exponent == 0 {
            return sign * (mantissa / 8) * 0.015625
        }
        return sign * (1 + mantissa / 8) * pow(2, Float(exponent) - 7)
    }

    /// `decode_e8m0_scale`: 2^(s - 127), with byte 0 meaning zero.
    public static func e8m0(_ byte: UInt8) -> Float {
        byte == 0 ? 0 : exp2(Float(byte) - 127)
    }
}
