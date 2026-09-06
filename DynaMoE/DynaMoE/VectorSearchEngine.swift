//
//  VectorSearchEngine.swift
//  DynaMoE
//
//  Phase 1: Semantic Codebase Indexing & Metal Vector Search (Local RAG)
//  - Metal-accelerated GPU batched cosine similarity kernel
//  - Native Apple Silicon sentence embedding generator (NLEmbedding)
//  - Okapi BM25 lexical inverted index with code identifier tokenization
//  - Reciprocal Rank Fusion (RRF) hybrid search coordinator
//

import Foundation
import Metal
import Accelerate
import NaturalLanguage

// MARK: - Codebase Embedding Engine (NLEmbedding)

public final class CodebaseEmbeddingEngine {
    public static let shared = CodebaseEmbeddingEngine()
    public static let embeddingDimension: Int = 512

    private let embedding: NLEmbedding?
    private let lock = NSLock()

    private init() {
        self.embedding = NLEmbedding.sentenceEmbedding(for: .english)
    }

    public var isAvailable: Bool {
        return embedding != nil
    }

    /// Generates a normalized 512-dimensional embedding vector for the provided text.
    public func embed(text: String) -> [Float]? {
        guard let embedding = embedding else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        // Take up to first 1000 characters for sentence embedding stability
        let prefix = String(trimmed.prefix(1000))
        let doubleVector: [Double]?
        lock.lock()
        doubleVector = embedding.vector(for: prefix)
        lock.unlock()

        guard let doubleVector = doubleVector else { return nil }

        var floatVector = doubleVector.map { Float($0) }
        normalizeInPlace(&floatVector)
        return floatVector
    }

    /// Generates embeddings for a batch of strings.
    public func embedBatch(texts: [String]) -> [[Float]?]? {
        guard isAvailable else { return nil }
        return texts.map { embed(text: $0) }
    }

    /// Normalizes a vector to unit L2 norm in-place so dot product equals cosine similarity.
    private func normalizeInPlace(_ vec: inout [Float]) {
        var normSq: Float = 0.0
        vDSP_svesq(vec, 1, &normSq, vDSP_Length(vec.count))
        let norm = sqrt(normSq)
        if norm > 1e-7 {
            var scale = 1.0 / norm
            vDSP_vsmul(vec, 1, &scale, &vec, 1, vDSP_Length(vec.count))
        }
    }
}

// MARK: - Metal GPU Vector Search Engine

public final class MetalVectorSearch {
    public static let shared = MetalVectorSearch()

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var pipelineState: MTLComputePipelineState?

    private var corpusBuffer: MTLBuffer?
    private var cachedNumVectors: Int = 0
    private let dimension: Int = CodebaseEmbeddingEngine.embeddingDimension

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) {
        self.device = device
        self.commandQueue = device?.makeCommandQueue()
        if let dev = device, let library = dev.makeDefaultLibrary() {
            if let fn = library.makeFunction(name: "vector_cosine_similarity_fp32") {
                self.pipelineState = try? dev.makeComputePipelineState(function: fn)
            }
        }
    }

    /// Loads or updates the corpus vector matrix in unified memory.
    /// All vectors must be normalized to unit L2 norm and have length `dimension` (512).
    public func updateCorpus(vectors: [[Float]]) -> Bool {
        guard let device = device else { return false }
        cachedNumVectors = vectors.count
        guard cachedNumVectors > 0 else {
            corpusBuffer = nil
            return true
        }

        let totalFloats = cachedNumVectors * dimension
        let byteSize = totalFloats * MemoryLayout<Float>.stride

        // Flatten vectors into single contiguous buffer
        var flatVectors = [Float](repeating: 0.0, count: totalFloats)
        for (i, vec) in vectors.enumerated() {
            let offset = i * dimension
            let copyCount = min(vec.count, dimension)
            for j in 0..<copyCount {
                flatVectors[offset + j] = vec[j]
            }
        }

        corpusBuffer = device.makeBuffer(
            bytes: flatVectors,
            length: byteSize,
            options: .storageModeShared
        )
        return corpusBuffer != nil
    }

    /// Executes batched cosine similarity on Apple Silicon GPU against the loaded corpus vectors.
    /// Returns the top-K highest scoring indices and their cosine similarity scores (range: -1.0 to 1.0).
    public func search(queryVector: [Float], topK: Int = 5) -> [(index: Int, score: Float)] {
        guard cachedNumVectors > 0 else { return [] }
        let effectiveTopK = min(topK, cachedNumVectors)

        // Try Metal GPU path
        if let device = device,
           let commandQueue = commandQueue,
           let pipelineState = pipelineState,
           let corpusBuffer = corpusBuffer {

            let queryByteSize = dimension * MemoryLayout<Float>.stride
            guard let queryBuffer = device.makeBuffer(bytes: queryVector, length: queryByteSize, options: .storageModeShared),
                  let outputScoresBuffer = device.makeBuffer(length: cachedNumVectors * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let encoder = commandBuffer.makeComputeCommandEncoder() else {
                return cpuFallbackSearch(queryVector: queryVector, topK: effectiveTopK)
            }

            encoder.setComputePipelineState(pipelineState)
            encoder.setBuffer(queryBuffer, offset: 0, index: 0)
            encoder.setBuffer(corpusBuffer, offset: 0, index: 1)
            encoder.setBuffer(outputScoresBuffer, offset: 0, index: 2)

            var dim = UInt32(dimension)
            var count = UInt32(cachedNumVectors)
            encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 4)

            let threadsPerGrid = MTLSize(width: cachedNumVectors, height: 1, depth: 1)
            let threadgroupWidth = min(pipelineState.maxTotalThreadsPerThreadgroup, 256)
            let threadsPerThreadgroup = MTLSize(width: threadgroupWidth, height: 1, depth: 1)

            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            encoder.endEncoding()

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let scoresPtr = outputScoresBuffer.contents().bindMemory(to: Float.self, capacity: cachedNumVectors)
            return extractTopK(from: scoresPtr, count: cachedNumVectors, topK: effectiveTopK)
        }

        // Fallback to CPU calculation
        return cpuFallbackSearch(queryVector: queryVector, topK: effectiveTopK)
    }

    private func extractTopK(from ptr: UnsafePointer<Float>, count: Int, topK: Int) -> [(index: Int, score: Float)] {
        var scored: [(index: Int, score: Float)] = []
        scored.reserveCapacity(count)
        for i in 0..<count {
            scored.append((index: i, score: ptr[i]))
        }
        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK))
    }

    private func cpuFallbackSearch(queryVector: [Float], topK: Int) -> [(index: Int, score: Float)] {
        guard let corpusBuffer = corpusBuffer else { return [] }
        let corpusPtr = corpusBuffer.contents().bindMemory(to: Float.self, capacity: cachedNumVectors * dimension)

        var scored: [(index: Int, score: Float)] = []
        scored.reserveCapacity(cachedNumVectors)

        for i in 0..<cachedNumVectors {
            let vecPtr = corpusPtr.advanced(by: i * dimension)
            var dot: Float = 0.0
            vDSP_dotpr(queryVector, 1, vecPtr, 1, &dot, vDSP_Length(dimension))
            scored.append((index: i, score: dot))
        }

        scored.sort { $0.score > $1.score }
        return Array(scored.prefix(topK))
    }
}

// MARK: - BM25 Lexical / Inverted Index

public final class BM25Index {
    public private(set) var numDocs: Int = 0
    public private(set) var avgDocLength: Float = 0.0
    private var docLengths: [Int] = []

    // Term -> [docIndex: termFrequency]
    private var invertedIndex: [String: [Int: Int]] = [:]

    // Okapi BM25 Hyperparameters
    public let k1: Float
    public let b: Float

    public init(k1: Float = 1.2, b: Float = 0.75) {
        self.k1 = k1
        self.b = b
    }

    /// Resets the index with a fresh collection of documents.
    public func index(documents: [String]) {
        invertedIndex.removeAll(keepingCapacity: true)
        numDocs = documents.count
        docLengths = [Int](repeating: 0, count: numDocs)

        var totalTokens = 0
        for (docIdx, doc) in documents.enumerated() {
            let tokens = BM25Index.tokenize(text: doc)
            docLengths[docIdx] = tokens.count
            totalTokens += tokens.count

            var termFreqs: [String: Int] = [:]
            for t in tokens {
                termFreqs[t, default: 0] += 1
            }

            for (t, freq) in termFreqs {
                if invertedIndex[t] == nil {
                    invertedIndex[t] = [:]
                }
                invertedIndex[t]?[docIdx] = freq
            }
        }

        avgDocLength = numDocs > 0 ? Float(totalTokens) / Float(numDocs) : 0.0
    }

    /// Performs Okapi BM25 lexical keyword search across the indexed corpus.
    public func search(query: String, topK: Int = 5) -> [(index: Int, score: Float)] {
        guard numDocs > 0 else { return [] }
        let queryTokens = BM25Index.tokenize(text: query)
        guard !queryTokens.isEmpty else { return [] }

        var docScores: [Int: Float] = [:]

        for term in queryTokens {
            guard let postings = invertedIndex[term] else { continue }
            let docFreq = postings.count
            // Okapi BM25 IDF: ln(1.0 + (N - n + 0.5) / (n + 0.5))
            let idf = log(1.0 + (Float(numDocs - docFreq) + 0.5) / (Float(docFreq) + 0.5))

            for (docIdx, tf) in postings {
                let docLen = Float(docLengths[docIdx])
                let denom = Float(tf) + k1 * (1.0 - b + b * (docLen / max(avgDocLength, 1.0)))
                let score = idf * (Float(tf) * (k1 + 1.0)) / max(denom, 1e-5)
                docScores[docIdx, default: 0.0] += score
            }
        }

        var results = docScores.map { (index: $0.key, score: $0.value) }
        results.sort { $0.score > $1.score }
        return Array(results.prefix(topK))
    }

    /// Code-aware tokenization: handles CamelCase, snake_case, identifiers, and syntax symbols.
    public static func tokenize(text: String) -> [String] {
        var tokens: [String] = []

        // Split by non-alphanumeric punctuation
        let rawWords = text.components(separatedBy: CharacterSet.alphanumerics.inverted)

        for word in rawWords {
            guard !word.isEmpty else { continue }
            let lowerWord = word.lowercased()
            tokens.append(lowerWord)

            // If word is CamelCase (contains uppercase after first char), split into sub-tokens
            let camelSubtokens = splitCamelCase(word)
            if camelSubtokens.count > 1 {
                for sub in camelSubtokens {
                    tokens.append(sub.lowercased())
                }
            }
        }

        return tokens
    }

    private static func splitCamelCase(_ word: String) -> [String] {
        var parts: [String] = []
        var current = ""
        for char in word {
            if char.isUppercase && !current.isEmpty {
                parts.append(current)
                current = String(char)
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty {
            parts.append(current)
        }
        return parts
    }
}

// MARK: - Hybrid Search Fusion (Reciprocal Rank Fusion - RRF)

public enum SearchMode: String, CaseIterable, Codable {
    case hybrid = "hybrid"
    case semantic = "semantic"
    case keyword = "keyword"
}

public struct HybridSearchResult {
    public let index: Int
    public let rrfScore: Float
    public let denseScore: Float?
    public let bm25Score: Float?
    public let denseRank: Int?
    public let bm25Rank: Int?
}

public enum HybridSearchFusion {
    /// Combines Dense Vector ranking and BM25 Lexical ranking via Reciprocal Rank Fusion:
    /// RRF_score(d) = 1 / (k + rank_dense) + 1 / (k + rank_bm25)
    public static func fuse(
        denseResults: [(index: Int, score: Float)],
        bm25Results: [(index: Int, score: Float)],
        topK: Int = 5,
        kRRF: Float = 60.0
    ) -> [HybridSearchResult] {
        var rrfScores: [Int: Float] = [:]
        var denseRanks: [Int: Int] = [:]
        var bm25Ranks: [Int: Int] = [:]
        var denseScores: [Int: Float] = [:]
        var bm25Scores: [Int: Float] = [:]

        for (rank, item) in denseResults.enumerated() {
            let r = rank + 1
            denseRanks[item.index] = r
            denseScores[item.index] = item.score
            rrfScores[item.index, default: 0.0] += 1.0 / (kRRF + Float(r))
        }

        for (rank, item) in bm25Results.enumerated() {
            let r = rank + 1
            bm25Ranks[item.index] = r
            bm25Scores[item.index] = item.score
            rrfScores[item.index, default: 0.0] += 1.0 / (kRRF + Float(r))
        }

        var results: [HybridSearchResult] = rrfScores.map { (idx, rrf) in
            HybridSearchResult(
                index: idx,
                rrfScore: rrf,
                denseScore: denseScores[idx],
                bm25Score: bm25Scores[idx],
                denseRank: denseRanks[idx],
                bm25Rank: bm25Ranks[idx]
            )
        }

        results.sort { $0.rrfScore > $1.rrfScore }
        return Array(results.prefix(topK))
    }
}
