//
//  CodebaseIndexer.swift
//  DynaMoE
//
//  Phase 1: Semantic Codebase Indexing & Metal Vector Search (Local RAG)
//  - AST Chunking & Metadata Management
//  - Incremental File Watcher for Real-Time On-Save Indexing
//  - Hybrid Search Coordinator (Dense Vector + BM25 via Reciprocal Rank Fusion)
//  - Disk Cache for Instant Startup
//  - Fully Asynchronous & Non-Blocking Background Execution
//

import Foundation
import Combine
import Metal

// MARK: - Codebase Chunk Model

public struct CodebaseChunk: Identifiable, Codable {
    public let id: UUID
    public let filePath: String         // Relative path within workspace
    public let absolutePath: String
    public let startLine: Int
    public let endLine: Int
    public let title: String
    public let content: String
    public let tokenCount: Int
    public var vector: [Float]?

    public init(
        id: UUID = UUID(),
        filePath: String,
        absolutePath: String,
        startLine: Int,
        endLine: Int,
        title: String,
        content: String,
        tokenCount: Int,
        vector: [Float]? = nil
    ) {
        self.id = id
        self.filePath = filePath
        self.absolutePath = absolutePath
        self.startLine = startLine
        self.endLine = endLine
        self.title = title
        self.content = content
        self.tokenCount = tokenCount
        self.vector = vector
    }
}

// MARK: - Search Result Structure

public struct CodebaseSearchResult: Identifiable {
    public var id: UUID { chunk.id }
    public let chunk: CodebaseChunk
    public let score: Float
    public let denseScore: Float?
    public let bm25Score: Float?
    public let searchMode: SearchMode
}

// MARK: - Incremental File Watcher

public final class CodebaseFileWatcher {
    private var source: DispatchSourceFileSystemObject?
    private var fileDescriptor: Int32 = -1
    private let queue = DispatchQueue(label: "com.dynamoe.filewatcher", qos: .utility)
    private var debounceWorkItem: DispatchWorkItem?
    private let debounceInterval: TimeInterval = 0.5
    private weak var indexer: CodebaseIndexer?

    public init(indexer: CodebaseIndexer) {
        self.indexer = indexer
    }

    public func startWatching(url: URL) {
        stopWatching()

        let standardPath = url.standardizedFileURL.path
        let dangerousPaths: Set<String> = [
            "/", "/System", "/Library", "/usr", "/bin", "/sbin", "/var", "/private", "/Users", "/Applications"
        ]
        guard !dangerousPaths.contains(standardPath) else { return }

        let fd = open(standardPath, O_EVTONLY)
        guard fd >= 0 else { return }
        fileDescriptor = fd

        let src = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fd,
            eventMask: [.write, .extend, .attrib, .link, .rename, .revoke],
            queue: queue
        )

        src.setEventHandler { [weak self] in
            self?.scheduleSync()
        }

        src.setCancelHandler { [weak self] in
            if let fd = self?.fileDescriptor, fd >= 0 {
                close(fd)
                self?.fileDescriptor = -1
            }
        }

        src.resume()
        source = src
    }

    public func stopWatching() {
        debounceWorkItem?.cancel()
        debounceWorkItem = nil
        if let src = source {
            src.cancel()
            source = nil
        }
        if fileDescriptor >= 0 {
            close(fileDescriptor)
            fileDescriptor = -1
        }
    }

    private func scheduleSync() {
        debounceWorkItem?.cancel()
        let item = DispatchWorkItem { [weak self] in
            guard let self = self, let indexer = self.indexer else { return }
            Task.detached(priority: .utility) {
                await indexer.incrementalSync()
            }
        }
        debounceWorkItem = item
        queue.asyncAfter(deadline: .now() + debounceInterval, execute: item)
    }

    deinit {
        stopWatching()
    }
}

// MARK: - Codebase Indexer

public final class CodebaseIndexer: ObservableObject {
    public static let shared = CodebaseIndexer()

    @Published public var isIndexing: Bool = false
    @Published public var indexingProgress: Double = 0.0
    @Published public var indexedFileCount: Int = 0
    @Published public var indexedChunkCount: Int = 0
    @Published public var lastIndexDurationMs: Double = 0.0
    @Published public var statusMessage: String = "No workspace indexed"

    public private(set) var activeWorkspaceURL: URL?
    private var chunks: [CodebaseChunk] = []
    private var fileModDates: [String: Date] = [:] // Relative path -> modified date
    private let bm25 = BM25Index()
    private lazy var watcher = CodebaseFileWatcher(indexer: self)
    private var indexingTask: Task<Void, Never>? = nil

    public static let supportedExtensions: Set<String> = [
        "swift", "metal", "h", "m", "c", "cpp", "hpp", "rs", "py", "js", "ts", "tsx", "jsx", "md", "json", "yaml", "yml", "sh", "zsh"
    ]

    public static let ignoredDirectories: Set<String> = [
        ".git", "build", "DerivedData", ".build", ".xcodeproj", ".xcworkspace", "node_modules", "Pods", ".DS_Store", ".venv", "venv", "__pycache__", "xcuserdata", ".dynamoe"
    ]

    private init() {}

    // MARK: - Public Indexing APIs

    /// Cancels any currently active indexing operation.
    public func cancelIndexing() {
        indexingTask?.cancel()
        indexingTask = nil
        watcher.stopWatching()
        Task { @MainActor in
            self.isIndexing = false
            self.statusMessage = "Indexing cancelled."
        }
    }

    /// Triggers asynchronous background indexing of the specified workspace directory.
    /// Does NOT block the caller or main thread.
    public func indexWorkspace(url: URL, forceRebuild: Bool = false) {
        indexingTask?.cancel()
        indexingTask = Task.detached(priority: .userInitiated) { [weak self] in
            await self?.performIndexWorkspace(url: url, forceRebuild: forceRebuild)
        }
    }

    /// Internal asynchronous implementation running off the main thread.
    public func performIndexWorkspace(url: URL, forceRebuild: Bool = false) async {
        let standardURL = url.standardizedFileURL
        let standardPath = standardURL.path

        // Guard against dangerous or root filesystem indexing
        let dangerousPaths: Set<String> = [
            "/", "/System", "/Library", "/usr", "/bin", "/sbin", "/var", "/private", "/Users", "/Applications"
        ]
        guard !dangerousPaths.contains(standardPath) && !standardPath.isEmpty else {
            await MainActor.run {
                self.isIndexing = false
                self.statusMessage = "Please select a specific project directory (system paths cannot be indexed)."
            }
            return
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        await MainActor.run {
            self.isIndexing = true
            self.indexingProgress = 0.05
            self.statusMessage = "Checking cache for \(standardURL.lastPathComponent)..."
            self.activeWorkspaceURL = standardURL
        }

        // 1. Check disk cache
        if !forceRebuild && loadCache(for: standardURL) {
            let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0
            await MainActor.run {
                self.isIndexing = false
                self.indexingProgress = 1.0
                self.indexedFileCount = self.fileModDates.count
                self.indexedChunkCount = self.chunks.count
                self.lastIndexDurationMs = elapsed
                self.statusMessage = "Loaded \(self.chunks.count) chunks from cache (\(String(format: "%.1f", elapsed)) ms)"
            }
            watcher.startWatching(url: standardURL)
            return
        }

        if Task.isCancelled { return }

        // 2. Scan directory
        await MainActor.run {
            self.statusMessage = "Scanning workspace files..."
            self.indexingProgress = 0.1
        }
        let scannedFiles = scanDirectory(url: standardURL)
        if Task.isCancelled { return }

        if scannedFiles.isEmpty {
            await MainActor.run {
                self.isIndexing = false
                self.statusMessage = "No supported code files found in directory."
            }
            return
        }

        await MainActor.run {
            self.statusMessage = "Found \(scannedFiles.count) files. Parsing AST chunks..."
            self.indexingProgress = 0.2
        }

        // 3. Chunk files
        var newChunks: [CodebaseChunk] = []
        var newModDates: [String: Date] = [:]
        var filesProcessed = 0
        let totalFiles = scannedFiles.count

        for (relPath, fileURL) in scannedFiles {
            if Task.isCancelled { return }

            if let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
               let modDate = attrs[.modificationDate] as? Date {
                newModDates[relPath] = modDate
            }

            do {
                let docResult = try SemanticDocumentReader.shared.readDocument(at: fileURL, maxChunkLength: 3000)
                for chunk in docResult.chunks {
                    let c = CodebaseChunk(
                        filePath: relPath,
                        absolutePath: fileURL.path,
                        startLine: chunk.startLine,
                        endLine: chunk.endLine,
                        title: "\(relPath):\(chunk.startLine)-\(chunk.endLine) \(chunk.title)",
                        content: chunk.content,
                        tokenCount: chunk.content.split(separator: " ").count,
                        vector: nil
                    )
                    newChunks.append(c)
                }
            } catch {
                continue
            }

            filesProcessed += 1
            if filesProcessed % 20 == 0 || filesProcessed == totalFiles {
                let ratio = 0.2 + (0.3 * (Double(filesProcessed) / Double(max(totalFiles, 1))))
                await MainActor.run {
                    self.statusMessage = "Parsed \(filesProcessed)/\(totalFiles) files (\(newChunks.count) chunks)..."
                    self.indexingProgress = ratio
                }
            }
        }

        if Task.isCancelled { return }

        // 4. Generate embeddings
        let totalChunks = newChunks.count
        await MainActor.run {
            self.statusMessage = "Generating embeddings for \(totalChunks) chunks on Apple Silicon..."
            self.indexingProgress = 0.5
        }

        for i in 0..<totalChunks {
            if Task.isCancelled { return }
            let textToEmbed = "\(newChunks[i].title)\n\(newChunks[i].content)"
            newChunks[i].vector = CodebaseEmbeddingEngine.shared.embed(text: textToEmbed)

            if i % 40 == 0 || i == totalChunks - 1 {
                let ratio = 0.5 + (0.45 * (Double(i + 1) / Double(max(totalChunks, 1))))
                await MainActor.run {
                    self.statusMessage = "Embedded \(i + 1)/\(totalChunks) chunks..."
                    self.indexingProgress = ratio
                }
            }
        }

        if Task.isCancelled { return }

        self.chunks = newChunks
        self.fileModDates = newModDates

        // 5. Update Metal GPU Vector Store
        let validVectors: [[Float]] = self.chunks.map { $0.vector ?? [Float](repeating: 0.0, count: CodebaseEmbeddingEngine.embeddingDimension) }
        _ = MetalVectorSearch.shared.updateCorpus(vectors: validVectors)

        // 6. Update BM25 Inverted Index
        let docTexts = self.chunks.map { "\($0.title)\n\($0.content)" }
        bm25.index(documents: docTexts)

        let elapsed = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        await MainActor.run {
            self.isIndexing = false
            self.indexingProgress = 1.0
            self.indexedFileCount = scannedFiles.count
            self.indexedChunkCount = self.chunks.count
            self.lastIndexDurationMs = elapsed
            self.statusMessage = "Indexed \(self.chunks.count) chunks across \(scannedFiles.count) files in \(String(format: "%.1f", elapsed)) ms"
        }

        saveCache(for: standardURL)
        watcher.startWatching(url: standardURL)
    }

    /// Incremental synchronization when a file changes on disk.
    public func incrementalSync() async {
        guard let url = activeWorkspaceURL else { return }
        let scanned = scanDirectory(url: url)
        var changed = false

        for (relPath, fileURL) in scanned {
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
                  let modDate = attrs[.modificationDate] as? Date else { continue }

            if let prevDate = fileModDates[relPath], prevDate == modDate {
                continue // File has not changed
            }

            // File is new or modified: re-index its chunks
            changed = true
            fileModDates[relPath] = modDate
            chunks.removeAll { $0.filePath == relPath }

            if let docResult = try? SemanticDocumentReader.shared.readDocument(at: fileURL, maxChunkLength: 3000) {
                for chunk in docResult.chunks {
                    var c = CodebaseChunk(
                        filePath: relPath,
                        absolutePath: fileURL.path,
                        startLine: chunk.startLine,
                        endLine: chunk.endLine,
                        title: "\(relPath):\(chunk.startLine)-\(chunk.endLine) \(chunk.title)",
                        content: chunk.content,
                        tokenCount: chunk.content.split(separator: " ").count,
                        vector: nil
                    )
                    c.vector = CodebaseEmbeddingEngine.shared.embed(text: "\(c.title)\n\(c.content)")
                    chunks.append(c)
                }
            }
        }

        // Check for deleted files
        let currentRelPaths = Set(scanned.keys)
        let previousRelPaths = Set(fileModDates.keys)
        let deleted = previousRelPaths.subtracting(currentRelPaths)
        if !deleted.isEmpty {
            changed = true
            for d in deleted {
                fileModDates.removeValue(forKey: d)
                chunks.removeAll { $0.filePath == d }
            }
        }

        if changed {
            let validVectors: [[Float]] = self.chunks.map { $0.vector ?? [Float](repeating: 0.0, count: CodebaseEmbeddingEngine.embeddingDimension) }
            _ = MetalVectorSearch.shared.updateCorpus(vectors: validVectors)
            let docTexts = self.chunks.map { "\($0.title)\n\($0.content)" }
            bm25.index(documents: docTexts)

            await MainActor.run {
                self.indexedFileCount = self.fileModDates.count
                self.indexedChunkCount = self.chunks.count
                self.statusMessage = "Updated index: \(self.chunks.count) chunks"
            }
            saveCache(for: url)
        }
    }

    // MARK: - Search API

    /// Executes Hybrid, Semantic, or Keyword search across the indexed codebase.
    public func search(
        query: String,
        workspace: URL? = nil,
        targetDirectory: String? = nil,
        fileExtensions: [String]? = nil,
        topK: Int = 5,
        mode: SearchMode = .hybrid
    ) async -> [CodebaseSearchResult] {
        // Auto-index if not yet indexed or workspace changed
        if let ws = workspace {
            let standardWS = ws.standardizedFileURL
            if activeWorkspaceURL == nil || activeWorkspaceURL?.path != standardWS.path || chunks.isEmpty {
                await performIndexWorkspace(url: standardWS)
            }
        }

        guard !chunks.isEmpty else { return [] }

        // Filter valid candidates by target directory or file extension if requested
        var candidateIndices = Set(0..<chunks.count)
        if let targetDir = targetDirectory, !targetDir.isEmpty {
            candidateIndices = candidateIndices.filter { idx in
                chunks[idx].filePath.hasPrefix(targetDir)
            }
        }
        if let exts = fileExtensions, !exts.isEmpty {
            let cleanExts = Set(exts.map { $0.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: ".")) })
            candidateIndices = candidateIndices.filter { idx in
                let chunkExt = (chunks[idx].filePath as NSString).pathExtension.lowercased()
                return cleanExts.contains(chunkExt)
            }
        }

        if candidateIndices.isEmpty { return [] }

        // 1. Dense Vector Search on Metal GPU
        var denseRanked: [(index: Int, score: Float)] = []
        if mode == .hybrid || mode == .semantic {
            if let queryVec = CodebaseEmbeddingEngine.shared.embed(text: query) {
                let rawDense = MetalVectorSearch.shared.search(queryVector: queryVec, topK: min(topK * 4, chunks.count))
                denseRanked = rawDense.filter { candidateIndices.contains($0.index) }
            }
        }

        // 2. BM25 Lexical Keyword Search
        var bm25Ranked: [(index: Int, score: Float)] = []
        if mode == .hybrid || mode == .keyword {
            let rawBM25 = bm25.search(query: query, topK: min(topK * 4, chunks.count))
            bm25Ranked = rawBM25.filter { candidateIndices.contains($0.index) }
        }

        // 3. Score Combination
        var results: [CodebaseSearchResult] = []

        switch mode {
        case .hybrid:
            let fused = HybridSearchFusion.fuse(
                denseResults: denseRanked,
                bm25Results: bm25Ranked,
                topK: topK
            )
            for f in fused {
                guard f.index < chunks.count else { continue }
                results.append(CodebaseSearchResult(
                    chunk: chunks[f.index],
                    score: f.rrfScore,
                    denseScore: f.denseScore,
                    bm25Score: f.bm25Score,
                    searchMode: .hybrid
                ))
            }

        case .semantic:
            for item in denseRanked.prefix(topK) {
                guard item.index < chunks.count else { continue }
                results.append(CodebaseSearchResult(
                    chunk: chunks[item.index],
                    score: item.score,
                    denseScore: item.score,
                    bm25Score: nil,
                    searchMode: .semantic
                ))
            }

        case .keyword:
            for item in bm25Ranked.prefix(topK) {
                guard item.index < chunks.count else { continue }
                results.append(CodebaseSearchResult(
                    chunk: chunks[item.index],
                    score: item.score,
                    denseScore: nil,
                    bm25Score: item.score,
                    searchMode: .keyword
                ))
            }
        }

        return results
    }

    // MARK: - Directory Scanning Helper

    private func scanDirectory(url: URL, maxFiles: Int = 10000) -> [String: URL] {
        var results: [String: URL] = [:]
        let fm = FileManager.default
        let rootURL = url.standardizedFileURL
        let rootPath = rootURL.path

        guard let enumerator = fm.enumerator(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return results }

        for case let fileURL as URL in enumerator {
            if results.count >= maxFiles || Task.isCancelled { break }

            let pathComponents = fileURL.pathComponents
            if pathComponents.contains(where: { CodebaseIndexer.ignoredDirectories.contains($0) }) {
                enumerator.skipDescendants()
                continue
            }

            let ext = fileURL.pathExtension.lowercased()
            if CodebaseIndexer.supportedExtensions.contains(ext) {
                let path = fileURL.standardizedFileURL.path
                var relPath = path
                if path.hasPrefix(rootPath) {
                    relPath = String(path.dropFirst(rootPath.count))
                    if relPath.hasPrefix("/") { relPath = String(relPath.dropFirst()) }
                }
                results[relPath] = fileURL
            }
        }

        return results
    }

    // MARK: - Disk Cache Persistence

    private func cacheFileURL(for workspaceURL: URL) -> URL? {
        guard let cachesDir = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else { return nil }
        let indexDir = cachesDir.appendingPathComponent("DynaMoE/Index", isDirectory: true)
        try? FileManager.default.createDirectory(at: indexDir, withIntermediateDirectories: true)

        let safeName = workspaceURL.path.replacingOccurrences(of: "/", with: "_")
        return indexDir.appendingPathComponent("\(safeName).json")
    }

    private struct IndexCacheData: Codable {
        let modDates: [String: Date]
        let chunks: [CodebaseChunk]
    }

    private func saveCache(for workspaceURL: URL) {
        guard let file = cacheFileURL(for: workspaceURL) else { return }
        let data = IndexCacheData(modDates: fileModDates, chunks: chunks)
        if let encoded = try? JSONEncoder().encode(data) {
            try? encoded.write(to: file, options: .atomic)
        }
    }

    private func loadCache(for workspaceURL: URL) -> Bool {
        guard let file = cacheFileURL(for: workspaceURL),
              let data = try? Data(contentsOf: file),
              let decoded = try? JSONDecoder().decode(IndexCacheData.self, from: data) else {
            return false
        }

        // Verify if any file has changed on disk
        let currentFiles = scanDirectory(url: workspaceURL)
        if currentFiles.count != decoded.modDates.count { return false }

        for (rel, fileURL) in currentFiles {
            guard let cachedDate = decoded.modDates[rel],
                  let attrs = try? FileManager.default.attributesOfItem(atPath: fileURL.path),
                  let currentMod = attrs[.modificationDate] as? Date,
                  abs(currentMod.timeIntervalSince(cachedDate)) < 1.0 else {
                return false
            }
        }

        self.chunks = decoded.chunks
        self.fileModDates = decoded.modDates

        let validVectors: [[Float]] = self.chunks.map { $0.vector ?? [Float](repeating: 0.0, count: CodebaseEmbeddingEngine.embeddingDimension) }
        _ = MetalVectorSearch.shared.updateCorpus(vectors: validVectors)
        let docTexts = self.chunks.map { "\($0.title)\n\($0.content)" }
        bm25.index(documents: docTexts)

        return true
    }
}
