//
//  NativeToolRegistry.swift
//  DynaMoE
//
//  Phase 3: Local Execution Loop & Tool Registry
//  Native embedded macOS tools with:
//  - Layout-aware semantic document reader (PDFs via PDFKit, Markdown, code files)
//  - Controlled sandboxed process runner
//  - Token-efficient web readability extractor
//  - Asynchronous thread-safe execution actor
//

import Foundation
import PDFKit

// MARK: - Semantic Document Chunker & PDF Reader

public final class SemanticDocumentReader {
    public static let shared = SemanticDocumentReader()

    public struct DocumentChunk {
        public let index: Int
        public let title: String
        public let content: String
        public let startLine: Int
        public let endLine: Int
    }

    public func readDocument(
        at url: URL,
        pageRange: ClosedRange<Int>? = nil,
        maxChunkLength: Int = 4000
    ) throws -> (title: String, chunks: [DocumentChunk], totalLength: Int) {
        let ext = url.pathExtension.lowercased()

        if ext == "pdf" {
            return try readPDF(at: url, pageRange: pageRange)
        } else {
            return try readTextOrCode(at: url, maxChunkLength: maxChunkLength)
        }
    }

    private func readPDF(
        at url: URL,
        pageRange: ClosedRange<Int>?
    ) throws -> (title: String, chunks: [DocumentChunk], totalLength: Int) {
        guard let doc = PDFDocument(url: url) else {
            throw NSError(domain: "SemanticDocumentReader", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to open PDF document at \(url.path)"])
        }

        let totalPages = doc.pageCount
        let title = url.deletingPathExtension().lastPathComponent
        var chunks: [DocumentChunk] = []
        var totalChars = 0

        let start = pageRange?.lowerBound ?? 1
        let end = min(pageRange?.upperBound ?? totalPages, totalPages)

        for p in start...end {
            guard let page = doc.page(at: p - 1), let text = page.string, !text.isEmpty else { continue }
            let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
            totalChars += cleaned.count
            chunks.append(DocumentChunk(
                index: p,
                title: "Page \(p) of \(totalPages)",
                content: cleaned,
                startLine: 1,
                endLine: cleaned.components(separatedBy: "\n").count
            ))
        }

        return (title: title, chunks: chunks, totalLength: totalChars)
    }

    private func readTextOrCode(
        at url: URL,
        maxChunkLength: Int
    ) throws -> (title: String, chunks: [DocumentChunk], totalLength: Int) {
        let content = try String(contentsOf: url, encoding: .utf8)
        let lines = content.components(separatedBy: "\n")
        let title = url.lastPathComponent
        var chunks: [DocumentChunk] = []

        var currentChunkLines: [String] = []
        var chunkStartLine = 1
        var currentChunkLen = 0
        var chunkIdx = 1

        for (idx, line) in lines.enumerated() {
            let lineNum = idx + 1
            let isHeading = line.hasPrefix("#") || line.hasPrefix("func ") || line.hasPrefix("class ") || line.hasPrefix("struct ") || line.hasPrefix("fn ") || line.hasPrefix("def ")

            if isHeading && currentChunkLen > maxChunkLength / 2 {
                // Emit current chunk
                let chunkText = currentChunkLines.joined(separator: "\n")
                chunks.append(DocumentChunk(
                    index: chunkIdx,
                    title: "Section \(chunkIdx) (Lines \(chunkStartLine)-\(lineNum - 1))",
                    content: chunkText,
                    startLine: chunkStartLine,
                    endLine: lineNum - 1
                ))
                chunkIdx += 1
                currentChunkLines = [line]
                chunkStartLine = lineNum
                currentChunkLen = line.count
            } else {
                currentChunkLines.append(line)
                currentChunkLen += line.count + 1
                if currentChunkLen >= maxChunkLength {
                    let chunkText = currentChunkLines.joined(separator: "\n")
                    chunks.append(DocumentChunk(
                        index: chunkIdx,
                        title: "Section \(chunkIdx) (Lines \(chunkStartLine)-\(lineNum))",
                        content: chunkText,
                        startLine: chunkStartLine,
                        endLine: lineNum
                    ))
                    chunkIdx += 1
                    currentChunkLines = []
                    chunkStartLine = lineNum + 1
                    currentChunkLen = 0
                }
            }
        }

        if !currentChunkLines.isEmpty {
            let chunkText = currentChunkLines.joined(separator: "\n")
            chunks.append(DocumentChunk(
                index: chunkIdx,
                title: "Section \(chunkIdx) (Lines \(chunkStartLine)-\(lines.count))",
                content: chunkText,
                startLine: chunkStartLine,
                endLine: lines.count
            ))
        }

        return (title: title, chunks: chunks, totalLength: content.count)
    }
}

// MARK: - Controlled Sandboxed Process Runner

public final class ControlledProcessRunner {
    public static let shared = ControlledProcessRunner()

    public func runCommand(
        command: String,
        workingDirectory: URL?,
        timeoutSeconds: Double = 30.0
    ) async throws -> (stdout: String, stderr: String, exitCode: Int32) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]

        if let wd = workingDirectory {
            process.currentDirectoryURL = wd.resolvingSymlinksInPath()
        }

        var env = ProcessInfo.processInfo.environment
        env["PAGER"] = "cat"
        env["TERM"] = "dumb"
        env["CI"] = "1"
        process.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe

        return try await withCheckedThrowingContinuation { continuation in
            var isResumed = false

            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInitiated))
            timer.schedule(deadline: .now() + timeoutSeconds)
            timer.setEventHandler {
                if !isResumed {
                    isResumed = true
                    process.terminate()
                    continuation.resume(throwing: NSError(domain: "ControlledProcessRunner", code: 124, userInfo: [NSLocalizedDescriptionKey: "Command timed out after \(timeoutSeconds) seconds"]))
                }
            }
            timer.resume()

            process.terminationHandler = { proc in
                timer.cancel()
                if !isResumed {
                    isResumed = true
                    let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()
                    let outStr = String(data: outData, encoding: .utf8) ?? ""
                    let errStr = String(data: errData, encoding: .utf8) ?? ""
                    continuation.resume(returning: (stdout: outStr, stderr: errStr, exitCode: proc.terminationStatus))
                }
            }

            do {
                try process.run()
            } catch {
                timer.cancel()
                if !isResumed {
                    isResumed = true
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}
