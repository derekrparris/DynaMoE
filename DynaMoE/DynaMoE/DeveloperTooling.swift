//
//  DeveloperTooling.swift
//  DynaMoE
//
//  Option 3: Deep Developer Tooling (Native Git & SourceKit-LSP)
//  Direct, syntax-aware code intelligence, version control, and self-healing compiler feedback.
//

import Foundation

// MARK: - Git Controller & Models

public struct GitStatusFile: Codable, Sendable {
    public let path: String
    public let statusCode: String
    public let isStaged: Bool
    public let isUntracked: Bool
    public let description: String
}

public struct GitStatusResult: Codable, Sendable {
    public let branch: String
    public let upstream: String?
    public let aheadCount: Int
    public let behindCount: Int
    public let stagedFiles: [GitStatusFile]
    public let unstagedFiles: [GitStatusFile]
    public let untrackedFiles: [GitStatusFile]
    public let hasChanges: Bool
    public let rawOutput: String
    public let summary: String
}

public struct GitDiffResult: Codable, Sendable {
    public let diff: String
    public let targetPath: String?
    public let isStaged: Bool
    public let lineCount: Int
    public let isTruncated: Bool
    public let summary: String
}

public struct GitCommitResult: Codable, Sendable {
    public let commitHash: String
    public let branch: String
    public let message: String
    public let summary: String
}

public final class GitController {
    public static let shared = GitController()

    private let gitURL = URL(fileURLWithPath: "/usr/bin/git")

    private init() {}

    // MARK: - Git Status

    public func status(workingDirectory: URL) async throws -> GitStatusResult {
        let (exitCode, stdout, stderr) = try await AgentHarness.runProcess(
            executableURL: gitURL,
            arguments: ["status", "--porcelain=v1", "-b"],
            currentDirectory: workingDirectory,
            timeoutSeconds: 30
        )

        guard exitCode == 0 else {
            throw NSError(
                domain: "GitController",
                code: Int(exitCode),
                userInfo: [NSLocalizedDescriptionKey: "Git status failed: \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"]
            )
        }

        var branchName = "HEAD"
        var upstreamName: String? = nil
        var ahead = 0
        var behind = 0
        var staged: [GitStatusFile] = []
        var unstaged: [GitStatusFile] = []
        var untracked: [GitStatusFile] = []

        let lines = stdout.components(separatedBy: "\n").filter { !$0.isEmpty }

        for line in lines {
            if line.hasPrefix("## ") {
                // Parse branch line, e.g. "## main...origin/main [ahead 1, behind 2]" or "## Initial commit on main"
                let branchSpec = String(line.dropFirst(3)).trimmingCharacters(in: .whitespacesAndNewlines)
                if branchSpec.contains("...") {
                    let parts = branchSpec.components(separatedBy: "...")
                    branchName = parts[0]
                    let remainder = parts.count > 1 ? parts[1] : ""
                    if remainder.contains(" ") {
                        let upParts = remainder.components(separatedBy: " ")
                        upstreamName = upParts[0]
                        if remainder.contains("ahead") {
                            let match = remainder.range(of: #"ahead (\d+)"#, options: .regularExpression)
                            if let match = match {
                                let sub = String(remainder[match])
                                ahead = Int(sub.components(separatedBy: " ").last ?? "0") ?? 0
                            }
                        }
                        if remainder.contains("behind") {
                            let match = remainder.range(of: #"behind (\d+)"#, options: .regularExpression)
                            if let match = match {
                                let sub = String(remainder[match])
                                behind = Int(sub.components(separatedBy: " ").last ?? "0") ?? 0
                            }
                        }
                    } else {
                        upstreamName = remainder
                    }
                } else {
                    branchName = branchSpec
                }
                continue
            }

            guard line.count >= 3 else { continue }
            let indexCode = line.prefix(1)
            let workTreeCode = line.dropFirst(1).prefix(1)
            let filePath = String(line.dropFirst(3)).trimmingCharacters(in: .whitespaces)

            if indexCode == "?" && workTreeCode == "?" {
                let file = GitStatusFile(path: filePath, statusCode: "??", isStaged: false, isUntracked: true, description: "Untracked")
                untracked.append(file)
            } else {
                if indexCode != " " && indexCode != "?" {
                    let desc = describeStatusCode(String(indexCode))
                    staged.append(GitStatusFile(path: filePath, statusCode: String(indexCode), isStaged: true, isUntracked: false, description: "Staged \(desc)"))
                }
                if workTreeCode != " " && workTreeCode != "?" {
                    let desc = describeStatusCode(String(workTreeCode))
                    unstaged.append(GitStatusFile(path: filePath, statusCode: String(workTreeCode), isStaged: false, isUntracked: false, description: "Unstaged \(desc)"))
                }
            }
        }

        let hasChanges = !staged.isEmpty || !unstaged.isEmpty || !untracked.isEmpty
        var summaryLines = [
            "On branch \(branchName)" + (upstreamName != nil ? " (tracking \(upstreamName!))" : ""),
            "Ahead: \(ahead), Behind: \(behind)"
        ]
        if staged.isEmpty && unstaged.isEmpty && untracked.isEmpty {
            summaryLines.append("Working tree clean (no staged or unstaged changes).")
        } else {
            summaryLines.append("Changes: \(staged.count) staged, \(unstaged.count) unstaged, \(untracked.count) untracked.")
        }

        return GitStatusResult(
            branch: branchName,
            upstream: upstreamName,
            aheadCount: ahead,
            behindCount: behind,
            stagedFiles: staged,
            unstagedFiles: unstaged,
            untrackedFiles: untracked,
            hasChanges: hasChanges,
            rawOutput: stdout,
            summary: summaryLines.joined(separator: "\n")
        )
    }

    private func describeStatusCode(_ code: String) -> String {
        switch code {
        case "M": return "modified"
        case "A": return "added"
        case "D": return "deleted"
        case "R": return "renamed"
        case "C": return "copied"
        case "U": return "updated but unmerged"
        default: return code
        }
    }

    // MARK: - Git Diff

    public func diff(
        workingDirectory: URL,
        path: String? = nil,
        staged: Bool = false,
        target: String? = nil,
        maxLines: Int = 500
    ) async throws -> GitDiffResult {
        var args = ["diff"]
        if staged {
            args.append("--staged")
        }
        if let target = target, !target.isEmpty {
            args.append(target)
        }
        if let path = path, !path.isEmpty {
            args.append("--")
            args.append(path)
        }

        let (exitCode, stdout, stderr) = try await AgentHarness.runProcess(
            executableURL: gitURL,
            arguments: args,
            currentDirectory: workingDirectory,
            timeoutSeconds: 30
        )

        guard exitCode == 0 else {
            throw NSError(
                domain: "GitController",
                code: Int(exitCode),
                userInfo: [NSLocalizedDescriptionKey: "Git diff failed: \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"]
            )
        }

        let lines = stdout.components(separatedBy: "\n")
        let isTruncated = lines.count > maxLines
        let diffSnippet = isTruncated ? lines.prefix(maxLines).joined(separator: "\n") + "\n\n... [Diff truncated at \(maxLines) lines]" : stdout

        let targetLabel = path ?? (staged ? "staged changes" : "working tree")
        let summary = lines.isEmpty || stdout.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "No diff detected for \(targetLabel)."
            : "Diff for \(targetLabel): \(lines.count) lines."

        return GitDiffResult(
            diff: diffSnippet,
            targetPath: path,
            isStaged: staged,
            lineCount: lines.count,
            isTruncated: isTruncated,
            summary: summary
        )
    }

    // MARK: - Git Commit

    public func commit(
        workingDirectory: URL,
        message: String,
        stageAll: Bool = false,
        paths: [String]? = nil
    ) async throws -> GitCommitResult {
        let cleanMessage = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanMessage.isEmpty else {
            throw NSError(
                domain: "GitController",
                code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Safety Rail: Commit message cannot be empty."]
            )
        }

        // Safety Rail: Reject command-injection flags
        let dangerousSubstrings = ["--no-verify", "--amend", "--force", "-f", "--reset-author"]
        for flag in dangerousSubstrings {
            if cleanMessage.contains(flag) {
                throw NSError(
                    domain: "GitController",
                    code: 403,
                    userInfo: [NSLocalizedDescriptionKey: "Safety Rail: Commit message contains forbidden flag '\(flag)'."]
                )
            }
        }

        // 1. Stage requested files if specified
        if let paths = paths, !paths.isEmpty {
            var addArgs = ["add", "--"]
            addArgs.append(contentsOf: paths)
            let (addCode, _, addErr) = try await AgentHarness.runProcess(
                executableURL: gitURL,
                arguments: addArgs,
                currentDirectory: workingDirectory,
                timeoutSeconds: 30
            )
            guard addCode == 0 else {
                throw NSError(
                    domain: "GitController",
                    code: Int(addCode),
                    userInfo: [NSLocalizedDescriptionKey: "Git add failed: \(addErr.trimmingCharacters(in: .whitespacesAndNewlines))"]
                )
            }
        } else if stageAll {
            let (addCode, _, addErr) = try await AgentHarness.runProcess(
                executableURL: gitURL,
                arguments: ["add", "-A"],
                currentDirectory: workingDirectory,
                timeoutSeconds: 30
            )
            guard addCode == 0 else {
                throw NSError(
                    domain: "GitController",
                    code: Int(addCode),
                    userInfo: [NSLocalizedDescriptionKey: "Git add -A failed: \(addErr.trimmingCharacters(in: .whitespacesAndNewlines))"]
                )
            }
        }

        // 2. Verify that staged changes exist
        let statusRes = try await status(workingDirectory: workingDirectory)
        guard !statusRes.stagedFiles.isEmpty else {
            throw NSError(
                domain: "GitController",
                code: 400,
                userInfo: [NSLocalizedDescriptionKey: "Safety Rail: No changes are staged to commit. Set 'stage_all: true' or specify 'paths' to stage files before committing."]
            )
        }

        // 3. Create the commit
        let (commitCode, commitOut, commitErr) = try await AgentHarness.runProcess(
            executableURL: gitURL,
            arguments: ["commit", "-m", cleanMessage],
            currentDirectory: workingDirectory,
            timeoutSeconds: 30
        )

        guard commitCode == 0 else {
            throw NSError(
                domain: "GitController",
                code: Int(commitCode),
                userInfo: [NSLocalizedDescriptionKey: "Git commit failed: \(commitErr.trimmingCharacters(in: .whitespacesAndNewlines))"]
            )
        }

        // Parse short commit hash from output, e.g. "[main 7a2b9c1] commit message"
        var hash = "HEAD"
        let lines = commitOut.components(separatedBy: "\n")
        if let firstLine = lines.first, let match = firstLine.range(of: #"\[([^\s]+)\s+([a-f0-9]+)\]"#, options: .regularExpression) {
            let sub = String(firstLine[match])
            let parts = sub.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").components(separatedBy: " ")
            if parts.count >= 2 {
                hash = parts[1]
            }
        }

        let summary = "Committed [\(hash)] on \(statusRes.branch): \(cleanMessage)"
        return GitCommitResult(
            commitHash: hash,
            branch: statusRes.branch,
            message: cleanMessage,
            summary: summary
        )
    }
}

// MARK: - Symbol Intelligence Engine & Models

public struct SymbolDefinition: Codable, Sendable {
    public let name: String
    public let kind: String
    public let filePath: String
    public let line: Int
    public let column: Int
    public let signature: String
    public let snippet: String
}

public struct SymbolReference: Codable, Sendable {
    public let name: String
    public let filePath: String
    public let line: Int
    public let column: Int
    public let lineContent: String
    public let isCallSite: Bool
}

public final class SymbolIntelligenceEngine {
    public static let shared = SymbolIntelligenceEngine()

    private init() {}

    private let supportedExtensions = ["swift", "metal", "h", "hpp", "c", "cpp", "m", "mm", "rs", "py"]

    // MARK: - Find Symbol Definition

    public func findDefinition(
        symbolName: String,
        inDirectory directory: URL,
        filterPath: String? = nil,
        maxResults: Int = 10
    ) async -> [SymbolDefinition] {
        let cleanSymbol = symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanSymbol.isEmpty else { return [] }

        var results: [SymbolDefinition] = []

        // Compile regex pattern for definitions
        // Matches: class, struct, enum, protocol, actor, typealias, func, let, var, kernel, vertex, fragment, typedef
        let escaped = NSRegularExpression.escapedPattern(for: cleanSymbol)
        let pattern = #"(?m)^[\t ]*(?:(?:public|private|fileprivate|internal|open|final|static|class|mutating|override|lazy|weak|async|kernel|vertex|fragment|typedef)\s+)*(class|struct|enum|protocol|actor|typealias|func|var|let|kernel|vertex|fragment)\s+\b\#(escaped)\b"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }

        let fileManager = FileManager.default
        let baseDir = filterPath != nil ? URL(fileURLWithPath: filterPath!, relativeTo: directory) : directory

        var isDir: ObjCBool = false
        if !fileManager.fileExists(atPath: baseDir.path, isDirectory: &isDir) {
            return []
        }

        let filesToScan: [URL]
        if !isDir.boolValue {
            filesToScan = [baseDir]
        } else {
            filesToScan = enumerateSourceFiles(in: baseDir)
        }

        for fileURL in filesToScan {
            guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else { continue }
            let nsContent = content as NSString
            let fullRange = NSRange(location: 0, length: nsContent.length)

            let matches = regex.matches(in: content, options: [], range: fullRange)
            for m in matches {
                guard m.range.location != NSNotFound else { continue }

                // Determine line and column
                let prefix = nsContent.substring(to: m.range.location)
                let linesBefore = prefix.components(separatedBy: "\n")
                let lineNum = linesBefore.count
                let colNum = (linesBefore.last?.count ?? 0) + 1

                var kind = "symbol"
                if m.numberOfRanges > 1 && m.range(at: 1).location != NSNotFound {
                    kind = nsContent.substring(with: m.range(at: 1))
                }

                let lineRange = nsContent.lineRange(for: m.range)
                let lineContent = nsContent.substring(with: lineRange).trimmingCharacters(in: .newlines)

                // Build context snippet (up to 3 surrounding lines)
                let allLines = content.components(separatedBy: "\n")
                let startIdx = max(0, lineNum - 1)
                let endIdx = min(allLines.count - 1, lineNum + 2)
                let snippet = allLines[startIdx...endIdx].joined(separator: "\n")

                results.append(SymbolDefinition(
                    name: cleanSymbol,
                    kind: kind,
                    filePath: fileURL.path,
                    line: lineNum,
                    column: colNum,
                    signature: lineContent,
                    snippet: snippet
                ))

                if results.count >= maxResults { return results }
            }
        }

        return results
    }

    // MARK: - Find Symbol References

    public func findReferences(
        symbolName: String,
        inDirectory directory: URL,
        filterPath: String? = nil,
        maxResults: Int = 30
    ) async -> [SymbolReference] {
        let cleanSymbol = symbolName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanSymbol.isEmpty else { return [] }

        var results: [SymbolReference] = []

        let escaped = NSRegularExpression.escapedPattern(for: cleanSymbol)
        let pattern = #"\b\#(escaped)\b"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []) else {
            return []
        }

        let fileManager = FileManager.default
        let baseDir = filterPath != nil ? URL(fileURLWithPath: filterPath!, relativeTo: directory) : directory

        var isDir: ObjCBool = false
        if !fileManager.fileExists(atPath: baseDir.path, isDirectory: &isDir) {
            return []
        }

        let filesToScan: [URL]
        if !isDir.boolValue {
            filesToScan = [baseDir]
        } else {
            filesToScan = enumerateSourceFiles(in: baseDir)
        }

        for fileURL in filesToScan {
            guard let content = try? String(contentsOf: fileURL, encoding: .utf8) else { continue }
            let lines = content.components(separatedBy: "\n")

            for (idx, line) in lines.enumerated() {
                let nsLine = line as NSString
                let matches = regex.matches(in: line, options: [], range: NSRange(location: 0, length: nsLine.length))

                for m in matches {
                    let col = m.range.location + 1
                    let trimmed = line.trimmingCharacters(in: .whitespaces)

                    // Skip import statements
                    if trimmed.hasPrefix("import ") { continue }

                    let isCallSite = line.contains("\(cleanSymbol)(") || line.contains("\(cleanSymbol).")

                    results.append(SymbolReference(
                        name: cleanSymbol,
                        filePath: fileURL.path,
                        line: idx + 1,
                        column: col,
                        lineContent: trimmed,
                        isCallSite: isCallSite
                    ))

                    if results.count >= maxResults { return results }
                }
            }
        }

        return results
    }

    private func enumerateSourceFiles(in directory: URL) -> [URL] {
        var result: [URL] = []
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        let forbiddenSubstrings = [".build", "DerivedData", ".git", "Pods", "Carthage", ".xcodeproj", ".xcworkspace"]

        for case let url as URL in enumerator {
            let path = url.path
            if forbiddenSubstrings.contains(where: { path.contains($0) }) {
                continue
            }
            if supportedExtensions.contains(url.pathExtension.lowercased()) {
                result.append(url)
            }
        }
        return result
    }
}

// MARK: - Lint & Diagnostics Engine (Self-Healing Loop)

public struct LintDiagnosticItem: Codable, Sendable {
    public let filePath: String
    public let line: Int
    public let column: Int
    public let severity: String // "error", "warning", "note"
    public let message: String
}

public struct LintDiagnosticReport: Codable, Sendable {
    public let filePath: String
    public let hasErrors: Bool
    public let hasWarnings: Bool
    public let diagnostics: [LintDiagnosticItem]
    public let readableSummary: String
    public let selfHealingPrompt: String?
}

public final class LintDiagnosticsEngine {
    public static let shared = LintDiagnosticsEngine()

    private init() {}

    /// Checks the given source file using fast native syntax/type verification.
    public static func checkFile(at fileURL: URL, workingDirectory: URL? = nil) async -> LintDiagnosticReport {
        let ext = fileURL.pathExtension.lowercased()
        let dir = workingDirectory ?? fileURL.deletingLastPathComponent()

        switch ext {
        case "swift":
            return await checkSwiftFile(fileURL: fileURL, workingDirectory: dir)

        case "metal":
            return await checkMetalFile(fileURL: fileURL, workingDirectory: dir)

        case "c", "cpp", "m", "mm", "h", "hpp":
            return await checkClangFile(fileURL: fileURL, workingDirectory: dir)

        default:
            return LintDiagnosticReport(
                filePath: fileURL.path,
                hasErrors: false,
                hasWarnings: false,
                diagnostics: [],
                readableSummary: "Diagnostics not supported for file extension '.\(ext)'.",
                selfHealingPrompt: nil
            )
        }
    }

    // MARK: - Swift Syntax Check (`swiftc -parse`)

    private static func checkSwiftFile(fileURL: URL, workingDirectory: URL) async -> LintDiagnosticReport {
        let xcrunURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        let (exitCode, _, stderr) = (try? await AgentHarness.runProcess(
            executableURL: xcrunURL,
            arguments: ["swiftc", "-parse", fileURL.path],
            currentDirectory: workingDirectory,
            timeoutSeconds: 15
        )) ?? (-1, "", "Failed to execute swiftc")

        let items = parseCompilerStderr(stderr, fallbackPath: fileURL.path)
        let errors = items.filter { $0.severity == "error" }
        let warnings = items.filter { $0.severity == "warning" }

        let hasErrors = exitCode != 0 || !errors.isEmpty
        let hasWarnings = !warnings.isEmpty

        let summary: String
        let selfHealing: String?

        if hasErrors {
            var lines = ["⚠️ Compiler diagnostics detected \(errors.count) error(s) in \(fileURL.lastPathComponent):"]
            for err in errors {
                lines.append("  • Line \(err.line):\(err.column) [error] \(err.message)")
            }
            summary = lines.joined(separator: "\n")
            selfHealing = """
            ⚠️ SELF-HEALING ACTION REQUIRED:
            The file '\(fileURL.lastPathComponent)' failed compilation after your edit:
            \(errors.map { "Line \($0.line):\($0.column) - \($0.message)" }.joined(separator: "\n"))
            Please inspect the error above and invoke file_edit to fix the syntax error immediately.
            """
        } else if hasWarnings {
            summary = "Compiled with \(warnings.count) warning(s) in \(fileURL.lastPathComponent)."
            selfHealing = nil
        } else {
            summary = "Syntax check passed cleanly (0 errors, 0 warnings)."
            selfHealing = nil
        }

        return LintDiagnosticReport(
            filePath: fileURL.path,
            hasErrors: hasErrors,
            hasWarnings: hasWarnings,
            diagnostics: items,
            readableSummary: summary,
            selfHealingPrompt: selfHealing
        )
    }

    // MARK: - Metal Syntax Check (`metal -fsyntax-only`)

    private static func checkMetalFile(fileURL: URL, workingDirectory: URL) async -> LintDiagnosticReport {
        let xcrunURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        let (exitCode, _, stderr) = (try? await AgentHarness.runProcess(
            executableURL: xcrunURL,
            arguments: ["metal", "-fsyntax-only", fileURL.path],
            currentDirectory: workingDirectory,
            timeoutSeconds: 15
        )) ?? (-1, "", "Failed to execute metal")

        let items = parseCompilerStderr(stderr, fallbackPath: fileURL.path)
        let errors = items.filter { $0.severity == "error" }
        let warnings = items.filter { $0.severity == "warning" }

        let hasErrors = exitCode != 0 || !errors.isEmpty
        let hasWarnings = !warnings.isEmpty

        let summary: String
        let selfHealing: String?

        if hasErrors {
            var lines = ["⚠️ Metal compiler diagnostics detected \(errors.count) error(s) in \(fileURL.lastPathComponent):"]
            for err in errors {
                lines.append("  • Line \(err.line):\(err.column) [error] \(err.message)")
            }
            summary = lines.joined(separator: "\n")
            selfHealing = """
            ⚠️ SELF-HEALING ACTION REQUIRED:
            Metal kernel compilation failed with error(s):
            \(errors.map { "Line \($0.line):\($0.column) - \($0.message)" }.joined(separator: "\n"))
            Please use file_edit to correct the Metal shader syntax.
            """
        } else {
            summary = "Metal shader syntax check passed cleanly."
            selfHealing = nil
        }

        return LintDiagnosticReport(
            filePath: fileURL.path,
            hasErrors: hasErrors,
            hasWarnings: hasWarnings,
            diagnostics: items,
            readableSummary: summary,
            selfHealingPrompt: selfHealing
        )
    }

    // MARK: - Clang Syntax Check (`clang -fsyntax-only`)

    private static func checkClangFile(fileURL: URL, workingDirectory: URL) async -> LintDiagnosticReport {
        let xcrunURL = URL(fileURLWithPath: "/usr/bin/xcrun")
        let (exitCode, _, stderr) = (try? await AgentHarness.runProcess(
            executableURL: xcrunURL,
            arguments: ["clang", "-fsyntax-only", fileURL.path],
            currentDirectory: workingDirectory,
            timeoutSeconds: 15
        )) ?? (-1, "", "Failed to execute clang")

        let items = parseCompilerStderr(stderr, fallbackPath: fileURL.path)
        let errors = items.filter { $0.severity == "error" }
        let hasErrors = exitCode != 0 || !errors.isEmpty

        let summary = hasErrors ? "Clang diagnostics: \(errors.count) error(s)." : "Clang check passed cleanly."
        let selfHealing = hasErrors ? "Please invoke file_edit to fix Clang errors: \(errors.map { $0.message }.joined(separator: "; "))" : nil

        return LintDiagnosticReport(
            filePath: fileURL.path,
            hasErrors: hasErrors,
            hasWarnings: false,
            diagnostics: items,
            readableSummary: summary,
            selfHealingPrompt: selfHealing
        )
    }

    // MARK: - Error Output Parser

    private static func parseCompilerStderr(_ stderr: String, fallbackPath: String) -> [LintDiagnosticItem] {
        var items: [LintDiagnosticItem] = []
        let lines = stderr.components(separatedBy: "\n")

        // Matches: /path/to/file.swift:14:22: error: message
        let regex = try? NSRegularExpression(pattern: #"^(.*?):(\d+):(\d+):\s+(error|warning|note):\s+(.*)$"#, options: [])

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }

            let nsLine = trimmed as NSString
            if let match = regex?.firstMatch(in: trimmed, options: [], range: NSRange(location: 0, length: nsLine.length)) {
                let filePath = nsLine.substring(with: match.range(at: 1))
                let lineNum = Int(nsLine.substring(with: match.range(at: 2))) ?? 1
                let colNum = Int(nsLine.substring(with: match.range(at: 3))) ?? 1
                let severity = nsLine.substring(with: match.range(at: 4))
                let message = nsLine.substring(with: match.range(at: 5))

                items.append(LintDiagnosticItem(
                    filePath: filePath.isEmpty ? fallbackPath : filePath,
                    line: lineNum,
                    column: colNum,
                    severity: severity,
                    message: message
                ))
            }
        }
        return items
    }
}
