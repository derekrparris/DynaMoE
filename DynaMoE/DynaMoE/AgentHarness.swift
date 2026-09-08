//
//  AgentHarness.swift
//  DynaMoE
//
//  Stateful Multi-Turn Agent Tool Harness for Qwen, DeepSeek & ChatML Models
//  Preserves exact ChatML syntax (<tool_call>, <|im_start|>, <|im_end|>) and maintains
//  persistent KV cache across tool turns without quadratic re-prefill.
//

import Foundation

// MARK: - Tool Definitions & Schemas

public struct ToolDefinition: Codable, Equatable {
    public let type: String
    public let function: ToolFunction

    public struct ToolFunction: Codable, Equatable {
        public let name: String
        public let description: String
        public let parameters: [String: AnyCodable]
    }

    public init(name: String, description: String, parameters: [String: AnyCodable]) {
        self.type = "function"
        self.function = ToolFunction(name: name, description: description, parameters: parameters)
    }
}

public struct ParsedToolCall: Equatable {
    public let name: String
    public let arguments: [String: Any]
    public let rawArguments: String
    public let rawText: String

    public static func == (lhs: ParsedToolCall, rhs: ParsedToolCall) -> Bool {
        return lhs.name == rhs.name && lhs.rawText == rhs.rawText
    }
}

public struct AnyCodable: Codable, Equatable {
    public let value: Any

    public init(_ value: Any) {
        self.value = value
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let val = try? container.decode(String.self) {
            value = val
        } else if let val = try? container.decode(Int.self) {
            value = val
        } else if let val = try? container.decode(Double.self) {
            value = val
        } else if let val = try? container.decode(Bool.self) {
            value = val
        } else if let val = try? container.decode([String: AnyCodable].self) {
            value = val.mapValues { $0.value }
        } else if let val = try? container.decode([AnyCodable].self) {
            value = val.map { $0.value }
        } else {
            value = ""
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        if let val = value as? String {
            try container.encode(val)
        } else if let val = value as? Int {
            try container.encode(val)
        } else if let val = value as? Double {
            try container.encode(val)
        } else if let val = value as? Bool {
            try container.encode(val)
        } else if let val = value as? [String: Any] {
            let wrapped = val.mapValues { AnyCodable($0) }
            try container.encode(wrapped)
        } else if let val = value as? [Any] {
            let wrapped = val.map { AnyCodable($0) }
            try container.encode(wrapped)
        }
    }

    public static func == (lhs: AnyCodable, rhs: AnyCodable) -> Bool {
        return String(describing: lhs.value) == String(describing: rhs.value)
    }
}

// MARK: - Agent Tool Protocol

public protocol AgentTool {
    var definition: ToolDefinition { get }
    func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool)
}

// MARK: - Core Coding & Shell Tool Implementations

/// Tool 1: shell_run — Executes shell commands with working directory & timeout
public final class ShellRunTool: AgentTool {
    public let definition = ToolDefinition(
        name: "shell_run",
        description: "Executes shell commands on the local macOS terminal via zsh. Use this to run scripts, compilers, git, or check system state. Output is captured and returned.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "command": [
                    "type": "string",
                    "description": "The exact shell command line string to execute."
                ],
                "cwd": [
                    "type": "string",
                    "description": "Optional directory path to execute the command in. Defaults to current workspace."
                ]
            ]),
            "required": AnyCodable(["command"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let command = arguments["command"] as? String, !command.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "Error: missing or empty 'command' parameter in shell_run"
            return (AgentHarness.toolErrorJSON(tool: "shell_run", error: err), nil, err, false)
        }

        let targetDir: URL
        if let customCwd = arguments["cwd"] as? String, !customCwd.isEmpty {
            targetDir = URL(fileURLWithPath: (customCwd as NSString).expandingTildeInPath)
        } else if let wd = workingDirectory {
            targetDir = wd
        } else {
            targetDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }

        let (stdout, stderr, exitCode) = try await ControlledProcessRunner.shared.runCommand(
            command: command,
            workingDirectory: targetDir,
            timeoutSeconds: 120.0
        )

        let cleanStdout = AgentHarness.truncateText(AgentHarness.sanitizeText(stdout.trimmingCharacters(in: .whitespacesAndNewlines)), limit: maxOutputLength)
        let cleanStderr = AgentHarness.truncateText(AgentHarness.sanitizeText(stderr.trimmingCharacters(in: .whitespacesAndNewlines)), limit: maxOutputLength)

        if exitCode == 0 {
            let res = AgentHarness.toolSuccessJSON(tool: "shell_run", data: [
                "stdout": cleanStdout.isEmpty ? "Command succeeded with no output." : cleanStdout,
                "exit_code": 0
            ])
            return (res, cleanStdout, nil, false)
        } else {
            let res = AgentHarness.toolErrorJSON(tool: "shell_run", error: cleanStderr.isEmpty ? cleanStdout : cleanStderr, extra: [
                "exit_code": exitCode,
                "stdout": cleanStdout
            ])
            let effectiveStderr = !cleanStderr.isEmpty ? cleanStderr : (!cleanStdout.isEmpty ? "Command exited with code \(exitCode): \(cleanStdout)" : "Command exited with error code \(exitCode)")
            return (res, cleanStdout, effectiveStderr, false)
        }
    }
}

/// Tool 2: file_read — Reads local files with line range slicing and line numbers
public final class FileReadTool: AgentTool {
    public let definition = ToolDefinition(
        name: "file_read",
        description: "Reads contents of a file on the local filesystem. Supports line range slicing (start_line, end_line) with 1-indexed line numbers to inspect large source files efficiently.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Path to the file to read (absolute or relative to workspace)."
                ],
                "start_line": [
                    "type": "integer",
                    "description": "Optional 1-indexed start line number."
                ],
                "end_line": [
                    "type": "integer",
                    "description": "Optional 1-indexed end line number."
                ]
            ]),
            "required": AnyCodable(["path"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let rawPath = arguments["path"] as? String, !rawPath.isEmpty else {
            let err = "Error: missing 'path' parameter in file_read"
            return (AgentHarness.toolErrorJSON(tool: "file_read", error: err), nil, err, false)
        }

        let resolvedPath = AgentHarness.resolvePath(rawPath, workingDirectory: workingDirectory)
        guard FileManager.default.fileExists(atPath: resolvedPath.path) else {
            let err = "File not found at path: \(resolvedPath.path)"
            return (AgentHarness.toolErrorJSON(tool: "file_read", error: err), nil, err, false)
        }

        if resolvedPath.pathExtension.lowercased() == "pdf" {
            do {
                let (title, chunks, totalChars) = try SemanticDocumentReader.shared.readDocument(at: resolvedPath, maxChunkLength: maxOutputLength)
                let combined = chunks.map { "=== \($0.title) ===\n\($0.content)" }.joined(separator: "\n\n")
                let truncated = AgentHarness.truncateText(AgentHarness.sanitizeText(combined), limit: maxOutputLength)
                let res = AgentHarness.toolSuccessJSON(tool: "file_read", data: [
                    "path": resolvedPath.path,
                    "title": title,
                    "total_characters": totalChars,
                    "page_chunks": chunks.count,
                    "content": truncated
                ])
                return (res, truncated, nil, false)
            } catch {
                let err = "Failed to parse PDF document: \(error.localizedDescription)"
                return (AgentHarness.toolErrorJSON(tool: "file_read", error: err), nil, err, false)
            }
        }

        do {
            let content = try String(contentsOf: resolvedPath, encoding: .utf8)
            let lines = content.components(separatedBy: "\n")
            let totalLines = lines.count

            var startLine = 1
            if let s = arguments["start_line"] as? Int {
                startLine = max(1, min(s, totalLines))
            }

            var endLine = totalLines
            if let e = arguments["end_line"] as? Int {
                endLine = max(startLine, min(e, totalLines))
            }

            var numberedLines: [String] = []
            for idx in (startLine - 1)..<endLine {
                numberedLines.append("\(idx + 1): \(lines[idx])")
            }

            let slicedText = numberedLines.joined(separator: "\n")
            let sanitized = AgentHarness.truncateText(AgentHarness.sanitizeText(slicedText), limit: maxOutputLength)

            let res = AgentHarness.toolSuccessJSON(tool: "file_read", data: [
                "path": resolvedPath.path,
                "total_lines": totalLines,
                "start_line": startLine,
                "end_line": endLine,
                "content": sanitized
            ])
            return (res, sanitized, nil, false)
        } catch {
            let err = "Failed to read file: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "file_read", error: err), nil, err, false)
        }
    }
}

/// Tool 3: file_write — Creates or overwrites files atomically
public final class FileWriteTool: AgentTool {
    public let definition = ToolDefinition(
        name: "file_write",
        description: "Creates a new file or overwrites an existing file with provided text content. Automatically creates parent directories.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Path to the file to create or overwrite."
                ],
                "content": [
                    "type": "string",
                    "description": "Full text content to write to the file."
                ]
            ]),
            "required": AnyCodable(["path", "content"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let rawPath = arguments["path"] as? String, !rawPath.isEmpty else {
            let err = "Error: missing 'path' parameter in file_write"
            return (AgentHarness.toolErrorJSON(tool: "file_write", error: err), nil, err, false)
        }
        guard let content = arguments["content"] as? String else {
            let err = "Error: missing 'content' parameter in file_write"
            return (AgentHarness.toolErrorJSON(tool: "file_write", error: err), nil, err, false)
        }

        let resolvedPath = AgentHarness.resolvePath(rawPath, workingDirectory: workingDirectory)
        do {
            let parentDir = resolvedPath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)
            try content.write(to: resolvedPath, atomically: true, encoding: .utf8)

            let byteCount = content.utf8.count
            let report = await LintDiagnosticsEngine.checkFile(at: resolvedPath, workingDirectory: workingDirectory)

            var msg = "Successfully wrote \(byteCount) bytes to \(resolvedPath.path)"
            var status = "written"
            if report.hasErrors {
                status = "written_with_syntax_errors"
                msg += "\n\n" + report.readableSummary + (report.selfHealingPrompt != nil ? "\n\n" + report.selfHealingPrompt! : "")
            } else if report.hasWarnings {
                status = "written_with_warnings"
                msg += "\n\n" + report.readableSummary
            }

            let res = AgentHarness.toolSuccessJSON(tool: "file_write", data: [
                "path": resolvedPath.path,
                "bytes_written": byteCount,
                "status": status,
                "compiler_diagnostics": report.diagnostics.map { [
                    "line": $0.line,
                    "column": $0.column,
                    "severity": $0.severity,
                    "message": $0.message
                ] },
                "self_healing_hint": report.selfHealingPrompt ?? ""
            ])
            return (res, msg, nil, false)
        } catch {
            let err = "Failed to write file: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "file_write", error: err), nil, err, false)
        }
    }
}

/// Tool 4: file_edit — Precise contiguous anchor search-and-replace (replace_file_content pattern)
public final class FileEditTool: AgentTool {
    public let definition = ToolDefinition(
        name: "file_edit",
        description: "Performs precise contiguous text replacement in a file. Provide the exact target_content to be replaced and replacement_content. Guarantees safety by checking for exact matches.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Path to the file to edit."
                ],
                "target_content": [
                    "type": "string",
                    "description": "Exact text substring in the file to find and replace."
                ],
                "replacement_content": [
                    "type": "string",
                    "description": "New text to substitute in place of target_content."
                ]
            ]),
            "required": AnyCodable(["path", "target_content", "replacement_content"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let rawPath = arguments["path"] as? String, !rawPath.isEmpty else {
            let err = "Error: missing 'path' parameter in file_edit"
            return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
        }
        guard let target = arguments["target_content"] as? String, !target.isEmpty else {
            let err = "Error: missing 'target_content' in file_edit"
            return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
        }
        guard let replacement = arguments["replacement_content"] as? String else {
            let err = "Error: missing 'replacement_content' in file_edit"
            return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
        }

        let resolvedPath = AgentHarness.resolvePath(rawPath, workingDirectory: workingDirectory)
        guard FileManager.default.fileExists(atPath: resolvedPath.path) else {
            let err = "File not found at path: \(resolvedPath.path)"
            return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
        }

        do {
            let existing = try String(contentsOf: resolvedPath, encoding: .utf8)
            guard existing.contains(target) else {
                let err = "target_content not found in \(resolvedPath.lastPathComponent). Ensure whitespace and indentation match exactly."
                return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
            }

            // Verify unique occurrence or replace first
            guard let range = existing.range(of: target) else {
                let err = "Could not locate target_content range in file."
                return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
            }

            let updated = existing.replacingCharacters(in: range, with: replacement)
            try updated.write(to: resolvedPath, atomically: true, encoding: .utf8)

            let report = await LintDiagnosticsEngine.checkFile(at: resolvedPath, workingDirectory: workingDirectory)

            var msg = "Successfully edited \(resolvedPath.lastPathComponent)"
            var status = "success"
            if report.hasErrors {
                status = "syntax_errors_detected"
                msg += "\n\n" + report.readableSummary + (report.selfHealingPrompt != nil ? "\n\n" + report.selfHealingPrompt! : "")
            } else if report.hasWarnings {
                status = "success_with_warnings"
                msg += "\n\n" + report.readableSummary
            }

            let res = AgentHarness.toolSuccessJSON(tool: "file_edit", data: [
                "path": resolvedPath.path,
                "status": status,
                "message": msg,
                "compiler_diagnostics": report.diagnostics.map { [
                    "line": $0.line,
                    "column": $0.column,
                    "severity": $0.severity,
                    "message": $0.message
                ] },
                "self_healing_hint": report.selfHealingPrompt ?? ""
            ])
            return (res, msg, nil, false)
        } catch {
            let err = "Failed to edit file: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "file_edit", error: err), nil, err, false)
        }
    }
}

/// Tool 5: find_files — Finds files matching glob / pattern in directory
public final class FindFilesTool: AgentTool {
    public let definition = ToolDefinition(
        name: "find_files",
        description: "Search for files and directories within a specified path matching a glob pattern or file extension.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "pattern": [
                    "type": "string",
                    "description": "Pattern or extension to match (e.g. '*.swift', 'ContentView', '*.metal')."
                ],
                "path": [
                    "type": "string",
                    "description": "Optional search directory. Defaults to workspace root."
                ],
                "max_depth": [
                    "type": "integer",
                    "description": "Optional maximum folder depth to search. Default 5."
                ]
            ]),
            "required": AnyCodable(["pattern"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let pattern = arguments["pattern"] as? String, !pattern.isEmpty else {
            let err = "Error: missing 'pattern' in find_files"
            return (AgentHarness.toolErrorJSON(tool: "find_files", error: err), nil, err, false)
        }

        let baseDir: URL
        if let customPath = arguments["path"] as? String, !customPath.isEmpty {
            baseDir = AgentHarness.resolvePath(customPath, workingDirectory: workingDirectory)
        } else if let wd = workingDirectory {
            baseDir = wd
        } else {
            baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }

        let maxDepth = (arguments["max_depth"] as? Int) ?? 5
        let escapedPattern = pattern.replacingOccurrences(of: "'", with: "\\'")
        let cmd = "find . -maxdepth \(maxDepth) -iname '\(escapedPattern)' -not -path '*/.*' | head -n 50"

        let (_, stdout, _) = try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", cmd],
            currentDirectory: baseDir,
            timeoutSeconds: 30
        )

        let matches = stdout.components(separatedBy: "\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        let res = AgentHarness.toolSuccessJSON(tool: "find_files", data: [
            "search_path": baseDir.path,
            "pattern": pattern,
            "count": matches.count,
            "matches": matches
        ])
        return (res, matches.joined(separator: "\n"), nil, false)
    }
}

/// Tool 6: grep_search — Fast regex and literal pattern search using ripgrep or grep
public final class GrepSearchTool: AgentTool {
    public let definition = ToolDefinition(
        name: "grep_search",
        description: "Searches for text or regular expression patterns across files in a directory using ripgrep (rg) or grep.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "query": [
                    "type": "string",
                    "description": "Text or regex pattern to search for."
                ],
                "path": [
                    "type": "string",
                    "description": "Optional directory or file path to search in."
                ],
                "case_insensitive": [
                    "type": "boolean",
                    "description": "Whether to perform case-insensitive search. Default true."
                ]
            ]),
            "required": AnyCodable(["query"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let query = arguments["query"] as? String, !query.isEmpty else {
            let err = "Error: missing 'query' in grep_search"
            return (AgentHarness.toolErrorJSON(tool: "grep_search", error: err), nil, err, false)
        }

        let baseDir: URL
        if let customPath = arguments["path"] as? String, !customPath.isEmpty {
            baseDir = AgentHarness.resolvePath(customPath, workingDirectory: workingDirectory)
        } else if let wd = workingDirectory {
            baseDir = wd
        } else {
            baseDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }

        let caseInsensitive = (arguments["case_insensitive"] as? Bool) ?? true
        let escapedQuery = query.replacingOccurrences(of: "'", with: "\\'")
        let flag = caseInsensitive ? "-i" : ""
        let cmd = "if command -v rg >/dev/null 2>&1; then rg \(flag) -n --max-count 50 -- '\(escapedQuery)' .; else grep -rn \(flag) --max-count=50 --exclude-dir=.* -- '\(escapedQuery)' .; fi"

        let (_, stdout, _) = try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", cmd],
            currentDirectory: baseDir,
            timeoutSeconds: 30
        )

        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(stdout.trimmingCharacters(in: .whitespacesAndNewlines)), limit: maxOutputLength)
        let lines = cleanOut.components(separatedBy: "\n").filter { !$0.isEmpty }

        let res = AgentHarness.toolSuccessJSON(tool: "grep_search", data: [
            "query": query,
            "count": lines.count,
            "results": lines
        ])
        return (res, cleanOut, nil, false)
    }
}

/// Manages local Headless Chrome / Chromium browser execution for web search and DOM extraction
public final class HeadlessChromeSearchEngine: @unchecked Sendable {
    public static let shared = HeadlessChromeSearchEngine()

    public static let knownBrowserPaths: [String] = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"
    ]

    /// Resolves the active browser binary path (from custom user preference, standard macOS paths, or PATH)
    public func resolveBinaryPath() -> String? {
        if let custom = UserDefaults.standard.string(forKey: "dynamoe_chrome_binary_path")?.trimmingCharacters(in: .whitespacesAndNewlines),
           !custom.isEmpty, FileManager.default.isExecutableFile(atPath: custom) {
            return custom
        }
        for path in Self.knownBrowserPaths {
            if FileManager.default.isExecutableFile(atPath: path) {
                return path
            }
        }
        // Fallback: check PATH using /usr/bin/which
        for bin in ["google-chrome", "chromium", "chrome"] {
            let proc = Process()
            proc.executableURL = URL(fileURLWithPath: "/usr/bin/which")
            proc.arguments = [bin]
            let pipe = Pipe()
            proc.standardOutput = pipe
            if let _ = try? proc.run() {
                proc.waitUntilExit()
                if proc.terminationStatus == 0 {
                    let out = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    if !out.isEmpty && FileManager.default.isExecutableFile(atPath: out) {
                        return out
                    }
                }
            }
        }
        return nil
    }

    /// Executes headless Chrome to dump the rendered DOM of a URL to a temporary file, avoiding 64KB pipe buffer limits
    public func dumpDOM(url: String, timeoutSeconds: Double = 15.0) async throws -> String {
        guard let binaryPath = resolveBinaryPath() else {
            throw NSError(domain: "HeadlessChromeSearchEngine", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "Headless Chrome / Chromium executable not found on macOS. Please install Google Chrome or specify the binary path in Settings."
            ])
        }

        let tmpFile = FileManager.default.temporaryDirectory.appendingPathComponent("dynamoe_dom_\(UUID().uuidString).html")
        FileManager.default.createFile(atPath: tmpFile.path, contents: nil)
        let fileHandle = try FileHandle(forWritingTo: tmpFile)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binaryPath)
        process.arguments = [
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-sync",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-features=Translate,OptimizationHints,MediaRouter",
            "--disable-default-apps",
            "--mute-audio",
            "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
            "--dump-dom",
            url
        ]
        process.standardOutput = fileHandle
        process.standardError = FileHandle.nullDevice

        return try await withCheckedThrowingContinuation { continuation in
            var isResumed = false
            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .userInitiated))
            timer.schedule(deadline: .now() + timeoutSeconds)
            timer.setEventHandler {
                if !isResumed {
                    isResumed = true
                    process.terminate()
                    try? fileHandle.close()
                    try? FileManager.default.removeItem(at: tmpFile)
                    continuation.resume(throwing: NSError(domain: "HeadlessChromeSearchEngine", code: 124, userInfo: [
                        NSLocalizedDescriptionKey: "Headless Chrome timed out after \(timeoutSeconds) seconds."
                    ]))
                }
            }
            timer.resume()

            process.terminationHandler = { proc in
                timer.cancel()
                if !isResumed {
                    isResumed = true
                    try? fileHandle.close()
                    do {
                        let html = try String(contentsOf: tmpFile, encoding: .utf8)
                        try? FileManager.default.removeItem(at: tmpFile)
                        continuation.resume(returning: html)
                    } catch {
                        try? FileManager.default.removeItem(at: tmpFile)
                        continuation.resume(throwing: error)
                    }
                }
            }

            do {
                try process.run()
            } catch {
                timer.cancel()
                if !isResumed {
                    isResumed = true
                    try? fileHandle.close()
                    try? FileManager.default.removeItem(at: tmpFile)
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// Unwraps search engine tracking redirects (such as Yahoo /RU=, Google /url?q=, DuckDuckGo /l/?uddg=) to direct canonical URLs
    public static func unwrapRedirectURL(_ rawUrl: String) -> String {
        var urlStr = rawUrl.trimmingCharacters(in: .whitespacesAndNewlines)

        // Yahoo redirect: /RU=(encoded_url)/RK=
        if let ruRange = urlStr.range(of: "/RU=", options: .caseInsensitive) {
            let afterRU = String(urlStr[ruRange.upperBound...])
            let encodedTarget: String
            if let rkRange = afterRU.range(of: "/RK=", options: .caseInsensitive) {
                encodedTarget = String(afterRU[..<rkRange.lowerBound])
            } else if let slashRange = afterRU.range(of: "/") {
                encodedTarget = String(afterRU[..<slashRange.lowerBound])
            } else {
                encodedTarget = afterRU
            }
            if let decoded = encodedTarget.removingPercentEncoding, decoded.hasPrefix("http") {
                urlStr = decoded
            }
        }
        // Google redirect: /url?q=(encoded_url)&
        else if urlStr.contains("/url?") && urlStr.contains("q=") {
            if let components = URLComponents(string: urlStr),
               let qItem = components.queryItems?.first(where: { $0.name == "q" }),
               let target = qItem.value, target.hasPrefix("http") {
                urlStr = target
            }
        }
        // DuckDuckGo redirect: /l/?uddg=(encoded_url)&
        else if urlStr.contains("uddg=") {
            if let components = URLComponents(string: urlStr),
               let uddgItem = components.queryItems?.first(where: { $0.name == "uddg" }),
               let target = uddgItem.value, target.hasPrefix("http") {
                urlStr = target
            }
        }

        return urlStr
    }

    /// Performs live web search using Headless Chrome instance
    public func search(query: String, maxResults: Int = 5) async throws -> [[String: String]] {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed) else {
            return []
        }
        let searchURL = "https://search.yahoo.com/search?p=\(encoded)"
        let html = try await dumpDOM(url: searchURL, timeoutSeconds: 15.0)

        var results: [[String: String]] = []
        var seenURLs = Set<String>()

        // 1. First attempt: Parse structured search result items
        let parts = html.components(separatedBy: "<li")
        for part in parts {
            guard part.contains("class=\"title") || part.contains("<h3") else { continue }

            guard let urlRange = part.range(of: "href=\"(https?://[^\"]+)\"", options: .regularExpression) else { continue }
            let rawUrl = String(part[urlRange])
                .replacingOccurrences(of: "href=\"", with: "")
                .replacingOccurrences(of: "\"", with: "")
                .replacingOccurrences(of: "&amp;", with: "&")

            let cleanUrl = Self.unwrapRedirectURL(rawUrl)
            guard !cleanUrl.contains("yahoo.com") && !cleanUrl.contains("yimg.com") && !seenURLs.contains(cleanUrl) else { continue }

            var title = ""
            if let titleRange = part.range(of: "<h3[^>]*>([\\s\\S]*?)</h3>", options: .regularExpression) {
                title = String(part[titleRange]).replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            } else {
                title = part.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            }
            title = stripHTML(title)
            guard !title.isEmpty && title.count >= 3 else { continue }

            var snippet = ""
            if let snipRange = part.range(of: "<div class=\"compText[^\"]*\"[^>]*>([\\s\\S]*?)</div>", options: .regularExpression) {
                snippet = stripHTML(String(part[snipRange]))
            } else if let snipRange = part.range(of: "<p[^>]*class=\"[^\"]*snippet[^\"]*\"[^>]*>([\\s\\S]*?)</p>", options: .regularExpression) {
                snippet = stripHTML(String(part[snipRange]))
            } else if let snipRange = part.range(of: "<p[^>]*>([\\s\\S]*?)</p>", options: .regularExpression) {
                snippet = stripHTML(String(part[snipRange]))
            }

            seenURLs.insert(cleanUrl)
            results.append([
                "title": title,
                "url": cleanUrl,
                "snippet": snippet
            ])
            if results.count >= maxResults { break }
        }

        // 2. Fallback: Parse organic <a> tags if list partition returned few results
        if results.isEmpty {
            let aPattern = #"<a\s+[^>]*href=\"(https?://[^\"]+)\"[^>]*>([\s\S]*?)</a>"#
            if let regex = try? NSRegularExpression(pattern: aPattern) {
                let nsHtml = html as NSString
                let matches = regex.matches(in: html, range: NSRange(location: 0, length: nsHtml.length))
                for m in matches {
                    let u = nsHtml.substring(with: m.range(at: 1))
                    let inner = nsHtml.substring(with: m.range(at: 2))
                    let t = stripHTML(inner)
                    let cleanU = Self.unwrapRedirectURL(u)
                    if t.count > 10 && !cleanU.contains("yahoo.com") && !cleanU.contains("yimg.com") && !cleanU.hasPrefix("#") && !seenURLs.contains(cleanU) {
                        seenURLs.insert(cleanU)
                        results.append([
                            "title": t,
                            "url": cleanU,
                            "snippet": ""
                        ])
                        if results.count >= maxResults { break }
                    }
                }
            }
        }

        return results
    }

    private func stripHTML(_ input: String) -> String {
        let withoutTags = input.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
        return withoutTags
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&#x27;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Tool 7: web_search — Performs live web search using Headless Chrome / Brave
public final class WebSearchTool: AgentTool {
    public let definition = ToolDefinition(
        name: "web_search",
        description: "Performs live web search using a local Headless Chrome browser instance for real-time information, documentation, libraries, news, and technical answers. Returns structured list of titles, URLs, and snippets.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "query": [
                    "type": "string",
                    "description": "The search query string to look up."
                ],
                "max_results": [
                    "type": "integer",
                    "description": "Optional maximum number of search results to return (1-10, default 5)."
                ]
            ]),
            "required": AnyCodable(["query"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let query = arguments["query"] as? String, !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "Missing or empty 'query' parameter."
            return (AgentHarness.toolErrorJSON(tool: "web_search", error: err), nil, err, false)
        }

        let maxResults = min(10, max(1, (arguments["max_results"] as? Int) ?? 5))
        let cleanQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)

        do {
            // Check for optional custom Brave Search API key
            let braveKey = UserDefaults.standard.string(forKey: "dynamoe_brave_search_api_key")?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            var results: [[String: String]] = []

            if !braveKey.isEmpty {
                results = try await searchBrave(query: cleanQuery, apiKey: braveKey, maxResults: maxResults)
            }

            // De facto search engine: Headless Chrome
            if results.isEmpty {
                results = try await HeadlessChromeSearchEngine.shared.search(query: cleanQuery, maxResults: maxResults)
            }

            if results.isEmpty {
                results = try await searchWikipedia(query: cleanQuery, maxResults: maxResults)
            }

            if results.isEmpty {
                let res = AgentHarness.toolSuccessJSON(tool: "web_search", data: ["query": cleanQuery, "count": 0, "results": []])
                return (res, res, nil, false)
            }

            var enrichedResults: [[String: Any]] = []
            for (idx, item) in results.enumerated() {
                let urlStr = item["url"] ?? ""
                let host = URL(string: urlStr)?.host?.replacingOccurrences(of: "www.", with: "") ?? ""
                let isOfficial = AgentHarness.isOfficialVendorDomain(host)
                var entry: [String: Any] = [
                    "position": idx + 1,
                    "title": item["title"] ?? "Untitled",
                    "url": urlStr,
                    "domain": host,
                    "is_official_domain": isOfficial
                ]
                if let snippet = item["snippet"], !snippet.isEmpty {
                    let cleanSnippet = AgentHarness.sanitizeText(snippet.trimmingCharacters(in: .whitespacesAndNewlines))
                    entry["snippet"] = cleanSnippet.count > 300 ? String(cleanSnippet.prefix(300)) + "..." : cleanSnippet
                }
                enrichedResults.append(entry)
            }

            let res = AgentHarness.toolSuccessJSON(tool: "web_search", data: [
                "query": cleanQuery,
                "count": results.count,
                "results": enrichedResults
            ])
            return (res, res, nil, false)
        } catch {
            let err = "Web search failed: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "web_search", error: err), nil, err, false)
        }
    }

    private func searchBrave(query: String, apiKey: String, maxResults: Int) async throws -> [[String: String]] {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://api.search.brave.com/res/v1/web/search?q=\(encoded)&count=\(maxResults)") else {
            return []
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return [] }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let web = json["web"] as? [String: Any],
              let items = web["results"] as? [[String: Any]] else { return [] }

        var results: [[String: String]] = []
        for item in items.prefix(maxResults) {
            let title = (item["title"] as? String) ?? ""
            let url = (item["url"] as? String) ?? ""
            let desc = (item["description"] as? String) ?? ""
            if !url.isEmpty {
                results.append([
                    "title": stripHTML(title),
                    "url": url,
                    "snippet": stripHTML(desc)
                ])
            }
        }
        return results
    }

    private func searchWikipedia(query: String, maxResults: Int) async throws -> [[String: String]] {
        guard let encoded = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=\(encoded)&utf8=&format=json") else {
            return []
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else { return [] }

        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let q = json["query"] as? [String: Any],
              let search = q["search"] as? [[String: Any]] else { return [] }

        var results: [[String: String]] = []
        for item in search.prefix(maxResults) {
            let title = (item["title"] as? String) ?? ""
            let snippet = (item["snippet"] as? String) ?? ""
            let pageId = item["pageid"] as? Int ?? 0
            let pageUrl = "https://en.wikipedia.org/?curid=\(pageId)"
            if !title.isEmpty {
                results.append([
                    "title": title,
                    "url": pageUrl,
                    "snippet": stripHTML(snippet)
                ])
            }
        }
        return results
    }

    private func stripHTML(_ input: String) -> String {
        let withoutTags = input.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
        return withoutTags
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&#x27;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Tool 8: web_fetch — Reads content from a web URL
public final class WebFetchTool: AgentTool {
    public let definition = ToolDefinition(
        name: "web_fetch",
        description: "Fetches and reads the textual content of a web page given a public URL. Automatically renders JavaScript via Headless Chrome, cleans out navigation/modals/templates, converts HTML tables into Markdown tables, and extracts primary article or specification content.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "url": [
                    "type": "string",
                    "description": "The absolute HTTP or HTTPS URL to fetch."
                ],
                "query": [
                    "type": "string",
                    "description": "Optional search keywords to filter and prioritize within the page (e.g. 'processor, memory, gpu'). When provided, matching sections and specification tables are prioritized first."
                ],
                "max_length": [
                    "type": "integer",
                    "description": "Optional maximum character length of returned content (defaults to 6,000 characters, up to 20,000)."
                ]
            ]),
            "required": AnyCodable(["url"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let urlStr = arguments["url"] as? String,
              let url = URL(string: urlStr.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.scheme == "http" || url.scheme == "https" else {
            let err = "Invalid URL: please provide an absolute http:// or https:// URL."
            return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
        }

        let queryFilter = (arguments["query"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
        let customMax = arguments["max_length"] as? Int
        let defaultLimit = max(6000, maxOutputLength)
        let limit = customMax.map { min(20000, max(500, $0)) } ?? defaultLimit

        do {
            var html: String? = nil
            // If headless Chrome is available, dump DOM to capture client-side JavaScript rendering
            if HeadlessChromeSearchEngine.shared.resolveBinaryPath() != nil {
                html = try? await HeadlessChromeSearchEngine.shared.dumpDOM(url: url.absoluteString, timeoutSeconds: 15.0)
            }

            if html == nil {
                var request = URLRequest(url: url)
                request.timeoutInterval = 15
                request.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36", forHTTPHeaderField: "User-Agent")
                request.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")

                let (data, response) = try await URLSession.shared.data(for: request)
                guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode),
                      let fetched = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) else {
                    let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                    let err = "Failed to fetch webpage (HTTP status \(status))."
                    return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
                }
                html = fetched
            }

            guard let pageHtml = html else {
                let err = "Failed to retrieve content for URL: \(url.absoluteString)"
                return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
            }

            var pageTitle = url.host ?? "Web Page"
            if let titleRange = pageHtml.range(of: "<title[^>]*>(.*?)</title>", options: [.regularExpression, .caseInsensitive]) {
                let rawTitle = String(pageHtml[titleRange])
                pageTitle = Self.stripHTML(rawTitle)
            }

            let fullCleaned = Self.cleanHTMLStructure(pageHtml)
            var contentToReturn: String

            if let q = queryFilter, !q.isEmpty {
                contentToReturn = Self.filterContentByQuery(fullCleaned, query: q, limit: limit)
            } else if fullCleaned.count > limit {
                contentToReturn = String(fullCleaned.prefix(limit)) + "\n\n... [Content truncated at \(limit) characters]"
            } else {
                contentToReturn = fullCleaned
            }

            let domain = url.host?.replacingOccurrences(of: "www.", with: "") ?? ""
            let isOfficial = AgentHarness.isOfficialVendorDomain(domain)

            var detectedDate: String? = nil
            if let dateRange = pageHtml.range(of: #"(?is)<time[^>]*>(.*?)</time>"#, options: .regularExpression) {
                let rawDate = Self.stripHTML(String(pageHtml[dateRange])).trimmingCharacters(in: .whitespacesAndNewlines)
                if !rawDate.isEmpty && rawDate.count < 50 {
                    detectedDate = rawDate
                }
            }
            if detectedDate == nil, let metaRange = pageHtml.range(of: #"(?is)<meta[^>]+property=[\"']article:published_time[\"'][^>]+content=[\"']([^\"']+)[\"']"#, options: .regularExpression) {
                let snippet = String(pageHtml[metaRange])
                if let contentMatch = snippet.range(of: #"content=[\"']([^\"']+)[\"']"#, options: .regularExpression) {
                    let rawContent = String(snippet[contentMatch])
                        .replacingOccurrences(of: "content=\"", with: "")
                        .replacingOccurrences(of: "content='", with: "")
                        .trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                    if !rawContent.isEmpty {
                        detectedDate = rawContent
                    }
                }
            }
            if detectedDate == nil {
                let prefixText = String(fullCleaned.prefix(1500))
                if let match = prefixText.range(of: #"(?i)(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+20\d{2}"#, options: .regularExpression) {
                    detectedDate = String(prefixText[match])
                }
            }

            var resultData: [String: Any] = [
                "url": url.absoluteString,
                "domain": domain,
                "is_official_domain": isOfficial,
                "title": pageTitle,
                "content_length": contentToReturn.count,
                "content": contentToReturn
            ]
            if let pubDate = detectedDate {
                resultData["published_date"] = pubDate
            }

            let res = AgentHarness.toolSuccessJSON(tool: "web_fetch", data: resultData)
            return (res, res, nil, false)
        } catch {
            let err = "Failed to fetch web content: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
        }
    }

    /// Converts HTML <table> elements into clean GitHub Flavored Markdown tables
    public static func convertTablesToMarkdown(html: String) -> String {
        let tablePattern = #"(?is)<table[^>]*>(.*?)</table>"#
        guard let tableRegex = try? NSRegularExpression(pattern: tablePattern) else { return html }

        let nsHtml = html as NSString
        let matches = tableRegex.matches(in: html, range: NSRange(location: 0, length: nsHtml.length))
        guard !matches.isEmpty else { return html }

        let ms = NSMutableString(string: html)

        for match in matches.reversed() {
            let fullMatchRange = match.range
            let innerTableHtml = nsHtml.substring(with: match.range(at: 1))

            let rowPattern = #"(?is)<tr[^>]*>(.*?)</tr>"#
            guard let rowRegex = try? NSRegularExpression(pattern: rowPattern) else { continue }
            let nsInner = innerTableHtml as NSString
            let rowMatches = rowRegex.matches(in: innerTableHtml, range: NSRange(location: 0, length: nsInner.length))
            guard !rowMatches.isEmpty else { continue }

            var markdownRows: [[String]] = []

            for rowMatch in rowMatches {
                let rowContent = nsInner.substring(with: rowMatch.range(at: 1))
                let cellPattern = #"(?is)<(td|th)[^>]*>(.*?)</\1>"#
                guard let cellRegex = try? NSRegularExpression(pattern: cellPattern) else { continue }
                let nsRow = rowContent as NSString
                let cellMatches = cellRegex.matches(in: rowContent, range: NSRange(location: 0, length: nsRow.length))
                guard !cellMatches.isEmpty else { continue }

                var cells: [String] = []
                for cellMatch in cellMatches {
                    let rawCell = nsRow.substring(with: cellMatch.range(at: 2))
                    let cleanCell = rawCell
                        .replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
                        .replacingOccurrences(of: "\n", with: " ")
                        .replacingOccurrences(of: "|", with: "\\|")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    cells.append(cleanCell.isEmpty ? "-" : cleanCell)
                }
                if !cells.isEmpty {
                    markdownRows.append(cells)
                }
            }

            guard !markdownRows.isEmpty else { continue }

            let maxCols = markdownRows.map { $0.count }.max() ?? 0
            guard maxCols > 0 else { continue }

            var tableMd = "\n\n"
            for (idx, row) in markdownRows.enumerated() {
                var paddedRow = row
                while paddedRow.count < maxCols {
                    paddedRow.append("-")
                }
                let line = "| " + paddedRow.joined(separator: " | ") + " |"
                tableMd += line + "\n"

                if idx == 0 {
                    let separator = "| " + Array(repeating: "---", count: maxCols).joined(separator: " | ") + " |"
                    tableMd += separator + "\n"
                }
            }
            tableMd += "\n"

            ms.replaceCharacters(in: fullMatchRange, with: tableMd)
        }

        return String(ms)
    }

    /// Strips non-content markup, templates, dropdowns, forms, and scripts from page HTML into clean markdown/text
    public static func cleanHTMLStructure(_ html: String) -> String {
        var pageHtml = html

        // Strip comments
        pageHtml = pageHtml.replacingOccurrences(of: "(?s)<!--.*?-->", with: "", options: .regularExpression)

        // Strip non-content / navigation / template tags:
        // script, style, nav, header, footer, svg, noscript, select, option, form, button, template, dialog, aside, iframe
        pageHtml = pageHtml.replacingOccurrences(
            of: "(?is)<(script|style|nav|header|footer|svg|noscript|select|option|form|button|template|dialog|aside|iframe)[^>]*>.*?</\\1>",
            with: "",
            options: .regularExpression
        )

        // Strip standalone/void tags or leftover tags
        pageHtml = pageHtml.replacingOccurrences(of: "(?is)<(input|meta|link|svg|path)[^>]*>", with: "", options: .regularExpression)

        // Strip accessibility / visually-hidden elements (e.g. screen-reader text like "opens in new window")
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<[^>]+class=[\"'][^\"']*(visually-hidden|sr-only|screen-reader-text|a11y-only)[^\"']*[\"'][^>]*>.*?</[^>]+>"#,
            with: "",
            options: .regularExpression
        )

        // Strip hidden elements: aria-hidden="true" or hidden attribute or presentation role
        pageHtml = pageHtml.replacingOccurrences(of: "(?is)<[^>]+aria-hidden=[\"']true[\"'][^>]*>.*?</[^>]+>", with: "", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: #"(?is)<[^>]+role=[\"'](presentation|none)[\"'][^>]*>.*?</[^>]+>"#, with: "", options: .regularExpression)

        // Strip common web scraper noise phrases
        let noisePhrases = [
            #"(?i)opens in (a )?new (window|tab)"#,
            #"(?i)skip to (main )?content"#,
            #"(?i)share this (article|page|story)"#,
            #"(?i)cookie (settings|preferences|notice|policy)"#,
            #"(?i)all rights reserved\."#
        ]
        for phrase in noisePhrases {
            pageHtml = pageHtml.replacingOccurrences(of: phrase, with: "", options: .regularExpression)
        }

        // Strip unhydrated JavaScript template tokens like {MBN_2026_MAIN}, {price.display.smart}, {{model.name}}, etc.
        pageHtml = pageHtml.replacingOccurrences(of: #"\{[A-Za-z0-9_$.]+\}"#, with: "", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: #"\{[A-Z0-9_]+_MAIN[^\}]*\}"#, with: "", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: #"\{\{[^}]+\}\}"#, with: "", options: .regularExpression)

        // Extract primary content body if <main>, <article>, or role="main" exists and is sufficiently rich
        if let mainRange = pageHtml.range(of: "(?is)<(main|article)[^>]*>([\\s\\S]*?)</\\1>", options: .regularExpression) {
            let mainSnippet = String(pageHtml[mainRange])
            if mainSnippet.count > 300 {
                pageHtml = mainSnippet
            }
        } else if let mainDivRange = pageHtml.range(of: #"(?is)<div[^>]+role=[\"']main[\"'][^>]*>([\s\S]*?)</div>"#, options: .regularExpression) {
            let divSnippet = String(pageHtml[mainDivRange])
            if divSnippet.count > 300 {
                pageHtml = divSnippet
            }
        }

        // Convert HTML tables to Markdown tables BEFORE stripping block tags
        pageHtml = convertTablesToMarkdown(html: pageHtml)

        // Convert headings to Markdown headings
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h1[^>]*>([\\s\\S]*?)</h1>", with: "\n\n# $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h2[^>]*>([\\s\\S]*?)</h2>", with: "\n\n## $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h3[^>]*>([\\s\\S]*?)</h3>", with: "\n\n### $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h[4-6][^>]*>([\\s\\S]*?)</h[4-6]>", with: "\n\n#### $1\n\n", options: .regularExpression)

        // Convert list items
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<li[^>]*>([\\s\\S]*?)</li>", with: "\n- $1", options: .regularExpression)

        // Convert bold / strong
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<(strong|b)[^>]*>([\\s\\S]*?)</\\1>", with: "**$2**", options: .regularExpression)

        // Replace other block tags with newlines
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)</?(p|div|section|blockquote|pre|code)[^>]*>", with: "\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<br\\s*/?>", with: "\n", options: .regularExpression)

        // Strip remaining HTML tags
        let text = stripHTML(pageHtml)

        // Clean up excessive whitespace, blank lines, and orphan markdown tokens
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        var resultLines: [String] = []
        var consecutiveBlanks = 0
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            // Strip orphan markdown headers (e.g. line is just "#", "##", "###", "####")
            let withoutHash = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "# \t"))
            if withoutHash.isEmpty && trimmed.hasPrefix("#") {
                continue
            }
            // Strip orphan bullet points (e.g. line is just "-", "*", "+", "•")
            let withoutBullet = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "-*+• \t"))
            if withoutBullet.isEmpty && (trimmed.hasPrefix("-") || trimmed.hasPrefix("*") || trimmed.hasPrefix("+") || trimmed.hasPrefix("•")) {
                continue
            }

            if trimmed.isEmpty {
                consecutiveBlanks += 1
                if consecutiveBlanks <= 1 {
                    resultLines.append("")
                }
            } else {
                consecutiveBlanks = 0
                resultLines.append(trimmed)
            }
        }

        return resultLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Prioritizes content sections matching query keywords
    public static func filterContentByQuery(_ content: String, query: String, limit: Int) -> String {
        let cleanQuery = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanQuery.isEmpty else {
            return String(content.prefix(limit))
        }

        let terms = cleanQuery.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 2 }
        guard !terms.isEmpty else {
            return String(content.prefix(limit))
        }

        // Split content into blocks by double newlines or markdown headings
        let rawBlocks = content.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var scoredBlocks: [(block: String, score: Int)] = []
        for block in rawBlocks {
            let lower = block.lowercased()
            var score = 0
            for term in terms {
                if lower.contains(term) {
                    score += 10
                    // Bonus if it's in a markdown table or heading
                    if block.contains("|") || block.hasPrefix("#") {
                        score += 5
                    }
                }
            }
            scoredBlocks.append((block, score))
        }

        let matching = scoredBlocks.filter { $0.score > 0 }.sorted { $0.score > $1.score }
        let nonMatching = scoredBlocks.filter { $0.score == 0 }

        if matching.isEmpty {
            return String(content.prefix(limit))
        }

        var prioritized: [String] = []
        prioritized.append("### Key Sections Matching \"\(cleanQuery)\":\n")
        var currentLength = prioritized[0].count

        for item in matching {
            if currentLength + item.block.count + 2 > limit {
                break
            }
            prioritized.append(item.block)
            currentLength += item.block.count + 2
        }

        // Fill remaining budget with other context
        if currentLength < limit && !nonMatching.isEmpty {
            prioritized.append("\n### Additional Page Context:\n")
            currentLength += prioritized.last!.count
            for item in nonMatching {
                if currentLength + item.block.count + 2 > limit {
                    break
                }
                prioritized.append(item.block)
                currentLength += item.block.count + 2
            }
        }

        return prioritized.joined(separator: "\n\n")
    }

    public static func stripHTML(_ input: String) -> String {
        let withoutTags = input.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
        return withoutTags
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&#x27;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// Tool 9: complete — Signals task completion with summary
public final class CompleteTool: AgentTool {
    public let definition = ToolDefinition(
        name: "complete",
        description: "Signals that the user's task or requested instructions are fully completed.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "summary": [
                    "type": "string",
                    "description": "Comprehensive summary of actions taken and final results."
                ]
            ]),
            "required": AnyCodable(["summary"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let summary = (arguments["summary"] as? String) ?? "Task completed."
        let sanitized = AgentHarness.sanitizeText(summary)
        let res = AgentHarness.toolSuccessJSON(tool: "complete", data: ["summary": sanitized])
        return (res, sanitized, nil, true)
    }
}

/// Tool 10: codebase_search — Hybrid Vector & Lexical search across codebase
public final class CodebaseSearchTool: AgentTool {
    public let definition = ToolDefinition(
        name: "codebase_search",
        description: "Semantically and lexically searches the indexed codebase using Metal-accelerated hybrid vector search (Dense Vector + BM25). Retrieves exact code snippets and AST chunks with file paths and line numbers without filling the context window.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "query": [
                    "type": "string",
                    "description": "The search query describing the functionality, concept, error, or identifier to find (e.g. 'where is KV cache allocated?', 'runTokenForward implementation')."
                ],
                "target_directory": [
                    "type": "string",
                    "description": "Optional subdirectory relative to the workspace to restrict the search."
                ],
                "file_extensions": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Optional list of file extensions to filter by (e.g. ['swift', 'metal'])."
                ],
                "top_k": [
                    "type": "integer",
                    "description": "Number of snippets to retrieve (default 5, max 15)."
                ],
                "search_mode": [
                    "type": "string",
                    "enum": ["hybrid", "semantic", "keyword"],
                    "description": "Search mode to use: 'hybrid' (combines BM25 and vector embeddings), 'semantic' (pure Metal vector search), or 'keyword' (BM25 keyword search)."
                ]
            ]),
            "required": AnyCodable(["query"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let query = arguments["query"] as? String, !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "Missing or empty 'query' parameter."
            return (AgentHarness.toolErrorJSON(tool: "codebase_search", error: err), nil, err, false)
        }

        let targetDir = arguments["target_directory"] as? String
        let fileExts = arguments["file_extensions"] as? [String]
        let topK = min(15, max(1, (arguments["top_k"] as? Int) ?? 5))
        let modeStr = (arguments["search_mode"] as? String)?.lowercased() ?? "hybrid"
        let mode: SearchMode = modeStr == "semantic" ? .semantic : (modeStr == "keyword" ? .keyword : .hybrid)

        let targetWorkspace: URL
        if let wd = workingDirectory {
            targetWorkspace = wd
        } else {
            targetWorkspace = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        let searchResults = await CodebaseIndexer.shared.search(
            query: query,
            workspace: targetWorkspace,
            targetDirectory: targetDir,
            fileExtensions: fileExts,
            topK: topK,
            mode: mode
        )
        let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

        if searchResults.isEmpty {
            let msg = "No codebase matches found for \"\(query)\"."
            let res = AgentHarness.toolSuccessJSON(tool: "codebase_search", data: [
                "query": query,
                "count": 0,
                "results": [],
                "duration_ms": elapsedMs
            ])
            return (res, msg, nil, false)
        }

        var readableOutput = "Found \(searchResults.count) codebase result(s) for \"\(query)\" (\(String(format: "%.1f", elapsedMs)) ms, mode: \(mode.rawValue)):\n\n"
        var jsonResults: [[String: Any]] = []

        for (idx, r) in searchResults.enumerated() {
            let snippet = AgentHarness.truncateText(r.chunk.content, limit: 1000)
            readableOutput += "### [\(idx + 1)] \(r.chunk.filePath):\(r.chunk.startLine)-\(r.chunk.endLine) (score: \(String(format: "%.3f", r.score)))\n"
            readableOutput += "```\((r.chunk.filePath as NSString).pathExtension)\n"
            readableOutput += snippet + "\n"
            readableOutput += "```\n\n"

            jsonResults.append([
                "file": r.chunk.filePath,
                "lines": "L\(r.chunk.startLine)-L\(r.chunk.endLine)",
                "title": r.chunk.title,
                "score": r.score,
                "snippet": snippet
            ])
        }

        let cleanStdout = AgentHarness.truncateText(readableOutput.trimmingCharacters(in: .whitespacesAndNewlines), limit: maxOutputLength)
        let res = AgentHarness.toolSuccessJSON(tool: "codebase_search", data: [
            "query": query,
            "count": searchResults.count,
            "results": jsonResults,
            "duration_ms": elapsedMs
        ])
        return (res, cleanStdout, nil, false)
    }
}

// MARK: - Subagent Delegation Tools

public final class SpawnSubagentTool: AgentTool {
    public let definition = ToolDefinition(
        name: "spawn_subagent",
        description: "Spawns an isolated background subagent with a dedicated role (e.g. Codebase Researcher, Test Runner, Shader Optimizer) to handle focused multi-step tasks without filling your primary context window.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "role": [
                    "type": "string",
                    "description": "Role title of the subagent (e.g. 'Codebase Researcher', 'Test Runner', 'Shader Optimizer', 'Documentation Writer')"
                ],
                "task_description": [
                    "type": "string",
                    "description": "Detailed, actionable instructions for what the subagent must execute and analyze."
                ],
                "allowed_tools": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "List of tools the subagent is permitted to call (optional). Defaults to safe read & research tools."
                ],
                "context_summary": [
                    "type": "string",
                    "description": "Optional context, constraints, or code references to seed the subagent's isolated context."
                ],
                "run_in_background": [
                    "type": "boolean",
                    "description": "Set true to execute asynchronously in the background task manager; set false (default) to wait for the subagent to complete and return its synthesized report."
                ]
            ]),
            "required": AnyCodable(["role", "task_description"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let role = arguments["role"] as? String, !role.isEmpty else {
            let err = "Missing or empty required argument: 'role'"
            return (AgentHarness.toolErrorJSON(tool: "spawn_subagent", error: err), nil, err, false)
        }
        guard let taskDesc = arguments["task_description"] as? String, !taskDesc.isEmpty else {
            let err = "Missing or empty required argument: 'task_description'"
            return (AgentHarness.toolErrorJSON(tool: "spawn_subagent", error: err), nil, err, false)
        }

        var allowedTools: [String] = []
        if let toolsArray = arguments["allowed_tools"] as? [String] {
            allowedTools = toolsArray
        }
        let contextSummary = arguments["context_summary"] as? String
        let runInBackground = (arguments["run_in_background"] as? Bool) ?? false

        let subagent = SubagentManager.shared.spawn(
            role: role,
            taskDescription: taskDesc,
            allowedTools: allowedTools,
            contextSummary: contextSummary,
            workingDirectory: workingDirectory
        )

        if runInBackground {
            let msg = "Subagent [\(role)] spawned in background (ID: \(subagent.id.uuidString)). Monitor in Task Manager drawer or via get_subagent_status."
            let res = AgentHarness.toolSuccessJSON(tool: "spawn_subagent", data: [
                "subagent_id": subagent.id.uuidString,
                "role": subagent.role,
                "status": "running",
                "run_in_background": true,
                "message": msg
            ])
            return (res, msg, nil, false)
        } else {
            // Synchronous delegation: wait for subagent to finish
            let summary = await subagent.waitForCompletion()
            let cleanSummary = AgentHarness.truncateText(AgentHarness.sanitizeText(summary), limit: maxOutputLength)
            let res = AgentHarness.toolSuccessJSON(tool: "spawn_subagent", data: [
                "subagent_id": subagent.id.uuidString,
                "role": subagent.role,
                "status": subagent.status.rawValue,
                "duration_seconds": subagent.executionDurationSeconds,
                "summary": cleanSummary
            ])
            return (res, cleanSummary, nil, false)
        }
    }
}

public final class GetSubagentStatusTool: AgentTool {
    public let definition = ToolDefinition(
        name: "get_subagent_status",
        description: "Checks the live execution status, step transcript, and final summary of a spawned subagent by its ID.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "subagent_id": [
                    "type": "string",
                    "description": "The UUID of the spawned subagent."
                ]
            ]),
            "required": AnyCodable(["subagent_id"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let idStr = arguments["subagent_id"] as? String, let uuid = UUID(uuidString: idStr) else {
            let err = "Invalid or missing 'subagent_id' UUID."
            return (AgentHarness.toolErrorJSON(tool: "get_subagent_status", error: err), nil, err, false)
        }

        guard let subagent = SubagentManager.shared.getSubagent(byId: uuid) else {
            let err = "Subagent with ID '\(idStr)' not found."
            return (AgentHarness.toolErrorJSON(tool: "get_subagent_status", error: err), nil, err, false)
        }

        let stepsSummary = subagent.transcript.map {
            "Step \($0.stepIndex): \($0.actionName) (\(String(format: "%.2f", $0.durationSeconds))s)"
        }.joined(separator: "\n")

        var readable = "Subagent: \(subagent.role) [\(subagent.status.displayName)]\n"
        readable += "Status: \(subagent.liveStatusText)\n"
        readable += "Steps Completed: \(subagent.transcript.count)\n"
        if !subagent.finalSummary.isEmpty {
            readable += "\nSummary:\n\(subagent.finalSummary)"
        }

        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(readable), limit: maxOutputLength)
        let res = AgentHarness.toolSuccessJSON(tool: "get_subagent_status", data: [
            "subagent_id": subagent.id.uuidString,
            "role": subagent.role,
            "status": subagent.status.rawValue,
            "live_status": subagent.liveStatusText,
            "steps_count": subagent.transcript.count,
            "steps_summary": stepsSummary,
            "summary": subagent.finalSummary,
            "duration_seconds": subagent.executionDurationSeconds
        ])
        return (res, cleanOut, nil, false)
    }
}

public final class SendSubagentMessageTool: AgentTool {
    public let definition = ToolDefinition(
        name: "send_subagent_message",
        description: "Sends a follow-up directive, instruction, or clarification to a running or completed subagent.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "subagent_id": [
                    "type": "string",
                    "description": "The UUID of the spawned subagent."
                ],
                "message": [
                    "type": "string",
                    "description": "The directive or message content to deliver to the subagent."
                ]
            ]),
            "required": AnyCodable(["subagent_id", "message"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let idStr = arguments["subagent_id"] as? String, let uuid = UUID(uuidString: idStr) else {
            let err = "Invalid or missing 'subagent_id' UUID."
            return (AgentHarness.toolErrorJSON(tool: "send_subagent_message", error: err), nil, err, false)
        }
        guard let msg = arguments["message"] as? String, !msg.isEmpty else {
            let err = "Missing or empty 'message'."
            return (AgentHarness.toolErrorJSON(tool: "send_subagent_message", error: err), nil, err, false)
        }

        guard let subagent = SubagentManager.shared.getSubagent(byId: uuid) else {
            let err = "Subagent with ID '\(idStr)' not found."
            return (AgentHarness.toolErrorJSON(tool: "send_subagent_message", error: err), nil, err, false)
        }

        SubagentManager.shared.sendMessage(toSubagentId: uuid, sender: "coordinator", content: msg)
        let readable = "Message delivered to subagent [\(subagent.role)] (ID: \(idStr))."
        let res = AgentHarness.toolSuccessJSON(tool: "send_subagent_message", data: [
            "subagent_id": idStr,
            "role": subagent.role,
            "delivered": true
        ])
        return (res, readable, nil, false)
    }
}

public final class ListSubagentsTool: AgentTool {
    public let definition = ToolDefinition(
        name: "list_subagents",
        description: "Lists all active and completed child subagents, their roles, IDs, statuses, and execution durations.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "status_filter": [
                    "type": "string",
                    "enum": ["all", "running", "completed", "failed"],
                    "description": "Filter subagents: 'all' (default), 'running', 'completed', or 'failed'"
                ]
            ]),
            "required": AnyCodable([])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let filter = (arguments["status_filter"] as? String)?.lowercased() ?? "all"
        let allSubagents = SubagentManager.shared.allSubagents

        let filtered = allSubagents.filter { s in
            if filter == "all" { return true }
            if filter == "running" { return s.status == .running || s.status == .pending }
            if filter == "completed" { return s.status == .completed }
            if filter == "failed" { return s.status == .failed || s.status == .cancelled }
            return true
        }

        var jsonList: [[String: Any]] = []
        var readableLines: [String] = []

        for s in filtered {
            jsonList.append([
                "id": s.id.uuidString,
                "role": s.role,
                "status": s.status.rawValue,
                "duration_seconds": s.executionDurationSeconds,
                "steps_count": s.transcript.count
            ])
            readableLines.append("- [\(s.role)] ID: \(s.id.uuidString) | Status: \(s.status.displayName) | Steps: \(s.transcript.count) | Duration: \(String(format: "%.2f", s.executionDurationSeconds))s")
        }

        let readable = readableLines.isEmpty ? "No subagents matching filter '\(filter)'." : readableLines.joined(separator: "\n")
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(readable), limit: maxOutputLength)
        let res = AgentHarness.toolSuccessJSON(tool: "list_subagents", data: [
            "count": filtered.count,
            "filter": filter,
            "subagents": jsonList
        ])
        return (res, cleanOut, nil, false)
    }
}

// MARK: - Option 3: Deep Developer Tooling Tools

/// Tool 15: git_status — Inspects repository working tree status, staged/unstaged changes, and branch tracking
public final class GitStatusTool: AgentTool {
    public let definition = ToolDefinition(
        name: "git_status",
        description: "Checks git repository status. Shows current branch, ahead/behind tracking, staged files, unstaged modifications, and untracked files.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Optional directory inside repository. Defaults to current workspace."
                ]
            ])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let targetDir: URL
        if let customPath = arguments["path"] as? String, !customPath.isEmpty {
            targetDir = AgentHarness.resolvePath(customPath, workingDirectory: workingDirectory)
        } else if let wd = workingDirectory {
            targetDir = wd
        } else {
            targetDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        }

        do {
            let res = try await GitController.shared.status(workingDirectory: targetDir)
            let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(res.summary), limit: maxOutputLength)
            let json = AgentHarness.toolSuccessJSON(tool: "git_status", data: [
                "branch": res.branch,
                "upstream": res.upstream as Any,
                "ahead": res.aheadCount,
                "behind": res.behindCount,
                "staged_count": res.stagedFiles.count,
                "unstaged_count": res.unstagedFiles.count,
                "untracked_count": res.untrackedFiles.count,
                "has_changes": res.hasChanges,
                "staged_files": res.stagedFiles.map { ["path": $0.path, "status": $0.statusCode] },
                "unstaged_files": res.unstagedFiles.map { ["path": $0.path, "status": $0.statusCode] },
                "untracked_files": res.untrackedFiles.map { ["path": $0.path] }
            ])
            return (json, cleanOut, nil, false)
        } catch {
            let err = error.localizedDescription
            return (AgentHarness.toolErrorJSON(tool: "git_status", error: err), nil, err, false)
        }
    }
}

/// Tool 16: git_diff — Formats repository diffs for working tree, staged changes, or target branches
public final class GitDiffTool: AgentTool {
    public let definition = ToolDefinition(
        name: "git_diff",
        description: "Inspects file differences (git diff). Supports inspecting unstaged changes, staged changes, specific file paths, or diffs against a commit/branch.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Optional specific file or folder path to inspect."
                ],
                "staged": [
                    "type": "boolean",
                    "description": "If true, shows diff for staged changes (--staged). Default false."
                ],
                "target": [
                    "type": "string",
                    "description": "Optional commit hash or branch name to compare against (e.g. 'main', 'HEAD~1')."
                ]
            ])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let baseDir = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let path = arguments["path"] as? String
        let staged = (arguments["staged"] as? Bool) ?? false
        let target = arguments["target"] as? String

        do {
            let res = try await GitController.shared.diff(
                workingDirectory: baseDir,
                path: path,
                staged: staged,
                target: target,
                maxLines: 500
            )
            let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(res.diff), limit: maxOutputLength)
            let json = AgentHarness.toolSuccessJSON(tool: "git_diff", data: [
                "target_path": res.targetPath as Any,
                "is_staged": res.isStaged,
                "line_count": res.lineCount,
                "is_truncated": res.isTruncated,
                "summary": res.summary
            ])
            return (json, cleanOut, nil, false)
        } catch {
            let err = error.localizedDescription
            return (AgentHarness.toolErrorJSON(tool: "git_diff", error: err), nil, err, false)
        }
    }
}

/// Tool 17: git_commit — Stages files and creates a git commit with safety rails
public final class GitCommitTool: AgentTool {
    public let definition = ToolDefinition(
        name: "git_commit",
        description: "Creates a git commit with safety rails. Stages designated files or all tracked modifications and records commit with message.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "message": [
                    "type": "string",
                    "description": "Clear commit message describing the change."
                ],
                "stage_all": [
                    "type": "boolean",
                    "description": "If true, stages all changes (git add -A) before committing. Default false."
                ],
                "paths": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Optional list of specific file paths to stage before committing."
                ]
            ]),
            "required": AnyCodable(["message"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let message = arguments["message"] as? String, !message.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "Error: missing or empty 'message' parameter in git_commit"
            return (AgentHarness.toolErrorJSON(tool: "git_commit", error: err), nil, err, false)
        }

        let stageAll = (arguments["stage_all"] as? Bool) ?? false
        let paths = arguments["paths"] as? [String]
        let baseDir = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        do {
            let res = try await GitController.shared.commit(
                workingDirectory: baseDir,
                message: message,
                stageAll: stageAll,
                paths: paths
            )
            let json = AgentHarness.toolSuccessJSON(tool: "git_commit", data: [
                "commit_hash": res.commitHash,
                "branch": res.branch,
                "message": res.message,
                "status": "committed"
            ])
            return (json, res.summary, nil, false)
        } catch {
            let err = error.localizedDescription
            return (AgentHarness.toolErrorJSON(tool: "git_commit", error: err), nil, err, false)
        }
    }
}

/// Tool 18: find_symbol_definition — Locates declaration/definition of a symbol
public final class FindSymbolDefinitionTool: AgentTool {
    public let definition = ToolDefinition(
        name: "find_symbol_definition",
        description: "Locates where a symbol (class, struct, enum, protocol, func, kernel, typealias) is defined across the codebase with exact file and line coordinates.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "symbol_name": [
                    "type": "string",
                    "description": "Exact identifier or function name to locate definition for."
                ],
                "path": [
                    "type": "string",
                    "description": "Optional subdirectory or file to search within."
                ],
                "max_results": [
                    "type": "integer",
                    "description": "Maximum number of definition matches to return. Default 10."
                ]
            ]),
            "required": AnyCodable(["symbol_name"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let symbolName = arguments["symbol_name"] as? String, !symbolName.isEmpty else {
            let err = "Error: missing 'symbol_name' parameter in find_symbol_definition"
            return (AgentHarness.toolErrorJSON(tool: "find_symbol_definition", error: err), nil, err, false)
        }

        let baseDir = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let filterPath = arguments["path"] as? String
        let maxResults = (arguments["max_results"] as? Int) ?? 10

        let defs = await SymbolIntelligenceEngine.shared.findDefinition(
            symbolName: symbolName,
            inDirectory: baseDir,
            filterPath: filterPath,
            maxResults: maxResults
        )

        var lines: [String] = []
        for d in defs {
            let shortPath = d.filePath.replacingOccurrences(of: baseDir.path + "/", with: "")
            lines.append("• [\(d.kind)] \(d.name) at \(shortPath):\(d.line):\(d.column)")
            lines.append("  Signature: \(d.signature)")
            lines.append("  ```\n\(d.snippet)\n  ```")
        }

        let readable = lines.isEmpty ? "No definitions found for symbol '\(symbolName)'." : lines.joined(separator: "\n")
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(readable), limit: maxOutputLength)

        let json = AgentHarness.toolSuccessJSON(tool: "find_symbol_definition", data: [
            "symbol": symbolName,
            "count": defs.count,
            "definitions": defs.map { [
                "name": $0.name,
                "kind": $0.kind,
                "file": $0.filePath,
                "line": $0.line,
                "column": $0.column,
                "signature": $0.signature
            ] }
        ])
        return (json, cleanOut, nil, false)
    }
}

/// Tool 19: find_references — Locates all reference/call sites of a symbol across the project
public final class FindReferencesTool: AgentTool {
    public let definition = ToolDefinition(
        name: "find_references",
        description: "Finds all usages, references, and call sites of a symbol across the project.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "symbol_name": [
                    "type": "string",
                    "description": "Symbol name or function to find references for."
                ],
                "path": [
                    "type": "string",
                    "description": "Optional subdirectory or file to search within."
                ],
                "max_results": [
                    "type": "integer",
                    "description": "Maximum number of references to return. Default 30."
                ]
            ]),
            "required": AnyCodable(["symbol_name"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let symbolName = arguments["symbol_name"] as? String, !symbolName.isEmpty else {
            let err = "Error: missing 'symbol_name' parameter in find_references"
            return (AgentHarness.toolErrorJSON(tool: "find_references", error: err), nil, err, false)
        }

        let baseDir = workingDirectory ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let filterPath = arguments["path"] as? String
        let maxResults = (arguments["max_results"] as? Int) ?? 30

        let refs = await SymbolIntelligenceEngine.shared.findReferences(
            symbolName: symbolName,
            inDirectory: baseDir,
            filterPath: filterPath,
            maxResults: maxResults
        )

        var lines: [String] = []
        for r in refs {
            let shortPath = r.filePath.replacingOccurrences(of: baseDir.path + "/", with: "")
            let callMarker = r.isCallSite ? " [call site]" : ""
            lines.append("• \(shortPath):\(r.line):\(r.column)\(callMarker): \(r.lineContent)")
        }

        let readable = lines.isEmpty ? "No references found for symbol '\(symbolName)'." : lines.joined(separator: "\n")
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(readable), limit: maxOutputLength)

        let json = AgentHarness.toolSuccessJSON(tool: "find_references", data: [
            "symbol": symbolName,
            "count": refs.count,
            "references": refs.map { [
                "file": $0.filePath,
                "line": $0.line,
                "column": $0.column,
                "content": $0.lineContent,
                "is_call_site": $0.isCallSite
            ] }
        ])
        return (json, cleanOut, nil, false)
    }
}

/// Tool 20: lint_diagnostics — Checks source files for syntax and compiler diagnostics
public final class LintDiagnosticsTool: AgentTool {
    public let definition = ToolDefinition(
        name: "lint_diagnostics",
        description: "Runs native compiler syntax diagnostics on a source file (.swift, .metal, .c, .cpp) to verify compilation and report errors.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "path": [
                    "type": "string",
                    "description": "Path to the source file to check."
                ]
            ]),
            "required": AnyCodable(["path"])
        ]
    )

    public func execute(
        arguments: [String: Any],
        workingDirectory: URL?,
        maxOutputLength: Int
    ) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let rawPath = arguments["path"] as? String, !rawPath.isEmpty else {
            let err = "Error: missing 'path' parameter in lint_diagnostics"
            return (AgentHarness.toolErrorJSON(tool: "lint_diagnostics", error: err), nil, err, false)
        }

        let resolvedPath = AgentHarness.resolvePath(rawPath, workingDirectory: workingDirectory)
        guard FileManager.default.fileExists(atPath: resolvedPath.path) else {
            let err = "File not found at path: \(resolvedPath.path)"
            return (AgentHarness.toolErrorJSON(tool: "lint_diagnostics", error: err), nil, err, false)
        }

        let report = await LintDiagnosticsEngine.checkFile(at: resolvedPath, workingDirectory: workingDirectory)
        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(report.readableSummary), limit: maxOutputLength)

        let json = AgentHarness.toolSuccessJSON(tool: "lint_diagnostics", data: [
            "path": report.filePath,
            "has_errors": report.hasErrors,
            "has_warnings": report.hasWarnings,
            "diagnostics_count": report.diagnostics.count,
            "diagnostics": report.diagnostics.map { [
                "line": $0.line,
                "column": $0.column,
                "severity": $0.severity,
                "message": $0.message
            ] },
            "self_healing_hint": report.selfHealingPrompt ?? ""
        ])
        return (json, cleanOut, nil, false)
    }
}

// MARK: - Agent Harness Coordinator

public final class AgentHarness {
    public static let shared = AgentHarness()

    public private(set) var tools: [String: AgentTool] = [:]
    public var defaultWorkingDirectory: URL? = nil
    public var maxToolOutputLength: Int = 4000
    public var maxAgentSteps: Int = 15

    private init() {
        registerDefaultTools()
    }

    public func registerDefaultTools() {
        registerTool(ShellRunTool())
        registerTool(FileReadTool())
        registerTool(FileWriteTool())
        registerTool(FileEditTool())
        registerTool(FindFilesTool())
        registerTool(GrepSearchTool())
        registerTool(WebSearchTool())
        registerTool(WebFetchTool())
        registerTool(CodebaseSearchTool())
        registerTool(SpawnSubagentTool())
        registerTool(GetSubagentStatusTool())
        registerTool(SendSubagentMessageTool())
        registerTool(ListSubagentsTool())
        registerTool(GitStatusTool())
        registerTool(GitDiffTool())
        registerTool(GitCommitTool())
        registerTool(FindSymbolDefinitionTool())
        registerTool(FindReferencesTool())
        registerTool(LintDiagnosticsTool())
        registerTool(CompleteTool())
        GrammarConstrainedSampler.shared.registerTools(availableToolDefinitions)
    }

    public func registerTool(_ tool: AgentTool) {
        tools[tool.definition.function.name] = tool
        GrammarConstrainedSampler.shared.registerTools(availableToolDefinitions)
    }

    public static func isStateChanging(toolName: String) -> Bool {
        return toolName == "file_write" || toolName == "file_edit" || toolName == "shell_run" || toolName == "git_commit"
    }

    public var availableToolDefinitions: [ToolDefinition] {
        return Array(tools.values.map { $0.definition })
    }

    // MARK: - Prompt Formatting & ChatML Generation

    /// Formats the current date, time, and timezone context for temporal awareness in agent prompts.
    public static func formattedDateTimeContext(date: Date = Date(), timeZone: TimeZone = .current) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.locale = Locale(identifier: "en_US")
        dateFormatter.timeZone = timeZone
        dateFormatter.dateFormat = "EEEE, MMMM d, yyyy 'at' h:mm a zzz"
        let readableDateTime = dateFormatter.string(from: date)

        let isoFormatter = ISO8601DateFormatter()
        isoFormatter.timeZone = timeZone
        let isoDateTime = isoFormatter.string(from: date)

        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let year = calendar.component(.year, from: date)

        let tzName = timeZone.identifier
        let tzAbbr = timeZone.abbreviation(for: date) ?? ""

        return """
        # Current Date & Time
        - Reference Date & Time: \(readableDateTime) (\(isoDateTime))
        - Current Year: \(year)
        - Timezone: \(tzName)\(tzAbbr.isEmpty ? "" : " (\(tzAbbr))")
        - Temporal Context: Today's reference date is \(readableDateTime). When conducting web research (`web_search`), fetching web pages (`web_fetch`), synthesizing recent developments, or reasoning about relative dates ("today", "yesterday", "recent", "this year"), always use \(year) and this reference timestamp.
        """
    }

    public func buildSystemPrompt(baseSystem: String, modelName: String? = nil, currentDate: Date = Date()) -> String {
        var cleanBase = baseSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleanBase.isEmpty {
            cleanBase = "You are an expert AI software engineering and reasoning assistant with direct access to local macOS development tools."
        }

        var prompt = ""

        // Inject Current Date & Time at the top if not already provided in baseSystem
        if !cleanBase.contains("Current Date & Time") {
            prompt += AgentHarness.formattedDateTimeContext(date: currentDate) + "\n\n"
        }

        prompt += "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
        for tool in availableToolDefinitions {
            if let data = try? JSONEncoder().encode(tool),
               let jsonStr = String(data: data, encoding: .utf8) {
                prompt += jsonStr + "\n"
            }
        }
        prompt += "</tools>\n\n"
        prompt += """
        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - When the user asks to inspect, read, edit, modify, or process a file, or run terminal commands, you MUST call the function immediately without conversational promises
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls

        Web Research & Grounding Guidelines:
        - Structured Tool Responses: Tools return structured JSON objects containing metadata (such as `domain`, `is_official_domain`, `published_date`, `title`) and `content`. Use these keys to evaluate source authority.
        - Temporal Grounding & Live Reality: Your pre-training weights have a fixed past knowledge cutoff. You are operating live in real-time in the current year. Hardware chips (e.g. Apple Silicon M4, M5 Max, M5 Ultra), memory capacities (e.g. 512GB unified memory), operating system versions, and corporate executives (e.g. Apple's Johny Srouji, John Ternus, etc.) will frequently be newer than your pre-training weights.
        - Authoritative Official Domain Reality: When content is retrieved from official vendor domains (e.g. apple.com, developer.apple.com, github.com) via `web_fetch` or `web_search`, TREAT IT AS GROUND-TRUTH FACTUAL REALITY.
        - Never Reject Live Data: NEVER dismiss, ignore, or second-guess official web results as "hallucinated", "parody", "machine-generated", or "corrupted" simply because chip numbers, specifications, release dates, or executive names are newer than your internal knowledge base. Live web data always supersedes frozen pre-training memory.
        - Press Releases & Marketing Phrasing: Vendor announcements often repeat key performance metrics across multiple sections (e.g. "up to 3.9x faster AI performance" in summaries, quotes, and comparison lists) and use idiomatic expressions (e.g. "rounds out connectivity"). Do not mistake marketing repetition or real executive quotes for machine errors.
        - Neutral Queries First: When searching for current products, technical specs, or news, formulate objective, neutral queries (e.g. "Apple Mac mini official current specifications", NOT "Mac mini M5 Max 2025"). Never embed unverified chip numbers or future years into your initial queries.
        - Strict URL Grounding: ONLY fetch URLs returned by `web_search`. NEVER invent, guess, or synthesize article numbers or support URLs (e.g., support.apple.com/en-us/104942), as they will lead to 404s or unrelated topics.
        - Official vs. Speculative Rumors: Distinguish between official shipping hardware (on vendor domains like apple.com, official documentation, or verified reviews) versus speculative rumors ("rumored", "expected to", "leaks", "concept").
        - Targeted In-Page Queries: Use `web_fetch(url: "...", query: "...")` with relevant keywords (e.g. `query: "processor, gpu, memory"`) to directly retrieve the exact specification tables or sections on dense pages.
        </IMPORTANT>
        """

        if !cleanBase.isEmpty {
            prompt += "\n\n" + cleanBase
        }

        return prompt
    }

    public func formatToolResponseTurn(responses: [String], includeThinkSuffix: Bool = false) -> String {
        var turn = "<|im_start|>user\n"
        for r in responses {
            turn += "<tool_response>\n\(r)\n</tool_response>\n"
        }
        turn += "<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkSuffix {
            turn += "<think>\n"
        }
        return turn
    }

    public func detectUncalledActionIntent(content: String, thinking: String?) -> Bool {
        let contentLower = content.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        let thinkingLower = (thinking ?? "").lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        if contentLower.count > 400 {
            return false
        }

        let actionPatterns = [
            "take a look at",
            "look at the",
            "look at this",
            "look into",
            "read the",
            "reading the",
            "start by reading",
            "read this",
            "inspect the",
            "inspecting the",
            "check the",
            "checking the",
            "examine the",
            "add a column",
            "add the column",
            "add column",
            "edit the",
            "modifying the",
            "modify the",
            "update the",
            "change the",
            "search for",
            "search the web",
            "look up",
            "run the",
            "execute the",
            "create the",
            "write to",
            "write the",
            "i'll start by",
            "let me start by",
            "i will start by",
            "let me first",
            "first, i will",
            "first, let me",
            "first i'll",
            "to begin, i will",
            "to begin, let me"
        ]

        for p in actionPatterns {
            if contentLower.contains(p) || thinkingLower.contains(p) {
                return true
            }
        }

        return false
    }

    public func formatActionContinuationTurn(includeThinkSuffix: Bool = false) -> String {
        var turn = "<|im_start|>user\n"
        turn += "Please call the function now using <tool_call><function=...><parameter=...>...</parameter></function></tool_call> to execute your action.\n"
        turn += "<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkSuffix {
            turn += "<think>\n"
        }
        return turn
    }

    // MARK: - Tolerant Tool Call Parser (XML + JSON + Multi-Call + Truncation Recovery)

    public func parseToolCalls(from text: String) -> (calls: [ParsedToolCall], brokenFragments: [String]) {
        var calls: [ParsedToolCall] = []
        var broken: [String] = []

        // 1. Native XML function calls (<function=name>...<parameter=k>v</parameter>...)
        let xmlCalls = parseAllXMLFunctionCalls(text)
        if !xmlCalls.isEmpty {
            return (xmlCalls, [])
        }

        // 2. Standard <tool_call>...</tool_call> tags (supporting both XML and JSON payloads)
        let toolCallRegex = try? NSRegularExpression(pattern: "<tool_call>([\\s\\S]*?)</tool_call>", options: [])
        let nsText = text as NSString
        let matches = toolCallRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []

        for m in matches {
            guard m.numberOfRanges >= 2 else { continue }
            let innerRange = m.range(at: 1)
            let rawMatch = nsText.substring(with: m.range(at: 0))
            let innerText = nsText.substring(with: innerRange).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !innerText.isEmpty else { continue }

            let innerXml = parseAllXMLFunctionCalls(innerText)
            if !innerXml.isEmpty {
                calls.append(contentsOf: innerXml)
            } else if let parsed = extractBalancedJSON(innerText) {
                if let name = parsed["name"] as? String {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: rawMatch))
                } else if let name = parsed["tool"] as? String {
                    let args = (parsed["parameters"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["parameters"] != nil) ? String(describing: parsed["parameters"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: rawMatch))
                } else {
                    broken.append(innerText)
                }
            } else {
                broken.append(innerText)
            }
        }

        // 3. Check for unclosed or truncated <tool_call>
        if calls.isEmpty && text.contains("<tool_call>") {
            let parts = text.components(separatedBy: "<tool_call>")
            for part in parts.dropFirst() {
                let candidate = part.replacingOccurrences(of: "</tool_call>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                let candXml = parseAllXMLFunctionCalls(candidate)
                if !candXml.isEmpty {
                    calls.append(contentsOf: candXml)
                } else if let parsed = extractBalancedJSON(candidate), let name = parsed["name"] as? String {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: candidate))
                } else if !candidate.isEmpty {
                    broken.append(candidate)
                }
            }
        }

        // 4. Fallback: Check if output contains raw JSON with a recognized tool name
        if calls.isEmpty {
            let knownToolNames = Set(availableToolDefinitions.map { $0.function.name })
            if let parsed = extractBalancedJSON(text), let name = parsed["name"] as? String {
                if knownToolNames.contains(name) {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: text))
                }
            }
        }

        return (calls, broken)
    }

    public func parseAllXMLFunctionCalls(_ text: String) -> [ParsedToolCall] {
        var calls: [ParsedToolCall] = []
        let fnRegex = try? NSRegularExpression(pattern: "<function=([^>]+)>([\\s\\S]*?)(?:</function>|(?=<function=)|$)", options: [])
        let nsText = text as NSString
        let matches = fnRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []

        for m in matches {
            guard m.numberOfRanges >= 3 else { continue }
            let fnName = nsText.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let body = nsText.substring(with: m.range(at: 2))
            let rawMatch = nsText.substring(with: m.range(at: 0))
            guard !fnName.isEmpty else { continue }

            var args: [String: Any] = [:]
            let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
            if let data = trimmedBody.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                args = json
            } else if let parsed = extractBalancedJSON(body) {
                args = parsed
            } else {
                let paramRegex = try? NSRegularExpression(pattern: "<parameter=([^>]+)>([\\s\\S]*?)(?:</parameter>|$)", options: [])
                let nsBody = body as NSString
                let pMatches = paramRegex?.matches(in: body, options: [], range: NSRange(location: 0, length: nsBody.length)) ?? []

                for pm in pMatches {
                    guard pm.numberOfRanges >= 3 else { continue }
                    let pName = nsBody.substring(with: pm.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let pValStr = nsBody.substring(with: pm.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)

                    if let data = pValStr.data(using: .utf8),
                       let jsonVal = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed),
                       (jsonVal is [String: Any] || jsonVal is [Any] || jsonVal is NSNumber) {
                        args[pName] = jsonVal
                    } else {
                        args[pName] = pValStr
                    }
                }
            }

            let rawArgsData = try? JSONSerialization.data(withJSONObject: args, options: [.prettyPrinted])
            let rawArgsStr = rawArgsData.flatMap { String(data: $0, encoding: .utf8) } ?? ""
            calls.append(ParsedToolCall(name: fnName, arguments: args, rawArguments: rawArgsStr, rawText: rawMatch))
        }

        return calls
    }

    /// String & escape aware balanced bracket JSON scanner
    public func extractBalancedJSON(_ text: String) -> [String: Any]? {
        var currentIndex = text.startIndex
        while currentIndex < text.endIndex, let startIdx = text[currentIndex...].firstIndex(of: "{") {
            let sub = text[startIdx...]
            var depth = 0
            var inString = false
            var isEscaped = false
            var endIdx: String.Index? = nil

            for i in sub.indices {
                let c = sub[i]
                if inString {
                    if isEscaped {
                        isEscaped = false
                    } else if c == "\\" {
                        isEscaped = true
                    } else if c == "\"" {
                        inString = false
                    }
                } else {
                    if c == "\"" {
                        inString = true
                    } else if c == "{" {
                        depth += 1
                    } else if c == "}" {
                        depth -= 1
                        if depth == 0 {
                            endIdx = i
                            break
                        }
                    }
                }
            }

            if let end = endIdx {
                let jsonSubstring = String(sub[...end])
                if let data = jsonSubstring.data(using: .utf8),
                   let dict = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
                   dict["name"] != nil {
                    return dict
                }
            }

            currentIndex = text.index(after: startIdx)
        }
        return nil
    }

    public func generateRecoveryPrompt(brokenFragment: String, maxTokens: Int) -> String {
        let cleanFrag = AgentHarness.truncateText(AgentHarness.sanitizeText(brokenFragment), limit: 400)
        return """
        Your previous <tool_call> could not be parsed as JSON (it was either malformed or truncated mid-token).
        Raw fragment:
        \(cleanFrag)

        Please re-issue the tool call as a single valid JSON object inside <tool_call></tool_call> tags. If the parameters or scripts are large, consider breaking them into smaller actions.
        """
    }

    // MARK: - Tool Execution

    public func executeTool(call: ParsedToolCall, workingDirectory: URL? = nil, maxOutputLength: Int? = nil) async -> (resultJSON: String, record: ToolCallRecord, isCompleted: Bool) {
        let startTime = CFAbsoluteTimeGetCurrent()
        let wd = workingDirectory ?? defaultWorkingDirectory
        let strArgs = call.arguments.mapValues { String(describing: $0) }
        let effectiveMaxLen = maxOutputLength ?? self.maxToolOutputLength

        guard let tool = tools[call.name] else {
            let err = "Unknown tool '\(call.name)'"
            let json = AgentHarness.toolErrorJSON(tool: call.name, error: err)
            let rec = ToolCallRecord(
                name: call.name,
                arguments: strArgs,
                rawArguments: call.rawArguments,
                status: .error,
                output: nil,
                error: err,
                executionDurationSeconds: CFAbsoluteTimeGetCurrent() - startTime
            )
            return (json, rec, false)
        }

        do {
            let (json, stdout, stderr, isCompleted) = try await tool.execute(arguments: call.arguments, workingDirectory: wd, maxOutputLength: effectiveMaxLen)
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let status: ToolExecutionStatus = (stderr != nil && !stderr!.isEmpty) ? .error : .success

            let rec = ToolCallRecord(
                name: call.name,
                arguments: strArgs,
                rawArguments: call.rawArguments,
                status: status,
                output: stdout,
                error: stderr,
                executionDurationSeconds: duration
            )
            return (json, rec, isCompleted)
        } catch {
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let err = error.localizedDescription
            let json = AgentHarness.toolErrorJSON(tool: call.name, error: err)
            let rec = ToolCallRecord(
                name: call.name,
                arguments: strArgs,
                rawArguments: call.rawArguments,
                status: .error,
                output: nil,
                error: err,
                executionDurationSeconds: duration
            )
            return (json, rec, false)
        }
    }

    // MARK: - Helpers & Utilities

    public static func resolvePath(_ rawPath: String, workingDirectory: URL?) -> URL {
        let expanded = (rawPath as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded)
        }
        if let wd = workingDirectory {
            return wd.appendingPathComponent(expanded)
        }
        return URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(expanded)
    }

    public static func sanitizeText(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: "<|im_start|>", with: "[im_start]")
        s = s.replacingOccurrences(of: "<|im_end|>", with: "[im_end]")
        s = s.replacingOccurrences(of: "<tool_call>", with: "[tool_call]")
        s = s.replacingOccurrences(of: "</tool_call>", with: "[/tool_call]")
        s = s.replacingOccurrences(of: "<tool_response>", with: "[tool_response]")
        s = s.replacingOccurrences(of: "</tool_response>", with: "[/tool_response]")
        return s
    }

    public static func truncateText(_ text: String, limit: Int) -> String {
        guard text.count > limit else { return text }
        let headCount = max(100, limit - 400)
        let tailCount = 300
        let head = text.prefix(headCount)
        let tail = text.suffix(tailCount)
        let truncated = text.count - (headCount + tailCount)
        return "\(head)\n\n... [truncated \(truncated) characters] ...\n\n\(tail)"
    }

    public static func isOfficialVendorDomain(_ host: String) -> Bool {
        let h = host.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !h.isEmpty else { return false }
        let officialSuffixes = [
            "apple.com", "microsoft.com", "github.com", "google.com",
            "openai.com", "anthropic.com", "nvidia.com", "meta.com",
            "wikipedia.org", "w3.org", "ietf.org", "kernel.org",
            "python.org", "rust-lang.org", "swift.org", "developer.apple.com"
        ]
        if officialSuffixes.contains(where: { h == $0 || h.hasSuffix("." + $0) }) {
            return true
        }
        if h.hasSuffix(".gov") || h.hasSuffix(".edu") {
            return true
        }
        return false
    }

    public static func toolSuccessJSON(tool: String, data: [String: Any]) -> String {
        let dict: [String: Any] = [
            "status": "success",
            "tool": tool,
            "result": data
        ]
        if let jsonBytes = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted]),
           let str = String(data: jsonBytes, encoding: .utf8) {
            return str
        }
        return "{\"status\": \"success\", \"tool\": \"\(tool)\"}"
    }

    public static func toolErrorJSON(tool: String, error: String, extra: [String: Any]? = nil) -> String {
        var dict: [String: Any] = [
            "status": "error",
            "tool": tool,
            "error": error
        ]
        if let extra = extra {
            for (k, v) in extra { dict[k] = v }
        }
        if let jsonBytes = try? JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted]),
           let str = String(data: jsonBytes, encoding: .utf8) {
            return str
        }
        return "{\"status\": \"error\", \"tool\": \"\(tool)\", \"error\": \"\(error)\"}"
    }

    public static func runProcess(
        executableURL: URL,
        arguments: [String],
        currentDirectory: URL,
        timeoutSeconds: TimeInterval = 120
    ) async throws -> (exitCode: Int32, stdout: String, stderr: String) {
        let process = Process()
        let stdoutPipe = Pipe()
        let stderrPipe = Pipe()

        // 1. Expand environment PATH for macOS GUI applications and set non-interactive flags
        var env = ProcessInfo.processInfo.environment
        let userHome = FileManager.default.homeDirectoryForCurrentUser.path
        let currentPath = env["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin"
        let extraPaths = [
            "/opt/homebrew/bin",
            "/opt/homebrew/sbin",
            "/usr/local/bin",
            "/usr/local/sbin",
            "\(userHome)/.cargo/bin",
            "\(userHome)/.local/bin"
        ]
        let fullPath = (extraPaths + [currentPath]).joined(separator: ":")
        env["PATH"] = fullPath
        env["HOMEBREW_NO_AUTO_UPDATE"] = "1"
        env["HOMEBREW_NO_INSTALL_CLEANUP"] = "1"
        env["HOMEBREW_NO_ENV_HINTS"] = "1"
        env["CI"] = "1"
        env["TERM"] = "dumb"
        env["PAGER"] = "cat"
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["NONINTERACTIVE"] = "1"
        env["DEBIAN_FRONTEND"] = "noninteractive"

        process.environment = env
        process.executableURL = executableURL
        process.arguments = arguments
        process.currentDirectoryURL = currentDirectory
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe

        final class ProcessState: @unchecked Sendable {
            let lock = NSLock()
            var stdoutData = Data()
            var stderrData = Data()
            var isFinished = false
            var isTimedOut = false
            var isCancelled = false
            var timerWorkItem: DispatchWorkItem?

            func cancelTimer() {
                timerWorkItem?.cancel()
                timerWorkItem = nil
            }
        }

        let state = ProcessState()

        // Asynchronously drain stdout and stderr without blocking to prevent 64KB pipe buffer deadlocks
        stdoutPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty {
                state.lock.lock()
                state.stdoutData.append(chunk)
                state.lock.unlock()
            }
        }

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let chunk = handle.availableData
            if !chunk.isEmpty {
                state.lock.lock()
                state.stderrData.append(chunk)
                state.lock.unlock()
            }
        }

        let cleanupPipes: @Sendable () -> Void = {
            stdoutPipe.fileHandleForReading.readabilityHandler = nil
            stderrPipe.fileHandleForReading.readabilityHandler = nil

            let remOut = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
            let remErr = stderrPipe.fileHandleForReading.readDataToEndOfFile()

            state.lock.lock()
            state.stdoutData.append(remOut)
            state.stderrData.append(remErr)
            state.lock.unlock()
        }

        let killProcessTree: @Sendable () -> Void = {
            let pid = process.processIdentifier
            if pid > 0 {
                kill(-pid, SIGTERM)
                kill(pid, SIGTERM)
            }
            process.terminate()

            DispatchQueue.global().asyncAfter(deadline: .now() + 0.5) {
                if process.isRunning && pid > 0 {
                    kill(-pid, SIGKILL)
                    kill(pid, SIGKILL)
                }
            }
        }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if timeoutSeconds > 0 {
                    let item = DispatchWorkItem {
                        state.lock.lock()
                        guard !state.isFinished else {
                            state.lock.unlock()
                            return
                        }
                        state.isTimedOut = true
                        state.lock.unlock()

                        killProcessTree()
                    }
                    state.timerWorkItem = item
                    DispatchQueue.global().asyncAfter(deadline: .now() + timeoutSeconds, execute: item)
                }

                DispatchQueue.global(qos: .userInitiated).async {
                    do {
                        try process.run()
                        process.waitUntilExit()

                        state.cancelTimer()
                        cleanupPipes()

                        state.lock.lock()
                        state.isFinished = true
                        let timedOut = state.isTimedOut
                        let cancelled = state.isCancelled
                        let finalStdout = String(data: state.stdoutData, encoding: .utf8) ?? ""
                        let finalStderr = String(data: state.stderrData, encoding: .utf8) ?? ""
                        let exitCode = process.terminationStatus
                        state.lock.unlock()

                        if timedOut {
                            let msg = "Command timed out after \(Int(timeoutSeconds)) seconds."
                            continuation.resume(returning: (124, finalStdout, msg))
                        } else if cancelled {
                            continuation.resume(returning: (130, finalStdout, "Command was cancelled by user."))
                        } else {
                            continuation.resume(returning: (exitCode, finalStdout, finalStderr))
                        }
                    } catch {
                        state.cancelTimer()
                        cleanupPipes()

                        state.lock.lock()
                        state.isFinished = true
                        state.lock.unlock()

                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            state.lock.lock()
            state.isCancelled = true
            state.lock.unlock()
            killProcessTree()
        }
    }
}
