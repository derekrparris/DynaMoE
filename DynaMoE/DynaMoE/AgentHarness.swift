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

public extension AgentTool {
    /// One-word classification used to group `tools_discover` results.
    var catalogCategory: String { Self.category(for: definition.function.name) }

    /// Short one-line synopsis shown in `tools_discover` listings.
    var catalogSummary: String {
        let d = definition.function.description
        return d.count > 110 ? String(d.prefix(107)) + "..." : d
    }

    /// Estimated prompt tokens consumed when this tool's schema is loaded into context.
    func approximatePromptTokens() -> Int {
        guard let data = try? JSONEncoder().encode(definition) else { return 0 }
        return max(1, data.count / 4)
    }

    static func category(for name: String) -> String {
        switch name {
        case "find_files", "grep_search", "codebase_search": return "search"
        case "find_symbol_definition", "find_references": return "analysis"
        case "web_search", "web_fetch": return "web"
        case "git_status", "git_diff", "git_commit": return "git"
        case "spawn_subagent", "get_subagent_status", "send_subagent_message", "list_subagents": return "subagents"
        case "lint_diagnostics": return "quality"
        default: return "core"
        }
    }
}

// MARK: - Core Coding & Shell Tool Implementations

/// Tool 1: shell_run — Executes shell commands with working directory & timeout
public final class ShellRunTool: AgentTool {
    public let definition = ToolDefinition(
        name: "shell_run",
        description: "Executes shell commands on the local macOS terminal via zsh. Use this to run scripts, compilers, git, or check system state. Output is captured and returned. When fetching web pages, prefer raw text endpoints (e.g. raw.githubusercontent.com/OWNER/REPO/HEAD/path) over rendered HTML pages; large HTML responses are auto-converted to plain text and truncated.",
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

        var rawStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if AgentHarness.looksLikeHTML(rawStdout), rawStdout.utf8.count > 2048 {
            let converted = AgentHarness.htmlToPlainText(rawStdout)
            print("🛠 [shell_run] HTML output converted: \(rawStdout.utf8.count) -> \(converted.utf8.count) chars")
            rawStdout = "[HTML converted to plain text: \(rawStdout.utf8.count) -> \(converted.utf8.count) chars]\n\(converted)"
        }
        let cleanStdout = AgentHarness.truncateText(AgentHarness.sanitizeText(rawStdout), limit: maxOutputLength)
        let cleanStderr = AgentHarness.truncateText(AgentHarness.sanitizeText(stderr.trimmingCharacters(in: .whitespacesAndNewlines)), limit: maxOutputLength)

        if exitCode == 0 {
            print("🛠 [shell_run] exit=0 cmd='\(String(command.prefix(100)))'")
            let res = AgentHarness.toolSuccessJSON(tool: "shell_run", data: [
                "stdout": cleanStdout.isEmpty ? "Command succeeded with no output." : cleanStdout,
                "exit_code": 0
            ])
            return (res, cleanStdout, nil, false)
        } else {
            print("🛠 [shell_run] exit=\(exitCode) cmd='\(String(command.prefix(100)))' stderr='\(String(cleanStderr.prefix(160)))' stdout='\(String(cleanStdout.prefix(80)))'")
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

        // Known binary formats surface as a structured "unsupported type" result instead
        // of a generic UTF-8 decode error, so the model knows the read never happened.
        if let fileKind = AgentHarness.binaryFileKind(at: resolvedPath) {
            let err = "file_read cannot display binary content (detected: \(fileKind))."
            let res = AgentHarness.toolErrorJSON(tool: "file_read", error: err, extra: [
                "file_kind": fileKind,
                "hint": "Use shell_run with `file \"\(resolvedPath.path)\"` to identify the format, or a format-specific command to extract text."
            ])
            return (res, nil, err, false)
        }

        do {
            let data = try Data(contentsOf: resolvedPath)
            let isUTF8 = String(data: data, encoding: .utf8) != nil
            guard let content = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) else {
                let err = "Failed to decode file as text (neither UTF-8 nor Latin-1). It is likely binary."
                return (AgentHarness.toolErrorJSON(tool: "file_read", error: err), nil, err, false)
            }
            let lines = content.components(separatedBy: "\n")
            let totalLines = lines.count

            var startLine = 1
            if let s = arguments["start_line"] as? Int {
                startLine = max(1, min(s, totalLines))
            } else if let s = arguments["start_line"], let si = Int(String(describing: s)) {
                startLine = max(1, min(si, totalLines))
            }

            var endLine = totalLines
            if let e = arguments["end_line"] as? Int {
                endLine = max(startLine, min(e, totalLines))
            } else if let e = arguments["end_line"], let ei = Int(String(describing: e)) {
                endLine = max(startLine, min(ei, totalLines))
            }

            // Head-heavy line-budget slicing: show as many leading lines as the character
            // budget allows, keep a short tail for orientation, and mark everything in
            // between with an explicit marker plus a paging hint the model can act on.
            let charBudget = max(400, maxOutputLength)
            let tailLineCount = (endLine - startLine + 1) > 2 ? 2 : 0
            let tailStartIdx = max(startLine - 1, endLine - tailLineCount)

            var tailLines: [String] = []
            for idx in tailStartIdx..<endLine {
                var lineText = "\(idx + 1): \(lines[idx])"
                if lineText.utf8.count > 600 {
                    lineText = String(lineText.prefix(600)) + " <<<LINE TRUNCATED>>>"
                }
                tailLines.append(lineText)
            }
            let tailText = tailLines.joined(separator: "\n")
            let tailCost = tailText.utf8.count + 72

            var bodyLines: [String] = []
            var usedBytes = 0
            var lastIncludedIdx = startLine - 2

            for idx in (startLine - 1)..<tailStartIdx {
                let lineText = "\(idx + 1): \(lines[idx])"
                let cost = lineText.utf8.count + 1
                if usedBytes + cost > charBudget - tailCost { break }
                bodyLines.append(lineText)
                usedBytes += cost
                lastIncludedIdx = idx
            }

            if bodyLines.isEmpty && tailStartIdx > startLine - 1 {
                var lineText = "\(startLine): \(lines[startLine - 1])"
                if lineText.utf8.count > max(80, charBudget / 2) {
                    lineText = String(lineText.prefix(max(80, charBudget / 2))) + " <<<LINE TRUNCATED>>>"
                }
                bodyLines.append(lineText)
                lastIncludedIdx = startLine - 1
            }

            let omittedCount = max(0, tailStartIdx - lastIncludedIdx - 1)
            var contentParts: [String] = []
            if !bodyLines.isEmpty { contentParts.append(bodyLines.joined(separator: "\n")) }
            if omittedCount > 0 {
                contentParts.append("<<<TRUNCATED: lines \(lastIncludedIdx + 2)-\(tailStartIdx) of \(totalLines) omitted>>>")
            }
            if !tailText.isEmpty && tailStartIdx > lastIncludedIdx { contentParts.append(tailText) }

            let slicedText = contentParts.joined(separator: "\n")
            let sanitized = AgentHarness.sanitizeText(slicedText)

            var resultData: [String: Any] = [
                "path": resolvedPath.path,
                "total_lines": totalLines,
                "start_line": startLine,
                "end_line": endLine,
                "content": sanitized
            ]
            if !isUTF8 {
                resultData["encoding_note"] = "File is not valid UTF-8; decoded as ISO Latin-1 (some bytes may display incorrectly)."
            }
            if omittedCount > 0 {
                let nextStart = lastIncludedIdx + 2
                resultData["truncated"] = true
                resultData["omitted_lines"] = omittedCount
                resultData["next_start_line"] = nextStart
                resultData["continuation_hint"] = "Call file_read again with start_line=\(nextStart) to read the next chunk."
            }

            let res = AgentHarness.toolSuccessJSON(tool: "file_read", data: resultData)
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
            var searchEngineName = "yahoo"

            if !braveKey.isEmpty {
                results = try await searchBrave(query: cleanQuery, apiKey: braveKey, maxResults: maxResults)
                searchEngineName = "brave"
            }

            // De facto search engine: Headless Chrome
            if results.isEmpty {
                results = try await HeadlessChromeSearchEngine.shared.search(query: cleanQuery, maxResults: maxResults)
                searchEngineName = "yahoo"
            }

            if results.isEmpty {
                results = try await searchWikipedia(query: cleanQuery, maxResults: maxResults)
                searchEngineName = "wikipedia"
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
                    "is_official_domain": isOfficial,
                    "source": searchEngineName
                ]
                if isOfficial {
                    entry["ground_truth_notice"] = "Verified official vendor domain (\(host)). Content is authentic ground truth."
                }
                if let snippet = item["snippet"], !snippet.isEmpty {
                    let cleanSnippet = AgentHarness.sanitizeText(snippet.trimmingCharacters(in: .whitespacesAndNewlines))
                    entry["snippet"] = cleanSnippet.count > 300 ? String(cleanSnippet.prefix(300)) + "..." : cleanSnippet
                }
                enrichedResults.append(entry)
            }

            let res = AgentHarness.toolSuccessJSON(tool: "web_search", data: [
                "query": cleanQuery,
                "count": results.count,
                "search_engine": searchEngineName,
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
                    "description": "Optional search keywords to filter sections on exceptionally large web pages (over 20,000 characters). Omit this parameter for standard articles and spec pages to receive the full, clean document in natural reading order."
                ],
                "max_length": [
                    "type": "integer",
                    "description": "Optional maximum character length of returned content (defaults to the configured tool-output token budget, up to 30,000 characters)."
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
        // Respect the configured token budget (passed in as a character budget), with a
        // floor for usable research snippets and the historical 24k-char ceiling.
        let defaultLimit = max(4000, min(24000, maxOutputLength))
        let limit = customMax.map { min(40000, max(1000, $0)) } ?? defaultLimit

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

            if fullCleaned.count <= limit {
                // If the entire cleaned page fits within the budget, ALWAYS preserve full natural reading order.
                // Never shuffle, slice, or fragment a document that fits in context.
                contentToReturn = fullCleaned
            } else if let q = queryFilter, !q.isEmpty {
                // For oversized documents where a query was explicitly provided, extract matching sections
                // while strictly preserving original sequential document order.
                contentToReturn = Self.filterContentByQuery(fullCleaned, query: q, limit: limit)
            } else {
                let prefix = fullCleaned.prefix(limit)
                let truncatedText: String
                if let lastBreak = prefix.lastIndex(where: { $0 == "\n" || $0 == "." }) {
                    truncatedText = String(prefix[...lastBreak])
                } else {
                    truncatedText = String(prefix)
                }
                contentToReturn = truncatedText + "\n\n... [Content truncated cleanly at section boundary (\(truncatedText.count) characters)]"
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

            let sanitizedContent = AgentHarness.sanitizeText(contentToReturn)
            let sanitizedTitle = AgentHarness.sanitizeText(pageTitle)
            var resultData: [String: Any] = [
                "url": url.absoluteString,
                "domain": domain,
                "is_official_domain": isOfficial,
                "title": sanitizedTitle,
                "content_length": contentToReturn.count,
                // Headings only — the full body already ships in `content`; duplicating
                // section bodies doubled every fetch's context cost.
                "section_headings": WebFetchTool.sectionHeadings(from: contentToReturn),
                "content": sanitizedContent
            ]
            if let pubDate = detectedDate {
                resultData["published_date"] = pubDate
            }
            if isOfficial {
                resultData["ground_truth_notice"] = "Verified official vendor content from \(domain). All executive quotes, specifications, benchmark multipliers, and pricing tiers represent authentic ground truth."
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

        // Strip non-content / navigation / template / media tags:
        // script, style, nav, header, footer, svg, noscript, select, option, form, button, template, dialog, aside, iframe, picture, figure
        pageHtml = pageHtml.replacingOccurrences(
            of: "(?is)<(script|style|nav|header|footer|svg|noscript|select|option|form|button|template|dialog|aside|iframe|picture|figure)[^>]*>.*?</\\1>",
            with: "",
            options: .regularExpression
        )

        // Strip article share bars, media downloads, copy-text containers, press contacts, and related stories
        let articleNoisePatterns = [
            #"(?is)<div[^>]+class=[\"'][^\"']*(?:nr-article-share|sharesheet|docsanddownloads|presscontacts|social-share|share-bar|share-component|article-list component-content)[^\"']*[\"'][\s\S]*?(?=<div[^>]+class=[\"'][^\"']*(?:pagebody|pagetitle|article-subhead|category component)|<footer|<section|$)"#,
            #"(?is)<div[^>]+data-copy-content[\s\S]*?</div>\s*</div>\s*</div>"#,
            #"(?is)<div[^>]+data-component-list=[\"']CopyText[\"'][\s\S]*?</div>\s*</div>"#,
            #"(?is)<div[^>]+class=[\"']*docsanddownloads[^\"']*[\"'][\s\S]*?</div>\s*</div>\s*</div>"#,
            #"(?is)<div[^>]+class=[\"']*(?:sharesheet|presscontacts)[^\"']*[\"'][\s\S]*?</div>\s*</div>"#
        ]
        for p in articleNoisePatterns {
            pageHtml = pageHtml.replacingOccurrences(of: p, with: "", options: .regularExpression)
        }

        // Strip standalone/void tags or leftover tags
        pageHtml = pageHtml.replacingOccurrences(of: "(?is)<(input|meta|link|svg|path)[^>]*>", with: "", options: .regularExpression)

        // Strip accessibility / visuallyhidden elements, table header placeholders, and diagram image callout pins
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<[^>]+class=[\"'][^\"']*(?:visually-?hidden|techspecs-columnheader|techspecs-header-row|caption-wrapper|image-wrapper|sr-only|screen-reader-text|a11y-only)[^\"']*[\"'][^>]*>.*?</[^>]+>"#,
            with: "",
            options: .regularExpression
        )
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<div[^>]+class=[\"'][^\"']*(?:caption-wrapper|image-wrapper|techspecs-header-row)[^\"']*[\"'][^>]*>.*?</div[^>]*>"#,
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

        // Strip article noise components again if embedded inside <main> or <article>
        for p in articleNoisePatterns {
            pageHtml = pageHtml.replacingOccurrences(of: p, with: "", options: .regularExpression)
        }

        // Convert standalone strong div/p to ## Heading (e.g. <div class="pagebody-copy"><strong>Pricing and Availability</strong></div>)
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<(?:div|p)[^>]*>\s*<(?:strong|b)[^>]*>([^<]{3,80})</(?:strong|b)>\s*</(?:div|p)>"#,
            with: "\n\n## $1\n\n",
            options: .regularExpression
        )

        // Convert HTML tables to Markdown tables BEFORE stripping block tags
        pageHtml = convertTablesToMarkdown(html: pageHtml)

        // Format techspecs Price row into an explicit, structured section
        if let priceRegex = try? NSRegularExpression(pattern: #"(?is)<div[^>]+class=[\"'][^\"']*techspecs-row[^\"]*[\"'][^>]*>(?:(?!<div class=\"techspecs-section).)*?Price.*?</div>\s*</div>"#) {
            let nsStr = pageHtml as NSString
            let matches = priceRegex.matches(in: pageHtml, range: NSRange(location: 0, length: nsStr.length))
            for match in matches.reversed() {
                let matchStr = nsStr.substring(with: match.range)
                let priceExtractor = try? NSRegularExpression(pattern: #"\$[\d,]+"#)
                let priceMatches = priceExtractor?.matches(in: matchStr, range: NSRange(location: 0, length: (matchStr as NSString).length)) ?? []
                var prices: [String] = []
                for pm in priceMatches {
                    prices.append((matchStr as NSString).substring(with: pm.range))
                }
                var replacement = "\n\n## Price\n"
                if prices.count >= 2 {
                    replacement += "- Base (M5 Max): \(prices[0])\n- High-End (M5 Ultra): \(prices[1])\n\n"
                } else if !prices.isEmpty {
                    replacement += prices.joined(separator: " | ") + "\n\n"
                }
                pageHtml = (pageHtml as NSString).replacingCharacters(in: match.range, with: replacement)
            }
        }

        // Format techspecs Finish row
        if let finishRegex = try? NSRegularExpression(pattern: #"(?is)<div[^>]+class=[\"'][^\"']*techspecs-row[^\"]*[\"'][^>]*>(?:(?!<div class=\"techspecs-section).)*?Finish.*?</div>\s*</div>"#) {
            let nsStr = pageHtml as NSString
            let matches = finishRegex.matches(in: pageHtml, range: NSRange(location: 0, length: nsStr.length))
            for match in matches.reversed() {
                pageHtml = (pageHtml as NSString).replacingCharacters(in: match.range, with: "\n\n## Finish\nSilver\n\n")
            }
        }

        // Convert techspecs-rowheader to ## Header
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<div[^>]+class=[\"'][^\"']*(?:techspecs-rowheader|rowheader)[^\"']*[\"'][^>]*>([\s\S]*?)</div>"#,
            with: "\n\n## $1\n\n",
            options: .regularExpression
        )

        // Convert techspecs-subheader to ### Subheader
        pageHtml = pageHtml.replacingOccurrences(
            of: #"(?is)<(p|div)[^>]+class=[\"'][^\"']*techspecs-subheader[^\"]*[\"'][^>]*>([\s\S]*?)</\1>"#,
            with: "\n\n### $2\n\n",
            options: .regularExpression
        )

        // Convert headings to Markdown headings
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h1[^>]*>([\\s\\S]*?)</h1>", with: "\n\n# $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h2[^>]*>([\\s\\S]*?)</h2>", with: "\n\n## $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h3[^>]*>([\\s\\S]*?)</h3>", with: "\n\n### $1\n\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<h[4-6][^>]*>([\\s\\S]*?)</h[4-6]>", with: "\n\n#### $1\n\n", options: .regularExpression)

        // Convert list items
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<li[^>]*>([\\s\\S]*?)</li>", with: "\n- $1", options: .regularExpression)

        // Convert bold / strong
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<(strong|b)[^>]*>([\\s\\S]*?)</\\1>", with: "**$2**", options: .regularExpression)

        // Convert definition lists <dt> and <dd>
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<dt[^>]*>([\\s\\S]*?)</dt>", with: "\n\n### $1\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)</?dd[^>]*>", with: "\n", options: .regularExpression)

        // Convert section heading containers (e.g., class="...section-header...", class="...section-title...", etc.)
        pageHtml = pageHtml.replacingOccurrences(of: #"(?i)<div[^>]*class=[\"'][^\"']*(?:section-header|section-title|spec-header|headline-reduced|category-header)[^\"']*[\"'][^>]*>([\s\S]*?)</div>"#, with: "\n\n## $1\n\n", options: .regularExpression)

        // Replace other block tags with newlines
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)</?(p|div|section|blockquote|pre|code)[^>]*>", with: "\n", options: .regularExpression)
        pageHtml = pageHtml.replacingOccurrences(of: "(?i)<br\\s*/?>", with: "\n", options: .regularExpression)

        // Strip remaining HTML tags
        let text = stripHTML(pageHtml)

        // Clean up excessive whitespace, blank lines, and orphan markdown tokens
        let rawLines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        var resultLines: [String] = []
        var consecutiveBlanks = 0
        for line in rawLines {
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
                // Tighten list items: remove preceding blank line if both this and previous item are list bullets
                if (trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("• ")) &&
                    resultLines.last == "" &&
                    resultLines.count >= 2 {
                    let prevNonBlank = resultLines[resultLines.count - 2]
                    if prevNonBlank.hasPrefix("- ") || prevNonBlank.hasPrefix("* ") || prevNonBlank.hasPrefix("• ") {
                        resultLines.removeLast()
                    }
                }
                resultLines.append(trimmed)
            }
        }

        return resultLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Extracts structured sections from Markdown content based on # and ## headings
    public static func extractStructuredSections(from markdown: String) -> [[String: String]] {
        var sections: [[String: String]] = []
        let lines = markdown.components(separatedBy: .newlines)
        var currentHeading = "Overview"
        var currentContentLines: [String] = []

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("## ") || trimmed.hasPrefix("# ") {
                let headingText = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "#* \t"))
                if !currentContentLines.isEmpty {
                    let body = currentContentLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
                    if !body.isEmpty {
                        sections.append(["heading": currentHeading, "details": body])
                    }
                    currentContentLines.removeAll()
                }
                currentHeading = headingText
            } else {
                currentContentLines.append(line)
            }
        }

        if !currentContentLines.isEmpty {
            let body = currentContentLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !body.isEmpty {
                sections.append(["heading": currentHeading, "details": body])
            }
        }

        return sections
    }

    /// Extracts just the markdown headings of a cleaned page — a compact table of contents
    /// that does not duplicate the body content.
    public static func sectionHeadings(from markdown: String) -> [String] {
        markdown.components(separatedBy: .newlines)
            .filter { $0.hasPrefix("## ") || $0.hasPrefix("# ") }
            .map { $0.trimmingCharacters(in: CharacterSet(charactersIn: "#* \t")) }
            .filter { !$0.isEmpty }
    }

    /// Extracts content sections matching query keywords while strictly preserving natural document order
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

        var matchingBlocks: [String] = []
        var currentLength = 0
        var lastHeading: String? = nil

        let headerPrefix = "### Key Sections Matching \"\(cleanQuery)\":\n"
        matchingBlocks.append(headerPrefix)
        currentLength += headerPrefix.count

        // Iterate through blocks in their ORIGINAL sequential document order.
        // Never sort by keyword score, which destroys linear reading order and creates confusing permutations.
        for block in rawBlocks {
            if block.hasPrefix("#") {
                lastHeading = block
            }

            let lower = block.lowercased()
            let matches = terms.contains { lower.contains($0) }
            if matches {
                // If there's an associated parent heading we haven't included yet, prepend it
                if let heading = lastHeading, !matchingBlocks.contains(heading) {
                    if currentLength + heading.count + 2 <= limit {
                        matchingBlocks.append(heading)
                        currentLength += heading.count + 2
                    }
                }

                if !matchingBlocks.contains(block) {
                    if currentLength + block.count + 2 > limit {
                        matchingBlocks.append("... [Additional sections truncated to stay within character limit]")
                        break
                    }
                    matchingBlocks.append(block)
                    currentLength += block.count + 2
                }
            }
        }

        if matchingBlocks.count <= 1 {
            // No matching blocks found
            return String(content.prefix(limit))
        }

        return matchingBlocks.joined(separator: "\n\n")
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

// MARK: - Tool Registry Management (tools_discover / tools_load / tools_unload)

/// Lists installed-but-unloaded tools grouped by category so the model can decide what to load.
public final class ToolDiscoverTool: AgentTool {
    public let definition = ToolDefinition(
        name: "tools_discover",
        description: "Lists additional tools that are installed but not currently loaded, grouped by category with estimated token cost. Call this to inspect what other capabilities are available, then use tools_load (with a batched 'names' array) to activate the ones a task needs.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "category": [
                    "type": "string",
                    "enum": ["search", "analysis", "web", "git", "subagents", "quality"],
                    "description": "Optional category filter; omit to list all groups."
                ],
                "query": [
                    "type": "string",
                    "description": "Optional keyword to filter tool names and descriptions."
                ]
            ]),
            "required": AnyCodable([])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let categoryFilter = (arguments["category"] as? String)?.lowercased()
        let queryFilter = (arguments["query"] as? String)?.lowercased()
        let harness = AgentHarness.shared

        var groups: [String: [[String: Any]]] = [:]
        for name in harness.tools.keys.sorted() {
            guard let tool = harness.tools[name], harness.loadedTools[name] == nil else { continue }
            let cat = tool.catalogCategory
            if let cf = categoryFilter, !cf.isEmpty, cf != "all", cat != cf { continue }
            let summary = tool.catalogSummary
            if let qf = queryFilter, !qf.isEmpty, !(name.contains(qf) || summary.lowercased().contains(qf)) { continue }
            groups[cat, default: []].append([
                "name": name,
                "summary": summary,
                "estimated_prompt_tokens": tool.approximatePromptTokens()
            ])
        }

        let order = ["search", "analysis", "web", "git", "subagents", "quality", "core"]
        var lines: [String] = []
        var total = 0
        for cat in order {
            guard let items = groups[cat], !items.isEmpty else { continue }
            lines.append("## \(cat)")
            for item in items {
                guard let n = item["name"] as? String,
                      let s = item["summary"] as? String,
                      let t = item["estimated_prompt_tokens"] as? Int else { continue }
                lines.append("- \(n) (~\(t) tokens): \(s)")
                total += 1
            }
        }
        let readable = lines.isEmpty ? "All installed tools are currently loaded. Nothing to discover." : lines.joined(separator: "\n")

        var groupsJSON: [[String: Any]] = []
        for cat in order {
            if let items = groups[cat], !items.isEmpty {
                groupsJSON.append(["category": cat, "tools": items])
            }
        }

        let res = AgentHarness.toolSuccessJSON(tool: "tools_discover", data: [
            "installed_count": harness.tools.count,
            "loaded_count": harness.loadedTools.count,
            "available_count": total,
            "groups": groupsJSON
        ])
        return (res, readable, nil, false)
    }
}

/// Loads an installed tool's schema into context so the model may call it directly.
public final class ToolLoadTool: AgentTool {
    public let definition = ToolDefinition(
        name: "tools_load",
        description: "Loads installed tools' full schemas into context so they can be called directly. Use tools_discover to list available tools first. Accepts a single 'name' or a 'names' array. Idempotent: loading an already-loaded tool is a no-op success. Prefer loading all tools needed for a task in ONE call (a single 'names' array), since each load event rewrites the tool block and triggers a context re-prefill.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "name": [
                    "type": "string",
                    "description": "Exact name of a single tool to load (e.g. 'web_search', 'git_diff')."
                ],
                "names": [
                    "type": "array",
                    "items": ["type": "string"],
                    "description": "Batch load: exact names of multiple tools to load in one call. Preferred over repeated single-name calls."
                ]
            ]),
            "required": AnyCodable([])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let harness = AgentHarness.shared

        // Resolve batch ('names') or single ('name') form.
        var names: [String] = []
        if let batch = arguments["names"] as? [String] {
            names = batch.compactMap { ($0 as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        }
        if let single = arguments["name"] as? String {
            let trimmed = single.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { names.append(trimmed) }
        }
        if names.isEmpty {
            let err = "Missing 'name' or 'names' parameter in tools_load. Run tools_discover to list available tools."
            return (AgentHarness.toolErrorJSON(tool: "tools_load", error: err), nil, err, false)
        }

        var loadedResults: [[String: Any]] = []
        var schemas: [String: String] = [:]
        var errors: [String] = []
        for name in names {
            if harness.loadedTools[name] != nil {
                loadedResults.append([
                    "tool_name": name,
                    "registration": false,
                    "already_loaded": true
                ])
                continue
            }
            do {
                let definition = try harness.loadTool(named: name)
                let schemaData = try? JSONEncoder().encode(definition)
                schemas[name] = schemaData.flatMap { String(data: $0, encoding: .utf8) } ?? ""
                loadedResults.append([
                    "tool_name": name,
                    "registration": true,
                    "already_loaded": false
                ])
            } catch {
                errors.append("Failed to load tool '\(name)': \(error.localizedDescription)")
            }
        }

        if loadedResults.isEmpty {
            let err = errors.joined(separator: "; ")
            return (AgentHarness.toolErrorJSON(tool: "tools_load", error: err), nil, err, false)
        }

        let newlyLoaded = loadedResults.filter { ($0["registration"] as? Bool) == true }.map { $0["tool_name"] as? String ?? "" }
        let msg: String
        if newlyLoaded.count == 1, let only = newlyLoaded.first {
            msg = "Tool '\(only)' loaded. You may now call it directly using the provided schema."
        } else if newlyLoaded.count > 1 {
            msg = "Tools \(newlyLoaded.map { "'\($0)'" }.joined(separator: ", ")) loaded. You may now call them directly using the provided schemas."
        } else {
            msg = "All requested tools were already loaded."
        }

        var data: [String: Any] = [
            "tools": loadedResults,
            "loaded_count": harness.loadedTools.count,
            "message": msg
        ]
        if !schemas.isEmpty {
            data["schemas"] = schemas
        }
        if !errors.isEmpty {
            data["errors"] = errors
        }
        let res = AgentHarness.toolSuccessJSON(tool: "tools_load", data: data)
        return (res, msg, errors.isEmpty ? nil : errors.joined(separator: "; "), false)
    }
}

/// Unloads a loaded tool's schema from context, freeing prompt tokens.
public final class ToolUnloadTool: AgentTool {
    public let definition = ToolDefinition(
        name: "tools_unload",
        description: "Removes a loaded tool's schema from context to free prompt tokens. Fundamental tools (shell_run, complete, tools_*) cannot be unloaded. Use sparingly: like tools_load, unloading rewrites the tool block and triggers a context re-prefill, so prefer unloading at task boundaries rather than between individual steps. Use tools_load to re-enable a tool later.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "name": [
                    "type": "string",
                    "description": "Exact name of the loaded tool to unload (e.g. 'web_fetch')."
                ]
            ]),
            "required": AnyCodable(["name"])
        ]
    )

    public func execute(arguments: [String: Any], workingDirectory: URL?, maxOutputLength: Int) async throws -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        guard let rawName = arguments["name"] as? String,
              let name = (rawName as? String)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !name.isEmpty else {
            let err = "Missing or empty 'name' parameter in tools_unload."
            return (AgentHarness.toolErrorJSON(tool: "tools_unload", error: err), nil, err, false)
        }

        let harness = AgentHarness.shared
        do {
            let result = try harness.unloadTool(named: name)
            let msg: String
            if result.wasLoaded {
                msg = "Tool '\(name)' unloaded. Call tools_load to re-enable it later."
            } else {
                msg = "Tool '\(name)' was not loaded; nothing to unload."
            }
            let res = AgentHarness.toolSuccessJSON(tool: "tools_unload", data: [
                "tool_name": name,
                "de_registration": result.wasLoaded,
                "was_loaded": result.wasLoaded,
                "loaded_count": harness.loadedTools.count,
                "message": msg
            ])
            return (res, msg, nil, false)
        } catch {
            let err = "Failed to unload tool '\(name)': \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "tools_unload", error: err), nil, err, false)
        }
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
            parentSessionId: AgentHarness.shared.currentSessionId,
            workingDirectory: workingDirectory
        )

        if runInBackground {
            let msg = "Subagent [\(role)] spawned in background (ID: \(subagent.id.uuidString)). You are responsible for its outcome: when you need its results, call get_subagent_status with wait=true, then present the report to the user in your reply. The user never contacts the subagent directly."
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
            await MainActor.run { subagent.isSummaryRelayed = true }
            let cleanSummary = AgentHarness.truncateText(AgentHarness.sanitizeText(summary), limit: maxOutputLength)
            let res = AgentHarness.toolSuccessJSON(tool: "spawn_subagent", data: [
                "subagent_id": subagent.id.uuidString,
                "role": subagent.role,
                "status": subagent.status.rawValue,
                "duration_seconds": subagent.executionDurationSeconds,
                "summary": cleanSummary,
                "directive": "Present this report to the user now, then answer their original question with it."
            ])
            return (res, cleanSummary, nil, false)
        }
    }
}

public final class GetSubagentStatusTool: AgentTool {
    public let definition = ToolDefinition(
        name: "get_subagent_status",
        description: "Checks the live execution status, step transcript, and final summary of a spawned subagent by its ID. Returns immediately by default; set wait=true to block until the subagent reaches a terminal state (recommended before presenting a background subagent's results to the user).",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "subagent_id": [
                    "type": "string",
                    "description": "The UUID of the spawned subagent."
                ],
                "wait": [
                    "type": "boolean",
                    "description": "Set true to block until the subagent completes (or times out) instead of returning the current snapshot immediately."
                ],
                "timeout_seconds": [
                    "type": "number",
                    "description": "Max seconds to wait when wait=true. Default 120, capped at 300."
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

        // Optional blocking wait for terminal state (used for background subagents).
        let shouldWait = (arguments["wait"] as? Bool) ?? false
        if shouldWait {
            let requestedTimeout = (arguments["timeout_seconds"] as? Double) ?? 120.0
            let timeout = min(max(requestedTimeout, 1.0), 300.0)
            let deadline = Date().addingTimeInterval(timeout)
            while !Task.isCancelled {
                let s = subagent.status
                if s == .completed || s == .failed || s == .cancelled { break }
                if Date() >= deadline { break }
                try? await Task.sleep(nanoseconds: 500_000_000)
            }
        }

        let stepsSummary = subagent.transcript.map {
            "Step \($0.stepIndex): \($0.actionName) (\(String(format: "%.2f", $0.durationSeconds))s)"
        }.joined(separator: "\n")

        let isTerminal = subagent.status == .completed || subagent.status == .failed || subagent.status == .cancelled
        var readable = "Subagent: \(subagent.role) [\(subagent.status.displayName)]\n"
        readable += "Status: \(subagent.liveStatusText)\n"
        readable += "Steps Completed: \(subagent.transcript.count)\n"
        if !subagent.finalSummary.isEmpty {
            readable += "\nSummary:\n\(subagent.finalSummary)\n"
        }
        if isTerminal {
            if subagent.isSummaryRelayed {
                readable += "\nThis report has already been delivered into the conversation; present it to the user and answer their question with it.\n"
            } else {
                readable += "\nIMPORTANT: Relay these results to the user now in your reply — the user cannot see this tool output or the subagent directly.\n"
            }
        } else if !shouldWait {
            readable += "\nStill running. Call again with wait=true to block until it finishes, then present its results to the user.\n"
        }

        var data: [String: Any] = [
            "subagent_id": subagent.id.uuidString,
            "role": subagent.role,
            "status": subagent.status.rawValue,
            "live_status": subagent.liveStatusText,
            "steps_count": subagent.transcript.count,
            "steps_summary": stepsSummary,
            "summary": subagent.finalSummary,
            "duration_seconds": subagent.executionDurationSeconds
        ]
        if isTerminal {
            data["directive"] = "Present this report to the user in your reply, then answer their original question with it. Do not end your turn without delivering these results."
        }

        let cleanOut = AgentHarness.truncateText(AgentHarness.sanitizeText(readable), limit: maxOutputLength)
        let res = AgentHarness.toolSuccessJSON(tool: "get_subagent_status", data: data)

        if isTerminal && !subagent.finalSummary.isEmpty {
            await MainActor.run { subagent.isSummaryRelayed = true }
        }
        return (res, cleanOut, nil, false)
    }
}

public final class SendSubagentMessageTool: AgentTool {
    public let definition = ToolDefinition(
        name: "send_subagent_message",
        description: "Sends a follow-up directive, instruction, or clarification to a running or completed subagent. The subagent consumes directives between execution steps: file paths in a directive are read, shell commands in a directive are run, and the results are folded into its final report.",
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

    /// Every installed tool, whether loaded or not. This is the searchable catalog.
    public private(set) var tools: [String: AgentTool] = [:]
    /// Subset of `tools` currently exposed to the model in the prompt and grammar.
    public private(set) var loadedTools: [String: AgentTool] = [:]
    public var defaultWorkingDirectory: URL? = nil
    /// Token budget for a single tool result (converted to characters conservatively
    /// via `charBudget(forTokenBudget:)` before tools truncate their output).
    public var maxToolOutputLength: Int = 4000
    public var maxAgentSteps: Int = 15
    /// Session that owns the in-flight agent run; stamped onto spawned subagents.
    public var currentSessionId: UUID? = nil

    // MARK: - Web Search Loop Guard
    // Termination for tool loops is structural, not a matter of model capability: `web_search`
    // never sets `isCompleted`, so the only non-cap exit is the model deciding to stop calling
    // tools. These guards escalate repeated identical queries and runaway search budgets so any
    // model (weak or strong) is reeled in.

    public enum WebSearchGuardAction {
        case none
        case answerNowDirective
        case forceSynthesis
    }

    /// Normalized query → number of times it has been issued in the current run.
    public private(set) var webSearchQueryCount: [String: Int] = [:]
    /// Total web_search calls in the current run.
    public private(set) var totalSearchesInRun: Int = 0
    /// Once true, `web_search` is unloaded from the prompt/grammar and every further call is a
    /// hard stop that forces a final synthesis turn.
    public private(set) var webSearchDisabled = false
    /// Action decided for the most recently executed tool call.
    public private(set) var lastSearchGuardAction: WebSearchGuardAction = .none
    /// Max distinct-ish web_search calls allowed per run before disabling further searches.
    public var searchBudgetPerRun: Int = 6
    /// Issuing the same normalized query this many times triggers the repeat guard.
    public var searchRepeatLimit: Int = 2

    /// Resets guard state at the start of a new task and re-exposes `web_search` in case a
    /// previous run unloaded it.
    public func beginAgentSearchGuard() {
        webSearchQueryCount.removeAll()
        totalSearchesInRun = 0
        webSearchDisabled = false
        lastSearchGuardAction = .none
        consecutiveEmptyToolCalls = 0
        if tools["web_search"] != nil && loadedTools["web_search"] == nil {
            _ = try? loadTool(named: "web_search")
        }
    }

    // MARK: - Degenerate Tool Call Guard
    // Small models occasionally emit a structurally valid tool call whose required
    // arguments are all empty (e.g. shell_run with an empty command). Executing it
    // burns a HITL approval prompt and a step on a guaranteed error; tracking the
    // streak lets the harness escalate to a forced synthesis turn instead of looping.

    /// Required argument keys per tool (from each tool's JSON schema).
    public static let requiredValueArgumentKeys: [String: [String]] = [
        "shell_run": ["command"],
        "file_read": ["path"],
        "file_write": ["path", "content"],
        "file_edit": ["path", "target_content", "replacement_content"],
        "find_files": ["pattern"],
        "grep_search": ["query"],
        "web_search": ["query"],
        "web_fetch": ["url"],
        "complete": ["summary"],
        "tools_unload": ["name"],
        "codebase_search": ["query"],
        "spawn_subagent": ["role", "task_description"],
        "get_subagent_status": ["subagent_id"],
        "send_subagent_message": ["subagent_id", "message"],
        "git_commit": ["message"],
        "find_symbol_definition": ["symbol_name"],
        "find_references": ["symbol_name"],
        "lint_diagnostics": ["path"]
    ]

    /// True only when EVERY required argument is missing or whitespace-empty —
    /// i.e. the call is an empty scaffold. Partially-filled calls pass through so
    /// each tool's own validation stays in charge.
    public static func hasEmptyRequiredArguments(toolName: String, arguments: [String: Any]) -> Bool {
        guard let required = requiredValueArgumentKeys[toolName] else { return false }
        for key in required {
            if let v = arguments[key] as? String {
                if !v.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return false }
            } else if arguments[key] != nil {
                return false
            }
        }
        return true
    }

    /// Consecutive tool calls (across agent steps) with empty required arguments.
    public private(set) var consecutiveEmptyToolCalls: Int = 0

    public func recordToolCallValidity(_ hadUsableArgs: Bool) {
        if hadUsableArgs {
            consecutiveEmptyToolCalls = 0
        } else {
            consecutiveEmptyToolCalls += 1
        }
    }

    /// Tools seeded into the prompt on startup — the "fundamental" set.
    /// Kept minimal: every schema here costs prefill tokens on every turn, and each
    /// load/unload event invalidates the KV-cache prefix (full re-prefill). The model
    /// discovers and loads everything else on demand via tools_discover/tools_load.
    public static let coreLoadedToolNames: Set<String> = [
        "shell_run", "complete",
        "tools_discover", "tools_load", "tools_unload"
    ]

    /// Tools the model may never unload through `tools_unload`.
    public static let nonUnloadableToolNames: Set<String> = [
        "shell_run", "complete",
        "tools_discover", "tools_load", "tools_unload"
    ]

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
        registerTool(ToolDiscoverTool())
        registerTool(ToolLoadTool())
        registerTool(ToolUnloadTool())

        // Seed the fundamental core set into the active prompt/grammar.
        for name in Self.coreLoadedToolNames where tools[name] != nil {
            loadedTools[name] = tools[name]
        }
        syncGrammar()
    }

    public func registerTool(_ tool: AgentTool) {
        tools[tool.definition.function.name] = tool
        // Re-registering an already-loaded tool keeps it loaded.
        if loadedTools[tool.definition.function.name] != nil {
            loadedTools[tool.definition.function.name] = tool
        }
        syncGrammar()
    }

    // MARK: - Tool Loading Lifecycle

    public enum ToolLoadError: LocalizedError {
        case unknownTool(String)
        case nonUnloadable(String)

        public var errorDescription: String? {
            switch self {
            case .unknownTool(let name):
                return "Unknown tool '\(name)'. Call tools_discover to see available installed tools."
            case .nonUnloadable(let name):
                return "Tool '\(name)' is a fundamental tool and cannot be unloaded."
            }
        }
    }

    /// Moves an installed tool into the loaded set and re-syncs the grammar sampler.
    @discardableResult
    public func loadTool(named name: String) throws -> ToolDefinition {
        guard let tool = tools[name] else {
            throw ToolLoadError.unknownTool(name)
        }
        let didChange = loadedTools[name] == nil
        loadedTools[name] = tool
        if didChange {
            syncGrammar()
        }
        return tool.definition
    }

    /// Moves a loaded tool back into the catalog and re-syncs the grammar sampler.
    @discardableResult
    public func unloadTool(named name: String) throws -> (name: String, wasLoaded: Bool) {
        guard tools[name] != nil else {
            throw ToolLoadError.unknownTool(name)
        }
        guard !Self.nonUnloadableToolNames.contains(name) else {
            throw ToolLoadError.nonUnloadable(name)
        }
        let wasLoaded = loadedTools.removeValue(forKey: name) != nil
        if wasLoaded {
            syncGrammar()
        }
        return (name, wasLoaded)
    }

    /// Resets the loaded set back to the fundamental core tools.
    public func resetLoadedToolsToCore() {
        loadedTools.removeAll()
        for name in Self.coreLoadedToolNames where tools[name] != nil {
            loadedTools[name] = tools[name]
        }
        syncGrammar()
    }

    private func syncGrammar() {
        GrammarConstrainedSampler.shared.registerTools(availableToolDefinitions)
    }

    public static func isStateChanging(toolName: String) -> Bool {
        return toolName == "file_write" || toolName == "file_edit" || toolName == "shell_run" || toolName == "git_commit"
    }

    /// Definitions currently exposed to the model (drives the prompt AND the grammar mask).
    public var availableToolDefinitions: [ToolDefinition] {
        return loadedTools.keys.sorted().compactMap { loadedTools[$0]?.definition }
    }

    /// Full catalog of every installed tool.
    public var allToolDefinitions: [ToolDefinition] {
        return tools.keys.sorted().compactMap { tools[$0]?.definition }
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

    public func buildSystemPrompt(baseSystem: String, modelName: String? = nil, currentDate: Date = Date(), isLingModel: Bool = false) -> String {
        var cleanBase = baseSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleanBase.isEmpty {
            cleanBase = "You are an expert AI software engineering and reasoning assistant with direct access to local macOS development tools."
        }

        var prompt = ""

        // Inject Current Date & Time at the top if not already provided in baseSystem
        if !cleanBase.contains("Current Date & Time") {
            prompt += AgentHarness.formattedDateTimeContext(date: currentDate) + "\n\n"
        }

        prompt += "# Tools\n\nOnly the following functions are currently loaded and callable:\n\n<tools>\n"
        for tool in availableToolDefinitions {
            if let data = try? JSONEncoder().encode(tool),
               let jsonStr = String(data: data, encoding: .utf8) {
                prompt += jsonStr + "\n"
            }
        }
        prompt += "</tools>\n\n"

        let unloadedCount = tools.count - loadedTools.count
        if unloadedCount > 0 {
            prompt += """
            NOTE: \(unloadedCount) additional tool(s) are installed but NOT loaded, so their schemas are not shown above and they cannot currently be called.

            To discover what else you can do:
            - Call `tools_discover` to list unloaded tools grouped by category (with estimated token cost).
            - Call `tools_load` with a `names` array to register multiple tool schemas into context in ONE call. IMPORTANT: every load/unload rewrites the tool block and triggers a re-prefill of the conversation, so decide up front which tools a task needs and load them all in a single batched call. Avoid loading tools one per step or unloading mid-task.

            """
        }

        prompt += """

        # Subagent Result Delivery Policy

        You are the COORDINATOR of the multi-agent system. Subagents are workers you delegate to — the user only talks to you, and subagents never talk to the user.

        - After spawning a subagent, YOU are responsible for retrieving its final report (via `get_subagent_status`, or the `[SUBAGENT_RESULT]` blocks the harness injects after tool turns) and presenting the results to the user in your own words. Never tell the user to "wait for the subagent", "monitor the drawer", or check anything themselves.
        - A subagent run is NOT finished work. Your turn is only complete once you have relayed the substance of its report (findings, summary, answer) in a reply addressed to the user.
        - Use `get_subagent_status` with `wait: true` to block until a background subagent finishes, then present its results. Use `send_subagent_message` to redirect a subagent that has drifted from its goal.
        - When you relay a subagent's results, always end with a final answer to the user. Do not end your turn after spawning, checking, or acknowledging a subagent — the task is only done when the user has received the outcome.
        """

        prompt += """

        # Tool Result Format & Reading Guidance

        Tool results are delivered as PLAIN TEXT, not JSON: a `[tool_name] status` header line, `key: value` metadata lines, and one or more `--- field ---` sections containing the raw content verbatim. Read the `--- content ---` section as the actual file/page text.

        Truncation markers (`<<<TRUNCATED: ... omitted>>>`, `<<<LINE TRUNCATED>>>`) mark spans cut for context budget — they are harness annotations, NOT file content. When a result reports `next_start_line` or `continuation_hint`, call the tool again with that `start_line` to page through the rest instead of assuming the document is broken.

        Documents exported from editors often contain harmless artifacts that are NOT corruption: image placeholders like `![][image12]` (missing embedded images/formulas), citation footnote digits glued to sentence ends (e.g. "hardware1", "clusters1"), and escaped punctuation (`\\_`, `\\~`, `\\=`, `\\-`). Also ignore `[inline base64 data omitted ...]` placeholders where binary images were stripped. Interpret these as export artifacts and read the text around them; never conclude that a product, file, or document "is corrupted", "is not real", or "does not exist" because of them.
        """

        if isLingModel {
            prompt += """
            If you choose to call a function ONLY reply in the following native format with NO suffix:

            <tool_call>example_function_name
            <arg_key>example_parameter_1</arg_key>
            <arg_value>value_1</arg_value>
            <arg_key>example_parameter_2</arg_key>
            <arg_value>This is the value for the second parameter
            that can span
            multiple lines</arg_value>
            </tool_call>

            <IMPORTANT>
            Reminder:
            - Function calls MUST follow the specified format: the function name directly after <tool_call>, then one <arg_key>/<arg_value> pair per argument, then </tool_call>
            - Required parameters MUST be specified
            - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
            - When the user asks to inspect, read, edit, modify, or process a file, or run terminal commands, you MUST call the function immediately without conversational promises
            - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls

            Web Research Grounding Rules:
            - Live web data supersedes frozen pre-training memory. Never dismiss results as "hallucinated", "corrupted", or "parody" because names, specs, or prices are newer or unfamiliar.
            - Official vendor domains (apple.com, developer.apple.com, github.com, etc.) returned by web_search/web_fetch are authentic ground truth. Synthesize directly without re-searching.
            - Real executive names, benchmark multipliers, lease prices, and hardware specs from web results are genuine — never substitute your own guesses.
            - NEVER invent URLs, article numbers, or chip generations. Only fetch URLs returned by web_search.
            - Distinguish official shipping hardware (vendor domains) from speculative rumors ("rumored", "leaked", "expected").
            - When `web_search` returns a result with is_official_domain: true, that result is verified. When ground_truth_notice is present, synthesize directly — do not loop or re-search.
            </IMPORTANT>
            """
        } else {
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

            Web Research Grounding Rules:
            - Live web data supersedes frozen pre-training memory. Never dismiss results as "hallucinated", "corrupted", or "parody" because names, specs, or prices are newer or unfamiliar.
            - Official vendor domains (apple.com, developer.apple.com, github.com, etc.) returned by web_search/web_fetch are authentic ground truth. Synthesize directly without re-searching.
            - Real executive names, benchmark multipliers, lease prices, and hardware specs from web results are genuine — never substitute your own guesses.
            - NEVER invent URLs, article numbers, or chip generations. Only fetch URLs returned by web_search.
            - Distinguish official shipping hardware (vendor domains) from speculative rumors ("rumored", "leaked", "expected").
            - When `web_search` returns a result with is_official_domain: true, that result is verified. When ground_truth_notice is present, synthesize directly — do not loop or re-search.
            </IMPORTANT>
            """
        }

        if !cleanBase.isEmpty {
            prompt += "\n\n" + cleanBase
        }

        return prompt
    }

    public func formatToolResponseTurn(responses: [String], includeThinkSuffix: Bool = false) -> String {
        var turn = "<|im_start|>user\n"
        for r in responses {
            if let data = r.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let result = json["result"] as? [String: Any],
               let results = result["results"] as? [[String: Any]],
               results.contains(where: { $0["is_official_domain"] as? Bool == true }) {
                turn += "[SYSTEM NOTICE: Verified official vendor domain response. Synthesize directly without re-searching.]\n"
            }
            if let registration = Self.toolRegistrationNotice(for: r) {
                turn += registration + "\n"
            }
            turn += "<tool_response>\n\(Self.renderToolResultForModel(r))\n</tool_response>\n"
        }
        turn += "<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkSuffix {
            turn += "<think>\n"
        }
        return turn
    }

    /// Ling/Bailing-3.0-native equivalent of `formatToolResponseTurn`. Ling's chat template
    /// expects tool results wrapped in `<role>OBSERVATION</role> ... <|role_end|>` followed by
    /// a fresh `<role>ASSISTANT</role>` turn, not ChatML `<|im_start|>user` / `<|im_end|>` tags.
    /// Mixing the two dialects leaves the model without a recognizable assistant turn boundary.
    public func formatLingToolResponseTurn(responses: [String], thinkingEnabled: Bool = true, includeAssistantPrefix: Bool = true) -> String {
        var turn = "<role>OBSERVATION</role>\n"
        for r in responses {
            if let data = r.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let result = json["result"] as? [String: Any],
               let results = result["results"] as? [[String: Any]],
               results.contains(where: { $0["is_official_domain"] as? Bool == true }) {
                turn += "[SYSTEM NOTICE: Verified official vendor domain response. Synthesize directly without re-searching.]\n"
            }
            if let registration = Self.toolRegistrationNotice(for: r) {
                turn += registration + "\n"
            }
            turn += "<tool_response>\n\(Self.renderToolResultForModel(r))\n</tool_response>\n"
        }
        turn += "<|role_end|>"
        guard includeAssistantPrefix else { return turn }
        turn += "\n<role>ASSISTANT</role>"
        turn += thinkingEnabled ? "\n<think>" : "\n<think></think>"
        return turn
    }

    /// Spark 2.5 (DeepSeek-style) tool response turn: <｜start▁of▁sentence｜><|Tool|> <tool_response>{result}</tool_response> per result, ending with <｜end▁of▁sentence｜>
    public func formatSparkToolResponseTurn(responses: [String]) -> String {
        var turn = "<｜start▁of▁sentence｜><|Tool|>"
        for r in responses {
            turn += "<tool_response>\(Self.renderToolResultForModel(r))</tool_response>"
        }
        turn += "<｜end▁of▁sentence｜>"
        return turn
    }

    /// Turns a `tools_load` / `tools_unload` result into an explicit context registration notice
    /// so the newly enabled (or disabled) schema is visible to the model before its next response.
    private static func toolRegistrationNotice(for responseJSON: String) -> String? {
        guard let data = responseJSON.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let result = json["result"] as? [String: Any] else { return nil }

        if result["registration"] as? Bool == true,
           let name = result["tool_name"] as? String {
            if let schema = result["schema"] as? String, !schema.isEmpty {
                return "[TOOL_REGISTRATION] Tool \"\(name)\" is now loaded and may be called directly. Schema:\n\(schema)"
            }
            return "[TOOL_REGISTRATION] Tool \"\(name)\" is now loaded and may be called directly."
        }
        if result["de_registration"] as? Bool == true,
           let name = result["tool_name"] as? String {
            return "[TOOL_DEREGISTRATION] Tool \"\(name)\" has been unloaded and may no longer be called. Call tools_load to re-enable it."
        }
        return nil
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
        turn += "If you intended to take an action, execute it now by emitting a complete tool call like <tool_call><function=file_read><parameter=path>/Users/example.txt</parameter></function></tool_call>. Do not echo the schema template itself. Otherwise, just reply to the user and end your turn.\n"
        turn += "<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkSuffix {
            turn += "<think>\n"
        }
        return turn
    }

    /// Ling/Bailing-3.0-native action-continuation nudge using `<role>HUMAN</role>` /
    /// `<role>ASSISTANT</role>` boundaries and the native `<tool_call>name<arg_key>/<arg_value>`
    /// call shape. Used when the model described an action in prose but never emitted a call.
    public func formatLingActionContinuationTurn(thinkingEnabled: Bool = true) -> String {
        var turn = "<role>HUMAN</role>"
        turn += "If you intended to take an action, execute it now by emitting a complete tool call like <tool_call>file_read\n<arg_key>path</arg_key>\n<arg_value>/Users/example.txt</arg_value>\n</tool_call>. Do not echo the schema template itself. Otherwise, just reply to the user and end your turn."
        turn += "<|role_end|>\n<role>ASSISTANT</role>"
        turn += thinkingEnabled ? "\n<think>" : "\n<think></think>"
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
            let knownToolNames = Set(allToolDefinitions.map { $0.function.name })
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
        // maxOutputLength (and maxToolOutputLength) is a TOKEN budget; tools work in
        // characters, so convert with a conservative chars-per-token estimate.
        let tokenBudget = maxOutputLength ?? self.maxToolOutputLength
        let effectiveMaxLen = AgentHarness.charBudget(forTokenBudget: tokenBudget)

        guard let tool = tools[call.name] else {
            let err = "Unknown tool '\(call.name)'"
            let json = AgentHarness.toolErrorJSON(tool: call.name, error: err)
            let rec = ToolCallRecord(
                name: call.name,
                arguments: strArgs,
                rawArguments: call.rawArguments,
                status: .error,
                output: AgentHarness.renderToolResultForModel(json),
                error: err,
                executionDurationSeconds: CFAbsoluteTimeGetCurrent() - startTime
            )
            return (json, rec, false)
        }

        do {
            let baseResult: (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool)
            if call.name == "web_search" {
                let raw = try await tool.execute(arguments: call.arguments, workingDirectory: wd, maxOutputLength: effectiveMaxLen)
                baseResult = await applyWebSearchLoopGuard(
                    query: (call.arguments["query"] as? String) ?? "",
                    baseJSON: raw.resultJSON,
                    baseStdout: raw.stdout,
                    baseStderr: raw.stderr,
                    baseIsCompleted: raw.isCompleted
                )
            } else {
                baseResult = try await tool.execute(arguments: call.arguments, workingDirectory: wd, maxOutputLength: effectiveMaxLen)
            }
            let json = baseResult.resultJSON
            let stdout = baseResult.stdout
            let stderr = baseResult.stderr
            let isCompleted = baseResult.isCompleted
            let duration = CFAbsoluteTimeGetCurrent() - startTime
            let status: ToolExecutionStatus = (stderr != nil && !stderr!.isEmpty) ? .error : .success

            // Store the exact model-facing rendering so history reconstruction embeds
            // byte-identical context (live turn and rebuilt turn always agree).
            let rendered = AgentHarness.renderToolResultForModel(json)
            let rec = ToolCallRecord(
                name: call.name,
                arguments: strArgs,
                rawArguments: call.rawArguments,
                status: status,
                output: rendered,
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
                output: AgentHarness.renderToolResultForModel(json),
                error: err,
                executionDurationSeconds: duration
            )
            return (json, rec, false)
        }
    }

    // MARK: - Web Search Loop Guard Logic

    /// Runs after the raw `web_search` tool result is produced. Detects repeated identical
    /// queries and runaway search budgets, escalates via a `guardrail` directive embedded in the
    /// tool JSON, unloads `web_search` from the prompt/grammar once the model is refusing to
    /// answer, and finally hard-stops (sets `isCompleted`) so the harness runs a forced synthesis
    /// turn instead of spinning forever.
    private func applyWebSearchLoopGuard(query: String, baseJSON: String, baseStdout: String?, baseStderr: String?, baseIsCompleted: Bool) async -> (resultJSON: String, stdout: String?, stderr: String?, isCompleted: Bool) {
        let clean = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else {
            lastSearchGuardAction = .none
            return (baseJSON, baseStdout, baseStderr, baseIsCompleted)
        }

        let normalized = clean.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let prior = webSearchQueryCount[normalized] ?? 0
        let issuedNow = prior + 1
        webSearchQueryCount[normalized] = issuedNow
        totalSearchesInRun += 1

        var action: WebSearchGuardAction = .none
        var notice: String? = nil
        var isCompleted = baseIsCompleted

        if webSearchDisabled {
            // The model kept searching after being told to stop — end the run decisively.
            action = .forceSynthesis
            isCompleted = true
            notice = "[guardrail] Web search is DISABLED for this task. You already have the complete set of search results in the conversation above. Do NOT issue any further tool calls. Compose your full final answer now, using only the results already gathered."
        } else if issuedNow >= searchRepeatLimit {
            webSearchDisabled = true
            _ = try? unloadTool(named: "web_search")
            action = .answerNowDirective
            notice = "[guardrail] You have already searched this exact query \(issuedNow) times in this task; re-searching cannot produce new information. Web search is now DISABLED. Produce your final answer immediately from the results already in the conversation."
        } else if totalSearchesInRun >= searchBudgetPerRun {
            webSearchDisabled = true
            _ = try? unloadTool(named: "web_search")
            action = .answerNowDirective
            notice = "[guardrail] Search budget exhausted (limit \(searchBudgetPerRun) per task). Web search is now DISABLED. Produce your final answer immediately from the results already in the conversation."
        } else if hasOfficialGroundTruth(in: baseJSON) {
            action = .answerNowDirective
            notice = "[guardrail] These results include verified official vendor ground truth and are authentic and authoritative. Do not search again — compose your final answer now from them."
        }

        lastSearchGuardAction = action

        guard let notice = notice else {
            return (baseJSON, baseStdout, baseStderr, isCompleted)
        }

        var augmented = baseJSON
        if var obj = try? JSONSerialization.jsonObject(with: Data(baseJSON.utf8)) as? [String: Any] {
            var result = (obj["result"] as? [String: Any]) ?? [:]
            result["guardrail"] = notice
            let actionName: String
            switch action {
            case .none: actionName = "none"
            case .answerNowDirective: actionName = "answer_now"
            case .forceSynthesis: actionName = "force_synthesis"
            }
            result["guardrail_action"] = actionName
            obj["result"] = result
            if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted]),
               let str = String(data: data, encoding: .utf8) {
                augmented = str
            }
        }
        return (augmented, baseStdout, baseStderr, isCompleted)
    }

    /// True when the `web_search` result JSON contains at least one official vendor / .gov / .edu
    /// domain, i.e. authentic ground truth that means further searching is pointless.
    private func hasOfficialGroundTruth(in resultJSON: String) -> Bool {
        guard let obj = try? JSONSerialization.jsonObject(with: Data(resultJSON.utf8)) as? [String: Any],
              let result = obj["result"] as? [String: Any],
              let results = result["results"] as? [[String: Any]] else { return false }
        return results.contains { ($0["is_official_domain"] as? Bool) == true }
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

    /// Identifies files that cannot be meaningfully rendered as text, so tools can return
    /// a structured "unsupported type" result instead of a generic decoding error.
    /// Returns a short human-readable kind name (e.g. "PNG image"), or nil when the file
    /// looks like readable text.
    public static func binaryFileKind(at url: URL) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8192), !prefix.isEmpty else { return nil }
        let bytes = [UInt8](prefix)

        let signatures: [([UInt8], String)] = [
            ([0x25, 0x50, 0x44, 0x46], "PDF document"),
            ([0x50, 0x4B, 0x03, 0x04], "ZIP archive (.zip/.docx/.xlsx/.pptx/.jar)"),
            ([0x50, 0x4B, 0x05, 0x06], "ZIP archive (.zip/.docx/.xlsx/.pptx/.jar)"),
            ([0x89, 0x50, 0x4E, 0x47], "PNG image"),
            ([0xFF, 0xD8, 0xFF], "JPEG image"),
            ([0x47, 0x49, 0x46, 0x38], "GIF image"),
            ([0x42, 0x4D], "BMP image"),
            ([0x52, 0x49, 0x46, 0x46], "RIFF container (WAV/AVI/WebP)"),
            ([0x49, 0x44, 0x33], "MP3 audio"),
            ([0x4F, 0x67, 0x67, 0x53], "Ogg media"),
            ([0x66, 0x4C, 0x61, 0x43], "FLAC audio"),
            ([0xD0, 0xCF, 0x11, 0xE0], "Legacy Microsoft Office document (.doc/.xls/.ppt)"),
            ([0x7F, 0x45, 0x4C, 0x46], "ELF binary"),
            ([0x4D, 0x5A], "Windows executable"),
            ([0xCA, 0xFE, 0xBA, 0xBE], "Java class / Mach-O fat binary"),
            ([0xFE, 0xED, 0xFA, 0xCE], "Mach-O binary"),
            ([0xCF, 0xFA, 0xED, 0xFE], "Mach-O binary"),
            ([0x1F, 0x8B], "gzip archive"),
            ([0x42, 0x5A, 0x68], "bzip2 archive"),
            ([0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00], "xz archive"),
            ([0x28, 0xB5, 0x2F, 0xFD], "zstd archive"),
            ([0x04, 0x22, 0x4D, 0x18], "lz4 archive"),
            ([0x53, 0x51, 0x4C, 0x69], "SQLite database"),
            ([0x00, 0x61, 0x73, 0x6D], "WebAssembly binary")
        ]
        for (sig, kind) in signatures where bytes.count >= sig.count && Array(bytes.prefix(sig.count)) == sig {
            return kind
        }

        // Heuristic: NUL bytes, or dense control characters, mean non-text content.
        let sampled = bytes.prefix(4096)
        if sampled.contains(0x00) { return "binary data" }
        let controlCount = sampled.filter { $0 < 0x20 && $0 != 0x09 && $0 != 0x0A && $0 != 0x0D && $0 != 0x1B }.count
        if controlCount > sampled.count / 4 { return "binary data" }
        return nil
    }

    /// Matches inline `data:...;base64,...` payloads of 256+ base64 characters.
    private static let inlineBase64Regex: NSRegularExpression? = {
        try? NSRegularExpression(pattern: "data:[a-zA-Z0-9.+/-]{1,64};base64,([A-Za-z0-9+/=]{256,})", options: [])
    }()

    /// Matches standalone base64 runs of 768+ characters (embedded binary blobs that
    /// no model can read and that otherwise flood the truncation tail with gibberish).
    private static let longBase64Regex: NSRegularExpression? = {
        try? NSRegularExpression(pattern: "[A-Za-z0-9+/=]{768,}", options: [])
    }()

    public static func sanitizeText(_ text: String) -> String {
        var s = text
        s = s.replacingOccurrences(of: "<|im_start|>", with: "[im_start]")
        s = s.replacingOccurrences(of: "<|im_end|>", with: "[im_end]")
        s = s.replacingOccurrences(of: "<tool_call>", with: "[tool_call]")
        s = s.replacingOccurrences(of: "</tool_call>", with: "[/tool_call]")
        s = s.replacingOccurrences(of: "<tool_response>", with: "[tool_response]")
        s = s.replacingOccurrences(of: "</tool_response>", with: "[/tool_response]")
        // Scrub inline base64 (data URIs first, then any remaining standalone run)
        // so binary payloads become compact placeholders instead of garbling context.
        if let re = Self.inlineBase64Regex {
            let ns = NSMutableString(string: s)
            for m in re.matches(in: s as String, options: [], range: NSRange(location: 0, length: ns.length)).reversed() {
                let b64Len = m.range(at: 1).length
                ns.replaceCharacters(in: m.range, with: "[inline base64 data omitted (\(b64Len) chars)]")
            }
            s = ns as String
        }
        if let re2 = Self.longBase64Regex {
            let ns2 = NSMutableString(string: s)
            for m2 in re2.matches(in: s as String, options: [], range: NSRange(location: 0, length: ns2.length)).reversed() {
                ns2.replaceCharacters(in: m2.range, with: "[long base64 run omitted (\(m2.range.length) chars)]")
            }
            s = ns2 as String
        }
        return s
    }

    /// Conservative characters-per-token estimate for mixed prose/code output.
    /// Used to convert the token-based tool-output budget into a character budget.
    public static func charBudget(forTokenBudget tokens: Int) -> Int {
        max(600, tokens * 5 / 2)
    }

    private static let htmlScriptStyleRegex = try? NSRegularExpression(
        pattern: "<(?i:script|style|noscript|svg|template|head)(?=[\\s/>])[^>]*>.*?</(?i:script|style|noscript|svg|template|head)\\s*>|<!--.*?-->|<(?i:script|style|noscript|svg|template|head)(?=[\\s/>])[^>]*/>",
        options: [.dotMatchesLineSeparators]
    )
    private static let htmlTagRegex = try? NSRegularExpression(pattern: "<[^<>]{0,400}>", options: [.dotMatchesLineSeparators])
    private static let htmlNumericEntityRegex = try? NSRegularExpression(pattern: "&#(x?)([0-9a-fA-F]+);", options: [])
    private static let htmlNoiseLineRegex = try? NSRegularExpression(
        pattern: "(?m)^[ \\t]*(?:(?:skip to(?: main)? content)|share|menu|search(?: button)?|sign (?:in|up)|log (?:in|out)|advertisement)[ \\t]*$",
        options: [.caseInsensitive]
    )
    private static let htmlAnchorRegex = try? NSRegularExpression(
        pattern: "<a\\b[^>]*>(.*?)</a>",
        options: [.dotMatchesLineSeparators, .caseInsensitive]
    )
    /// Boilerplate phrases are removed only when they constitute an ENTIRE link
    /// span (navigation), never as words inside prose, commands, or code. The
    /// anchor-delimiting step tags link text with bracket sentinels the generic
    /// tag stripper cannot produce.
    private static let htmlNoiseLinkRegex = try? NSRegularExpression(
        pattern: "⟦\\s*(?:(?:skip to(?: main)? content)|share|menu|search(?: button)?|sign (?:in|up)|log (?:in|out)|advertisement)\\s*⟧",
        options: [.caseInsensitive]
    )

    private static let htmlVocabRegex = try? NSRegularExpression(
        pattern: "<(?i:div|span|p[\\s>]|a[\\s>]|li[\\s>]|ul|ol|meta|link|script|style|table|img|br|h[1-6][\\s>]|header|footer|nav|section|form|input|button)",
        options: []
    )

    /// Heuristic HTML detection for tool output. A doctype/html/body marker is
    /// definitive; an XML declaration opts out; otherwise output must show both
    /// tag density and HTML-vocabulary tags, so JSON, XML data, and shell text
    /// never reach the conversion path.
    public static func looksLikeHTML(_ text: String) -> Bool {
        if text.range(of: "<!doctype html", options: [.regularExpression, .caseInsensitive]) != nil { return true }
        if text.range(of: "<html[\\s>]", options: [.regularExpression, .caseInsensitive]) != nil { return true }
        if text.range(of: "<body[\\s>]", options: [.regularExpression, .caseInsensitive]) != nil { return true }
        guard text.utf8.count > 512 else { return false }
        if text.range(of: "<\\?xml", options: [.regularExpression, .caseInsensitive]) != nil { return false }
        guard let re = htmlTagRegex, let vocab = htmlVocabRegex else { return false }
        let tagCount = re.numberOfMatches(in: text, options: [], range: NSRange(text.startIndex..., in: text))
        guard tagCount >= 12 else { return false }
        return vocab.firstMatch(in: text, options: [], range: NSRange(text.startIndex..., in: text)) != nil
    }

    /// Converts HTML to compact plain text for the model context: drops
    /// script/style/head blocks, turns block-level boundaries into newlines,
    /// strips remaining tags, decodes entities, and collapses whitespace.
    /// Cuts a fetched web page from ~29KB of markup to its readable content.
    public static func htmlToPlainText(_ html: String) -> String {
        var s = html

        // 1. Drop invisible/bulk blocks (scripts, styles, metadata, comments).
        if let re = htmlScriptStyleRegex {
            s = re.stringByReplacingMatches(in: s, options: [], range: NSRange(s.startIndex..., in: s), withTemplate: " ")
        }

        // 2. Mark navigation link spans before tags are stripped, so boilerplate
        //    removal can be restricted to link text instead of matching words
        //    anywhere in the document (prose and code stay untouched).
        if let re = htmlAnchorRegex {
            s = re.stringByReplacingMatches(in: s, options: [], range: NSRange(s.startIndex..., in: s), withTemplate: " ⟦$1⟧ ")
        }

        // 3. Newlines at block boundaries so text does not glue together.
        let blockClosers = ["</p>", "</div>", "</li>", "</tr>", "</ul>", "</ol>", "</table>", "</section>", "</article>", "</header>", "</footer>", "</nav>", "</blockquote>", "</pre>", "</h1>", "</h2>", "</h3>", "</h4>", "</h5>", "</h6>", "</dd>", "</dt>", "<br>", "<br/>", "<br />", "<hr>", "<hr/>", "<hr />"]
        for closer in blockClosers {
            s = s.replacingOccurrences(of: closer, with: "\n", options: .caseInsensitive)
        }

        // 4. Strip every remaining tag (replaced with a space so inline
        //    siblings like nav links stay separate words).
        if let re = htmlTagRegex {
            s = re.stringByReplacingMatches(in: s, options: [], range: NSRange(s.startIndex..., in: s), withTemplate: " ")
        }

        // 5. Decode entities: numeric first (covers the long tail), then the
        //    common named set.
        if let re = htmlNumericEntityRegex {
            let ns = NSMutableString(string: s)
            for m in re.matches(in: s, options: [], range: NSRange(s.startIndex..., in: s)).reversed() {
                let isHex = m.range(at: 1).length > 0
                let digitsRange = m.range(at: 2)
                guard let digits = Range(digitsRange, in: s), let value = UInt32(String(s[digits]), radix: isHex ? 16 : 10),
                      let scalar = Unicode.Scalar(value) else { continue }
                ns.replaceCharacters(in: m.range, with: String(Character(scalar)))
            }
            s = ns as String
        }
        let named: [String: String] = [
            "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"", "&apos;": "'",
            "&nbsp;": " ", "&copy;": "©", "&reg;": "®", "&trade;": "™",
            "&mdash;": "—", "&ndash;": "–", "&hellip;": "…", "&middot;": "·",
            "&rsquo;": "'", "&lsquo;": "'", "&rdquo;": "\"", "&ldquo;": "\"",
            "&laquo;": "«", "&raquo;": "»", "&deg;": "°", "&times;": "×"
        ]
        for (entity, replacement) in named {
            s = s.replacingOccurrences(of: entity, with: replacement)
        }

        // 6. Drop boilerplate: exact-match link spans, then common single-word
        //    noise lines, then collapse whitespace runs.
        if let re = htmlNoiseLinkRegex {
            s = re.stringByReplacingMatches(in: s, options: [], range: NSRange(s.startIndex..., in: s), withTemplate: " ")
        }
        s = s.replacingOccurrences(of: "⟦", with: " ").replacingOccurrences(of: "⟧", with: " ")
        if let re = htmlNoiseLineRegex {
            s = re.stringByReplacingMatches(in: s, options: [], range: NSRange(s.startIndex..., in: s), withTemplate: "")
        }
        var collapsed = ""
        var blankRun = 0
        for line in s.components(separatedBy: "\n") {
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)
            if trimmedLine.isEmpty {
                blankRun += 1
                if blankRun <= 1 { collapsed += "\n" }
            } else {
                blankRun = 0
                collapsed += trimmedLine + "\n"
            }
        }
        return collapsed.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Truncates oversized text for the model context. The head keeps as much as the
    /// budget allows (cut at a line boundary when possible), a short tail preserves
    /// the end of the output, and the splice is wrapped in an unambiguous marker that
    /// cannot be mistaken for file content.
    public static func truncateText(_ text: String, limit: Int) -> String {
        let byteLimit = max(200, limit)
        guard text.utf8.count > byteLimit else { return text }

        let headBudget = max(100, byteLimit - 400)
        let tailBudget = min(300, max(0, byteLimit / 4))
        let utf8 = text.utf8

        // Head: end after the last newline within budget (single-line fallback: hard cut).
        let hardHeadEnd = utf8.index(utf8.startIndex, offsetBy: min(headBudget, utf8.count), limitedBy: utf8.endIndex) ?? utf8.endIndex
        var headEnd = hardHeadEnd
        if headEnd < utf8.endIndex {
            var search = headEnd
            while search > utf8.startIndex {
                search = utf8.index(before: search)
                if utf8[search] == 0x0A {
                    headEnd = utf8.index(after: search)
                    break
                }
            }
        }

        // Tail: start right after the first newline at/after the tail lower bound.
        var tailStart = utf8.endIndex
        if tailBudget > 0 {
            let lower = utf8.index(utf8.endIndex, offsetBy: -min(tailBudget, utf8.count), limitedBy: utf8.startIndex) ?? utf8.startIndex
            var search = lower
            while search < utf8.endIndex {
                if utf8[search] == 0x0A {
                    search = utf8.index(after: search)
                    break
                }
                search = utf8.index(after: search)
            }
            tailStart = search
        }

        let head = String(decoding: utf8[utf8.startIndex..<headEnd], as: UTF8.self)
        let tail = String(decoding: utf8[tailStart..<utf8.endIndex], as: UTF8.self)

        if headEnd >= tailStart {
            let omitted = text.utf8.count - head.utf8.count
            return head + "\n<<<TRUNCATED: \(omitted) characters omitted>>>"
        }
        let omitted = text.utf8.count - head.utf8.count - tail.utf8.count
        return head + "\n<<<TRUNCATED: \(omitted) characters omitted>>>\n" + tail
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

    public static func cleanHTMLStructure(_ html: String) -> String {
        return WebFetchTool.cleanHTMLStructure(html)
    }

    public static func extractStructuredSections(from markdown: String) -> [[String: String]] {
        return WebFetchTool.extractStructuredSections(from: markdown)
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

    // MARK: - Tool Result Rendering (model-facing observations)

    /// Renders a tool result JSON into the plain, human-readable form the model should see.
    /// Observations are emitted as text rather than JSON so file content (HTML, code, CSV)
    /// reaches the model without `\"` / `\/` / literal-`\n` escaping — the leading cause of
    /// "garbled" reads on anything quote-dense.
    ///
    /// Idempotent: input that is not a tool-result JSON object (already-rendered text,
    /// plain prose) is returned unchanged, so results can be stored rendered and re-embedded.
    public static func renderToolResultForModel(_ resultJSON: String) -> String {
        guard let data = resultJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let toolName = obj["tool"] as? String, !toolName.isEmpty else {
            return resultJSON
        }
        let status = (obj["status"] as? String) ?? "success"
        var lines: [String] = []

        guard let result = obj["result"] as? [String: Any] else {
            if status == "error" {
                lines.append("[\(toolName)] ERROR: \(obj["error"] as? String ?? "unknown error")")
            } else {
                lines.append("[\(toolName)] \(status)")
            }
            for (k, v) in obj.sorted(by: { $0.key < $1.key }) where k != "status" && k != "tool" && k != "error" {
                Self.appendRendered(key: k, value: v, indent: 0, to: &lines)
            }
            return lines.joined(separator: "\n")
        }

        if status == "error" {
            lines.append("[\(toolName)] ERROR: \(obj["error"] as? String ?? result["error"] as? String ?? "unknown error")")
        } else {
            lines.append("[\(toolName)] success")
        }
        // Metadata first (short values), long text bodies last, in stable key order.
        let bodyKeys = result.keys.filter { Self.isLongTextValue(result[$0]) }.sorted()
        let metaKeys = result.keys.filter { !Self.isLongTextValue(result[$0]) }.sorted()
        for k in metaKeys {
            Self.appendRendered(key: k, value: result[k], indent: 0, to: &lines)
        }
        for k in bodyKeys {
            guard let v = result[k] else { continue }
            lines.append("")
            lines.append("--- \(k) ---")
            Self.appendLongBody(value: v, indent: 0, to: &lines)
        }
        return lines.joined(separator: "\n")
    }

    private static func isLongTextValue(_ value: Any?) -> Bool {
        guard let s = value as? String else { return false }
        return s.contains("\n") || s.count > 240
    }

    private static func appendRendered(key: String, value: Any?, indent: Int, to lines: inout [String]) {
        let pad = String(repeating: "  ", count: indent)
        switch value {
        case .none:
            lines.append("\(pad)\(key): (nil)")
        case let n as NSNumber:
            // NSNumber bridges BOTH integers and booleans; distinguish via CFBoolean so
            // `start_line: 1` does not render as `start_line: true`.
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                lines.append("\(pad)\(key): \(n.boolValue ? "true" : "false")")
            } else {
                lines.append("\(pad)\(key): \(n.stringValue)")
            }
        case let s as String:
            if s.contains("\n") || s.count > 240 {
                lines.append("\(pad)\(key):")
                Self.appendLongBody(value: s, indent: indent + 1, to: &lines)
            } else {
                lines.append("\(pad)\(key): \(s.isEmpty ? "(empty)" : s)")
            }
        case let a as [Any]:
            if a.isEmpty {
                lines.append("\(pad)\(key): (none)")
            } else if a.allSatisfy({ Self.isCompactScalar($0) }) {
                lines.append("\(pad)\(key): \(a.map { Self.renderCompact($0) }.joined(separator: ", "))")
            } else if let dicts = a as? [[String: Any]] {
                for (i, d) in dicts.enumerated() {
                    lines.append("\(pad)\(key) #\(i + 1):")
                    for (k, v) in d.sorted(by: { $0.key < $1.key }) {
                        Self.appendRendered(key: k, value: v, indent: indent + 1, to: &lines)
                    }
                }
            } else {
                lines.append("\(pad)\(key): \(Self.renderCompact(a))")
            }
        case let d as [String: Any]:
            lines.append("\(pad)\(key):")
            for (k, v) in d.sorted(by: { $0.key < $1.key }) {
                Self.appendRendered(key: k, value: v, indent: indent + 1, to: &lines)
            }
        default:
            lines.append("\(pad)\(key): \(String(describing: value))")
        }
    }

    private static func appendLongBody(value: Any, indent: Int, to lines: inout [String]) {
        let pad = String(repeating: "  ", count: indent)
        if let s = value as? String {
            let clean = Self.sanitizeText(s)
            if indent == 0 {
                lines.append(clean)
            } else {
                lines.append(clean.components(separatedBy: "\n").map { pad + $0 }.joined(separator: "\n"))
            }
        } else {
            lines.append(pad + Self.renderCompact(value))
        }
    }

    private static func isCompactScalar(_ value: Any) -> Bool {
        value is String || value is NSNumber
    }

    private static func renderCompact(_ value: Any) -> String {
        switch value {
        case let s as String:
            return s
        case let n as NSNumber:
            if CFGetTypeID(n) == CFBooleanGetTypeID() {
                return n.boolValue ? "true" : "false"
            }
            return n.stringValue
        default:
            if JSONSerialization.isValidJSONObject([value]),
               let data = try? JSONSerialization.data(withJSONObject: [value], options: [.sortedKeys, .withoutEscapingSlashes]),
               let str = String(data: data, encoding: .utf8) {
                return truncateText(String(str.dropFirst(1).dropLast(2)), limit: 2000)
            }
            return String(describing: value)
        }
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
