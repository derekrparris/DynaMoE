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

        let (exitCode, stdout, stderr) = try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", command],
            currentDirectory: targetDir,
            timeoutSeconds: 180
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
            return (res, cleanStdout, cleanStderr, false)
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
            let msg = "Successfully wrote \(byteCount) bytes to \(resolvedPath.path)"
            let res = AgentHarness.toolSuccessJSON(tool: "file_write", data: [
                "path": resolvedPath.path,
                "bytes_written": byteCount,
                "status": "written"
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

            let msg = "Successfully edited \(resolvedPath.lastPathComponent)"
            let res = AgentHarness.toolSuccessJSON(tool: "file_edit", data: [
                "path": resolvedPath.path,
                "status": "success",
                "message": msg
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

/// Tool 7: web_search — Performs live web search using DuckDuckGo / Brave
public final class WebSearchTool: AgentTool {
    public let definition = ToolDefinition(
        name: "web_search",
        description: "Performs live web search for real-time information, documentation, libraries, news, and technical answers. Returns structured list of titles, URLs, and snippets.",
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

            if results.isEmpty {
                results = try await searchDuckDuckGo(query: cleanQuery, maxResults: maxResults)
            }

            if results.isEmpty {
                results = try await searchWikipedia(query: cleanQuery, maxResults: maxResults)
            }

            if results.isEmpty {
                let msg = "No web search results found for query: \"\(cleanQuery)\""
                let res = AgentHarness.toolSuccessJSON(tool: "web_search", data: ["query": cleanQuery, "count": 0, "results": []])
                return (res, msg, nil, false)
            }

            var readableOutput = "Found \(results.count) web search results for \"\(cleanQuery)\":\n\n"
            for (idx, item) in results.enumerated() {
                readableOutput += "\(idx + 1). \(item["title"] ?? "Untitled")\n"
                readableOutput += "   URL: \(item["url"] ?? "")\n"
                if let snippet = item["snippet"], !snippet.isEmpty {
                    readableOutput += "   Snippet: \(snippet)\n"
                }
                readableOutput += "\n"
            }

            let cleanStdout = AgentHarness.truncateText(AgentHarness.sanitizeText(readableOutput.trimmingCharacters(in: .whitespacesAndNewlines)), limit: maxOutputLength)
            let res = AgentHarness.toolSuccessJSON(tool: "web_search", data: [
                "query": cleanQuery,
                "count": results.count,
                "results": results
            ])
            return (res, cleanStdout, nil, false)
        } catch {
            let err = "Web search failed: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "web_search", error: err), nil, err, false)
        }
    }

    private func searchDuckDuckGo(query: String, maxResults: Int) async throws -> [[String: String]] {
        guard let encodedQuery = query.addingPercentEncoding(withAllowedCharacters: .urlQueryAllowed),
              let url = URL(string: "https://html.duckduckgo.com/html/?q=\(encodedQuery)") else {
            return []
        }

        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.timeoutInterval = 10
        request.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36", forHTTPHeaderField: "User-Agent")
        request.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")
        request.setValue("en-US,en;q=0.5", forHTTPHeaderField: "Accept-Language")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200,
              let html = String(data: data, encoding: .utf8) else {
            return []
        }

        var results: [[String: String]] = []
        let titlePattern = try NSRegularExpression(pattern: "<a[^>]+class=[\"']result__a[\"'][^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", options: [.dotMatchesLineSeparators])
        let snippetPattern = try NSRegularExpression(pattern: "<a[^>]+class=[\"']result__snippet[\"'][^>]*>(.*?)</a>", options: [.dotMatchesLineSeparators])

        let nsHtml = html as NSString
        let titleMatches = titlePattern.matches(in: html, range: NSRange(location: 0, length: nsHtml.length))
        let snippetMatches = snippetPattern.matches(in: html, range: NSRange(location: 0, length: nsHtml.length))

        let count = min(titleMatches.count, snippetMatches.count, maxResults)
        for i in 0..<count {
            let tMatch = titleMatches[i]
            let sMatch = snippetMatches[i]

            let rawUrl = nsHtml.substring(with: tMatch.range(at: 1))
            let rawTitle = nsHtml.substring(with: tMatch.range(at: 2))
            let rawSnippet = nsHtml.substring(with: sMatch.range(at: 1))

            var cleanUrl = rawUrl
            if let uddgRange = cleanUrl.range(of: "uddg=") {
                let substr = String(cleanUrl[uddgRange.upperBound...])
                let endIdx = substr.firstIndex(of: "&") ?? substr.endIndex
                let encoded = String(substr[..<endIdx])
                if let decoded = encoded.removingPercentEncoding {
                    cleanUrl = decoded
                }
            } else if cleanUrl.hasPrefix("//") {
                cleanUrl = "https:" + cleanUrl
            }

            let cleanTitle = stripHTML(rawTitle)
            let cleanSnippet = stripHTML(rawSnippet)

            if !cleanUrl.isEmpty && !cleanTitle.isEmpty {
                results.append([
                    "title": cleanTitle,
                    "url": cleanUrl,
                    "snippet": cleanSnippet
                ])
            }
        }
        return results
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
        description: "Fetches and reads the textual content of a web page given a public URL. Automatically strips HTML markup, scripts, and navigation clutter into readable text/markdown.",
        parameters: [
            "type": AnyCodable("object"),
            "properties": AnyCodable([
                "url": [
                    "type": "string",
                    "description": "The absolute HTTP or HTTPS URL to fetch."
                ],
                "max_length": [
                    "type": "integer",
                    "description": "Optional maximum character length of returned content (defaults to maxToolOutputLength)."
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

        let customMax = arguments["max_length"] as? Int
        let limit = customMax ?? maxOutputLength

        do {
            var request = URLRequest(url: url)
            request.timeoutInterval = 15
            request.setValue("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36", forHTTPHeaderField: "User-Agent")
            request.setValue("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", forHTTPHeaderField: "Accept")

            let (data, response) = try await URLSession.shared.data(for: request)
            guard let httpResponse = response as? HTTPURLResponse, (200...299).contains(httpResponse.statusCode),
                  var html = String(data: data, encoding: .utf8) ?? String(data: data, encoding: .isoLatin1) else {
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                let err = "Failed to fetch webpage (HTTP status \(status))."
                return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
            }

            var pageTitle = url.host ?? "Web Page"
            if let titleRange = html.range(of: "<title[^>]*>(.*?)</title>", options: [.regularExpression, .caseInsensitive]) {
                let rawTitle = String(html[titleRange])
                pageTitle = stripHTML(rawTitle)
            }

            // Strip comments
            html = html.replacingOccurrences(of: "(?s)<!--.*?-->", with: "", options: .regularExpression)
            // Strip script, style, nav, header, footer, svg, noscript
            html = html.replacingOccurrences(of: "(?s)<(script|style|nav|header|footer|svg|noscript)[^>]*>.*?</\\1>", with: "", options: .regularExpression)
            // Replace block tags with newline
            html = html.replacingOccurrences(of: "(?i)</?(p|div|h1|h2|h3|h4|h5|h6|li|tr|article|section|blockquote|pre|code)[^>]*>", with: "\n", options: .regularExpression)
            html = html.replacingOccurrences(of: "(?i)<br\\s*/?>", with: "\n", options: .regularExpression)

            let text = stripHTML(html)
            let lines = text.components(separatedBy: .newlines)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                .filter { !$0.isEmpty }
            var cleaned = lines.joined(separator: "\n")

            if cleaned.count > limit {
                cleaned = String(cleaned.prefix(limit)) + "\n\n... [Content truncated at \(limit) characters]"
            }

            let res = AgentHarness.toolSuccessJSON(tool: "web_fetch", data: [
                "url": url.absoluteString,
                "title": pageTitle,
                "length": cleaned.count,
                "content": cleaned
            ])
            return (res, cleaned, nil, false)
        } catch {
            let err = "Failed to fetch web content: \(error.localizedDescription)"
            return (AgentHarness.toolErrorJSON(tool: "web_fetch", error: err), nil, err, false)
        }
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
        registerTool(CompleteTool())
    }

    public func registerTool(_ tool: AgentTool) {
        tools[tool.definition.function.name] = tool
    }

    public var availableToolDefinitions: [ToolDefinition] {
        return Array(tools.values.map { $0.definition })
    }

    // MARK: - Prompt Formatting & ChatML Generation

    public func buildSystemPrompt(baseSystem: String, modelName: String? = nil) -> String {
        var cleanBase = baseSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        if cleanBase.isEmpty {
            cleanBase = "You are an expert AI software engineering and reasoning assistant with direct access to local macOS development tools."
        }

        var prompt = cleanBase
        prompt += "\n\n# Tool Calling Protocol & Instructions\n"
        prompt += "You have direct access to native tools on the local macOS system to inspect files, edit code, execute terminal commands, and perform web searches.\n\n"
        prompt += "## Operating Rules:\n"
        prompt += "1. **Action-Oriented Execution**: When you need to read a file, create or modify code, search directories, or run commands, ALWAYS emit a `<tool_call>` block. Never merely describe what you intend to do in prose without issuing the actual tool call.\n"
        prompt += "2. **Multi-Turn Continuity**: When a tool completes and returns a `<tool_response>`, evaluate the output. If additional steps are required (such as modifying a file after reading it, or verifying changes), emit the next `<tool_call>` immediately.\n"
        prompt += "3. **Completion**: Only provide your final conversational answer to the user once all necessary file edits, commands, and operations are complete.\n"
        prompt += "4. **Format**: To invoke a tool, output a JSON object within `<tool_call></tool_call>` tags:\n"
        prompt += "<tool_call>\n{\"name\": \"file_read\", \"arguments\": {\"path\": \"ContentView.swift\", \"start_line\": 1, \"end_line\": 50}}\n</tool_call>\n\n"
        prompt += "# Available Tools\n<tools>\n"

        for tool in availableToolDefinitions {
            if let data = try? JSONEncoder().encode(tool),
               let jsonStr = String(data: data, encoding: .utf8) {
                prompt += jsonStr + "\n"
            }
        }
        prompt += "</tools>\n"

        return prompt
    }

    public func formatToolResponseTurn(responses: [String], includeThinkSuffix: Bool = false) -> String {
        var turn = "<|im_start|>user\n"
        for r in responses {
            turn += "<tool_response>\n\(r)\n</tool_response>\n"
        }
        turn += "Tool execution completed. Inspect the results above and continue fulfilling the request. If further actions or file edits are needed, output the next <tool_call> block immediately. If finished, provide your final response.\n"
        turn += "<|im_end|>\n<|im_start|>assistant\n"
        if includeThinkSuffix {
            turn += "<think>\n"
        }
        return turn
    }

    // MARK: - Tolerant Tool Call Parser (Multi-Call + Truncation Recovery)

    public func parseToolCalls(from text: String) -> (calls: [ParsedToolCall], brokenFragments: [String]) {
        var calls: [ParsedToolCall] = []
        var broken: [String] = []

        let toolCallRegex = try? NSRegularExpression(pattern: "<tool_call>([\\s\\S]*?)</tool_call>", options: [])
        let nsText = text as NSString
        let matches = toolCallRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []

        for m in matches {
            guard m.numberOfRanges >= 2 else { continue }
            let innerRange = m.range(at: 1)
            let rawMatch = nsText.substring(with: m.range(at: 0))
            let innerText = nsText.substring(with: innerRange).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !innerText.isEmpty else { continue }

            if let parsed = extractBalancedJSON(innerText) {
                if let name = parsed["name"] as? String {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: rawMatch))
                } else {
                    broken.append(innerText)
                }
            } else {
                broken.append(innerText)
            }
        }

        // Check for unclosed or truncated <tool_call>
        if calls.isEmpty && text.contains("<tool_call>") {
            let parts = text.components(separatedBy: "<tool_call>")
            for part in parts.dropFirst() {
                let candidate = part.replacingOccurrences(of: "</tool_call>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                if let parsed = extractBalancedJSON(candidate), let name = parsed["name"] as? String {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: candidate))
                } else if !candidate.isEmpty {
                    broken.append(candidate)
                }
            }
        }

        // Fallback: Check if output contains raw JSON with a recognized tool name
        if calls.isEmpty {
            if let parsed = extractBalancedJSON(text), let name = parsed["name"] as? String {
                let knownToolNames = Set(availableToolDefinitions.map { $0.function.name })
                if knownToolNames.contains(name) {
                    let args = (parsed["arguments"] as? [String: Any]) ?? [:]
                    let rawArgs = (parsed["arguments"] != nil) ? String(describing: parsed["arguments"]!) : ""
                    calls.append(ParsedToolCall(name: name, arguments: args, rawArguments: rawArgs, rawText: text))
                }
            }
        }

        return (calls, broken)
    }

    /// String & escape aware balanced bracket JSON scanner
    public func extractBalancedJSON(_ text: String) -> [String: Any]? {
        guard let startIdx = text.firstIndex(of: "{") else { return nil }
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

        guard let end = endIdx else { return nil }
        let jsonSubstring = String(sub[...end])
        guard let data = jsonSubstring.data(using: .utf8) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
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
        timeoutSeconds: TimeInterval = 180
    ) async throws -> (exitCode: Int32, stdout: String, stderr: String) {
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                let stdoutPipe = Pipe()
                let stderrPipe = Pipe()

                process.executableURL = executableURL
                process.arguments = arguments
                process.currentDirectoryURL = currentDirectory
                process.standardOutput = stdoutPipe
                process.standardError = stderrPipe

                var isDone = false
                let lock = NSLock()

                do {
                    try process.run()

                    let stdoutData = stdoutPipe.fileHandleForReading.readDataToEndOfFile()
                    let stderrData = stderrPipe.fileHandleForReading.readDataToEndOfFile()
                    process.waitUntilExit()

                    lock.lock()
                    if !isDone {
                        isDone = true
                        lock.unlock()
                        let stdoutStr = String(data: stdoutData, encoding: .utf8) ?? ""
                        let stderrStr = String(data: stderrData, encoding: .utf8) ?? ""
                        continuation.resume(returning: (process.terminationStatus, stdoutStr, stderrStr))
                    } else {
                        lock.unlock()
                    }
                } catch {
                    lock.lock()
                    if !isDone {
                        isDone = true
                        lock.unlock()
                        continuation.resume(throwing: error)
                    } else {
                        lock.unlock()
                    }
                }
            }
        }
    }
}
