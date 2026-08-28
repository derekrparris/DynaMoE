//
//  AgentHarness.swift
//  DynaMoE
//
//  Stateful Multi-Turn Agent Tool Harness for Qwen & ChatML Models
//  Preserves exact ChatML syntax (<tool_call>, <|im_start|>, <|im_end|>) and maintains
//  persistent KV cache across tool turns without quadratic re-prefill.
//

import Foundation

public struct ToolDefinition: Codable {
    public let type: String
    public let function: ToolFunction

    public struct ToolFunction: Codable {
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
    public let rawText: String

    public static func == (lhs: ParsedToolCall, rhs: ParsedToolCall) -> Bool {
        return lhs.name == rhs.name && lhs.rawText == rhs.rawText
    }
}

public struct AnyCodable: Codable {
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
        }
    }
}

public final class AgentHarness {
    public static let shared = AgentHarness()

    public static let defaultSystemPrompt = """
    You are a helpful assistant with access to local system tools on macOS.
    Whenever you need to interact with files or execute commands, generate a <tool_call> block.
    """

    public private(set) var availableTools: [ToolDefinition] = []

    private init() {
        registerDefaultTools()
    }

    private func registerDefaultTools() {
        let shellTool = ToolDefinition(
            name: "shell_run",
            description: "Execute a shell command on macOS and return its stdout and stderr.",
            parameters: [
                "type": AnyCodable("object"),
                "properties": AnyCodable([
                    "command": [
                        "type": "string",
                        "description": "The exact shell command line string to execute."
                    ]
                ]),
                "required": AnyCodable(["command"])
            ]
        )

        let readTool = ToolDefinition(
            name: "file_read",
            description: "Read text contents of a file on the local filesystem.",
            parameters: [
                "type": AnyCodable("object"),
                "properties": AnyCodable([
                    "path": [
                        "type": "string",
                        "description": "Absolute path to the file to read."
                    ]
                ]),
                "required": AnyCodable(["path"])
            ]
        )

        availableTools = [shellTool, readTool]
    }

    // MARK: - Prompt Formatting (ChatML)

    /// Builds standard ChatML formatted initial prompt including system prompt and tool schemas
    public func formatInitialChatML(system: String = defaultSystemPrompt, userMessage: String) -> String {
        var prompt = "<|im_start|>system\n\(system)\n"
        if !availableTools.isEmpty {
            prompt += "\n# Tools\nYou have access to the following tools:\n```json\n"
            if let data = try? JSONEncoder().encode(availableTools),
               let jsonStr = String(data: data, encoding: .utf8) {
                prompt += jsonStr
            }
            prompt += "\n```\n"
        }
        prompt += "<|im_end|>\n"
        prompt += "<|im_start|>user\n\(userMessage)<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt
    }

    /// Formats a tool execution response as an incremental continuation turn
    public func formatToolResponseTurn(toolName: String, response: String) -> String {
        return "<tool_response>\n\(response)\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
    }

    // MARK: - Tool Call Extraction & Execution

    /// Extracts tool call JSON blocks from generated model output
    public func parseToolCalls(from text: String) -> [ParsedToolCall] {
        var results: [ParsedToolCall] = []
        let pattern = "<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else {
            return results
        }

        let nsString = text as NSString
        let matches = regex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))

        for match in matches {
            guard match.numberOfRanges >= 2 else { continue }
            let fullRange = match.range(at: 0)
            let jsonRange = match.range(at: 1)
            let rawMatch = nsString.substring(with: fullRange)
            let jsonStr = nsString.substring(with: jsonRange)

            if let data = jsonStr.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let name = json["name"] as? String {
                let args = (json["arguments"] as? [String: Any]) ?? [:]
                results.append(ParsedToolCall(name: name, arguments: args, rawText: rawMatch))
            }
        }

        return results
    }

    /// Executes a parsed tool call locally and returns the execution string result
    public func executeToolCall(_ call: ParsedToolCall) -> String {
        switch call.name {
        case "shell_run":
            guard let cmd = call.arguments["command"] as? String else {
                return "Error: missing 'command' parameter in shell_run"
            }
            return runShellCommand(cmd)

        case "file_read":
            guard let path = call.arguments["path"] as? String else {
                return "Error: missing 'path' parameter in file_read"
            }
            return readFileContents(path)

        default:
            return "Error: Unknown tool '\(call.name)'"
        }
    }

    private func runShellCommand(_ command: String) -> String {
        let process = Process()
        let pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-c", command]
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
            process.waitUntilExit()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            return String(data: data, encoding: .utf8) ?? ""
        } catch {
            return "Execution failed: \(error.localizedDescription)"
        }
    }

    private func readFileContents(_ path: String) -> String {
        do {
            return try String(contentsOfFile: path, encoding: .utf8)
        } catch {
            return "Read failed: \(error.localizedDescription)"
        }
    }
}
