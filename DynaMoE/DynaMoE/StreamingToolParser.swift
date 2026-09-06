//
//  StreamingToolParser.swift
//  DynaMoE
//
//  Phase 2: Streaming AST Parser & Template Translation
//  Universal chat template adapter (Qwen XML, JSON, Llama 3).
//  Pre-execution catching intercepts generation the exact moment </tool_call> is emitted,
//  instantly freezing token generation without waiting for EOS or post-call chatter.
//

import Foundation

public enum ToolCallFormat: String, CaseIterable, Codable {
    case qwenXML = "qwen_xml"
    case hermeticJSON = "hermetic_json"
    case llama3 = "llama_3"
}

public final class StreamingToolParser {
    public static let shared = StreamingToolParser()

    // Structural Delimiters
    public static let qwenToolCallOpen = "<tool_call>"
    public static let qwenToolCallClose = "</tool_call>"
    public static let qwenFunctionClose = "</function>"
    public static let llamaTagOpen = "<|python_tag|>"
    public static let llamaTagClose = "</|python_tag|>"

    public init() {}

    /// Checks if the stream has completed a tool call and should immediately freeze token generation.
    /// This is Pre-Execution Catching: zero wasted tokens and zero post-tool hallucination.
    public func shouldFreezeGeneration(
        accumulatedText: String,
        deltaText: String,
        format: ToolCallFormat = .qwenXML
    ) -> Bool {
        switch format {
        case .qwenXML:
            // Check if </tool_call> has been closed
            if accumulatedText.contains(Self.qwenToolCallClose) || deltaText.contains(Self.qwenToolCallClose) {
                return true
            }
            // Fallback: If </function> was closed and model is inside a tool_call block
            if accumulatedText.contains(Self.qwenToolCallOpen) && (accumulatedText.contains(Self.qwenFunctionClose) || deltaText.contains(Self.qwenFunctionClose)) {
                return true
            }
            return false

        case .hermeticJSON:
            if accumulatedText.contains(Self.qwenToolCallOpen) {
                let slice = accumulatedText.components(separatedBy: Self.qwenToolCallOpen).last ?? ""
                if slice.contains(Self.qwenToolCallClose) {
                    return true
                }
                // Check balanced JSON braces within tool call
                let trimmed = slice.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.hasPrefix("{") && trimmed.hasSuffix("}") && trimmed.count > 10 {
                    if let _ = try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)) {
                        return true
                    }
                }
            }
            return false

        case .llama3:
            if accumulatedText.contains(Self.llamaTagClose) || deltaText.contains(Self.llamaTagClose) {
                return true
            }
            return false
        }
    }

    /// Parses all completed tool calls from the stream text across supported formats.
    public func parseStreamingToolCalls(
        from text: String,
        format: ToolCallFormat = .qwenXML
    ) -> (calls: [ParsedToolCall], brokenFragments: [String]) {
        // 1. Check for Qwen XML with JSON arguments body: <function=name>{"arg": "val"}</function>
        let fnRegex = try? NSRegularExpression(pattern: "<function=([^>]+)>([\\s\\S]*?)</function>", options: [])
        let nsText = text as NSString
        let fnMatches = fnRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []

        var fnCalls: [ParsedToolCall] = []
        for m in fnMatches {
            guard m.numberOfRanges >= 3 else { continue }
            let name = nsText.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let body = nsText.substring(with: m.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
            let rawMatch = nsText.substring(with: m.range(at: 0))

            if let data = body.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                fnCalls.append(ParsedToolCall(name: name, arguments: json, rawArguments: body, rawText: rawMatch))
            }
        }
        if !fnCalls.isEmpty {
            return (calls: fnCalls, brokenFragments: [])
        }

        // 2. First try standard AgentHarness XML parser
        let agentCalls = AgentHarness.shared.parseToolCalls(from: text)
        if !agentCalls.calls.isEmpty {
            return agentCalls
        }

        // 3. Hermetic JSON format inside <tool_call>{"tool": "...", "parameters": {...}}</tool_call>
        let toolCallRegex = try? NSRegularExpression(pattern: "<tool_call>([\\s\\S]*?)</tool_call>", options: [])
        let tcMatches = toolCallRegex?.matches(in: text, options: [], range: NSRange(location: 0, length: nsText.length)) ?? []
        var hermeticCalls: [ParsedToolCall] = []
        for m in tcMatches {
            guard m.numberOfRanges >= 2 else { continue }
            let inner = nsText.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
            let raw = nsText.substring(with: m.range(at: 0))
            if let data = inner.data(using: .utf8),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let name = (json["tool"] as? String) ?? (json["name"] as? String) ?? ""
                let args = (json["parameters"] as? [String: Any]) ?? (json["arguments"] as? [String: Any]) ?? [:]
                if !name.isEmpty {
                    hermeticCalls.append(ParsedToolCall(name: name, arguments: args, rawArguments: inner, rawText: raw))
                }
            }
        }
        if !hermeticCalls.isEmpty {
            return (calls: hermeticCalls, brokenFragments: [])
        }

        // 4. Try Llama 3 <|python_tag|> format
        if text.contains(Self.llamaTagOpen) {
            var llamaCalls: [ParsedToolCall] = []
            let parts = text.components(separatedBy: Self.llamaTagOpen)
            for part in parts.dropFirst() {
                let snippet = part.components(separatedBy: Self.llamaTagClose).first ?? part
                if let call = parseLlamaFunctionCall(snippet) {
                    llamaCalls.append(call)
                }
            }
            if !llamaCalls.isEmpty {
                return (calls: llamaCalls, brokenFragments: [])
            }
        }

        return agentCalls
    }

    private func parseLlamaFunctionCall(_ raw: String) -> ParsedToolCall? {
        // Example: call:file_read(path="/path/to/file")
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let openParen = trimmed.firstIndex(of: "("),
              let closeParen = trimmed.lastIndex(of: ")"),
              openParen < closeParen else { return nil }

        var fnName = String(trimmed[..<openParen]).trimmingCharacters(in: .whitespaces)
        if fnName.hasPrefix("call:") {
            fnName = String(fnName.dropFirst(5)).trimmingCharacters(in: .whitespaces)
        }

        let argsStr = String(trimmed[trimmed.index(after: openParen)..<closeParen])
        var parsedArgs: [String: Any] = [:]

        // Simple key=value comma splitter
        let pairs = argsStr.components(separatedBy: ",")
        for pair in pairs {
            let kv = pair.components(separatedBy: "=")
            if kv.count == 2 {
                let k = kv[0].trimmingCharacters(in: .whitespaces)
                var v = kv[1].trimmingCharacters(in: .whitespaces)
                if (v.hasPrefix("\"") && v.hasSuffix("\"")) || (v.hasPrefix("'") && v.hasSuffix("'")) {
                    v = String(v.dropFirst().dropLast())
                }
                parsedArgs[k] = v
            }
        }

        return ParsedToolCall(name: fnName, arguments: parsedArgs, rawArguments: argsStr, rawText: trimmed)
    }
}
