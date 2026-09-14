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

    /// Rewrites Ling/Bailing-3.0-native tool calls into the canonical internal
    /// `<tool_call><function=name>...</function></tool_call>` shape so the rest of the parser
    /// pipeline (and any downstream grammar/recovery logic) can execute them.
    ///
    /// Ling emits the function name directly after the opening tag, followed by
    /// `<arg_key>k</arg_key>` / `<arg_value>v</arg_value>` pairs with no `<function=...>` wrapper:
    ///
    ///     <tool_call>shell_run
    ///     <arg_key>command</arg_key>
    ///     <arg_value>ls -la</arg_value>
    ///     </tool_call>
    ///
    /// Blocks that already carry a `<function=...>` wrapper (Qwen XML) or a JSON payload
    /// (hermetic JSON) are left untouched. A bare-name block is only rewritten when what follows
    /// the name is empty, starts with JSON, or contains arg_key/arg_value pairs — prose inside a
    /// stray `<tool_call>` tag is still reported as a broken fragment instead of a phantom tool.
    /// Truncated blocks (no closing tag) are handled too, since generation may freeze mid-stream.
    public static func normalizeBareNameToolCalls(_ raw: String) -> String {
        guard raw.contains(qwenToolCallOpen) else { return raw }
        let blockPattern = "<tool_call>([\\s\\S]*?)(</tool_call>|$)"
        guard let blockRegex = try? NSRegularExpression(pattern: blockPattern, options: []),
              let nameRegex = try? NSRegularExpression(pattern: "^\\s*([A-Za-z_][A-Za-z0-9_.\\-]*)", options: []) else {
            return raw
        }

        var out = raw
        let ns = out as NSString
        let matches = blockRegex.matches(in: out, options: [], range: NSRange(location: 0, length: ns.length))
        for m in matches.reversed() {
            guard m.numberOfRanges >= 3 else { continue }
            let body = ns.substring(with: m.range(at: 1))
            guard !body.contains("<function=") else { continue }

            let nsBody = body as NSString
            guard let nameMatch = nameRegex.firstMatch(in: body, options: [], range: NSRange(location: 0, length: nsBody.length)),
                  nameMatch.numberOfRanges >= 2 else { continue }
            let name = nsBody.substring(with: nameMatch.range(at: 1))
            let remainder = nsBody.substring(from: nameMatch.range(at: 0).length)
                .trimmingCharacters(in: .whitespacesAndNewlines)

            let trimmedRemainder = remainder.trimmingCharacters(in: .whitespacesAndNewlines)
            let looksLikeArguments = trimmedRemainder.isEmpty
                || trimmedRemainder.hasPrefix("{")
                || trimmedRemainder.hasPrefix("[")
                || trimmedRemainder.contains("<arg_key")
                || trimmedRemainder.contains("<arg_value")
            guard looksLikeArguments else { continue }

            let closing = ns.substring(with: m.range(at: 2))
            let wrappedBody = remainder.isEmpty
                ? "<tool_call>\n<function=\(name)>\n</function>\n\(closing)"
                : "<tool_call>\n<function=\(name)>\n\(remainder)\n</function>\n\(closing)"
            out = (out as NSString).replacingCharacters(in: m.range(at: 0), with: wrappedBody)
        }
        return out
    }

    /// Rewrites the model's native Qwen-native argument dialect into the canonical internal
    /// <parameter=k>v</parameter> form. Some checkpoints emit <arg_key>name</arg_key> followed by
    /// <arg_value>value</arg_value> (possibly with newlines/comments between them) instead of
    /// the parameter= form that AgentHarness.parseToolCalls recognizes; previously such calls
    /// were silently dropped and only the surrounding prose appeared in the reply. This pass
    /// converts every completed pair into a single parameter block and also defends against a
    /// trailing ech(<arg_value>…</arg_value>) whose key never arrived on the current buffer.
    ///
    /// Ling/Bailing 3.0 emits its function name directly after `<tool_call>` with no
    /// `<function=...>` wrapper, so `normalizeBareNameToolCalls` runs first to rewrap those blocks.
    public static func normalizeArgKeyDialect(_ raw: String) -> String {
        var out = normalizeBareNameToolCalls(raw)
        guard out.contains("arg_key") || out.contains("arg_value") else { return out }
        // Collapse each <arg_key>k</arg_key> ... <arg_value>v</arg_value> pair into one block.
        // Pair might span newlines and contain inner whitespace/comments; match loosely.
        let pairPattern = "<arg_key>\\s*([^<]+?)\\s*</arg_key>\\s*<arg_value>\\s*([\\s\\S]*?)\\s*</arg_value>"
        if let re = try? NSRegularExpression(pattern: pairPattern, options: []) {
            let ns = out as NSString
            let all = re.matches(in: out, options: [], range: NSRange(location: 0, length: ns.length))
            for m in all.reversed() {
                guard m.numberOfRanges >= 3 else { continue }
                let key = ns.substring(with: m.range(at: 1))
                let val = ns.substring(with: m.range(at: 2))
                out = (out as NSString).replacingCharacters(in: m.range(at: 0),
                    with: "<parameter=\(key)>\(val)</parameter>")
            }
        }
        // Any <arg_value>v</arg_value> that lost its key (very deep stream buffering) folds
        // into a position-independent parameter named by its value only — the tolerant parser
        // will still match <function=name>… with at least one usable parameter.
        let orphanPattern = "<arg_value>\\s*([\\s\\S]*?)\\s*</arg_value>"
        if let re2 = try? NSRegularExpression(pattern: orphanPattern, options: []) {
            let ns = out as NSString
            let orphan = re2.matches(in: out, options: [], range: NSRange(location: 0, length: ns.length))
            for m in orphan.reversed() {
                guard m.numberOfRanges >= 2 else { continue }
                let val = ns.substring(with: m.range(at: 1))
                out = (out as NSString).replacingCharacters(in: m.range(at: 0),
                    with: "<parameter=value>\(val)</parameter>")
            }
        }
        return out
    }

    /// Decodes a canonical `<parameter=...>` value the same way `AgentHarness.parseAllXMLFunctionCalls`
    /// does: JSON objects, arrays, and numbers become structured values, everything else stays a
    /// plain string. Required for Ling's template, which JSON-encodes non-string argument values
    /// (e.g. arrays for `tools_load`'s `names`, booleans for flags, numbers for counts).
    private static func decodeParameterValue(_ rawValue: String) -> Any {
        var value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.count >= 2, value.hasPrefix("\""), value.hasSuffix("\"") {
            value = String(value.dropFirst().dropLast())
        }
        if let data = value.data(using: .utf8),
           let jsonVal = try? JSONSerialization.jsonObject(with: data, options: .fragmentsAllowed),
           (jsonVal is [String: Any] || jsonVal is [Any] || jsonVal is NSNumber) {
            return jsonVal
        }
        return value
    }

    /// Parses all completed tool calls from the stream text across supported formats.
    public func parseStreamingToolCalls(
        from text: String,
        format: ToolCallFormat = .qwenXML
    ) -> (calls: [ParsedToolCall], brokenFragments: [String]) {
        // 0. Dialect normalization: some models emit native Qwen argument XML inside the
        //    function body using <arg_key>k</arg_key> / <arg_value>v</arg_value> pairs instead
        //    of the internal <parameter=k>v</parameter> form. Rewrite those into canonical
        //    <parameter=...> blocks so the tolerant parser actually executes the call instead
        //    of silently dropping it (which previously left only prose in the reply).
        var text = Self.normalizeArgKeyDialect(text)

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

        // 1.5. <function=name> bodies carrying parameter dialects the canonical parser
        //      doesn't recognize: either the generic-instruct shape
        //      <parameter>key</parameter> value </parameter> or the canonical
        //      <parameter=key>value</parameter>. Previously such calls parsed as
        //      name-only with empty arguments and were rejected as degenerate.
        if fnCalls.isEmpty && !fnMatches.isEmpty {
            var dialectCalls: [ParsedToolCall] = []
            for m in fnMatches {
                guard m.numberOfRanges >= 3 else { continue }
                let name = nsText.substring(with: m.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                let body = nsText.substring(with: m.range(at: 2))
                let rawMatch = nsText.substring(with: m.range(at: 0))
                guard !name.isEmpty else { continue }
                var args: [String: Any] = [:]
                var rawParts: [String] = []
                // Parameters are terminated by </parameter>; split and pair each
                // opening marker with the value text that follows it.
                let chunks = body.components(separatedBy: "</parameter>")
                var pendingKey: String? = nil
                for chunk in chunks {
                    let trimmedChunk = chunk.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let key = pendingKey {
                        if !key.isEmpty && !trimmedChunk.isEmpty {
                            args[key] = Self.decodeParameterValue(trimmedChunk)
                        }
                        pendingKey = nil
                    }
                    if let eqRange = chunk.range(of: "<parameter=") {
                        // Canonical dialect: <parameter=key>value (value may be inline
                        // or arrive in the next chunk before the closing delimiter).
                        let afterEq = chunk[eqRange.upperBound...]
                        if let gt = afterEq.range(of: ">") {
                            let key = String(afterEq[..<gt.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                            let inlineValue = String(afterEq[gt.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                            if !key.isEmpty, !inlineValue.isEmpty {
                                args[key] = Self.decodeParameterValue(inlineValue)
                            } else if !key.isEmpty {
                                pendingKey = key
                            }
                        }
                    } else if let openRange = chunk.range(of: "<parameter>") {
                        // Key-only dialect: the key runs to the end of this chunk and
                        // its value arrives in the next chunk.
                        let key = String(chunk[openRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
                        if !key.isEmpty { pendingKey = key }
                    }
                }
                if !args.isEmpty {
                    rawParts.append(body)
                    dialectCalls.append(ParsedToolCall(
                        name: name,
                        arguments: args,
                        rawArguments: rawParts.joined(separator: "\n"),
                        rawText: rawMatch
                    ))
                }
            }
            if !dialectCalls.isEmpty {
                return (calls: dialectCalls, brokenFragments: [])
            }
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
