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
    public static let qwenFunctionOpen = "<function="
    public static let qwenFunctionClose = "</function>"
    public static let llamaTagOpen = "<|python_tag|>"
    public static let llamaTagClose = "</|python_tag|>"
    // Gemma 4 native tool call: <|tool_call>call:name{key:value,...}<tool_call|>
    public static let gemmaToolCallOpen = "<|tool_call>"
    public static let gemmaToolCallClose = "<tool_call|>"
    // Gemma 4 string-quote delimiter used inside tool-call arguments.
    fileprivate static let gemmaQuote = "<|\"|>"

    // Parameter/argument value spans: tag-like text inside them is data, not structure.
    private static let parameterOpen = "<parameter="
    private static let parameterOpenBare = "<parameter>"
    private static let parameterClose = "</parameter>"
    private static let argValueOpen = "<arg_value>"
    private static let argValueClose = "</arg_value>"

    public init() {}

    /// True when `text` contains a `<tool_call>` opener whose block never closed —
    /// the shape the pre-execution freeze leaves behind whenever it cuts
    /// generation at `</function>` (which is the common case: 7 of 9 calls in one
    /// observed run). Callers close the block before the text is spliced into the
    /// next prompt; left unterminated, the model's own history becomes a stack of
    /// structurally invalid call examples and it starts inventing shapes.
    public static func hasUnclosedToolCallBlock(_ text: String) -> Bool {
        unclosedToolCallCount(text) > 0
    }

    /// How many `<tool_call>` blocks a turn left unterminated. Count-based rather
    /// than "last block" so a turn that froze two calls gets both closed, not one.
    ///
    /// Only structural openers count. Parameter/argument values are unconstrained
    /// by the grammar, so a shell command, URL, or search query echoing a literal
    /// `<tool_call>` is data, not a block opener; counting it appended an extra
    /// `</tool_call>` and persisted an unmatched closer into the next prompt,
    /// the history corruption this suffix exists to prevent.
    public static func unclosedToolCallCount(_ text: String) -> Int {
        // A parameter whose closer was split across lines (`</parameter=\n>`) never
        // matches the literal `</parameter>` terminator below, so the value scan
        // runs to the end of the turn and the structural `</tool_call>` closers
        // hiding inside it are never counted — every call then looks unclosed and
        // the caller appends a duplicate closer. Normalize the closer first so the
        // scanner can return to structure.
        let text = repairSplitParameterClosers(text)
        var depth = 0
        var valueTerminator: String? = nil
        var index = text.startIndex
        while index < text.endIndex {
            let remaining = text[index...]
            if let terminator = valueTerminator {
                if remaining.hasPrefix(terminator) {
                    index = text.index(index, offsetBy: terminator.count)
                    valueTerminator = nil
                } else {
                    index = text.index(after: index)
                }
                continue
            }
            if remaining.hasPrefix(parameterOpen) {
                valueTerminator = parameterClose
                index = text.index(index, offsetBy: parameterOpen.count)
                continue
            }
            if remaining.hasPrefix(parameterOpenBare) {
                valueTerminator = parameterClose
                index = text.index(index, offsetBy: parameterOpenBare.count)
                continue
            }
            if remaining.hasPrefix(argValueOpen) {
                valueTerminator = argValueClose
                index = text.index(index, offsetBy: argValueOpen.count)
                continue
            }
            if remaining.hasPrefix(qwenToolCallOpen) {
                depth += 1
                index = text.index(index, offsetBy: qwenToolCallOpen.count)
                continue
            }
            if remaining.hasPrefix(qwenToolCallClose) {
                if depth > 0 { depth -= 1 }
                index = text.index(index, offsetBy: qwenToolCallClose.count)
                continue
            }
            index = text.index(after: index)
        }
        return depth
    }

    /// The closing tags a raw (frozen or truncated) turn is missing, innermost
    /// first, so a committed turn always reads back as a structurally valid example.
    ///
    /// The pre-execution freeze and the max-token cap both routinely cut a call
    /// before its closers. `unclosedToolCallCount` only sees block depth, so it
    /// appends a bare `</tool_call>` and leaves the inner tags dangling — observed
    /// live: the model opened `<parameter=command>`, ended the turn with no
    /// `</parameter></function>`, and the committed example was a parameter tag that
    /// never closed, which the next steps imitated. This scanner tracks the open
    /// parameter/function/tool-call tags (skipping parameter values, which are data)
    /// and returns exactly the closers needed, in order.
    public static func structuralClosureTags(forRawDecodedTurn text: String) -> [String] {
        let text = repairSplitParameterClosers(text)
        // Expected closers, outermost first.
        var stack: [String] = []
        var valueTerminator: String? = nil
        var index = text.startIndex
        while index < text.endIndex {
            let remaining = text[index...]
            if let terminator = valueTerminator {
                if remaining.hasPrefix(terminator) {
                    popFrom(&stack, tag: terminator)
                    valueTerminator = nil
                    index = text.index(index, offsetBy: terminator.count)
                } else if remaining.hasPrefix(qwenFunctionClose) || remaining.hasPrefix(qwenToolCallClose) {
                    // A closer here is usually the real end of the call: the value was
                    // cut before its own terminator and the closer must be handled as
                    // structure. But a value can legitimately CONTAIN a literal closer
                    // (a command echoing markup, a file body with XML), and treating
                    // that as structure silently drops the parameter/function closers
                    // from the committed turn. If the value's own terminator still
                    // appears before any further structural closer, this is value text;
                    // otherwise it is the real closer.
                    if valueCloseOccursBeforeNextStructuralCloser(in: text, from: index, terminator: terminator) {
                        index = text.index(after: index)
                    } else {
                        valueTerminator = nil
                    }
                } else {
                    index = text.index(after: index)
                }
                continue
            }
            if remaining.hasPrefix(parameterOpen) {
                stack.append(parameterClose); valueTerminator = parameterClose
                index = text.index(index, offsetBy: parameterOpen.count); continue
            }
            if remaining.hasPrefix(parameterOpenBare) {
                stack.append(parameterClose); valueTerminator = parameterClose
                index = text.index(index, offsetBy: parameterOpenBare.count); continue
            }
            if remaining.hasPrefix(argValueOpen) {
                stack.append(argValueClose); valueTerminator = argValueClose
                index = text.index(index, offsetBy: argValueOpen.count); continue
            }
            if remaining.hasPrefix(qwenFunctionOpen) {
                stack.append(qwenFunctionClose)
                index = text.index(index, offsetBy: qwenFunctionOpen.count); continue
            }
            if remaining.hasPrefix(qwenFunctionClose) {
                popFrom(&stack, tag: qwenFunctionClose)
                index = text.index(index, offsetBy: qwenFunctionClose.count); continue
            }
            if remaining.hasPrefix(qwenToolCallOpen) {
                stack.append(qwenToolCallClose)
                index = text.index(index, offsetBy: qwenToolCallOpen.count); continue
            }
            if remaining.hasPrefix(qwenToolCallClose) {
                popFrom(&stack, tag: qwenToolCallClose)
                index = text.index(index, offsetBy: qwenToolCallClose.count); continue
            }
            index = text.index(after: index)
        }
        // Innermost open tag was pushed last, so its closer must be appended first.
        return Array(stack.reversed())
    }

    /// Removes `tag` and everything opened after it: an out-of-order closer abandons
    /// its inner tags rather than producing a `</tool_call></parameter>` jumble.
    private static func popFrom(_ stack: inout [String], tag: String) {
        if let idx = stack.lastIndex(of: tag) {
            stack.removeSubrange(idx...)
        }
    }

    /// True when a value's own close tag (`terminator`) appears in `text` after `index`
    /// before any further structural closer. Distinguishes a literal closer embedded in
    /// a parameter value (editing XML, echoing markup) from the real end of the call,
    /// so the literal one does not silently drop the parameter/function closers.
    private static func valueCloseOccursBeforeNextStructuralCloser(
        in text: String,
        from index: String.Index,
        terminator: String
    ) -> Bool {
        var i = text.index(after: index)
        while i < text.endIndex {
            let remaining = text[i...]
            if remaining.hasPrefix(terminator) { return true }
            if remaining.hasPrefix(qwenFunctionClose) || remaining.hasPrefix(qwenToolCallClose) { return false }
            i = text.index(after: i)
        }
        return false
    }

    /// The tokens that must be appended after a turn's generated ids so the turn
    /// the model reads back next time is well-formed: the inner closers the freeze
    /// or token cap cut (`</parameter>`, `</function>`, `</tool_call>`), then the
    /// model's end tag when the turn never emitted one.
    ///
    /// MUST be fed the RAW decoded turn text. Feeding an already-normalized string
    /// makes `structuralClosureTags` read the closers the caller just added and
    /// skip appending them to the token stream, so the model sees a turn that the
    /// string path claims is closed but the token path left open (observed: every
    /// continuation in one run lost both its closer AND its end tag).
    public static func turnClosureSuffix(
        forRawDecodedTurn decodedTurn: String,
        endTag: String,
        encode: (String) throws -> [UInt32]
    ) -> [UInt32]? {
        var ids: [UInt32] = []
        for tag in structuralClosureTags(forRawDecodedTurn: decodedTurn) {
            guard let closeIds = try? encode(tag), !closeIds.isEmpty else { return nil }
            ids.append(contentsOf: closeIds)
        }
        if !decodedTurn.contains(endTag) {
            guard let endIds = try? encode(endTag), !endIds.isEmpty else { return nil }
            ids.append(contentsOf: endIds)
        }
        return ids
    }

    /// Checks if the stream has completed a tool call and should immediately freeze token generation.
    /// This is Pre-Execution Catching: zero wasted tokens and zero post-tool hallucination.
    public func shouldFreezeGeneration(
        accumulatedText: String,
        deltaText: String,
        format: ToolCallFormat = .qwenXML
    ) -> Bool {
        // Gemma 4 native tool calls freeze only once a complete, parseable block has
        // arrived. Checking the opener and closer independently could freeze on a stray
        // closer in prose followed by a later, incomplete opener that never parses.
        if !Self.parseGemmaBlocks(in: accumulatedText).isEmpty {
            return true
        }
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

    /// The model sometimes opens a parameter with a pipe separator and an
    /// attribute-style quote instead of the canonical `=`, and nests it inside the
    /// real opener:
    ///
    ///     <parameter=command>
    ///     <parameter|command="sed -n '2p' file.csv"; echo done
    ///     </parameter>
    ///
    /// Observed live: emitted right after a run of failing steps, then zsh read the
    /// first line as a stdin redirect from a file named `parameter`
    /// ("zsh:1: no such file or directory: parameter"). The model took that for the
    /// environment mangling its commands and spiralled, while the broken shape got
    /// committed verbatim to its own history as the next example to imitate.
    /// Rewrite the opener to the canonical nested form so `unwrapNestedParameterTag`
    /// descends into the real value and the committed turn reads back as valid
    /// markup. The trailing quote the model may or may not add is consumed when
    /// present, and nothing else is touched.
    public static func repairMalformedParameterOpeners(_ text: String) -> String {
        guard text.contains("<parameter|") else { return text }
        var result = text
        // Attribute-quoted form: <parameter|key="value" — the model opens a quote
        // after the `=` and (usually) closes it early, then keeps writing the rest
        // of the command outside it. Pair and drop both quotes so the value reads
        // as one command instead of leaving an unbalanced `"`.
        if let pairRe = try? NSRegularExpression(
            pattern: "<parameter\\|([A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*\"([^\"]*)\"",
            options: []
        ) {
            let ns = result as NSString
            result = pairRe.stringByReplacingMatches(
                in: result,
                options: [],
                range: NSRange(location: 0, length: ns.length),
                withTemplate: "<parameter=$1>$2"
            )
        }
        // Bare forms: <parameter|key>, <parameter|key=value, <parameter|key="value
        // with no closing quote. Consume the separator (and any opening quote) only.
        guard let re = try? NSRegularExpression(
            pattern: "<parameter\\|([A-Za-z_][A-Za-z0-9_]*)\\s*(?:=\\s*\"?|>)",
            options: []
        ) else { return result }
        let ns = result as NSString
        return re.stringByReplacingMatches(
            in: result,
            options: [],
            range: NSRange(location: 0, length: ns.length),
            withTemplate: "<parameter=$1>"
        )
    }

    /// A parameter opener the model duplicated inside its own value:
    /// `<parameter=command>` immediately followed by a second `<parameter=command>`.
    /// `repairMalformedParameterOpeners` canonicalizes the pipe dialect
    /// (`<parameter|command="…`) into exactly this shape, and the model also emits the
    /// plain duplicate directly. The real payload is the inner one, so the outer
    /// duplicate is dropped; otherwise the committed turn carries two openers and a
    /// single `</parameter>`, and `structuralClosureTags` will not balance it (tags
    /// inside a value are data), teaching the model a malformed example to imitate.
    /// Collapsing them at text level is equivalent to what `unwrapNestedParameterTag`
    /// already does for the value, just earlier so the markup stays well-formed too.
    public static func collapseRedundantNestedParameterOpeners(_ text: String) -> String {
        guard text.contains("<parameter=") else { return text }
        guard let re = try? NSRegularExpression(
            pattern: "(<parameter=([A-Za-z_][A-Za-z0-9_]*)>)[ \\t\\r\\n]*(<parameter=\\2>)",
            options: []
        ) else { return text }
        var current = text
        // Left-to-right replacement is non-overlapping, so a triple needs a repeat pass.
        for _ in 0..<4 {
            let ns = current as NSString
            let replaced = re.stringByReplacingMatches(
                in: current,
                options: [],
                range: NSRange(location: 0, length: ns.length),
                withTemplate: "$1"
            )
            if replaced == current { break }
            current = replaced
        }
        return current
    }

    /// The model occasionally splits the parameter close tag across lines —
    /// `</parameter` then `=` and/or whitespace, then `>` — because the grammar
    /// constrains call structure but not the byte-precise closer. Left alone the
    /// tolerant `<parameter=([^>]+)>([\s\S]*?)(?:</parameter>|$)` regex falls through
    /// to its end-of-input alternative and swallows the broken closer into the
    /// VALUE, so `shell_run` executes a command whose last line is a bare `>` and
    /// zsh dies with "parse error near '>'"; the model reads that as a command
    /// error, retries, and spirals. Rewrite the malformed closer back to canonical
    /// so the value ends where the model meant it to. Only whitespace may separate
    /// the `=` from the `>`, so a legitimate `</parameter=foo>` is left untouched.
    ///
    /// Also normalizes the pipe-separated opener dialect (see
    /// `repairMalformedParameterOpeners`) so every caller that repairs parameter
    /// markup gets both directions. Runs before the closer guard so a turn frozen
    /// mid-value still gets its opener fixed even when no `</parameter>` exists yet.
    public static func repairSplitParameterClosers(_ text: String) -> String {
        let repaired = collapseRedundantNestedParameterOpeners(repairMalformedParameterOpeners(text))
        guard repaired.contains("</parameter") else { return repaired }
        guard let re = try? NSRegularExpression(pattern: "</parameter\\s*=?\\s*>", options: []) else { return repaired }
        let ns = repaired as NSString
        return re.stringByReplacingMatches(
            in: repaired,
            options: [],
            range: NSRange(location: 0, length: ns.length),
            withTemplate: parameterClose
        )
    }

    /// Trims a shell heredoc the model over-ran. Observed live: after the intended
    /// terminator line (`PYEOF`) the model kept emitting the same delimiter a dozen
    /// more times, so zsh ran twelve failing `PYEOF` commands after the script
    /// succeeded, flipping a working command into `exit_code: 127` and feeding the
    /// loop a confusing error. Cut the value at the FIRST terminator when everything
    /// after it is only more copies of that terminator (or blank). A real command
    /// legitimately has live lines after a heredoc (`EOF` then `echo done`), so those
    /// are left untouched.
    public static func truncateHeredocOverrun(_ value: String) -> String {
        let lines = value.components(separatedBy: "\n")
        guard let opener = firstHeredocOpener(in: lines) else { return value }
        var index = opener.index + 1
        var terminator: Int? = nil
        while index < lines.count {
            let candidate = opener.stripsTabs ? String(lines[index].drop(while: { $0 == "\t" })) : lines[index]
            if candidate == opener.delimiter { terminator = index; break }
            index += 1
        }
        guard let term = terminator else { return value }
        let trailing = lines[(term + 1)...]
        guard !trailing.isEmpty else { return value }
        let allDuplicates = trailing.allSatisfy { line in
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            return trimmed.isEmpty || trimmed == opener.delimiter
        }
        guard allDuplicates else { return value }
        return lines[0...term].joined(separator: "\n")
    }

    /// First heredoc opener in a value: `<<EOF`, `<<-EOF` (tab-stripping), `<<'EOF'`,
    /// `<<"EOF"`, or `<<\EOF`. Returns its line index, delimiter, and tab-strip flag.
    private static func firstHeredocOpener(in lines: [String]) -> (index: Int, delimiter: String, stripsTabs: Bool)? {
        guard let re = try? NSRegularExpression(
            pattern: "<<(-?)\\s*(?:'([^']+)'|\"([^\"]+)\"|\\\\([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))",
            options: []
        ) else { return nil }
        for (i, line) in lines.enumerated() {
            let ns = line as NSString
            guard let m = re.firstMatch(in: line, options: [], range: NSRange(location: 0, length: ns.length)),
                  m.numberOfRanges >= 6 else { continue }
            for group in 2...5 where m.range(at: group).location != NSNotFound {
                let delimiter = ns.substring(with: m.range(at: group))
                if !delimiter.isEmpty {
                    return (i, delimiter, ns.substring(with: m.range(at: 1)) == "-")
                }
            }
        }
        return nil
    }

    /// Applies `truncateHeredocOverrun` to every `command` parameter value in a turn,
    /// so the committed example matches the command that actually ran. Restricted to
    /// the `command` key so a `file_write` body containing heredoc-like text is never
    /// touched.
    public static func truncateHeredocOverruns(inTurnText text: String) -> String {
        guard text.contains("<<") else { return text }
        guard let re = try? NSRegularExpression(pattern: "<parameter=([A-Za-z_][A-Za-z0-9_]*)>([\\s\\S]*?)</parameter>", options: []) else { return text }
        var result = text
        let ns = result as NSString
        let matches = re.matches(in: result, options: [], range: NSRange(location: 0, length: ns.length))
        for m in matches.reversed() {
            guard m.numberOfRanges >= 3 else { continue }
            guard ns.substring(with: m.range(at: 1)) == "command" else { continue }
            let value = ns.substring(with: m.range(at: 2))
            let truncated = truncateHeredocOverrun(value)
            guard truncated != value else { continue }
            result = (result as NSString).replacingCharacters(in: m.range(at: 2), with: truncated)
        }
        return result
    }

    /// Identical characters in a row that mark a collapse. Set well above any real
    /// banner/separator rule so a `####…` comment line is never mistaken for one.
    static let degenerateCharacterRun = 200
    /// A short motif copied for this many characters without interruption is a
    /// collapse. Real script text has no exact repeat this long.
    static let degenerateRepeatSpan = 160
    /// Longest motif period considered a repeat; genuine text repeats shorter
    /// fragments, never a 25+ character block verbatim.
    static let degenerateMaxPeriod = 24

    /// Detects the repetition collapse that ends a degenerating generation: the model
    /// stops producing new tokens and copies a short motif (`'''"""'''"""…`) until the
    /// context fills. Observed: a 12,647-character `shell_run` argument made almost
    /// entirely of `'''"""`, run as an unterminated Python heredoc, whose 22,357-
    /// character syntax error was then fed straight back as the next tool result — the
    /// collapse ate the rest of the run. A match means the text is not a command or
    /// program at all, so callers reject it instead of executing it.
    ///
    /// Returns a short description of the repeat when found, else nil.
    public static func repetitionCollapse(in text: String) -> String? {
        let chars = Array(text)
        guard chars.count >= degenerateRepeatSpan else { return nil }

        var runLength = 1
        var bestRun = 1
        var bestRunChar = chars[0]
        for i in 1..<chars.count {
            if chars[i] == chars[i - 1] {
                runLength += 1
                if runLength > bestRun { bestRun = runLength; bestRunChar = chars[i] }
            } else {
                runLength = 1
            }
        }
        if bestRun >= degenerateCharacterRun {
            return "\(bestRun) identical '\(bestRunChar)' characters in a row"
        }

        // A motif of period p repeated back to back shows up as a long run where
        // chars[i] == chars[i - p]. Walk every candidate period and keep the longest
        // uninterrupted match; a real program never matches for this many characters.
        var bestSpan = 0
        var bestPeriod = 0
        var bestEnd = 0
        for period in 1...degenerateMaxPeriod {
            var span = 0
            var i = period
            while i < chars.count {
                if chars[i] == chars[i - period] {
                    span += 1
                    if span > bestSpan { bestSpan = span; bestPeriod = period; bestEnd = i }
                } else {
                    span = 0
                }
                i += 1
            }
        }
        guard bestSpan >= degenerateRepeatSpan, bestEnd >= bestPeriod else { return nil }
        let motif = String(chars[(bestEnd - bestPeriod + 1)...bestEnd])
        return "a \(bestPeriod)-character motif (\(motif.debugDescription)) copied about \(bestSpan / bestPeriod) times"
    }

    /// Rewrites a collapsed `command` parameter in a turn to a short placeholder so
    /// the committed history never carries tens of thousands of repeated characters
    /// that the model would keep attending to (and imitating). Scoped to `command`
    /// only: a `file_write` body is real content even when long, and the caller
    /// re-encodes the whole turn whenever this changes anything.
    public static func compressDegenerateCommandArguments(inTurnText text: String) -> String {
        guard text.contains("</parameter>") else { return text }
        guard let re = try? NSRegularExpression(pattern: "<parameter=([A-Za-z_][A-Za-z0-9_]*)>([\\s\\S]*?)</parameter>", options: []) else { return text }
        var result = text
        let ns = result as NSString
        let matches = re.matches(in: result, options: [], range: NSRange(location: 0, length: ns.length))
        for m in matches.reversed() {
            guard m.numberOfRanges >= 3 else { continue }
            guard ns.substring(with: m.range(at: 1)) == "command" else { continue }
            let value = ns.substring(with: m.range(at: 2))
            guard let collapse = repetitionCollapse(in: value) else { continue }
            let replacement = "[collapsed generation omitted: \(value.count) characters, \(collapse)]"
            result = (result as NSString).replacingCharacters(in: m.range(at: 2), with: replacement)
        }
        return result
    }

    /// The model sometimes opens a parameter tag and then opens it AGAIN inside
    /// the value it is writing:
    ///
    ///     <parameter=command>
    ///     <parameter=command>curl -s https://example.com
    ///     </parameter>
    ///
    /// a shape it picked up imitating the unterminated calls the pre-execution
    /// freeze leaves in its own context. The inner payload is the real value
    /// (these are not knowingly nested structures), so descend into it rather
    /// than handing the tool a value that starts with literal markup. Observed
    /// cost of not doing this: `shell_run` executed the markup as a shell
    /// redirect ("zsh: no such file or directory: parameter=command") and
    /// `web_fetch` rejected it as an invalid URL, each time looping.
    ///
    /// A related loop: the model re-emits the bare key terminator as the value's
    /// first line (`<parameter=command>` then `command>`), so the command becomes
    /// `command>\npython3 …` and zsh fails with "parse error near '\n'". Both the
    /// bare `<parameter>` opener and that `key>` line are stripped here.
    ///
    /// Shared by both call parsers so they cannot disagree (the live agent loop
    /// parses with `parseStreamingToolCalls`, not the AgentHarness one).
    public static func unwrapNestedParameterTag(_ value: String) -> String {
        var v = value.trimmingCharacters(in: .whitespacesAndNewlines)
        var guardCount = 0
        while guardCount < 8 {
            if v.hasPrefix("<parameter=") {
                guard let open = v.range(of: ">") else { break }
                v = String(v[open.upperBound...])
                if let close = v.range(of: "</parameter>", options: .backwards) {
                    v = String(v[..<close.lowerBound])
                }
                v = v.trimmingCharacters(in: .whitespacesAndNewlines)
            } else if v.hasPrefix("<parameter>") {
                v = String(v.dropFirst("<parameter>".count)).trimmingCharacters(in: .whitespacesAndNewlines)
            } else if let newline = v.firstIndex(of: "\n") {
                // The model can re-emit tag-like markup as the value's first lines:
                // a lone key terminator ("command>") or a made-up tag ("<command>",
                // "</command>") instead of the content. Drop each such leading line
                // and descend, so the intended content survives if it is there and,
                // if the value is ONLY such markup, the caller sees an empty value
                // (which the empty-argument guard already handles) instead of a
                // command that fails with a shell parse error every step.
                let firstLine = v[..<newline].trimmingCharacters(in: .whitespaces)
                let rest = String(v[v.index(after: newline)...]).trimmingCharacters(in: .whitespacesAndNewlines)
                guard Self.isSpuriousMarkupLine(firstLine) else { break }
                v = rest
                if v.isEmpty { return "" }
            } else {
                break
            }
            guardCount += 1
        }
        if Self.isSpuriousMarkupLine(v) { return "" }
        return v
    }

    /// True for a lone markup token that is not real content: a bare `key>`
    /// terminator, or an angle-bracketed name (`<command>`, `</command>`). A real
    /// redirect (`> out.txt`, `< input.txt`) or `<<'EOF'` does not match (whole
    /// line, single token, no spaces).
    private static func isSpuriousMarkupLine(_ line: String) -> Bool {
        guard !line.isEmpty, !line.contains(" ") , !line.contains("\t") else { return false }
        var name = line
        if name.hasPrefix("</") {
            name = String(name.dropFirst(2))
        } else if name.hasPrefix("<") {
            name = String(name.dropFirst())
        }
        guard name.hasSuffix(">") else { return false }
        name = String(name.dropLast())
        guard let first = name.first, first.isLetter || first == "_" else { return false }
        return name.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" || $0 == "." || $0 == "-" }
    }

    /// Stores a parsed parameter, preferring the FIRST non-empty value when the model
    /// emits the same parameter name twice. Observed failure: a `shell_run` call with
    /// two `<parameter=command>` blocks — the real command first, then a human-readable
    /// echo ("echo check python pandas availability"). Last-wins silently dropped the
    /// real command and executed the echo, wasting the step. A later value may still
    /// replace an earlier one that collapsed to empty (noise-only), so both orderings
    /// recover the real value.
    static func assignParameter(_ args: inout [String: Any], key: String, value: Any) {
        guard !key.isEmpty else { return }
        if let existing = args[key] {
            let existingEmpty = (existing as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? false
            let newEmpty = (value as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? false
            if !(existingEmpty && !newEmpty) { return }
        }
        args[key] = value
    }

    /// Decodes a canonical `<parameter=...>` value the same way `AgentHarness.parseAllXMLFunctionCalls`
    /// does: JSON objects, arrays, and numbers become structured values, everything else stays a
    /// plain string. Required for Ling's template, which JSON-encodes non-string argument values
    /// (e.g. arrays for `tools_load`'s `names`, booleans for flags, numbers for counts).
    private static func decodeParameterValue(_ rawValue: String) -> Any {
        var value = unwrapNestedParameterTag(rawValue)
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
        // Gemma 4 native blocks are parsed FIRST: a Gemma string argument can contain
        // text that resembles another protocol, which the dialect normalization and the
        // earlier parsers below would otherwise claim. Parsing the outer Gemma block first
        // keeps argument data from being reinterpreted as a different tool protocol.
        if text.contains(Self.gemmaToolCallOpen) {
            let gemmaFirst = Self.parseGemmaBlocks(in: text)
            if !gemmaFirst.isEmpty {
                return (calls: gemmaFirst, brokenFragments: [])
            }
        }

        // 0. Dialect normalization: some models emit native Qwen argument XML inside the
        //    function body using <arg_key>k</arg_key> / <arg_value>v</arg_value> pairs instead
        //    of the internal <parameter=k>v</parameter> form. Rewrite those into canonical
        //    <parameter=...> blocks so the tolerant parser actually executes the call instead
        //    of silently dropping it (which previously left only prose in the reply).
        var text = Self.repairSplitParameterClosers(Self.normalizeArgKeyDialect(text))

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
                            Self.assignParameter(&args, key: key, value: Self.decodeParameterValue(trimmedChunk))
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
                                Self.assignParameter(&args, key: key, value: Self.decodeParameterValue(inlineValue))
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

        // 5. Gemma 4 native format (already tried first on the raw text; retry on the
        //    normalized text in case normalization revealed a complete block).
        if text.contains(Self.gemmaToolCallOpen) {
            let gemmaCalls = Self.parseGemmaBlocks(in: text)
            if !gemmaCalls.isEmpty {
                return (calls: gemmaCalls, brokenFragments: [])
            }
        }

        return agentCalls
    }

    /// Parses a Gemma 4 native tool call of the form
    /// `<|tool_call>call:name{key:value,...}<tool_call|>`. Argument values use Gemma's
    /// `<|"|>` string delimiter (also tolerates plain quotes), nested `{}`/`[]`, and bare
    /// literals. A call is only returned once its closing tag has arrived.
    static func parseGemmaToolCall(_ block: String) -> ParsedToolCall? {
        var body = block.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.hasPrefix(gemmaToolCallOpen) {
            body = String(body.dropFirst(gemmaToolCallOpen.count))
        }
        // A truncated call (no closing tag) only occurs on EOS or the token cap, where
        // executing partial arguments (a cut-off shell command) is unsafe. Require the
        // closer before returning a call.
        guard let closeRange = body.range(of: gemmaToolCallClose) else { return nil }
        body = String(body[..<closeRange.lowerBound])
        body = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if body.hasPrefix("call:") {
            body = String(body.dropFirst("call:".count))
        }

        guard let braceIdx = body.firstIndex(of: "{") else {
            // Name-only call (no arguments).
            let name = body.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !name.isEmpty, name.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" || $0 == "." || $0 == "-" }) else { return nil }
            return ParsedToolCall(name: name, arguments: [:], rawArguments: "", rawText: block)
        }

        let name = String(body[..<braceIdx]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return nil }
        let argsStr = String(body[braceIdx...])
        var parser = GemmaArgumentParser(argsStr)
        // The closing tag does not prove the argument object is complete: an unclosed
        // quote, brace, or bracket still yields accumulated values. Require the parser
        // to have closed every quote/container and consumed the whole argument string,
        // so a truncated command (a half-written shell line) cannot be executed.
        guard let parsed = parser.parseValue() as? [String: Any], parser.complete, parser.consumedAll else {
            return nil
        }
        return ParsedToolCall(name: name, arguments: parsed, rawArguments: argsStr, rawText: block)
    }

    /// Extracts and parses every complete Gemma 4 native call block in `text`.
    private static func parseGemmaBlocks(in text: String) -> [ParsedToolCall] {
        let blockPattern = "<\\|tool_call>([\\s\\S]*?)(<tool_call\\|>)"
        guard let blockRegex = try? NSRegularExpression(pattern: blockPattern, options: []) else { return [] }
        let ns = text as NSString
        var calls: [ParsedToolCall] = []
        for m in blockRegex.matches(in: text, options: [], range: NSRange(location: 0, length: ns.length)) {
            let block = ns.substring(with: m.range(at: 0))
            if let call = Self.parseGemmaToolCall(block) {
                calls.append(call)
            }
        }
        return calls
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

/// Recursive-descent parser for Gemma 4 tool-call argument bodies: a `{key:value,...}`
/// object using `<|"|>` string delimiters (plain quotes also tolerated), nested
/// `{}`/`[]`, and bare numbers/booleans/null. Values are returned as JSON-compatible
/// Foundation types so downstream argument coercion behaves like the other formats.
private struct GemmaArgumentParser {
    private let chars: [Character]
    private var idx = 0
    /// False once any quote or container reached the end of input without closing.
    /// A tool call is only executable when the whole argument body parsed cleanly.
    var complete = true

    init(_ text: String) { chars = Array(text) }

    private mutating func skipWhitespace() {
        while idx < chars.count, chars[idx].isWhitespace { idx += 1 }
    }
    private func peek() -> Character? { idx < chars.count ? chars[idx] : nil }

    /// True when the parser consumed the entire argument string (trailing whitespace aside).
    var consumedAll: Bool {
        var i = idx
        while i < chars.count, chars[i].isWhitespace { i += 1 }
        return i >= chars.count
    }

    private func matches(_ token: String) -> Bool {
        let t = Array(token)
        guard idx + t.count <= chars.count else { return false }
        for k in 0..<t.count where chars[idx + k] != t[k] { return false }
        return true
    }

    mutating func parseValue() -> Any? {
        skipWhitespace()
        guard let c = peek() else { return nil }
        if c == "{" { return parseObject() }
        if c == "[" { return parseArray() }
        if matches(StreamingToolParser.gemmaQuote) { return parseGemmaQuoted() }
        if c == "\"" || c == "'" { return parseQuoted(quote: c) }
        return parseBare()
    }

    private mutating func parseGemmaQuoted() -> String {
        idx += StreamingToolParser.gemmaQuote.count
        var out = ""
        var closed = false
        while idx < chars.count {
            if matches(StreamingToolParser.gemmaQuote) {
                idx += StreamingToolParser.gemmaQuote.count
                closed = true
                break
            }
            out.append(chars[idx]); idx += 1
        }
        if !closed { complete = false }
        return out
    }

    private mutating func parseQuoted(quote: Character) -> String {
        idx += 1
        var out = ""
        var closed = false
        while idx < chars.count {
            let ch = chars[idx]
            if ch == "\\", idx + 1 < chars.count {
                let next = chars[idx + 1]
                switch next {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                case "\\": out.append("\\")
                case "\"": out.append("\"")
                case "'": out.append("'")
                // Unknown escape: keep the backslash so a literal path or command
                // (e.g. a regex \d) is not silently altered when executed.
                default: out.append("\\"); out.append(next)
                }
                idx += 2
                continue
            }
            if ch == quote { idx += 1; closed = true; break }
            out.append(ch); idx += 1
        }
        if !closed { complete = false }
        return out
    }

    private mutating func parseObject() -> [String: Any] {
        var obj: [String: Any] = [:]
        idx += 1 // consume '{'
        var closed = false
        while idx < chars.count {
            skipWhitespace()
            if idx >= chars.count { break }
            if chars[idx] == "}" { idx += 1; closed = true; break }
            let key: String
            if chars[idx] == "\"" || chars[idx] == "'" { key = parseQuoted(quote: chars[idx]) }
            else if matches(StreamingToolParser.gemmaQuote) { key = parseGemmaQuoted() }
            else { key = parseBareKey() }
            skipWhitespace()
            // Require the colon separator and a value that advances the cursor: an
            // object such as `{command "x"}` or `{command:}` is malformed and must not
            // count as a clean parse, or malformed output could execute.
            guard peek() == ":" else { complete = false; break }
            idx += 1
            skipWhitespace()
            let valueStart = idx
            guard let value = parseValue(), idx > valueStart else { complete = false; break }
            if !key.isEmpty { obj[key] = value }
            skipWhitespace()
            if peek() == "," { idx += 1; continue }
            if peek() == "}" { idx += 1; closed = true; break }
            // Any other separator (a stray token, `;`, etc.) means the object is
            // malformed: mark it incomplete so it cannot execute.
            complete = false
            break
        }
        if !closed { complete = false }
        return obj
    }

    private mutating func parseArray() -> [Any] {
        var arr: [Any] = []
        idx += 1 // consume '['
        var closed = false
        while idx < chars.count {
            skipWhitespace()
            if idx >= chars.count { break }
            if chars[idx] == "]" { idx += 1; closed = true; break }
            if let value = parseValue() { arr.append(value) }
            skipWhitespace()
            if peek() == "," { idx += 1; continue }
            if peek() == "]" { idx += 1; closed = true; break }
            // Any other separator means the array is malformed.
            complete = false
            break
        }
        if !closed { complete = false }
        return arr
    }

    private mutating func parseBareKey() -> String {
        var out = ""
        while idx < chars.count {
            let c = chars[idx]
            if c == ":" || c == "," || c == "}" || c == "{" || c.isWhitespace { break }
            out.append(c); idx += 1
        }
        return out
    }

    private mutating func parseBare() -> Any {
        var out = ""
        while idx < chars.count {
            let c = chars[idx]
            if c == "," || c == "}" || c == "]" { break }
            out.append(c); idx += 1
        }
        let token = out.trimmingCharacters(in: .whitespacesAndNewlines)
        if token == "true" { return true }
        if token == "false" { return false }
        if token == "null" { return NSNull() }
        if let intValue = Int(token) { return intValue }
        if let doubleValue = Double(token) { return doubleValue }
        return token
    }
}
