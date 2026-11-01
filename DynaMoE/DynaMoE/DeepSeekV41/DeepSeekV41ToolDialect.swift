//
//  DeepSeekV41ToolDialect.swift
//  DynaMoE
//
//  How DeepSeek-V4.1-Flash calls a tool. Nothing about this is assumed: the
//  checkpoint ships no chat template, its `encoding/` package has no tool path, and
//  the vocabulary's own `tool`-`call` special tokens (nine of them, each a single
//  id) are NOT what the model emits.
//
//  `DeepSeekV41PublishedTurnTests` asked the model, and the PROMPT SHAPE decides
//  which dialect comes back:
//
//  * A V3.1-shaped tool block (a schematic built from the marker strings) made it
//    copy that schematic back as plain text, a `tool_call_name` tag around the
//    name and a `tool_call_arguments` tag around the JSON. With no schematic at
//    all it wrote a bare `tool_call` tag around `{"name": ..., "arguments": ...}`.
//
//  * The AGENT-shaped prompt this harness actually builds (the tool catalogue,
//    then the instruction block below) produced something else: the DSML tags,
//    which are what this checkpoint was post-trained on. They are built from one
//    vocabulary token, id 128825, spelled `dsmlTag` below as U+FF5C "DSML" U+FF5C.
//    The tag text around that token is ordinary text, and it carries a space after
//    the token: an opening DSML tag is `<` token ` invoke name="..."`, its closer
//    is `</` token ` invoke>`.
//
//  All three are recognized here, plus the vocabulary's own call container for
//  callers that send it. DSML is what the instructions teach, because DSML is what
//  the prompt this app builds actually receives.
//
//  The result side was measured too, and it does not discriminate: the same
//  conversation with the payload wrapped in the vocabulary's `tool`-`output` pair,
//  in a `tool_response`, in a plain tag, or bare all produced the identical answer.
//  The family's own output pair is therefore used, because it is the checkpoint's
//  own spelling and it tokenizes as one id.
//
//  This type is pure logic: render the instruction block, rewrite a generated call
//  into the canonical call block the shared parser already understands, decide when
//  generation should freeze, and render a result turn. Keeping the daisy-chain out
//  of `StreamingToolParser` means the agent loop, HITL gating, timeline and history
//  rendering all stay one code path.
//

import Foundation

nonisolated public enum DeepSeekV41ToolDialect {

    // MARK: - The shape the checkpoint emits

    public static let callNameOpen = "<tool_call_name>"
    public static let callNameClose = "</tool_call_name>"
    public static let callArgumentsOpen = "<tool_call_arguments>"
    public static let callArgumentsClose = "</tool_call_arguments>"

    /// The canonical block the shared parser consumes.
    static let canonicalOpen = "<tool_call>"
    static let canonicalClose = "</tool_call>"
    static let canonicalFunctionOpen = "<function="
    static let canonicalFunctionClose = "</function>"

    // MARK: - DSML, the dialect the agent-shaped prompt receives

    /// The vocabulary's DSML token (id 128825), U+FF5C "DSML" U+FF5C. Every DSML
    /// tag is this token inside ordinary text, so it is never spelled by hand: a
    /// rendered view strips the fullwidth bars and the result looks right and
    /// tokenizes wrong.
    public static let dsmlTag = "\u{FF5C}DSML\u{FF5C}"

    /// The multi-call container the model wraps its calls in.
    public static let dsmlCallsOpen = "<" + dsmlTag + " calls>"
    public static let dsmlCallsClose = "</" + dsmlTag + " calls>"

    /// One call: `<` tag ` invoke name="NAME">`, parameters, then the closer.
    public static let dsmlInvokeOpen = "<" + dsmlTag + " invoke name=\""
    public static let dsmlInvokeTail = "\">"
    public static let dsmlInvokeClose = "</" + dsmlTag + " invoke>"

    /// One argument: `<` tag ` parameter name="KEY" string="true">value`, then the
    /// closer. `string="true"` means the value is literal text; `string="false"`
    /// means it is JSON and must be handed on as a number, bool, array or object.
    public static let dsmlParameterOpen = "<" + dsmlTag + " parameter name=\""
    public static let dsmlParameterClose = "</" + dsmlTag + " parameter>"

    public static func dsmlInvokeTag(name: String) -> String {
        dsmlInvokeOpen + name + dsmlInvokeTail
    }

    public static func dsmlParameterTag(name: String, isString: Bool = true) -> String {
        dsmlParameterOpen + name + "\" string=\"" + (isString ? "true" : "false") + "\">"
    }

    // MARK: - Instructions

    /// The tool-format block appended to the system prompt for this family.
    ///
    /// It teaches DSML, the dialect the agent-shaped prompt was measured to
    /// receive (`#27`), and it is written with the tag builders rather than with
    /// literal tag text. The tool catalogue itself is rendered by the harness beside
    /// this block; what this adds is the call grammar only.
    public static func instructions() -> String {
        let callsOpen = dsmlCallsOpen
        let callsClose = dsmlCallsClose
        let invokeOpen = dsmlInvokeTag(name: "example_function_name")
        let invokeClose = dsmlInvokeClose
        let parameterOpen = dsmlParameterTag(name: "example_parameter_1")
        let parameterClose = dsmlParameterClose
        return """
        # Tool Call Format

        You can call tools. When you do, reply with ONLY the call block: a calls container
        holding one invoke per tool, and inside each invoke one parameter per argument,
        each carrying the argument's JSON name.

        \(callsOpen)
        \(invokeOpen)
        \(parameterOpen)value_1\(parameterClose)
        \(invokeClose)
        \(callsClose)

        Reminder:
        - `name` on an invoke MUST be an exact match to one of the available tools
        - Every parameter value is text unless its tag says `string="false"`, which marks
          a JSON number, boolean, array or object
        - Required parameters MUST be specified
        - You may reason in natural language BEFORE the call block, but NOT after it
        - When the user asks to inspect, read, edit, modify, or process a file, or run terminal commands, call the tool immediately without conversational promises
        - If no tool fits, answer the question like normal with your current knowledge and do not mention tool calls
        """
    }

    // MARK: - History reconstruction

    /// A call block in the shape the model itself emits, for a stored turn whose
    /// raw call text was lost (a message carried over from another dialect, an
    /// edited turn, a reloaded session).
    ///
    /// The arguments are the records' string values, so each one is rendered as a
    /// string parameter. A stored record has already lost the JSON type of its
    /// values, and claiming a number was a string is the safer error: the tool
    /// receives the same text either way, and `string="true"` is what the grammar
    /// says a plain value is.
    public static func callBlock(
        name: String,
        arguments: [String: String]
    ) -> String {
        var block = dsmlCallsOpen + "\n" + dsmlInvokeTag(name: name) + "\n"
        for key in arguments.keys.sorted() {
            let value = escapedParameterValue(arguments[key] ?? "")
            block += dsmlParameterTag(name: key) + value + dsmlParameterClose + "\n"
        }
        block += dsmlInvokeClose + "\n" + dsmlCallsClose
        return block
    }

    /// A parameter value cannot carry the parameter closer, or the call would
    /// close early. The real grammar has no escape for it, so the literal is
    /// broken with a zero-width no-break space, the same neutralization
    /// `DeepSeekV41PromptDialect.escaped` applies to forged role markers.
    static func escapedParameterValue(_ value: String) -> String {
        guard value.contains(dsmlParameterClose) else { return value }
        return value.replacingOccurrences(
            of: dsmlParameterClose,
            with: "</" + dsmlTag + "\u{FEFF} parameter>"
        )
    }

    // MARK: - Canonicalization

    /// Rewrites the model's own call syntax into canonical XML.
    ///
    /// Every recognized form becomes `<tool_call><function=name>{json}</function></tool_call>`,
    /// which is the shape `StreamingToolParser.parseStreamingToolCalls` already
    /// extracts, so nothing downstream needs a second dialect. Text outside the
    /// blocks is preserved: the model's preamble ("I'll check…") is still shown to
    /// the user, exactly as a Qwen call's preamble is.
    public static func canonicalized(
        _ text: String,
        markers: DeepSeekV41PromptMarkers = .familyDefault
    ) -> String {
        var out = canonicalizeDSMLBlocks(text)
        out = canonicalizeNameArgumentsBlocks(out)
        out = canonicalizeJSONBlocks(out)
        out = canonicalizeSpecialContainer(out, markers: markers)
        return out
    }

    /// The checkpoint's own call grammar: a `calls` container holding one `invoke`
    /// per tool, each carrying `parameter` elements.
    ///
    /// This is the form the agent-shaped prompt receives, so it is the one the loop
    /// actually executes. Everything except the DSML token itself is ordinary text,
    /// which is why the tags are matched as strings rather than as token ids.
    ///
    /// A container is optional: a single call is sometimes written as a bare
    /// `invoke`, so the scan looks for invokes rather than for the container. The
    /// container tags are removed afterwards so they never reach the bubble as
    /// prose. An invoke with no closer, or a parameter with no closer, is a
    /// generation that stopped mid-call and is left as text: executing a call the
    /// model never finished asking for is worse than showing the fragment.
    static func canonicalizeDSMLBlocks(_ text: String) -> String {
        guard text.contains(dsmlTag) else { return text }
        var out = ""
        var cursor = text.startIndex
        while let invoke = text.range(of: dsmlInvokeOpen, range: cursor..<text.endIndex) {
            out += text[cursor..<invoke.lowerBound]
            guard let nameEnd = text.range(of: dsmlInvokeTail, range: invoke.upperBound..<text.endIndex)
            else {
                out += text[invoke.lowerBound...]
                return out
            }
            let name = text[invoke.upperBound..<nameEnd.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let invokeEnd = text.range(
                of: dsmlInvokeClose, range: nameEnd.upperBound..<text.endIndex
            ) else {
                out += text[invoke.lowerBound...]
                return out
            }
            let body = text[nameEnd.upperBound..<invokeEnd.lowerBound]
            guard !name.isEmpty, let arguments = dsmlArguments(body) else {
                out += text[invoke.lowerBound...]
                return out
            }
            out += canonicalBlock(name: name, arguments: arguments)
            cursor = invokeEnd.upperBound
        }
        out += text[cursor...]
        return out
            .replacingOccurrences(of: dsmlCallsOpen, with: "")
            .replacingOccurrences(of: dsmlCallsClose, with: "")
    }

    /// One invoke's parameters as a JSON object, or nil when a parameter is
    /// unterminated.
    static func dsmlArguments(_ body: Substring) -> String? {
        var fields: [(String, Any)] = []
        var cursor = body.startIndex
        while let open = body.range(of: dsmlParameterOpen, range: cursor..<body.endIndex) {
            guard let nameEnd = body.range(of: "\"", range: open.upperBound..<body.endIndex)
            else { return nil }
            let name = String(body[open.upperBound..<nameEnd.lowerBound])
            guard let valueStart = body.range(
                of: ">", range: nameEnd.upperBound..<body.endIndex
            ), let close = body.range(
                of: dsmlParameterClose, range: valueStart.upperBound..<body.endIndex
            ) else { return nil }
            let header = body[nameEnd.upperBound..<valueStart.lowerBound]
            let isString = !header.contains("string=\"false\"")
            let raw = body[valueStart.upperBound..<close.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            fields.append((name, dsmlValue(raw, isString: isString)))
            cursor = close.upperBound
        }
        var object: [String: Any] = [:]
        for (key, value) in fields { object[key] = value }
        guard JSONSerialization.isValidJSONObject(object),
              let encoded = try? JSONSerialization.data(withJSONObject: object),
              let json = String(data: encoded, encoding: .utf8) else { return nil }
        return json
    }

    /// A parameter value: literal text, or JSON when the tag said `string="false"`.
    ///
    /// A `string="false"` body that does not parse is handed on as text rather than
    /// dropped: the tool's own schema validation reports a better error than a call
    /// that silently lost an argument.
    static func dsmlValue(_ raw: String, isString: Bool) -> Any {
        guard !isString, !raw.isEmpty, let data = raw.data(using: .utf8),
              let parsed = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed]
              ) else {
            return raw
        }
        return parsed
    }

    /// `<tool_call_name>N</tool_call_name>[ws]<tool_call_arguments>{json}</tool_call_arguments>`.
    ///
    /// A name without its argument block is still a call: a zero-argument tool is
    /// written that way, and the executor rejects a call whose required parameters
    /// are missing, which is a better error than silently dropping it.
    static func canonicalizeNameArgumentsBlocks(_ text: String) -> String {
        guard text.contains(callNameOpen) else { return text }
        var out = ""
        var cursor = text.startIndex
        while let open = text.range(of: callNameOpen, range: cursor..<text.endIndex) {
            out += text[cursor..<open.lowerBound]
            guard let close = text.range(of: callNameClose, range: open.upperBound..<text.endIndex)
            else {
                // An unterminated name is not a call; leave the text alone so the
                // turn reports it as noise rather than executing a partial name.
                out += text[open.lowerBound...]
                return out
            }
            let name = text[open.upperBound..<close.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var arguments = "{}"
            var next = close.upperBound
            if let argumentsOpen = text.range(
                of: callArgumentsOpen, range: next..<text.endIndex
            ), isOnlyWhitespace(text[next..<argumentsOpen.lowerBound]) {
                // An opener with no closer is a generation that stopped mid-call.
                // Emitting the name with empty arguments would execute a tool the
                // model never finished asking for, so the block is left as text.
                guard let argumentsClose = text.range(
                    of: callArgumentsClose, range: argumentsOpen.upperBound..<text.endIndex
                ) else {
                    out += text[open.lowerBound...]
                    return out
                }
                let body = text[argumentsOpen.upperBound..<argumentsClose.lowerBound]
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                arguments = body.isEmpty ? "{}" : body
                next = argumentsClose.upperBound
            }
            out += canonicalBlock(name: name, arguments: arguments)
            cursor = next
        }
        out += text[cursor...]
        return out
    }

    /// `<tool_call>{"name": …, "arguments": {…}}</tool_call>` — the form the model
    /// emits when it is not shown a schematic.
    static func canonicalizeJSONBlocks(_ text: String) -> String {
        guard text.contains(canonicalOpen) else { return text }
        var out = ""
        var cursor = text.startIndex
        while let open = text.range(of: canonicalOpen, range: cursor..<text.endIndex) {
            out += text[cursor..<open.lowerBound]
            guard let close = text.range(of: canonicalClose, range: open.upperBound..<text.endIndex)
            else {
                out += text[open.lowerBound...]
                return out
            }
            let body = text[open.upperBound..<close.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let parsed = jsonCall(body) else {
                // Already canonical XML, or not a call at all: hand it on untouched
                // so the shared parser decides.
                out += text[open.lowerBound..<close.upperBound]
                cursor = close.upperBound
                continue
            }
            out += canonicalBlock(name: parsed.name, arguments: parsed.arguments)
            cursor = close.upperBound
        }
        out += text[cursor...]
        return out
    }

    /// The vocabulary's own container:
    /// `<|tool▁calls▁begin|><|tool▁call▁begin|>name<|tool▁sep|>{json}<|tool▁call▁end|>…<|tool▁calls▁end|>`.
    ///
    /// The checkpoint does not emit it, but the tokens exist and an API caller may
    /// send it, so it is recognized rather than misread as prose.
    static func canonicalizeSpecialContainer(
        _ text: String,
        markers: DeepSeekV41PromptMarkers
    ) -> String {
        let callOpen = markers.toolCallBegin
        guard !callOpen.isEmpty, text.contains(callOpen) else { return text }
        var out = ""
        var cursor = text.startIndex
        while let open = text.range(of: callOpen, range: cursor..<text.endIndex) {
            out += text[cursor..<open.lowerBound]
            guard let callClose = text.range(
                of: markers.toolCallEnd, range: open.upperBound..<text.endIndex
            ) else {
                out += text[open.lowerBound...]
                return out
            }
            let body = text[open.upperBound..<callClose.lowerBound]
            guard let separator = body.range(of: markers.toolCallSeparator) else {
                out += text[open.lowerBound..<callClose.upperBound]
                cursor = callClose.upperBound
                continue
            }
            let name = body[body.startIndex..<separator.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var arguments = body[separator.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if arguments.hasPrefix("function") {
                arguments = arguments.dropFirst("function".count)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            out += canonicalBlock(
                name: name, arguments: arguments.isEmpty ? "{}" : String(arguments)
            )
            cursor = callClose.upperBound
        }
        out += text[cursor...]
        return out
    }

    /// `{"name": "...", "arguments": {...}}`, with the argument key spelled the ways
    /// different templates spell it.
    static func jsonCall(_ body: String) -> (name: String, arguments: String)? {
        guard body.hasPrefix("{"), let data = body.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = object["name"] as? String, !name.isEmpty else {
            return nil
        }
        let raw = object["arguments"] ?? object["args"] ?? object["parameters"] ?? [String: Any]()
        let arguments: Any
        if let text = raw as? String, let data = text.data(using: .utf8),
           let parsed = try? JSONSerialization.jsonObject(with: data) {
            arguments = parsed
        } else {
            arguments = raw
        }
        guard JSONSerialization.isValidJSONObject(arguments),
              let encoded = try? JSONSerialization.data(withJSONObject: arguments),
              let json = String(data: encoded, encoding: .utf8) else {
            return nil
        }
        return (name, json)
    }

    static func canonicalBlock(name: String, arguments: String) -> String {
        canonicalOpen + canonicalFunctionOpen + name + ">" + arguments + canonicalFunctionClose
            + canonicalClose
    }

    // MARK: - Freezing

    /// True once a call block has closed, so generation stops before the model
    /// narrates its own tool call.
    ///
    /// DSML closes on its calls container (`#27`: that is the last thing the model
    /// writes for an agent-shaped prompt), or on an invoke closer when it wrote a
    /// single call with no container. The plain-text form closes on the arguments
    /// tag; the JSON form closes on the canonical closer; the vocabulary container
    /// closes on its own end marker.
    public static func shouldFreeze(
        _ text: String,
        markers: DeepSeekV41PromptMarkers = .familyDefault
    ) -> Bool {
        if text.contains(dsmlCallsClose) { return true }
        if text.contains(dsmlInvokeClose), !text.contains(dsmlCallsOpen) { return true }
        if text.contains(callArgumentsClose) { return true }
        if text.contains(canonicalClose), jsonCall(argumentsBody(of: text)) != nil {
            return true
        }
        if !markers.toolCallsEnd.isEmpty, text.contains(markers.toolCallsEnd) {
            return true
        }
        return false
    }

    /// The body of the first complete `<tool_call>` block, for the JSON test above.
    static func argumentsBody(of text: String) -> String {
        guard let open = text.range(of: canonicalOpen),
              let close = text.range(of: canonicalClose, range: open.upperBound..<text.endIndex)
        else { return "" }
        return text[open.upperBound..<close.lowerBound]
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Results

    /// A tool result as the next turn sees it: the payload inside the family's own
    /// `tool▁output` pair.
    public static func resultTurn(
        payload: String,
        markers: DeepSeekV41PromptMarkers = .familyDefault
    ) -> String {
        markers.toolOutputBegin + payload + markers.toolOutputEnd
    }

    static func isOnlyWhitespace(_ slice: Substring) -> Bool {
        slice.allSatisfy { $0.isWhitespace }
    }
}
