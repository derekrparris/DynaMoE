//
//  Inkling2ToolDialect.swift
//  DynaMoE
//
//  How Inkling 2 Mini calls a tool, and how its results come back.
//
//  Unlike DeepSeek's, this is not measured from the weights: the checkpoint
//  SHIPS its template, and the template defines both halves of the protocol --
//
//      <|message_model|>NAME<|content_invoke_tool_json|>{"name":NAME,"args":{...}}<|end_message|>
//      <|message_tool|>NAME<|content_text|>RESULT<|end_message|>
//
//  so every literal here is read off `chat_template.jinja` and every shape is
//  pinned by a golden the template itself rendered (`Tools/inkling2_tool_goldens.py`).
//  What the template does NOT define is the model's behavior around the call:
//  whether it writes prose before or after one, and whether a half-written call
//  can be executed. The answers here are the conservative ones -- freeze once a
//  call is complete, never execute an incomplete one -- and they are the same
//  rules every other family's dialect follows.
//
//  Pure logic: no weights, no state. The agent loop, HITL gating, timeline and
//  history rendering all stay on their one shared path.
//

import Foundation

nonisolated public enum Inkling2ToolDialect {

    // MARK: - The checkpoint's own spellings

    /// The canonical block the shared parser consumes.
    static let canonicalOpen = "<tool_call>"
    static let canonicalClose = "</tool_call>"
    static let canonicalFunctionOpen = "<function="
    static let canonicalFunctionClose = "</function>"

    /// The JSON payload's keys, as the template writes them.
    static let nameKey = "name"
    static let argumentsKey = "args"

    // MARK: - Instructions

    /// The tool-format block appended to the system prompt for this family.
    ///
    /// The template's own spelling is taught, built from the marker constants
    /// rather than typed: a rendered view strips tag-like text, so a literal
    /// copied from one is how a dialect ends up teaching a tag the tokenizer
    /// does not have.
    public static func instructions(markers: Inkling2PromptMarkers = .familyDefault) -> String {
        let model = markers.model
        let invoke = markers.contentInvokeToolJSON
        let end = markers.endMessage
        return """
        # Tool Call Format

        You can call tools. When you do, write ONLY the call block: one model message per call,
        whose text is the tool's name, then the invoke marker and a JSON object holding the
        tool's name and its arguments.

        \(model)example_function_name\(invoke){"name":"example_function_name","args":{"example_parameter_1":"value_1"}}\(end)

        Reminder:
        - The name MUST be an exact match to one of the available tools
        - Every argument goes in `args`, keyed by its JSON name
        - Required parameters MUST be specified
        - You may reason in natural language BEFORE the call block, but NOT after it
        - If no tool fits, answer the question like normal with your current knowledge and do not mention tool calls
        """
    }

    // MARK: - Parsing what the model writes

    /// One call block: `<|message_model|>NAME<|content_invoke_tool_json|>JSON<|end_message|>`.
    ///
    /// The block is only a call when the JSON is complete and parses, when it
    /// names a tool, and when its `args` is an object. Anything else -- a
    /// truncated payload, a `name` with no arguments yet, a JSON string instead
    /// of an object -- is left in the text as prose: executing a call the model
    /// never finished asking for is worse than showing the fragment.
    static func callBlocks(
        in text: String, markers: Inkling2PromptMarkers
    ) -> [(name: String, arguments: [String: Any], raw: String)] {
        var calls: [(name: String, arguments: [String: Any], raw: String)] = []
        var cursor = text.startIndex
        while let open = text.range(of: markers.model, range: cursor..<text.endIndex) {
            guard let invoke = text.range(
                of: markers.contentInvokeToolJSON, range: open.upperBound..<text.endIndex
            ) else { return calls }
            let name = text[open.upperBound..<invoke.lowerBound]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard let close = text.range(
                of: markers.endMessage, range: invoke.upperBound..<text.endIndex
            ) else {
                // Still being written: the next delta re-parses it.
                return calls
            }
            let payload = String(text[invoke.upperBound..<close.lowerBound])
            if !name.isEmpty,
               let data = payload.data(using: .utf8),
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let declared = object[nameKey] as? String,
               declared == name,
               let arguments = object[argumentsKey] as? [String: Any] {
                calls.append((
                    name: name, arguments: arguments,
                    raw: String(text[open.lowerBound..<close.upperBound])
                ))
            }
            // Not a valid call: keep scanning after it, and leave the bytes.
            cursor = close.upperBound
        }
        return calls
    }

    /// Rewrites the model's call blocks into the canonical block the shared
    /// parser understands, leaving everything else untouched.
    public static func canonicalized(
        _ text: String, markers: Inkling2PromptMarkers = .familyDefault
    ) -> String {
        let calls = callBlocks(in: text, markers: markers)
        guard !calls.isEmpty else { return text }
        // The prose is kept in order around the calls: the calls are what the
        // executor needs, and the text between them is what the user reads.
        var output = ""
        var remaining = text[text.startIndex...]
        for call in calls {
            guard let range = remaining.range(of: call.raw) else { continue }
            output += remaining[..<range.lowerBound]
            output += canonicalBlock(name: call.name, arguments: call.arguments)
            remaining = remaining[range.upperBound...]
        }
        output += remaining
        return output
    }

    static func canonicalBlock(name: String, arguments: [String: Any]) -> String {
        var object: [String: Any] = [:]
        for (key, value) in arguments { object[key] = value }
        let json: String
        if JSONSerialization.isValidJSONObject(object),
           let data = try? JSONSerialization.data(withJSONObject: object),
           let text = String(data: data, encoding: .utf8) {
            json = text
        } else {
            json = "{}"
        }
        return canonicalOpen + canonicalFunctionOpen + name + ">" + json + canonicalFunctionClose
            + canonicalClose
    }

    // MARK: - Freezing

    /// True once a call block has closed, so generation stops before the model
    /// narrates its own tool call.
    ///
    /// The marker to key on is the one the template puts at the END of a call
    /// block, and only when a block is actually complete: an `end_message`
    /// alone also closes ordinary text, so the test is a parsed call, not the
    /// marker.
    public static func shouldFreeze(
        _ text: String, markers: Inkling2PromptMarkers = .familyDefault
    ) -> Bool {
        guard text.contains(markers.contentInvokeToolJSON) else { return false }
        return !callBlocks(in: text, markers: markers).isEmpty
    }

    // MARK: - Results

    /// A tool result as the next turn sees it: the family's own tool message,
    /// the tool's name after the role marker, and the payload as a text block.
    public static func resultTurn(
        name: String, payload: String, markers: Inkling2PromptMarkers = .familyDefault
    ) -> String {
        markers.tool + name + markers.contentText + payload + markers.endMessage
    }
}
