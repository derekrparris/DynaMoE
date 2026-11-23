//
//  Inkling2Prompt.swift
//  DynaMoE
//
//  The prompt dialect: how a conversation is serialized for Inkling 2 Mini.
//
//  This is not ChatML and not any other family's template. The checkpoint's
//  `chat_template.jinja` is a sequence of ROLE markers followed by TYPED CONTENT
//  blocks:
//
//      <|message_system|><|content_text|>...<|end_message|>
//      <|message_user|><|content_text|>...<|end_message|>
//      <|message_model|><|content_thinking|>...<|end_message|>
//      <|message_model|><|content_text|>...<|end_message|><|content_model_end_sampling|>
//
//  with two rules that are easy to miss and are pinned by goldens:
//
//    * A reasoning-effort line is emitted as its own SYSTEM message right
//      before the first non-system message -- not at the head of the prompt,
//      and not appended at the end. Its text is `Thinking effort level: X`
//      where X is `0` for 0.0 and the decimal otherwise (Python's `str` of the
//      float, which is what the template interpolates).
//    * A completed assistant turn ends with `<|content_model_end_sampling|>`
//      AFTER its content blocks, and that same token is the EOS. The template's
//      `{% generation %}` span covers exactly the assistant's own blocks, so a
//      training mask and a live turn agree about where the model's text is.
//
//  There is no assistant prefix: the template ends after the last non-assistant
//  message, so generation begins immediately after that message's
//  `<|end_message|>`.
//
//  Every marker literal is spelled from the tokenizer's own added-token names
//  (`<|message_user|>` and friends) rather than retyped from a rendered view:
//  AGENTS.md's "reading files safely" rule exists because a renderer can strip
//  a tag and silently yield ASCII that looks right and tokenizes wrong.
//
//  Tool declarations, call parsing and canonical replay are Phase 4. This file
//  renders the text path (system / user / assistant / named tool result), which
//  is what a plain chat turn and a history replay need.
//

import Foundation

// MARK: - Markers

nonisolated public struct Inkling2PromptMarkers: Sendable, Equatable {
    public let system: String
    public let user: String
    public let model: String
    public let tool: String
    public let contentText: String
    public let contentImage: String
    public let contentThinking: String
    public let contentModelEndSampling: String
    public let endMessage: String
    public let beginOfText: String
    /// The image placeholder token, `<|unused_200054|>`.
    public let imagePlaceholder: String
    /// Tool-call content types (rendered by the Phase 4 dialect).
    public let contentInvokeToolJSON: String
    public let contentInvokeToolText: String
    public let contentXML: String
    public let contentToolError: String

    /// The checkpoint's own added-token spellings, in its own order.
    public static let familyDefault = Inkling2PromptMarkers(
        system: "<|message_system|>",
        user: "<|message_user|>",
        model: "<|message_model|>",
        tool: "<|message_tool|>",
        contentText: "<|content_text|>",
        contentImage: "<|content_image|>",
        contentThinking: "<|content_thinking|>",
        contentModelEndSampling: "<|content_model_end_sampling|>",
        endMessage: "<|end_message|>",
        beginOfText: "<|begin_of_text|>",
        imagePlaceholder: "<|unused_200054|>",
        contentInvokeToolJSON: "<|content_invoke_tool_json|>",
        contentInvokeToolText: "<|content_invoke_tool_text|>",
        contentXML: "<|content_xml|>",
        contentToolError: "<|content_tool_error|>"
    )
}

// MARK: - Effort

/// The reasoning-effort vocabulary the checkpoint's template understands.
///
/// `none` is 0.0, `minimal` 0.1, `low` 0.2, `medium` 0.7, `high` 0.9, `xhigh`
/// and `max` 0.99; the default is 0.9. A numeric effort is accepted in
/// [0.0, 0.99]. This is not DeepSeek's 1...100 graded budget and must not be
/// forced into it.
nonisolated public enum Inkling2Effort: Sendable, Equatable {
    public static let defaultValue: Float = 0.9
    public static let range: ClosedRange<Float> = 0.0...0.99

    /// The template's own map, in its own order.
    public static let named: [(name: String, value: Float)] = [
        ("none", 0.0), ("minimal", 0.1), ("low", 0.2), ("medium", 0.7),
        ("high", 0.9), ("xhigh", 0.99), ("max", 0.99)
    ]

    public static func value(for name: String) -> Float? {
        let key = name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return named.first { $0.name == key }?.value
    }

    /// Clamps nothing: a value outside the range is refused, the way the
    /// template raises.
    public static func validate(_ value: Float) throws -> Float {
        guard value.isFinite, range.contains(value) else {
            throw Inkling2PromptError.effortOutOfRange(value)
        }
        return value
    }

    /// How the template interpolates the number: `0` for exactly zero, the
    /// shortest decimal otherwise. Swift's `String(describing:)` of a Float
    /// gives "0.9", "0.2", "0.99", "0.0" -> the zero case is special-cased.
    public static func rendered(_ value: Float) -> String {
        value == 0 ? "0" : String(describing: value)
    }
}

// MARK: - Tool declarations

/// One tool the model may call, as the template's `tool_declare` message
/// carries it.
///
/// The template accepts either a bare function or an OpenAI-style wrapper and
/// normalizes both to the same four keys, so this is the normalized form: what
/// a caller has is a name, a description and a JSON-schema parameter object.
nonisolated public struct Inkling2ToolDeclaration: Sendable, Equatable {
    public let name: String
    public let description: String
    /// The parameters as a JSON schema object, already in JSON-value form.
    public let parameters: [String: MCPJSONValue]
    /// `function` unless the caller says otherwise; the template defaults it.
    public let type: String

    public init(
        name: String,
        description: String = "",
        parameters: [String: MCPJSONValue] = [:],
        type: String = "function"
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
        self.type = type
    }

    /// From the app's own tool definition, whose parameters are `AnyCodable`.
    public init(_ definition: ToolDefinition) {
        self.name = definition.function.name
        self.description = definition.function.description
        self.parameters = definition.function.parameters.mapValues {
            MCPJSONValue(any: $0.value)
        }
        self.type = definition.type
    }

    /// From an OpenAI-route tool: the schema arrives as raw JSON text, which is
    /// re-parsed here so the template's `tojson` sees the client's own object
    /// rather than a string containing it.
    init(_ tool: OpenAIChatRequest.Tool) {
        let parameters: [String: MCPJSONValue]
        if let data = tool.function.parametersData,
           let parsed = MCPJSONValue(data: data),
           case .object(let fields) = parsed {
            parameters = fields
        } else {
            parameters = [:]
        }
        self.init(
            name: tool.function.name,
            description: tool.function.description ?? "",
            parameters: parameters,
            type: tool.type ?? "function"
        )
    }

    /// From an Anthropic-route tool, whose input schema is the same raw JSON.
    init(_ tool: AnthropicMessagesRequest.Tool) {
        let parameters: [String: MCPJSONValue]
        if let data = tool.inputSchemaData,
           let parsed = MCPJSONValue(data: data),
           case .object(let fields) = parsed {
            parameters = fields
        } else {
            parameters = [:]
        }
        self.init(name: tool.name, description: tool.description ?? "", parameters: parameters)
    }
}

// MARK: - JSON as the template writes it

/// The subset of JSON the template's `tojson` produces, byte for byte.
///
/// `tojson` in jinja2 is `json.dumps` with `sort_keys=True` and the caller's
/// separators, and markupsafe's wrapper does NOT escape non-ASCII -- so an
/// accented character stays literal while a newline becomes `\n`. Keys sort by
/// codepoint (Python's string comparison), which for valid UTF-8 is byte order.
nonisolated enum Inkling2TemplateJSON {
    /// Codepoint order, which is what Python's `sorted` on `str` does.
    static func keyOrder(_ left: String, _ right: String) -> Bool {
        left.unicodeScalars.lexicographicallyPrecedes(right.unicodeScalars) { a, b in
            a.value < b.value
        }
    }

    /// Compact JSON: `(",", ":")` separators, object keys sorted.
    static func compact(_ value: MCPJSONValue) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return flag ? "true" : "false"
        case .int(let number): return String(number)
        // Python prints a float with `repr`, the shortest round-trip form.
        // Swift's description is also shortest round-trip; they differ only in
        // exponent spelling, which no template value in this family reaches.
        case .double(let number): return String(describing: number)
        case .string(let text): return quoted(text)
        case .array(let items): return "[" + items.map(compact).joined(separator: ",") + "]"
        case .object(let fields):
            let keys = fields.keys.sorted(by: keyOrder)
            return "{" + keys.map { quoted($0) + ":" + compact(fields[$0] ?? .null) }
                .joined(separator: ",") + "}"
        }
    }

    /// Python's string escaping with `ensure_ascii=False`: quote, backslash, the
    /// five short control escapes, `\u00XX` for the rest below 0x20, and every
    /// other scalar literal.
    static func quoted(_ text: String) -> String {
        var output = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{08}": output += "\\b"
            case "\u{0C}": output += "\\f"
            default:
                if scalar.value < 0x20 {
                    output += String(format: "\\u%04x", scalar.value)
                } else {
                    output.unicodeScalars.append(scalar)
                }
            }
        }
        return output + "\""
    }
}

// MARK: - Errors

nonisolated public enum Inkling2PromptError: Error, Equatable, CustomStringConvertible {
    case effortOutOfRange(Float)
    case unknownRole(String)
    case imageNotSupportedHere(String)
    case invalidImageGrid(rows: Int, columns: Int)
    case emptyToolName

    public var description: String {
        switch self {
        case .effortOutOfRange(let value):
            return "reasoning_effort \(value) must be in 0.0...0.99"
        case .unknownRole(let role):
            return "unknown message role: \(role)"
        case .imageNotSupportedHere(let role):
            return "image content is only supported for user and tool messages, not \(role)"
        case .invalidImageGrid(let rows, let columns):
            return "an image grid of \(rows)x\(columns) cells has no language positions"
        case .emptyToolName:
            return "a tool message must name the tool whose result it carries"
        }
    }
}

// MARK: - Turns

/// One block of a message: the template's typed content.
nonisolated public enum Inkling2ContentPart: Sendable, Equatable {
    case text(String)
    /// An attachment, which the template renders as
    /// `<|content_image|><|unused_200054|>`. The grid is what the marker
    /// expands to after tokenization -- one marker per cell -- so it belongs to
    /// the part rather than to a caller's side array; the tower's rows travel
    /// in the turn plan (`Inkling2ImagePrompt`).
    case image(Inkling2ImageGrid)

    public static func parts(from content: String) -> [Inkling2ContentPart] { [.text(content)] }
}

/// One tool call in a conversation turn: the name and its already-parsed
/// arguments.
///
/// The template refuses a JSON STRING of arguments ("canonicalize upstream"),
/// so the arguments here are a mapping by construction -- a caller holding the
/// model's raw JSON parses it first, and an unparseable call never becomes one
/// of these (which is what keeps a truncated call from being executed).
nonisolated public struct Inkling2ToolCall: Sendable, Equatable {
    public let name: String
    /// The template iterates this with `dictsort`, so insertion order is not
    /// meaningful; `sortedKeys` at encode time is what the golden compares.
    public let arguments: [String: MCPJSONValue]

    public init(name: String, arguments: [String: MCPJSONValue]) {
        self.name = name
        self.arguments = arguments
    }
}

/// One conversation turn, in the shape the template consumes.
nonisolated public enum Inkling2Turn: Sendable, Equatable {
    case system(String)
    case user([Inkling2ContentPart])
    /// An assistant turn. `calls` are rendered after the content blocks, one
    /// `message_model` block each, and the turn's end-of-sampling marker comes
    /// after all of them -- so a call-only turn is one or more call blocks and
    /// then the marker, exactly as the template writes it.
    case assistant(reasoning: String?, content: String?, calls: [Inkling2ToolCall] = [])
    /// A tool RESULT. The template writes the tool's name straight after the
    /// role marker, so the name is part of the turn.
    case tool(name: String, content: [Inkling2ContentPart])

    /// The template's role key, for the error messages that name one.
    public var role: String {
        switch self {
        case .system: return "system"
        case .user: return "user"
        case .assistant: return "assistant"
        case .tool: return "tool"
        }
    }
}

// MARK: - Renderer

nonisolated public struct Inkling2Prompt: Sendable {

    public let markers: Inkling2PromptMarkers

    public init(markers: Inkling2PromptMarkers = .familyDefault) {
        self.markers = markers
    }

    /// Renders a conversation exactly as the checkpoint's template does.
    /// `effort` is the numeric reasoning effort; nil means the template's
    /// default (0.9). `tools` are declarations, rendered as the template's
    /// `tool_declare` system message at the HEAD of the prompt.
    public func render(
        turns: [Inkling2Turn], effort: Float? = nil, tools: [Inkling2ToolDeclaration] = []
    ) throws -> String {
        let resolvedEffort = try Inkling2Effort.validate(effort ?? Inkling2Effort.defaultValue)
        var output = ""
        if !tools.isEmpty { output += try toolDeclarationLine(tools) }
        var effortEmitted = false
        for turn in turns {
            if !effortEmitted && turn.role != "system" {
                output += try effortLine(resolvedEffort)
                effortEmitted = true
            }
            output += try render(turn: turn)
        }
        if !effortEmitted {
            // The template emits the line after the loop when every message was a
            // system message, so a system-only prompt still carries it.
            output += try effortLine(resolvedEffort)
        }
        return output
    }

    /// The tool declarations as the template writes them: one system message
    /// whose text begins `tool_declare`, then the specs as compact JSON with
    /// sorted keys.
    ///
    /// Each spec is `{description, name, parameters, type}` with the template's
    /// defaults -- an absent description is `""` and absent parameters are `{}`
    /// -- and `type` falls back to `function` only when the declaration has no
    /// type. The ordering is `sort_keys=true`, i.e. byte order of the keys, at
    /// every level.
    public func toolDeclarationLine(_ tools: [Inkling2ToolDeclaration]) throws -> String {
        let specs: [MCPJSONValue] = tools.map { tool in
            let parameters = tool.parameters.isEmpty
                ? MCPJSONValue.object([:])
                : MCPJSONValue.object(tool.parameters)
            return .object([
                "description": .string(tool.description),
                "name": .string(tool.name),
                "parameters": parameters,
                "type": .string(tool.type.isEmpty ? "function" : tool.type)
            ])
        }
        let json = Inkling2TemplateJSON.compact(.array(specs))
        return markers.system + "tool_declare" + markers.contentXML + json + markers.endMessage
    }

    /// The effort line as its own system message.
    public func effortLine(_ effort: Float) throws -> String {
        let value = try Inkling2Effort.validate(effort)
        return markers.system + markers.contentText
            + "Thinking effort level: " + Inkling2Effort.rendered(value)
            + markers.endMessage
    }

    /// One turn, without the effort line.
    ///
    /// The template writes `<|end_message|>` after EVERY content block and no
    /// marker after the role itself, so a turn's text ends with its last
    /// block's marker and there is nothing else to close.
    public func render(turn: Inkling2Turn) throws -> String {
        switch turn {
        case .system(let content):
            let body = try render(parts: [.text(content)], role: "system")
            return markers.system + body

        case .user(let parts):
            let body = try render(parts: parts, role: "user")
            return markers.user + body

        case .assistant(let reasoning, let content, let calls):
            var output = ""
            if let reasoning, !reasoning.isEmpty {
                output += markers.model + markers.contentThinking + reasoning + markers.endMessage
            }
            // An assistant turn with only reasoning or tool calls has no text
            // block, matching the template's condition.
            if let content, (!content.isEmpty || (reasoning == nil && calls.isEmpty)) {
                output += markers.model + markers.contentText + content + markers.endMessage
            }
            // One call block per call, each opening its own model message with
            // the tool's name. The arguments are re-encoded rather than
            // echoed: the template renders `{"name": <json>, "args": {<sorted
            // key>: <json>}}`, so a caller's own formatting never survives.
            for call in calls {
                let name = Inkling2TemplateJSON.compact(.string(call.name))
                var fields: [String] = []
                for key in call.arguments.keys.sorted(by: Inkling2TemplateJSON.keyOrder) {
                    fields.append(
                        Inkling2TemplateJSON.compact(.string(key)) + ":"
                            + Inkling2TemplateJSON.compact(call.arguments[key] ?? .null)
                    )
                }
                output += markers.model + call.name + markers.contentInvokeToolJSON
                    + "{\"name\":" + name + ",\"args\":{" + fields.joined(separator: ",") + "}}"
                    + markers.endMessage
            }
            output += markers.contentModelEndSampling
            return output

        case .tool(let name, let parts):
            guard !name.isEmpty else { throw Inkling2PromptError.emptyToolName }
            let body = try render(parts: parts, role: "tool")
            return markers.tool + name + body
        }
    }

    /// The typed content blocks of one message. Internal rather than private so
    /// the image-placement refusal can be exercised on a role the turn enum
    /// cannot express (an assistant turn carrying a picture).
    func render(parts: [Inkling2ContentPart], role: String) throws -> String {
        var output = ""
        for part in parts {
            switch part {
            case .text(let text):
                output += markers.contentText + text + markers.endMessage
            case .image(let grid):
                guard role == "user" || role == "tool" else {
                    throw Inkling2PromptError.imageNotSupportedHere(role)
                }
                // The template writes one placeholder per image; the cell count
                // is what it expands to after tokenization. A grid with no cells
                // would expand to nothing, so it is refused by name here rather
                // than shrinking the span the caller priced.
                guard grid.isValid else {
                    throw Inkling2PromptError.invalidImageGrid(
                        rows: grid.cellRows, columns: grid.cellColumns
                    )
                }
                output += markers.contentImage + markers.imagePlaceholder + markers.endMessage
            }
        }
        return output
    }

    /// Every attachment's grid in RENDER ORDER, which is the order the marker
    /// tokens appear in after tokenization.
    ///
    /// A caller holding encoded spans builds them in this order, so a picture
    /// cannot land on another picture's markers: `Inkling2ImagePrompt.expand`
    /// consumes placeholders and spans left to right and refuses a count
    /// mismatch, and this is what makes "left to right" the conversation's own
    /// order rather than a caller's guess.
    public static func imageGrids(in turns: [Inkling2Turn]) -> [Inkling2ImageGrid] {
        func grids(in parts: [Inkling2ContentPart]) -> [Inkling2ImageGrid] {
            parts.compactMap {
                if case .image(let grid) = $0 { return grid }
                return nil
            }
        }
        return turns.flatMap { turn -> [Inkling2ImageGrid] in
            switch turn {
            case .system, .assistant: return []
            case .user(let parts): return grids(in: parts)
            case .tool(_, let parts): return grids(in: parts)
            }
        }
    }

    /// What a live turn appends after the prompt before the model writes: the
    /// template has no assistant prefix, so nothing.
    public var assistantPrefix: String { "" }
}
