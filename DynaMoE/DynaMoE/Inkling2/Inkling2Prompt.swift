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

// MARK: - Errors

nonisolated public enum Inkling2PromptError: Error, Equatable, CustomStringConvertible {
    case effortOutOfRange(Float)
    case unknownRole(String)
    case imageNotSupportedHere(String)
    case emptyToolName

    public var description: String {
        switch self {
        case .effortOutOfRange(let value):
            return "reasoning_effort \(value) must be in 0.0...0.99"
        case .unknownRole(let role):
            return "unknown message role: \(role)"
        case .imageNotSupportedHere(let role):
            return "image content is only supported for user and tool messages, not \(role)"
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
    /// `<|content_image|><|unused_200054|>`; the tower runs later.
    case image

    public static func parts(from content: String) -> [Inkling2ContentPart] { [.text(content)] }
}

/// One conversation turn, in the shape the template consumes.
nonisolated public enum Inkling2Turn: Sendable, Equatable {
    case system(String)
    case user([Inkling2ContentPart])
    case assistant(reasoning: String?, content: String?)
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

    /// Renders a conversation exactly as the checkpoint's template does for the
    /// text path. `effort` is the numeric reasoning effort; nil means the
    /// template's default (0.9).
    public func render(turns: [Inkling2Turn], effort: Float? = nil) throws -> String {
        let resolvedEffort = try Inkling2Effort.validate(effort ?? Inkling2Effort.defaultValue)
        var output = ""
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

        case .assistant(let reasoning, let content):
            var output = ""
            if let reasoning, !reasoning.isEmpty {
                output += markers.model + markers.contentThinking + reasoning + markers.endMessage
            }
            // An assistant turn with only reasoning or tool calls has no text
            // block, matching the template's condition.
            if let content, (!content.isEmpty || reasoning == nil) {
                output += markers.model + markers.contentText + content + markers.endMessage
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
            case .image:
                guard role == "user" || role == "tool" else {
                    throw Inkling2PromptError.imageNotSupportedHere(role)
                }
                output += markers.contentImage + markers.imagePlaceholder + markers.endMessage
            }
        }
        return output
    }

    /// What a live turn appends after the prompt before the model writes: the
    /// template has no assistant prefix, so nothing.
    public var assistantPrefix: String { "" }
}
