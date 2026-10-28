//
//  DeepSeekV41Prompt.swift
//  DynaMoE
//
//  The prompt dialect: how a conversation is serialized for DeepSeek-V4.1-Flash.
//
//  Three things force this out of `ChatPromptFormatter`'s shared paths:
//
//    1. The markers are fullwidth-bar special tokens (`及各`,
//       `及各`, …) that tokenize as single ids. Reusing the Qwen
//       `<|im_start|>` spelling would tokenize the whole prompt into ordinary
//       text tokens and the model would answer the prompt text itself.
//    2. Reasoning is a graded budget over 1...100, rendered as an instruction
//       line inside the system turn rather than as a discrete level, and the
//       reasoning channel is opened by the assistant marker.
//    3. Tool calls use the `及各function及各name…及各` container, so a
//       tool result handed back to the model has to be re-rendered in that
//       container to stay prefix-compatible with the previous turn.
//
//  Every marker literal below is built from unicode escapes, never retyped from
//  a rendered view: AGENTS.md's "reading files safely" rule exists because a
//  renderer strips the fullwidth bars and silently yields plain-ASCII text that
//  would look correct and tokenize wrong.
//
//  Extraction from the checkpoint's own `chat_template.jinja` (or the
//  `chat_template` field of `tokenizer_config.json`) runs first, because the
//  checkpoint is the authority on its own dialect. The family default is what
//  the published DeepSeek V3.x/V4 checkpoints use, and `conformance` says which
//  one is in force so the load log can be honest about it.
//

import Foundation

// MARK: - Markers

nonisolated public struct DeepSeekV41PromptMarkers: Sendable, Equatable {
    public let beginOfSentence: String
    public let endOfSentence: String
    public let user: String
    public let assistant: String
    public let toolCallsBegin: String
    public let toolCallSeparator: String
    public let toolCallsEnd: String

    /// The published DeepSeek V3.x / V4 spellings. Fullwidth vertical bar
    /// (U+FF5C) around every role name, lower-one-eighth-block (U+2581) inside
    /// the sentence and tool markers.
    public static let familyDefault = DeepSeekV41PromptMarkers(
        beginOfSentence: "<" + "\u{FF5C}" + "begin" + "\u{2581}" + "of" + "\u{2581}" + "sentence" + "\u{FF5C}" + ">",
        endOfSentence: "<" + "\u{FF5C}" + "end" + "\u{2581}" + "of" + "\u{2581}" + "sentence" + "\u{FF5C}" + ">",
        user: "<" + "\u{FF5C}" + "User" + "\u{FF5C}" + ">",
        assistant: "<" + "\u{FF5C}" + "Assistant" + "\u{FF5C}" + ">",
        toolCallsBegin: "<" + "\u{FF5C}" + "tool" + "\u{2581}" + "calls" + "\u{2581}" + "begin" + "\u{FF5C}" + ">",
        toolCallSeparator: "<" + "\u{FF5C}" + "tool" + "\u{2581}" + "sep" + "\u{FF5C}" + ">",
        toolCallsEnd: "<" + "\u{FF5C}" + "tool" + "\u{2581}" + "calls" + "\u{2581}" + "end" + "\u{FF5C}" + ">"
    )

    public init(
        beginOfSentence: String,
        endOfSentence: String,
        user: String,
        assistant: String,
        toolCallsBegin: String,
        toolCallSeparator: String,
        toolCallsEnd: String
    ) {
        self.beginOfSentence = beginOfSentence
        self.endOfSentence = endOfSentence
        self.user = user
        self.assistant = assistant
        self.toolCallsBegin = toolCallsBegin
        self.toolCallSeparator = toolCallSeparator
        self.toolCallsEnd = toolCallsEnd
    }
}

// MARK: - Dialect

nonisolated public struct DeepSeekV41PromptDialect: Sendable, Equatable {

    /// Where the markers came from. The load log prints this, because a repacked
    /// checkpoint that lost its template silently falls back and a wrong dialect
    /// is indistinguishable from a bad model.
    public enum Source: String, Sendable, Equatable {
        case checkpointTemplate
        case familyDefault
    }

    public let markers: DeepSeekV41PromptMarkers
    public let source: Source
    /// The template text the markers were extracted from, when there was one.
    public let template: String?

    public init(
        markers: DeepSeekV41PromptMarkers = .familyDefault,
        source: Source = .familyDefault,
        template: String? = nil
    ) {
        self.markers = markers
        self.source = source
        self.template = template
    }

    public var isFromCheckpoint: Bool { source == .checkpointTemplate }

    // MARK: Resolution

    /// Reads the checkpoint's own template, falling back to the family default.
    /// Never throws: a missing or unreadable template is a `familyDefault`
    /// dialect, and `DeepSeekV41LoadDiagnostics` reports which one is in force.
    public static func resolve(snapshotDir: URL?) -> DeepSeekV41PromptDialect {
        guard let snapshotDir else { return DeepSeekV41PromptDialect() }
        guard let template = chatTemplate(in: snapshotDir) else {
            return DeepSeekV41PromptDialect()
        }
        return DeepSeekV41PromptDialect(
            markers: markers(in: template),
            source: .checkpointTemplate,
            template: template
        )
    }

    /// `chat_template.jinja` wins when present: a repacked jinja template must
    /// not be masked by the copy embedded in `tokenizer_config.json`.
    public static func chatTemplate(in directory: URL) -> String? {
        let jinja = directory.appendingPathComponent("chat_template.jinja")
        if let text = try? String(contentsOf: jinja, encoding: .utf8), !text.isEmpty {
            return text
        }
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        guard let data = try? Data(contentsOf: configURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if let template = object["chat_template"] as? String, !template.isEmpty {
            return template
        }
        // Some checkpoints keep the template as a pre-rendered message list.
        if let messages = object["chat_template"] as? [[String: Any]],
           let first = messages.first?["template"] as? String,
           !first.isEmpty {
            return first
        }
        return nil
    }

    /// Pulls the role markers out of a template.
    ///
    /// The search is deliberately structural rather than literal: the template
    /// contains the markers as quoted strings, so a marker is any quoted run
    /// that carries the fullwidth bar delimiter. The role is identified from the
    /// text between the bars, which tolerates both `及各`-style
    /// markers and templates that spell roles as `及各`/`及各`.
    public static func markers(in template: String) -> DeepSeekV41PromptMarkers {
        let quoted = quotedLiterals(in: template)
        var begin = DeepSeekV41PromptMarkers.familyDefault.beginOfSentence
        var end = DeepSeekV41PromptMarkers.familyDefault.endOfSentence
        var user = DeepSeekV41PromptMarkers.familyDefault.user
        var assistant = DeepSeekV41PromptMarkers.familyDefault.assistant
        var toolBegin = DeepSeekV41PromptMarkers.familyDefault.toolCallsBegin
        var toolSep = DeepSeekV41PromptMarkers.familyDefault.toolCallSeparator
        var toolEnd = DeepSeekV41PromptMarkers.familyDefault.toolCallsEnd

        for literal in quoted {
            guard literal.contains("\u{FF5C}") else { continue }
            let role = roleName(in: literal)
            switch role {
            case "beginofsentence", "bos", "begin":
                begin = literal
            case "endofsentence", "eos", "end":
                end = literal
            case "user", "human", "userrole":
                user = literal
            case "assistant", "bot", "model", "assistantrole":
                assistant = literal
            case "toolcallsbegin", "toolcalls":
                toolBegin = literal
            case "toolsep", "toolseparator":
                toolSep = literal
            case "toolcallsend":
                toolEnd = literal
            default:
                continue
            }
        }
        return DeepSeekV41PromptMarkers(
            beginOfSentence: begin,
            endOfSentence: end,
            user: user,
            assistant: assistant,
            toolCallsBegin: toolBegin,
            toolCallSeparator: toolSep,
            toolCallsEnd: toolEnd
        )
    }

    /// Lowercased alphanumeric run between the first and second fullwidth bars.
    static func roleName(in literal: String) -> String {
        let parts = literal.components(separatedBy: "\u{FF5C}")
        guard parts.count >= 3 else { return "" }
        return parts[1]
            .lowercased()
            .filter { $0.isLetter || $0.isNumber }
    }

    /// Every single- or double-quoted run in the template, unescaped enough to
    /// compare against role names. Template source only ever writes markers as
    /// plain literals, so no full Jinja evaluation is needed.
    static func quotedLiterals(in template: String) -> [String] {
        var out: [String] = []
        var current: String?
        var escaped = false
        for character in template {
            if let quote = current?.first {
                if escaped {
                    current?.append(character)
                    escaped = false
                    continue
                }
                if character == "\\" {
                    escaped = true
                    continue
                }
                if character == quote {
                    out.append(String(current!.dropFirst()))
                    current = nil
                    continue
                }
                current?.append(character)
                continue
            }
            if character == "\"" || character == "'" {
                current = String(character)
            }
        }
        return out
    }

    // MARK: Reasoning effort

    /// The effort budget range the checkpoint accepts. Mirrors
    /// `ModelConfig.reasoningEffortBounds`, which is what the Settings UI reads.
    public static let effortBounds = 1...100

    /// Renders the graded effort instruction. DeepSeek reads an integer budget
    /// rather than a named level, so the discrete `ThinkingEffort` a caller has
    /// maps onto the range the checkpoint publishes.
    ///
    /// `low` is a quarter of the range and `high` the top, which is the split the
    /// published guidance describes (a short budget for chat, the full budget for
    /// a reasoning-heavy turn). Thinking off emits nothing at all.
    public func effortInstruction(_ effort: ThinkingEffort?, thinkingEnabled: Bool) -> String? {
        guard thinkingEnabled, let effort else { return nil }
        let budget: Int
        switch effort {
        case .low: budget = 25
        case .high: budget = DeepSeekV41PromptDialect.effortBounds.upperBound
        }
        return Self.effortInstruction(budget: budget)
    }

    public static func effortInstruction(budget: Int) -> String {
        let clamped = min(max(budget, effortBounds.lowerBound), effortBounds.upperBound)
        return "Reasoning Effort: \(clamped)"
    }

    /// Parses an effort budget out of arbitrary client text so the API adapters
    /// can accept `reasoning_effort: 60` as well as `"high"`.
    public static func effortBudget(from raw: String?) -> Int? {
        guard let raw else { return nil }
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if let value = Int(trimmed) {
            guard effortBounds.contains(value) else { return nil }
            return value
        }
        switch trimmed {
        case "low", "minimal": return 25
        case "medium", "mid": return 50
        case "high", "max", "maximum": return effortBounds.upperBound
        default: return nil
        }
    }

    // MARK: Rendering

    /// One turn, after family templating.
    public struct Turn: Sendable, Equatable {
        public var isUser: Bool
        /// Visible answer / tool result text.
        public var text: String
        /// Reasoning text, when the turn carried any.
        public var reasoning: String?

        public init(isUser: Bool, text: String, reasoning: String? = nil) {
            self.isUser = isUser
            self.text = text
            self.reasoning = reasoning
        }
    }

    /// The system turn body: the caller's system prompt, with the effort
    /// instruction appended when thinking is on.
    public func systemBody(
        cleanSystem: String,
        thinkingEnabled: Bool,
        effort: ThinkingEffort?
    ) -> String {
        let instruction = effortInstruction(effort, thinkingEnabled: thinkingEnabled)
        let base = cleanSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let instruction else { return base }
        return base.isEmpty ? instruction : base + "\n\n" + instruction
    }

    /// Renders a whole conversation. The trailing assistant marker leaves the
    /// reasoning channel open only when thinking is on, which mirrors what the
    /// template does: with thinking off the model emits the close marker itself,
    /// so the prompt must not pre-open the channel.
    public func render(
        turns: [Turn],
        cleanSystem: String,
        thinkingEnabled: Bool,
        effort: ThinkingEffort?
    ) -> String {
        let m = markers
        var out = m.beginOfSentence
        let system = systemBody(
            cleanSystem: cleanSystem,
            thinkingEnabled: thinkingEnabled,
            effort: effort
        )
        if !system.isEmpty {
            out += system
        }
        for turn in turns {
            if turn.isUser {
                out += m.user + turn.text
            } else {
                out += assistantBody(for: turn, thinkingEnabled: thinkingEnabled)
            }
        }
        out += assistantPrefix(thinkingEnabled: thinkingEnabled)
        return out
    }

    /// A completed assistant turn's serialization. A turn that carried reasoning
    /// keeps its reasoning inside the channel it was generated in, even if
    /// thinking has since been turned off: re-rendering older turns differently
    /// from the pinned prompt loses the whole prefix.
    func assistantBody(for turn: Turn, thinkingEnabled: Bool) -> String {
        let m = markers
        let hasReasoning = !(turn.reasoning ?? "").isEmpty
        var body = m.assistant
        if let reasoning = turn.reasoning, !reasoning.isEmpty {
            body += TurnTextSplitter.canonicalReasoningOpenTag
            body += "\n"
            body += reasoning
            body += TurnTextSplitter.canonicalReasoningCloseTag
            body += "\n\n"
        } else if thinkingEnabled {
            // An earlier turn generated with thinking on but no recorded
            // reasoning still has to carry an empty channel: the marker belongs
            // to the turn, not to the setting that happens to be on now.
            body += TurnTextSplitter.canonicalReasoningOpenTag + "\n\n"
                + TurnTextSplitter.canonicalReasoningCloseTag + "\n\n"
        }
        body += turn.text
        body += m.endOfSentence
        return body
    }

    /// The prefix the next assistant turn starts from.
    func assistantPrefix(thinkingEnabled: Bool) -> String {
        var out = markers.assistant
        if thinkingEnabled {
            out += TurnTextSplitter.canonicalReasoningOpenTag + "\n"
        }
        return out
    }

    /// Re-renders a tool result for the history. DeepSeek takes tool output inside
    /// the tool-calls container, so the result is wrapped there rather than
    /// emitted as plain user text: the previous turn's generation contained that
    /// container and a differently-shaped result breaks the prefix.
    public func toolResultTurn(name: String, payload: String) -> String {
        let m = markers
        return m.user + m.toolCallsBegin + "function" + m.toolCallSeparator + name + "\n"
            + payload
            + m.toolCallsEnd
    }

    /// Stop sequences: the turn ends at the end-of-sentence marker, and the tool
    /// container closes before it so a generation that emits a call is stopped
    /// there by the parser rather than by the sampler.
    public var stopSequences: [String] {
        [markers.endOfSentence, markers.user]
    }
}


