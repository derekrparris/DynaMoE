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
    public let system: String
    public let user: String
    public let assistant: String
    public let toolCallsBegin: String
    public let toolCallSeparator: String
    public let toolCallsEnd: String
    /// The reasoning channel the template opens and closes. Defaults come from
    /// the checkpoint's declared special tokens, not from a renderer constant.
    public let reasoningOpen: String
    public let reasoningClose: String

    public static let defaultSystem = "<" + "\u{FF5C}" + "System" + "\u{FF5C}" + ">"
    public static let defaultReasoningOpen = TurnTextSplitter.canonicalReasoningOpenTag
    public static let defaultReasoningClose = TurnTextSplitter.canonicalReasoningCloseTag

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
        toolCallsEnd: "<" + "\u{FF5C}" + "tool" + "\u{2581}" + "calls" + "\u{2581}" + "end" + "\u{FF5C}" + ">",
        system: DeepSeekV41PromptMarkers.defaultSystem,
        reasoningOpen: DeepSeekV41PromptMarkers.defaultReasoningOpen,
        reasoningClose: DeepSeekV41PromptMarkers.defaultReasoningClose
    )

    public init(
        beginOfSentence: String,
        endOfSentence: String,
        user: String,
        assistant: String,
        toolCallsBegin: String,
        toolCallSeparator: String,
        toolCallsEnd: String,
        system: String = DeepSeekV41PromptMarkers.defaultSystem,
        reasoningOpen: String = DeepSeekV41PromptMarkers.defaultReasoningOpen,
        reasoningClose: String = DeepSeekV41PromptMarkers.defaultReasoningClose
    ) {
        self.beginOfSentence = beginOfSentence
        self.endOfSentence = endOfSentence
        self.system = system
        self.user = user
        self.assistant = assistant
        self.toolCallsBegin = toolCallsBegin
        self.toolCallSeparator = toolCallSeparator
        self.toolCallsEnd = toolCallsEnd
        self.reasoningOpen = reasoningOpen
        self.reasoningClose = reasoningClose
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
        let extracted = extractMarkers(in: template)
        // A template that names no marker proves nothing about the dialect. It
        // used to be reported as `checkpointTemplate` anyway, which made the
        // load log claim a checkpoint-specific dialect the renderer was not
        // actually using.
        guard extracted.matched > 0 else {
            return DeepSeekV41PromptDialect(template: template)
        }
        return DeepSeekV41PromptDialect(
            markers: extracted.markers,
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
        extractMarkers(in: template).markers
    }

    /// `markers(in:)` plus how many markers the template actually named, so a
    /// caller can tell an extracted dialect from the family fallback.
    static func extractMarkers(
        in template: String
    ) -> (markers: DeepSeekV41PromptMarkers, matched: Int) {
        let quoted = quotedLiterals(in: template)
        var begin = DeepSeekV41PromptMarkers.familyDefault.beginOfSentence
        var end = DeepSeekV41PromptMarkers.familyDefault.endOfSentence
        var system = DeepSeekV41PromptMarkers.familyDefault.system
        var user = DeepSeekV41PromptMarkers.familyDefault.user
        var assistant = DeepSeekV41PromptMarkers.familyDefault.assistant
        var toolBegin = DeepSeekV41PromptMarkers.familyDefault.toolCallsBegin
        var toolSep = DeepSeekV41PromptMarkers.familyDefault.toolCallSeparator
        var toolEnd = DeepSeekV41PromptMarkers.familyDefault.toolCallsEnd
        var reasoningOpen = DeepSeekV41PromptMarkers.defaultReasoningOpen
        var reasoningClose = DeepSeekV41PromptMarkers.defaultReasoningClose
        var matched = 0

        for literal in quoted {
            if literal.contains("\u{FF5C}") {
                switch roleName(in: literal) {
                case "beginofsentence", "bos", "begin":
                    begin = literal
                case "endofsentence", "eos", "end":
                    end = literal
                case "system", "sys", "systemrole":
                    system = literal
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
                matched += 1
                continue
            }
            // Reasoning tags are special tokens, not fullwidth-bar markers, so
            // they are discovered from their literal spelling instead.
            let lowered = literal.lowercased()
            guard literal.hasPrefix("<"), literal.hasSuffix(">"),
                  lowered.contains("think") || lowered.contains("reason") else { continue }
            if lowered.contains("end") || lowered.hasPrefix("</") {
                reasoningClose = literal
            } else {
                reasoningOpen = literal
            }
            matched += 1
        }
        return (
            DeepSeekV41PromptMarkers(
                beginOfSentence: begin,
                endOfSentence: end,
                user: user,
                assistant: assistant,
                toolCallsBegin: toolBegin,
                toolCallSeparator: toolSep,
                toolCallsEnd: toolEnd,
                system: system,
                reasoningOpen: reasoningOpen,
                reasoningClose: reasoningClose
            ),
            matched
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

    /// The effort budget range the checkpoint accepts. The UI keeps the
    /// discrete levels (25 for low, 100 for high) rather than exposing a
    /// continuous slider until the runtime can actually serve a turn.
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

    // MARK: Splitting recorded turns

    /// Splits a recorded assistant turn into reasoning and answer using the
    /// dialect's own channel markers. The shared splitter only knows the
    /// canonical tags, so a checkpoint that declares different ones would keep
    /// its reasoning in the visible answer.
    public func splitReasoning(
        _ raw: String
    ) -> (think: String?, resp: String) {
        var text = raw
        let canonicalOpen = TurnTextSplitter.canonicalReasoningOpenTag
        let canonicalClose = TurnTextSplitter.canonicalReasoningCloseTag
        if markers.reasoningOpen != canonicalOpen, !markers.reasoningOpen.isEmpty {
            text = text.replacingOccurrences(of: markers.reasoningOpen, with: canonicalOpen)
        }
        if markers.reasoningClose != canonicalClose, !markers.reasoningClose.isEmpty {
            text = text.replacingOccurrences(of: markers.reasoningClose, with: canonicalClose)
        }
        let split = TurnTextSplitter.splitThinkingAndResponse(
            raw: text,
            promptRequestsThinking: false
        )
        return (think: split.think, resp: split.resp)
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
            out += m.system + system
        }
        for turn in turns {
            if turn.isUser {
                out += m.user + escaped(turn.text)
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
        var body = m.assistant
        if let reasoning = turn.reasoning, !reasoning.isEmpty {
            body += m.reasoningOpen
            body += "\n"
            body += reasoning
            body += m.reasoningClose
            body += "\n\n"
        } else if thinkingEnabled {
            // An earlier turn generated with thinking on but no recorded
            // reasoning still has to carry an empty channel: the marker belongs
            // to the turn, not to the setting that happens to be on now.
            body += m.reasoningOpen + "\n\n" + m.reasoningClose + "\n\n"
        }
        body += turn.text
        body += m.endOfSentence
        return body
    }

    /// The prefix the next assistant turn starts from.
    func assistantPrefix(thinkingEnabled: Bool) -> String {
        var out = markers.assistant
        if thinkingEnabled {
            out += markers.reasoningOpen + "\n"
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
            + escaped(payload)
            + m.toolCallsEnd
    }

    /// Neutralizes marker literals in untrusted text (a user prompt or a tool
    /// payload) so it cannot forge a role or tool boundary. A zero-width
    /// no-break space after the leading `<` keeps the visible text intact while
    /// breaking the exact special-token spelling.
    public func escaped(_ text: String) -> String {
        guard !text.isEmpty else { return text }
        let literals = [
            markers.beginOfSentence, markers.endOfSentence, markers.system,
            markers.user, markers.assistant, markers.toolCallsBegin,
            markers.toolCallSeparator, markers.toolCallsEnd,
            markers.reasoningOpen, markers.reasoningClose
        ]
        .filter { !$0.isEmpty }
        .sorted { $0.count > $1.count }
        var out = text
        for literal in literals {
            guard let first = literal.first else { continue }
            out = out.replacingOccurrences(
                of: literal,
                with: String(first) + "\u{FEFF}" + String(literal.dropFirst())
            )
        }
        return out
    }

    /// Stop sequences: the turn ends at the end-of-sentence marker, and the tool
    /// container closes before it so a generation that emits a call is stopped
    /// there by the parser rather than by the sampler.
    public var stopSequences: [String] {
        [markers.endOfSentence, markers.user]
    }
}


