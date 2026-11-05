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
    /// The per-call pair inside the container. The checkpoint's tokenizer carries
    /// both (`128808`/`128809`), and a multi-call turn nests one pair per call.
    public let toolCallBegin: String
    public let toolCallEnd: String
    /// A tool RESULT's pair, and the container a batch of them sits in. These are
    /// the spellings the history renderer needs; the calls container is not one.
    public let toolOutputBegin: String
    public let toolOutputEnd: String
    public let toolOutputsBegin: String
    public let toolOutputsEnd: String
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
        toolCallsBegin: DeepSeekV41PromptMarkers.marker("tool", "calls", "begin"),
        toolCallSeparator: DeepSeekV41PromptMarkers.marker("tool", "sep"),
        toolCallsEnd: DeepSeekV41PromptMarkers.marker("tool", "calls", "end"),
        toolCallBegin: DeepSeekV41PromptMarkers.marker("tool", "call", "begin"),
        toolCallEnd: DeepSeekV41PromptMarkers.marker("tool", "call", "end"),
        toolOutputBegin: DeepSeekV41PromptMarkers.marker("tool", "output", "begin"),
        toolOutputEnd: DeepSeekV41PromptMarkers.marker("tool", "output", "end"),
        toolOutputsBegin: DeepSeekV41PromptMarkers.marker("tool", "outputs", "begin"),
        toolOutputsEnd: DeepSeekV41PromptMarkers.marker("tool", "outputs", "end"),
        system: DeepSeekV41PromptMarkers.defaultSystem,
        reasoningOpen: DeepSeekV41PromptMarkers.defaultReasoningOpen,
        reasoningClose: DeepSeekV41PromptMarkers.defaultReasoningClose
    )

    /// Builds a fullwidth-bar marker from its parts. Never spell one by hand in a
    /// source file: a rendered view strips the bars, so the result looks right and
    /// tokenizes wrong.
    public static func marker(_ parts: String...) -> String {
        marker(parts)
    }

    public static func marker(_ parts: [String]) -> String {
        "<" + "\u{FF5C}" + parts.joined(separator: "\u{2581}") + "\u{FF5C}" + ">"
    }

    public init(
        beginOfSentence: String,
        endOfSentence: String,
        user: String,
        assistant: String,
        toolCallsBegin: String,
        toolCallSeparator: String,
        toolCallsEnd: String,
        toolCallBegin: String = DeepSeekV41PromptMarkers.familyDefault.toolCallBegin,
        toolCallEnd: String = DeepSeekV41PromptMarkers.familyDefault.toolCallEnd,
        toolOutputBegin: String = DeepSeekV41PromptMarkers.familyDefault.toolOutputBegin,
        toolOutputEnd: String = DeepSeekV41PromptMarkers.familyDefault.toolOutputEnd,
        toolOutputsBegin: String = DeepSeekV41PromptMarkers.familyDefault.toolOutputsBegin,
        toolOutputsEnd: String = DeepSeekV41PromptMarkers.familyDefault.toolOutputsEnd,
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
        self.toolCallBegin = toolCallBegin
        self.toolCallEnd = toolCallEnd
        self.toolOutputBegin = toolOutputBegin
        self.toolOutputEnd = toolOutputEnd
        self.toolOutputsBegin = toolOutputsBegin
        self.toolOutputsEnd = toolOutputsEnd
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
        var callBegin = DeepSeekV41PromptMarkers.familyDefault.toolCallBegin
        var callEnd = DeepSeekV41PromptMarkers.familyDefault.toolCallEnd
        var outputBegin = DeepSeekV41PromptMarkers.familyDefault.toolOutputBegin
        var outputEnd = DeepSeekV41PromptMarkers.familyDefault.toolOutputEnd
        var outputsBegin = DeepSeekV41PromptMarkers.familyDefault.toolOutputsBegin
        var outputsEnd = DeepSeekV41PromptMarkers.familyDefault.toolOutputsEnd
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
                case "toolcallbegin":
                    callBegin = literal
                case "toolcallend":
                    callEnd = literal
                case "toolsep", "toolseparator":
                    toolSep = literal
                case "toolcallsend":
                    toolEnd = literal
                case "tooloutputbegin":
                    outputBegin = literal
                case "tooloutputend":
                    outputEnd = literal
                case "tooloutputsbegin":
                    outputsBegin = literal
                case "tooloutputsend":
                    outputsEnd = literal
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
                toolCallBegin: callBegin,
                toolCallEnd: callEnd,
                toolOutputBegin: outputBegin,
                toolOutputEnd: outputEnd,
                toolOutputsBegin: outputsBegin,
                toolOutputsEnd: outputsEnd,
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

    /// The checkpoint's own effort line, including the range note and the blank
    /// line that separates it from the caller's system text. The encoder in
    /// `encoding/encoding.py` writes exactly this string, and a prompt that leaves
    /// the note out still names a budget the model was never shown that way.
    public static func effortInstruction(budget: Int) -> String {
        let clamped = min(max(budget, effortBounds.lowerBound), effortBounds.upperBound)
        return "Reasoning Effort: \(clamped) "
            + "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
    }

    /// Parses an effort budget out of arbitrary text: a number in this checkpoint's
    /// range, or one of the words the menu and the API adapters already use.
    ///
    /// This is deliberately NOT what the API adapters call, and the reason is a type
    /// rather than an oversight. `ThinkingEffort` carries two levels, so a client's
    /// `reasoning_effort` is resolved onto low or high by
    /// `OpenAIAdapter.thinkingEffortLevel` and a NUMBER never reaches this function
    /// (`#96`). The words above are the same vocabulary that mapping uses, which is why
    /// `minimal` is 25 here and `.low` there: keeping the two in step is the point of
    /// saying it twice. A caller that does hold a budget (a stored profile, or a numeric
    /// surface if the enum ever grows one) gets the number back, or nil outside the range
    /// the checkpoint publishes.
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
        /// The turn is a tool RESULT, not a role turn. The checkpoint places a
        /// result in its own `tool▁output` pair directly after the call turn's
        /// end-of-sentence marker, with no role marker of its own, so it is
        /// rendered as a bare pair rather than as `<|User|>text`.
        public var isToolOutput: Bool

        public init(
            isUser: Bool,
            text: String,
            reasoning: String? = nil,
            isToolOutput: Bool = false
        ) {
            self.isUser = isUser
            self.text = text
            self.reasoning = reasoning
            self.isToolOutput = isToolOutput
        }
    }

    /// The system turn body: the effort instruction when thinking is on, then the
    /// caller's system prompt.
    ///
    /// The order is the checkpoint encoder's, not a preference: the effort line is
    /// the first thing after `<|System|>` and ends with a blank line, so a prompt
    /// that appends it after the system text puts the budget where the model reads
    /// it as part of the last instruction instead of ahead of everything.
    public func systemBody(
        cleanSystem: String,
        thinkingEnabled: Bool,
        effort: ThinkingEffort?
    ) -> String {
        let instruction = effortInstruction(effort, thinkingEnabled: thinkingEnabled)
        let base = cleanSystem.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let instruction else { return base }
        return base.isEmpty ? instruction : instruction + base
    }

    /// Renders a whole conversation. The trailing assistant marker opens the
    /// reasoning channel when thinking is on and closes it when thinking is off,
    /// exactly as the checkpoint's encoder does: in chat mode the model is trained
    /// to answer after `<|Assistant|> endslash think`, and leaving that token out
    /// asks a thinking model for an answer it was never told to start.
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
            if turn.isToolOutput {
                // A tool result rides the call turn's own closer, in the family's
                // output pair: `<|end▁of▁sentence|>` then the payload. The role
                // marker belongs to the NEXT assistant turn, which is what the
                // agent loop appends after a result.
                out += m.toolOutputBegin + escaped(turn.text) + m.toolOutputEnd
            } else if turn.isUser {
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
            // The channel's own content, exactly as it was generated. The
            // checkpoint's encoder inserts no whitespace around it, and neither
            // does this: a reasoning trace already begins and ends with what the
            // model emitted, and padding it here would re-render a pinned turn
            // differently from the text the cache was built on.
            body += m.reasoningOpen + reasoning + m.reasoningClose
        } else if thinkingEnabled {
            // An earlier turn generated with thinking on but no recorded
            // reasoning still has to carry an empty channel: the marker belongs
            // to the turn, not to the setting that happens to be on now.
            body += m.reasoningOpen + m.reasoningClose
        } else {
            // Chat mode: the assistant turn opens closed, which is the token the
            // encoder writes before every answer.
            body += m.reasoningClose
        }
        body += turn.text
        body += m.endOfSentence
        return body
    }

    /// The prefix the next assistant turn starts from.
    func assistantPrefix(thinkingEnabled: Bool) -> String {
        markers.assistant + (thinkingEnabled ? markers.reasoningOpen : markers.reasoningClose)
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
            markers.toolCallSeparator, markers.toolCallsEnd, markers.toolCallBegin,
            markers.toolCallEnd, markers.toolOutputBegin, markers.toolOutputEnd,
            markers.toolOutputsBegin, markers.toolOutputsEnd,
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


