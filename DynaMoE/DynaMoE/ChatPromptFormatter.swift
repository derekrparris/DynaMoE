//
//  ChatPromptFormatter.swift
//  DynaMoE
//
//  Per-family chat prompt templating, extracted verbatim from the former
//  inline implementation in `ContentView.startAutoregressiveGeneration` so it
//  can be exercised by both the chat UI and the OpenAI/Anthropic-compatible
//  server adapters. Pure string assembly: no Metal, no sessions, no SwiftUI.
//
//  CAUTION: the literals below embed model-family control tags (some with
//  multibyte lookalikes). Never retype them by hand; verify codepoints if a
//  render ever shows a stripped tag (see AGENTS.md "Reading files safely").
//

import Foundation

public enum ChatPromptFormatter {

    /// Builds the fully templated prompt string for one generation turn.
    /// `prompt` is either the plain user text or an already-formatted
    /// continuation (agent tool responses, synthesis directives) — the
    /// "already templated" probes match those custom prompts verbatim.
    public static func formatPrompt(
        prompt: String,
        cleanSystem: String,
        thinkingEnabled: Bool,
        modelSupportsThinking: Bool,
        modelConfig: ModelConfig?,
        thinkingEffort: ThinkingEffort? = nil,
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect(),
        inklingTools: [Inkling2ToolDeclaration] = []
    ) -> String {
        let thinkSuffix: String
        if thinkingEnabled {
            thinkSuffix = Self.openReasoningSuffix
        } else if modelSupportsThinking {
            thinkSuffix = Self.emptyReasoningSuffix
        } else {
            thinkSuffix = ""
        }

        let isLing = (modelConfig?.isLingModel == true)
        let formattedPrompt: String
        if modelConfig?.isInkling2Model == true {
            // Inkling 2: role markers plus typed content blocks, and an effort
            // line as its own system message. The template has no assistant
            // prefix -- generation begins after the last message's end marker --
            // so `thinkingEnabled` moves the EFFORT value, not a suffix.
            if prompt.contains(Inkling2PromptMarkers.familyDefault.user) {
                formattedPrompt = prompt
            } else {
                var turns: [Inkling2Turn] = []
                if !cleanSystem.isEmpty { turns.append(.system(cleanSystem)) }
                turns.append(.user([.text(prompt)]))
                formattedPrompt = (try? Inkling2Prompt().render(
                    turns: turns,
                    effort: Self.inkling2Effort(
                        thinkingEnabled: thinkingEnabled, thinkingEffort: thinkingEffort
                    ),
                    tools: inklingTools
                )) ?? prompt
            }
        } else if modelConfig?.isDeepSeekV41Model == true {
            // DeepSeek V4.1: fullwidth-bar role markers, and an effort budget
            // rendered inside the system turn rather than a named thinking level.
            if isAlreadyDeepSeekTemplated(prompt, markers: promptDialect.markers) {
                formattedPrompt = prompt
            } else {
                formattedPrompt = promptDialect.render(
                    turns: [.init(isUser: true, text: prompt)],
                    cleanSystem: cleanSystem,
                    thinkingEnabled: thinkingEnabled,
                    effort: thinkingEffort
                )
            }
        } else if isLing {
            if prompt.contains(Self.lingRoleOpen) {
                formattedPrompt = prompt
            } else {
                let lingThinking = thinkingEnabled ? "detailed thinking on" : "detailed thinking off"
                let sysPart = cleanSystem.isEmpty ? lingThinking : "\(cleanSystem)\n\(lingThinking)"
                let thinkTag = thinkingEnabled ? "\n<think>" : "\n<think></think>"
                formattedPrompt = Self.lingRoleOpen + "SYSTEM" + Self.lingRoleClose + sysPart
                    + Self.lingRoleEnd + Self.lingRoleOpen + "HUMAN" + Self.lingRoleClose + prompt
                    + Self.lingRoleEnd + Self.lingRoleOpen + "ASSISTANT" + Self.lingRoleClose + thinkTag
            }
        } else if modelConfig?.isSparkModel == true {
            // Spark 2.5 DeepSeek-style dialect; X2.6 adds the effort line to the
            // System block (never when thinking is off, per the template).
            if prompt.contains(Self.sparkBotMarker) || prompt.contains(Self.sparkUserMarker)
                || prompt.contains(Self.sparkSystemMarker) {
                formattedPrompt = prompt
            } else {
                let systemBody = Self.sparkSystemBody(
                    cleanSystem: cleanSystem,
                    thinkingEnabled: thinkingEnabled,
                    thinkingEffort: thinkingEffort
                )
                var p = Self.sparkStartOfSentence + Self.sparkSystemMarker + "\n" + systemBody + Self.sparkEndOfSentence
                p += Self.sparkStartOfSentence + Self.sparkUserMarker + prompt + Self.sparkEndOfSentence
                let sparkThink = thinkingEnabled ? "<think>" : "<think></think>"
                p += Self.sparkStartOfSentence + Self.sparkBotMarker + sparkThink
                formattedPrompt = p
            }
        } else if modelConfig?.isGemma4Model == true {
            // Gemma 4 canonical chat template
            if prompt.contains("<|turn>") {
                formattedPrompt = prompt
            } else {
                var p = "<bos>"
                if thinkingEnabled {
                    p += "<|turn>system\n<|think|>\n\(cleanSystem)<turn|>\n"
                } else if !cleanSystem.isEmpty {
                    p += "<|turn>system\n\(cleanSystem)<turn|>\n"
                }
                p += "<|turn>user\n\(prompt)<turn|>\n"
                if thinkingEnabled {
                    p += "<|turn>model\n"
                } else {
                    p += "<|turn>model\n" + Self.gemmaReasoningOpen + Self.gemmaReasoningClose
                }
                formattedPrompt = p
            }
        } else if prompt.contains("<|im_start|>") {
            var p = prompt
            if !cleanSystem.isEmpty && !prompt.contains("<|im_start|>system") {
                p = "<|im_start|>system\n\(cleanSystem)<|im_end|>\n" + p
            }
            let trimmed = p.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasSuffix("<|im_start|>assistant") {
                if thinkingEnabled {
                    p = trimmed + "\n<think>\n"
                } else if modelSupportsThinking {
                    p = trimmed + "\n<think>\n\n</think>\n\n"
                }
            }
            formattedPrompt = p
        } else if !cleanSystem.isEmpty {
            formattedPrompt = "<|im_start|>system\n\(cleanSystem)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n\(thinkSuffix)"
        } else {
            formattedPrompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n\(thinkSuffix)"
        }
        return formattedPrompt
    }

    /// Spark 2.5 (DeepSeek-style) end-of-sentence marker. Built from unicode
    /// escapes so a renderer can never mangle the fullwidth bars / block
    /// underscores (AGENTS.md: "Reading files safely"). `nonisolated` for the
    /// same reason as the framing below: `PromptMarkers` reads it from a
    /// nonisolated context, and default main-actor isolation would make that a
    /// concurrency diagnostic.
    public nonisolated static let sparkEndOfSentence = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"

    /// Spark 2.5's remaining turn framing. `nonisolated` because
    /// `PromptMarkers` sanitizes them from a nonisolated context, and named
    /// here so the sanitizer and the renderer cannot disagree byte for byte.
    public nonisolated static let sparkStartOfSentence = "<\u{FF5C}start\u{2581}of\u{2581}sentence\u{FF5C}>"
    public nonisolated static let sparkSystemMarker = "<|System|>"
    public nonisolated static let sparkUserMarker = "<|User|>"
    public nonisolated static let sparkBotMarker = "<|Bot|>"
    public nonisolated static let sparkToolMarker = "<|Tool|>"

    /// Ling/Bailing 3.0's role spans. The assistant turn boundary is
    /// `<role>ASSISTANT</role>`, so both halves of the span are control syntax.
    public nonisolated static let lingRoleOpen = "<role>"
    public nonisolated static let lingRoleClose = "</role>"
    public nonisolated static let lingRoleEnd = "<|role_end|>"

    /// Spark X2.6's thinking-level instruction, appended to the System block
    /// when the model's chat template exposes `reasoning_effort` (levels:
    /// low/high; thinking off emits nothing, matching the template). Verbatim
    /// from Spark-X2.6-4B's `chat_template.jinja` `reasoning_instructions`;
    /// the leading newline separates it from the system base line.
    public static func sparkThinkingEffortInstructions(level: ThinkingEffort) -> String {
        "\n`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low` and `high`.\nNow the system is invoked with `thinking_effort=\(level.rawValue)`."
    }

    /// Renders Spark's System block body the way the model's own template
    /// does: the default assistant sentence (with the effort instruction when
    /// thinking is on), then any tools block, then the caller's system text.
    /// Recognizes both the in-app `# Tools` section and the server's appended
    /// tool declaration so the instruction never lands after the tools.
    public static func sparkSystemBody(
        cleanSystem: String,
        thinkingEnabled: Bool,
        thinkingEffort: ThinkingEffort?
    ) -> String {
        let defaultSentence = "you are a helpful assistant."
        let effortLine = (thinkingEnabled ? thinkingEffort : nil)
            .map { sparkThinkingEffortInstructions(level: $0) } ?? ""
        let base = defaultSentence + effortLine
        guard !cleanSystem.isEmpty else { return base }
        // In-app agent prompt: [date/prefix]\n\n# Tools ... [client system].
        if let range = cleanSystem.range(of: "\n\n# Tools") {
            let prefix = String(cleanSystem[..<range.lowerBound])
            let rest = String(cleanSystem[range.lowerBound...])
            return base + (prefix.isEmpty ? "" : "\n\n" + prefix) + rest
        }
        // Server tool declaration: [client system]\n\nYou have access ...
        if let range = cleanSystem.range(of: "\n\nYou have access to the following functions.") {
            let client = String(cleanSystem[..<range.lowerBound])
            let tools = String(cleanSystem[range.lowerBound...])
            return base + tools + (client.isEmpty ? "" : "\n\n" + client)
        }
        return base + "\n\n" + cleanSystem
    }

    /// Nanbeige 4.3's tool-result block. The checkpoint's template has no tool
    /// role: a result rides a USER turn as `<tool_response>…</tool_response>`,
    /// one block per executed call, the turn closed once. The trailing newline
    /// matches what `AgentHarness.formatToolResponseTurn` emits live, so a
    /// rebuilt history and the live continuation stay byte-compatible (which is
    /// what lets a post-tool turn extend the previous turn's pinned prefix).
    public static func nanbeigeToolResponse(text: String) -> String {
        "<tool_response>\n" + text + "\n</tool_response>\n"
    }

    /// Nanbeige 4.3's own tool-call markup, used only to REBUILD a call whose
    /// raw text is no longer in the stored turn (the raw call survives in every
    /// turn this app produced). The family documents the `qwen3_coder` parser,
    /// which is this XML form: the arguments are `<parameter=name>` children of
    /// one `<function=name>` inside a `<tool_call>` container, each value on its
    /// own lines.
    public static func nanbeigeToolCallBlock(name: String, arguments: [String: String]) -> String {
        var block = "<tool_call>\n<function=" + name + ">\n"
        for key in arguments.keys.sorted() {
            block += "<parameter=" + key + ">\n" + (arguments[key] ?? "") + "\n</parameter>\n"
        }
        block += "</function>\n</tool_call>"
        return block
    }

    /// The System block a family renders, i.e. what a digest over "the system
    /// prompt the model saw" has to cover. Both dialects with a graded effort
    /// level put it inside that block (Spark appends `thinking_effort`, DeepSeek
    /// V4.1 prefixes its `Reasoning Effort:` line), so a level change has to
    /// change this string; otherwise a recorded raw turn stays trusted under a
    /// system prompt it was never generated with. A family with no graded level
    /// contributes the caller's text unchanged.
    public static func renderedSystemBody(
        cleanSystem: String,
        thinkingEnabled: Bool,
        thinkingEffort: ThinkingEffort?,
        modelConfig: ModelConfig?,
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect()
    ) -> String {
        if modelConfig?.isSparkModel == true {
            return sparkSystemBody(
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                thinkingEffort: thinkingEffort
            )
        }
        if modelConfig?.isDeepSeekV41Model == true {
            return promptDialect.systemBody(
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                effort: thinkingEffort
            )
        }
        return cleanSystem
    }

    /// Gemma 4's native reasoning channel markers. The model opens the channel
    /// itself after `<|turn>model` when thinking is on, so a completed turn's
    /// echoed reasoning must be re-wrapped in them: the canonical
    /// ` thinking`/`</think>` markers the adapters fold in are the *other*
    /// families' tags and would not reproduce the pinned generation.
    nonisolated static let gemmaChannelOpen = "<|channel>"
    nonisolated static let gemmaChannelClose = "<channel|>"
    nonisolated static let gemmaThinkMarker = "<|think|>"
    nonisolated static let gemmaReasoningOpen = gemmaChannelOpen + "thought\n"
    nonisolated static let gemmaReasoningClose = gemmaChannelClose

    /// Prefixes the protocol adapters use to re-render an assistant tool-call
    /// turn out of a client's echoed `tool_calls` / `tool_use` structure. A
    /// faithful echo of a tool-call turn only ever arrives in this form (never
    /// as the raw tagged markup the model emitted), so the server's turn memory
    /// has to recognize it to reproduce the pinned generation.
    public static let openAIToolCallEchoPrefix = "Function calls made:"
    public static let anthropicToolCallEchoPrefix = "Tool calls made:"

    /// The reasoning suffix a family's assistant prefix carries when its
    /// reasoning block is open, and when it is empty (opened and immediately
    /// closed). Assembled from the canonical tags so the literals are never
    /// retyped (AGENTS.md).
    static let openReasoningSuffix = TurnTextSplitter.canonicalReasoningOpenTag + "\n"
    static let emptyReasoningSuffix =
        TurnTextSplitter.canonicalReasoningOpenTag + "\n\n" + TurnTextSplitter.canonicalReasoningCloseTag + "\n\n"

    /// The Inkling 2 effort line's value. Thinking OFF renders 0.0 (`none`)
    /// rather than the template's 0.9 default: a turn that asked for no
    /// reasoning must not be handed the checkpoint's default effort, which is
    /// the template's fallback only when the caller passes nothing at all.
    static func inkling2Effort(thinkingEnabled: Bool, thinkingEffort: ThinkingEffort?) -> Float {
        guard thinkingEnabled else { return 0 }
        switch thinkingEffort ?? .high {
        case .low: return Inkling2Effort.value(for: "low") ?? 0.2
        case .high: return Inkling2Effort.value(for: "high") ?? 0.9
        }
    }

    /// One user or tool turn's text as Inkling 2's typed content parts, with each
    /// picture placed where the caller marked it.
    ///
    /// The checkpoint gives a picture its own content block, so an image is a
    /// PART and never a marker spliced into the text. The API server's decoders
    /// keep a message's content order as a `U+FFFC` sentinel per picture, which
    /// is what decides where the block goes and is never itself emitted; a turn
    /// with pictures but no sentinel has them move in front, which is the order
    /// the in-app composer writes and the checkpoint's own prompts use.
    ///
    /// A count mismatch renders NOTHING, which is fail-closed rather than
    /// reconciled: an empty turn leaves the prompt without the placeholders the
    /// attached pictures need, so the engine's own picture/placeholder check
    /// refuses the turn by name. Both entry points check this first; this is the
    /// backstop for a caller that assembled turns itself.
    nonisolated static func inklingContentParts(
        text: String,
        grids: [Inkling2ImageGrid]
    ) -> [Inkling2ContentPart] {
        let chunks = text.components(separatedBy: ServerImageParts.imagePlaceholder)
        let sentinels = chunks.count - 1
        guard sentinels == 0 || sentinels == grids.count else { return [] }
        guard sentinels > 0 else {
            // No sentinel: the pictures go in front, the order the in-app composer
            // writes and the order the checkpoint's own prompts use.
            var parts = grids.map { Inkling2ContentPart.image($0) }
            if !text.isEmpty { parts.append(.text(text)) }
            return parts.isEmpty ? [.text(text)] : parts
        }
        var parts: [Inkling2ContentPart] = []
        for (offset, chunk) in chunks.enumerated() {
            if offset > 0 { parts.append(.image(grids[offset - 1])) }
            if !chunk.isEmpty { parts.append(.text(chunk)) }
        }
        return parts
    }

    /// Whether a *historical* assistant turn is rendered with its reasoning block
    /// open.
    ///
    /// A turn whose own text carries reasoning was generated inside an open
    /// block, so it is re-opened here even when thinking has since been turned
    /// off: rendering it after an already-closed empty block would put its
    /// reasoning outside the block it belongs to and re-render the turn
    /// differently from the prompt the pin recorded. Every other turn follows
    /// this turn's setting, which is the only signal available for a turn that
    /// was generated without reasoning at all.
    private static func opensReasoningForHistory(
        thinkingEnabled: Bool,
        modelSupportsThinking: Bool,
        text: String
    ) -> Bool {
        guard modelSupportsThinking else { return false }
        return thinkingEnabled || TurnTextSplitter.containsReasoningClose(text)
    }

    /// Renders a *completed* historical assistant turn's body. When the family's
    /// assistant prefix opened a reasoning block (thinking enabled) a client's
    /// echo (the visible answer, not the reasoning) must not be presented as
    /// an unfinished thought, so the reasoning close marker is inserted ahead of
    /// it. A recorded raw generation (server-spliced, reasoning included)
    /// already carries the marker and is left untouched, keeping the pinned
    /// prompt+generation prefix intact.
    private static func completedAssistantBody(prefixOpensReasoning: Bool, text: String) -> String {
        guard prefixOpensReasoning else { return text }
        let close = TurnTextSplitter.canonicalReasoningCloseTag
        guard !text.contains(close) else { return text }
        return close + text
    }

    /// The Gemma counterpart of `completedAssistantBody`. Gemma's assistant
    /// prefix leaves the reasoning channel open (`<|turn>model\n`) and the model
    /// opens the channel itself, so the channel is the body's business: reasoning
    /// a client echoed back arrives in the flattened canonical form (`reasoning`
    /// + canonical close + answer) and is converted to the native markers here,
    /// whether or not this turn still has thinking enabled — the turn's own text
    /// is what says it was generated with reasoning. A turn without reasoning
    /// gets an explicitly closed, empty channel, and a body already carrying the
    /// native markers (a server-spliced raw generation) is left untouched.
    private static func completedGemmaBody(text: String) -> String {
        guard !text.contains(gemmaReasoningClose) else { return text }
        if let close = text.range(of: TurnTextSplitter.canonicalReasoningCloseTag) {
            return gemmaReasoningOpen
                + text[text.startIndex..<close.lowerBound]
                + gemmaReasoningClose
                + text[close.upperBound...]
        }
        return gemmaReasoningOpen + gemmaReasoningClose + text
    }

    /// One role-tagged turn of a conversation, before family templating.
    nonisolated public struct ConversationTurn: Sendable, Equatable {
        public enum Role: Sendable, Equatable {
            case user
            case assistant
        }

        /// What a turn is, beyond the role it is tagged with. Only DeepSeek V4.1
        /// distinguishes: the checkpoint has no tool role, so a result is a bare
        /// `tool▁output` pair after the call turn rather than a turn of its own.
        /// Every other family keeps rendering `role` + `text` exactly as before,
        /// which is why a tool result still carries `role == .user`.
        public enum Kind: Sendable, Equatable {
            case text
            /// A tool result, named by the tool that produced it. `text` is the
            /// payload the model reads.
            case toolResult(name: String)
        }

        public var role: Role
        public var text: String
        public var kind: Kind
        /// The pictures this turn carries, in content order.
        ///
        /// Only Inkling 2 renders these: its template gives a picture its own
        /// typed content block, so the grid is a PART of the turn rather than a
        /// marker spliced into `text` (which is how DeepSeek V4.1, the only
        /// other family with a tower, carries one). A family that does not read
        /// pictures ignores this and keeps rendering `text` alone.
        public var imageGrids: [Inkling2ImageGrid]

        public init(
            role: Role,
            text: String,
            kind: Kind = .text,
            imageGrids: [Inkling2ImageGrid] = []
        ) {
            self.role = role
            self.text = text
            self.kind = kind
            self.imageGrids = imageGrids
        }
    }

    /// Renders a multi-turn conversation with the model family's native
    /// role boundaries, mirroring `formatPrompt`'s per-family templates.
    /// Each assistant turn opens with the same assistant prefix the
    /// single-turn template emits (role marker plus the family's thinking
    /// markers): those tokens belong to the *prompt*, so the next turn must
    /// reproduce them to stay prefix-compatible with the previous turn's
    /// pinned prompt+generation.
    ///
    /// History used to be flattened into a single user block carrying
    /// synthetic "User:"/"Assistant:" labels, which re-tokenized the whole
    /// conversation every turn: turn N+1's prompt could no longer extend
    /// turn N's pin, so multi-turn requests fully re-prefilled — and hybrid
    /// (GDN) models, whose recurrent state only matches at the exact pinned
    /// token boundary, could not reuse it at all.
    public static func formatConversation(
        turns: [ConversationTurn],
        cleanSystem: String,
        thinkingEnabled: Bool,
        modelSupportsThinking: Bool,
        modelConfig: ModelConfig?,
        thinkingEffort: ThinkingEffort? = nil,
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect(),
        inklingTools: [Inkling2ToolDeclaration] = []
    ) -> String {
        guard !turns.isEmpty else { return "" }

        if modelConfig?.isInkling2Model == true {
            // The checkpoint's template renders roles and typed blocks and has
            // no assistant prefix, so the last turn is whatever the caller put
            // there -- a completed assistant turn replays with its
            // end-sampling marker, and generation starts right after it.
            let splitter = TurnTextSplitter.self
            var rendered: [Inkling2Turn] = []
            if !cleanSystem.isEmpty { rendered.append(.system(cleanSystem)) }
            for turn in turns {
                switch turn.kind {
                case .toolResult(let name):
                    rendered.append(.tool(
                        name: name,
                        content: inklingContentParts(text: turn.text, grids: turn.imageGrids)
                    ))
                case .text:
                    guard turn.role == .assistant else {
                        rendered.append(.user(
                            inklingContentParts(text: turn.text, grids: turn.imageGrids)
                        ))
                        continue
                    }
                    let text = turn.text
                    let reasoning: String?
                    let content: String
                    if splitter.containsReasoningClose(text) {
                        let split = splitter.splitThinkingAndResponse(raw: text, promptRequestsThinking: true)
                        reasoning = split.think.isEmpty ? nil : split.think
                        content = split.resp
                    } else {
                        reasoning = nil
                        content = text
                    }
                    rendered.append(.assistant(reasoning: reasoning, content: content))
                }
            }
            return (try? Inkling2Prompt().render(
                turns: rendered,
                effort: inkling2Effort(thinkingEnabled: thinkingEnabled, thinkingEffort: thinkingEffort),
                tools: inklingTools
            )) ?? ""
        }

        if modelConfig?.isDeepSeekV41Model == true {
            // A completed assistant turn is rendered from its own content: the
            // reasoning splitter already marks where its channel closed, so the
            // dialect re-opens the channel only for a turn that carried
            // reasoning, and always for the turn being generated now.
            let rendered = turns.map { turn -> DeepSeekV41PromptDialect.Turn in
                if case .toolResult = turn.kind {
                    // The result's payload, in the family's output pair, after the
                    // call turn's closer. No role marker: the checkpoint has no
                    // tool role, and the pair is what the loop appends live.
                    return .init(isUser: false, text: turn.text, isToolOutput: true)
                }
                guard turn.role == .assistant else {
                    return .init(isUser: true, text: turn.text)
                }
                let split = promptDialect.splitReasoning(turn.text)
                return .init(isUser: false, text: split.resp, reasoning: split.think)
            }
            return promptDialect.render(
                turns: rendered,
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                effort: thinkingEffort
            )
        }

        if modelConfig?.isLingModel == true {
            let lingThinking = thinkingEnabled ? "detailed thinking on" : "detailed thinking off"
            let sysPart = cleanSystem.isEmpty ? lingThinking : "\(cleanSystem)\n\(lingThinking)"
            // The Ling assistant prefix with its reasoning block open, and with
            // it empty (a completed turn that carried no reasoning).
            let openPrefix = Self.lingRoleOpen + "ASSISTANT" + Self.lingRoleClose + "\n" + TurnTextSplitter.canonicalReasoningOpenTag
            let emptyPrefix = openPrefix + TurnTextSplitter.canonicalReasoningCloseTag
            var p = Self.lingRoleOpen + "SYSTEM" + Self.lingRoleClose + sysPart + Self.lingRoleEnd
            for turn in turns {
                if turn.role == .user {
                    p += Self.lingRoleOpen + "HUMAN" + Self.lingRoleClose + turn.text + Self.lingRoleEnd
                } else {
                    let opens = thinkingEnabled || TurnTextSplitter.containsReasoningClose(turn.text)
                    p += (opens ? openPrefix : emptyPrefix)
                        + Self.completedAssistantBody(prefixOpensReasoning: opens, text: turn.text)
                        + Self.lingRoleEnd
                }
            }
            p += (thinkingEnabled ? openPrefix : emptyPrefix)
            return p
        }

        if modelConfig?.isSparkModel == true {
            let sysLine = Self.sparkSystemBody(
                cleanSystem: cleanSystem,
                thinkingEnabled: thinkingEnabled,
                thinkingEffort: thinkingEffort
            )
            let openPrefix = Self.sparkStartOfSentence + Self.sparkBotMarker + TurnTextSplitter.canonicalReasoningOpenTag
            let emptyPrefix = openPrefix + TurnTextSplitter.canonicalReasoningCloseTag
            var p = Self.sparkStartOfSentence + Self.sparkSystemMarker + "\n" + sysLine + Self.sparkEndOfSentence
            for turn in turns {
                if turn.role == .user {
                    p += Self.sparkStartOfSentence + Self.sparkUserMarker + turn.text + Self.sparkEndOfSentence
                } else {
                    let opens = thinkingEnabled || TurnTextSplitter.containsReasoningClose(turn.text)
                    p += (opens ? openPrefix : emptyPrefix)
                        + Self.completedAssistantBody(prefixOpensReasoning: opens, text: turn.text)
                        + Self.sparkEndOfSentence
                }
            }
            p += (thinkingEnabled ? openPrefix : emptyPrefix)
            return p
        }

        if modelConfig?.isGemma4Model == true {
            let trailingPrefix = "<|turn>model\n" + (thinkingEnabled ? "" : Self.gemmaReasoningOpen + Self.gemmaReasoningClose)
            var p = "<bos>"
            if thinkingEnabled {
                p += "<|turn>system\n<|think|>\n" + cleanSystem + "<turn|>\n"
            } else if !cleanSystem.isEmpty {
                p += "<|turn>system\n" + cleanSystem + "<turn|>\n"
            }
            for turn in turns {
                if turn.role == .user {
                    p += "<|turn>user\n" + turn.text + "<turn|>\n"
                } else {
                    // The model opens its own reasoning channel, so the body owns
                    // it: an echoed reasoning is converted to the native markers
                    // even when thinking has since been turned off, and a turn
                    // without reasoning gets an explicitly empty channel.
                    p += "<|turn>model\n" + Self.completedGemmaBody(text: turn.text) + "<turn|>\n"
                }
            }
            p += trailingPrefix
            return p
        }

        // The default dialect. A completed turn is rendered from its own text
        // whenever that text carries reasoning, because the turn's content is
        // what says its block was open: thinking that has since been disabled
        // must not push earlier reasoning out of the block it belongs to, which
        // would also re-render that turn differently from the prompt the pin
        // recorded.
        let emptySuffix = modelSupportsThinking ? Self.emptyReasoningSuffix : ""
        let emptyPrefix = "<|im_start|>assistant\n" + emptySuffix
        let openPrefix = "<|im_start|>assistant\n" + Self.openReasoningSuffix
        let assistantPrefix = thinkingEnabled ? openPrefix : emptyPrefix
        // Nanbeige 4.3 keeps a tool result in a USER turn wrapped in
        // `<tool_response>` (its template has no tool role at all); every other
        // ChatML family renders the payload as the turn's own text.
        //
        // The rest of this family's template needs no branch: its `<|im_start|>`
        // framing, its `Thinking` channel and its already-closed empty block are
        // byte-identical to the defaults above, and `preserve_thinking` is
        // deliberately left ON (always replay) so a recorded turn re-renders the
        // way it was generated.
        let wrapsToolResults = modelConfig?.isNanbeige43Model == true
        var p = ""
        if !cleanSystem.isEmpty {
            p += "<|im_start|>system\n" + cleanSystem + "<|im_end|>\n"
        }
        for turn in turns {
            if turn.role == .user {
                if wrapsToolResults, case .toolResult = turn.kind {
                    p += "<|im_start|>user\n" + Self.nanbeigeToolResponse(text: turn.text) + "<|im_end|>\n"
                } else {
                    p += "<|im_start|>user\n" + turn.text + "<|im_end|>\n"
                }
            } else {
                let opens = Self.opensReasoningForHistory(
                    thinkingEnabled: thinkingEnabled,
                    modelSupportsThinking: modelSupportsThinking,
                    text: turn.text
                )
                p += (opens ? openPrefix : emptyPrefix)
                    + Self.completedAssistantBody(prefixOpensReasoning: opens, text: turn.text)
                    + "<|im_end|>\n"
            }
        }
        p += assistantPrefix
        return p
    }

    /// A prompt that already carries the dialect's own role marker only arrives
    /// from the agent harness (which renders tool continuations itself), so it is
    /// passed through verbatim. Mirrors the `<role>` / `<|Bot|>` / `<|turn>`
    /// passthrough probes the other families use.
    ///
    /// The markers are the RESOLVED dialect's, not the family default: a checkpoint
    /// whose template renames them would otherwise be probed for one spelling and
    /// rendered in another, and every harness prompt would be wrapped twice.
    ///
    /// The probe requires the prompt to *begin* with a role/sentence marker. It
    /// used to pass any prompt that merely contained one, so a user message
    /// quoting `<|User|>` was treated as harness-authored and injected verbatim
    /// as a turn boundary.
    static func isAlreadyDeepSeekTemplated(
        _ prompt: String,
        markers: DeepSeekV41PromptMarkers = .familyDefault
    ) -> Bool {
        let m = markers
        let trimmed = prompt.drop { $0.isWhitespace }
        return [m.beginOfSentence, m.system, m.user, m.assistant]
            .filter { !$0.isEmpty }
            .contains { trimmed.hasPrefix($0) }
    }

    /// The tag that closes an assistant turn for this family, or nil when the
    /// family's assistant turns carry no explicit closer. The server drops a
    /// trailing closer from a recorded raw generation before splicing it into
    /// history, so the renderer does not close the turn twice.
    public static func assistantTurnCloseTag(
        modelConfig: ModelConfig?,
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect()
    ) -> String? {
        if modelConfig?.isDeepSeekV41Model == true { return promptDialect.markers.endOfSentence }
        if modelConfig?.isLingModel == true { return lingRoleEnd }
        if modelConfig?.isSparkModel == true { return nil }
        if modelConfig?.isGemma4Model == true { return "<turn|>" }
        return "<|im_end|>"
    }
}
