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
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect()
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
        if modelConfig?.isDeepSeekV41Model == true {
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
            if prompt.contains("<role>") {
                formattedPrompt = prompt
            } else {
                let lingThinking = thinkingEnabled ? "detailed thinking on" : "detailed thinking off"
                let sysPart = cleanSystem.isEmpty ? lingThinking : "\(cleanSystem)\n\(lingThinking)"
                let thinkTag = thinkingEnabled ? "\n<think>" : "\n<think></think>"
                formattedPrompt = "<role>SYSTEM</role>\(sysPart)<|role_end|><role>HUMAN</role>\(prompt)<|role_end|><role>ASSISTANT</role>\(thinkTag)"
            }
        } else if modelConfig?.isSparkModel == true {
            // Spark 2.5 DeepSeek-style dialect; X2.6 adds the effort line to the
            // System block (never when thinking is off, per the template).
            if prompt.contains("<|Bot|>") || prompt.contains("<|User|>") || prompt.contains("<|System|>") {
                formattedPrompt = prompt
            } else {
                let systemBody = Self.sparkSystemBody(
                    cleanSystem: cleanSystem,
                    thinkingEnabled: thinkingEnabled,
                    thinkingEffort: thinkingEffort
                )
                var p = "<｜start▁of▁sentence｜><|System|>\n" + systemBody + "<｜end▁of▁sentence｜>"
                p += "<｜start▁of▁sentence｜><|User|>\(prompt)<｜end▁of▁sentence｜>"
                let sparkThink = thinkingEnabled ? "<think>" : "<think></think>"
                p += "<｜start▁of▁sentence｜><|Bot|>\(sparkThink)"
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
    /// underscores (AGENTS.md: "Reading files safely").
    public static let sparkEndOfSentence = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>"

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

    /// Gemma 4's native reasoning channel markers. The model opens the channel
    /// itself after `<|turn>model` when thinking is on, so a completed turn's
    /// echoed reasoning must be re-wrapped in them: the canonical
    /// ` thinking`/`</think>` markers the adapters fold in are the *other*
    /// families' tags and would not reproduce the pinned generation.
    static let gemmaReasoningOpen = "<|channel>thought\n"
    static let gemmaReasoningClose = "<channel|>"

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
    public struct ConversationTurn: Sendable, Equatable {
        public enum Role: Sendable, Equatable {
            case user
            case assistant
        }

        public var role: Role
        public var text: String

        public init(role: Role, text: String) {
            self.role = role
            self.text = text
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
        promptDialect: DeepSeekV41PromptDialect = DeepSeekV41PromptDialect()
    ) -> String {
        guard !turns.isEmpty else { return "" }

        if modelConfig?.isDeepSeekV41Model == true {
            // A completed assistant turn is rendered from its own content: the
            // reasoning splitter already marks where its channel closed, so the
            // dialect re-opens the channel only for a turn that carried
            // reasoning, and always for the turn being generated now.
            let rendered = turns.map { turn -> DeepSeekV41PromptDialect.Turn in
                guard turn.role == .assistant else {
                    return .init(isUser: true, text: turn.text)
                }
                let split = TurnTextSplitter.splitThinkingAndResponse(
                    raw: turn.text,
                    promptRequestsThinking: false
                )
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
            let openPrefix = "<role>ASSISTANT</role>" + "\n" + TurnTextSplitter.canonicalReasoningOpenTag
            let emptyPrefix = openPrefix + TurnTextSplitter.canonicalReasoningCloseTag
            var p = "<role>SYSTEM</role>" + sysPart + "<|role_end|>"
            for turn in turns {
                if turn.role == .user {
                    p += "<role>HUMAN</role>" + turn.text + "<|role_end|>"
                } else {
                    let opens = thinkingEnabled || TurnTextSplitter.containsReasoningClose(turn.text)
                    p += (opens ? openPrefix : emptyPrefix)
                        + Self.completedAssistantBody(prefixOpensReasoning: opens, text: turn.text)
                        + "<|role_end|>"
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
            let openPrefix = "<｜start▁of▁sentence｜><|Bot|>" + TurnTextSplitter.canonicalReasoningOpenTag
            let emptyPrefix = openPrefix + TurnTextSplitter.canonicalReasoningCloseTag
            var p = "<｜start▁of▁sentence｜><|System|>\n" + sysLine + "<｜end▁of▁sentence｜>"
            for turn in turns {
                if turn.role == .user {
                    p += "<｜start▁of▁sentence｜><|User|>" + turn.text + "<｜end▁of▁sentence｜>"
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
        var p = ""
        if !cleanSystem.isEmpty {
            p += "<|im_start|>system\n" + cleanSystem + "<|im_end|>\n"
        }
        for turn in turns {
            if turn.role == .user {
                p += "<|im_start|>user\n" + turn.text + "<|im_end|>\n"
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
    static func isAlreadyDeepSeekTemplated(
        _ prompt: String,
        markers: DeepSeekV41PromptMarkers = .familyDefault
    ) -> Bool {
        let m = markers
        return prompt.contains(m.user)
            || prompt.contains(m.assistant)
            || prompt.contains(m.beginOfSentence)
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
        if modelConfig?.isLingModel == true { return "<|role_end|>" }
        if modelConfig?.isSparkModel == true { return nil }
        if modelConfig?.isGemma4Model == true { return "<turn|>" }
        return "<|im_end|>"
    }
}
