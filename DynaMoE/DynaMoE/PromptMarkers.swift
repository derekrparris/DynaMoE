//
//  PromptMarkers.swift
//  DynaMoE
//
//  Phase 8 — Skills.
//
//  Chat-template control markers are LIVE syntax once text is pasted into a
//  prompt: a skill `description:` or a tool result that contains `<|im_end|>`
//  closes the turn it landed in and opens a forged one. Before this file,
//  `AgentHarness.sanitizeText` replaced the ChatML pair and the tool tags and
//  nothing else, so Gemma's `<|turn>`, Llama 3's headers and DeepSeek's
//  fullwidth role markers passed straight through.
//
//  One pass, every family this app renders. A marker that survives is a
//  boundary, so the substitution is deliberately eager: over-replacing only
//  changes what the model reads, never who is speaking.
//

import Foundation

public nonisolated enum PromptMarkers {
    /// Ordered so a longer marker is replaced before any shorter marker that is
    /// a prefix of it (none today, but the tool tags make that easy to break).
    private static let substitutions: [(String, String)] = [
        // ChatML (Qwen and most in-app agent templates).
        ("<|im_start|>", "[im_start]"),
        ("<|im_end|>", "[im_end]"),
        // The in-app tool dialect.
        ("<tool_call>", "[tool_call]"),
        ("</tool_call>", "[/tool_call]"),
        ("<tool_response>", "[tool_response]"),
        ("</tool_response>", "[/tool_response]"),
        // Gemma turn boundaries and reasoning channel.
        ("<|turn>", "[turn]"),
        ("<turn|>", "[end_turn]"),
        (ChatPromptFormatter.gemmaChannelOpen, "[channel]"),
        (ChatPromptFormatter.gemmaReasoningClose, "[end_channel]"),
        (ChatPromptFormatter.gemmaThinkMarker, "[think]"),
        ("<bos>", "[bos]"),
        ("<|\"|>", "[quote]"),
        // Ling/Bailing 3.0 role spans. Both halves are control syntax: the
        // assistant turn boundary is `<role>ASSISTANT</role>`.
        (ChatPromptFormatter.lingRoleOpen, "[role]"),
        (ChatPromptFormatter.lingRoleClose, "[/role]"),
        (ChatPromptFormatter.lingRoleEnd, "[role_end]"),
        // Spark 2.5's framing (ASCII role markers plus fullwidth sentence
        // markers). Taken from the formatter's constants for the same reason
        // as DeepSeek's below.
        (ChatPromptFormatter.sparkStartOfSentence, "[start_of_sentence]"),
        (ChatPromptFormatter.sparkEndOfSentence, "[end_of_sentence]"),
        (ChatPromptFormatter.sparkSystemMarker, "[System]"),
        (ChatPromptFormatter.sparkUserMarker, "[User]"),
        (ChatPromptFormatter.sparkBotMarker, "[Bot]"),
        (ChatPromptFormatter.sparkToolMarker, "[Tool]"),
        // Llama 3 headers and text boundaries.
        ("<|start_header_id|>", "[start_header_id]"),
        ("<|end_header_id|>", "[end_header_id]"),
        ("<|begin_of_text|>", "[begin_of_text]"),
        ("<|endoftext|>", "[endoftext]"),
        ("<|eot_id|>", "[eot_id]"),
        // The canonical reasoning pair. These are the tags the adapters fold
        // into a rendered assistant turn, so text carrying them can close the
        // reasoning block it landed in and go on speaking outside it. Taken
        // from `TurnTextSplitter` so the sanitizer never retypes a tag literal
        // (AGENTS.md: "Reading files safely").
        (TurnTextSplitter.canonicalReasoningOpenTag, "[think]"),
        (TurnTextSplitter.canonicalReasoningCloseTag, "[/think]"),
        // Ornith's reasoning span, which `TurnTextSplitter` already treats as a
        // boundary when it splits a turn.
        ("<thought>", "[think]"),
        ("</thought>", "[/think]"),
        ("<|thought|>", "[think]"),
        ("</|thought|>", "[/think]"),
        // Hermes / Qwen function-calling framing: some templates emit these
        // instead of the in-app tool tags, and they are control syntax either
        // way.
        ("<tool_call>", "[tool_call]"),
        ("</tool_call>", "[/tool_call]"),
        ("<tool_response>", "[tool_response]"),
        ("</tool_response>", "[/tool_response]"),
        // DeepSeek V4.1 spells its special tokens with FULLWIDTH vertical bars
        // (U+FF5C) and BLOCK LOW LINE (U+2581). Taken from the dialect's own
        // constants so a source edit cannot silently ASCII-ize them.
        (DeepSeekV41PromptMarkers.familyDefault.beginOfSentence, "[begin_of_sentence]"),
        (DeepSeekV41PromptMarkers.familyDefault.endOfSentence, "[end_of_sentence]"),
        (DeepSeekV41PromptMarkers.familyDefault.system, "[System]"),
        (DeepSeekV41PromptMarkers.familyDefault.user, "[User]"),
        (DeepSeekV41PromptMarkers.familyDefault.assistant, "[Assistant]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolCallsBegin, "[tool_calls]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolCallsEnd, "[/tool_calls]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolCallBegin, "[tool_call_begin]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolCallEnd, "[tool_call_end]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolOutputBegin, "[tool_output]"),
        (DeepSeekV41PromptMarkers.familyDefault.toolOutputEnd, "[/tool_output]"),
    ]

    /// The subset that is control syntax for THIS app's own templates: the
    /// ChatML pair and the in-app tool tags. Everything else in
    /// `substitutions` belongs to a model family whose markers are inert
    /// unless that family is the one rendering.
    ///
    /// Local file content gets this pass and not the broad one. A file the
    /// model is asked to edit must come back byte-faithful: `ChatPromptFormatter`
    /// is full of role-span and system-marker literals, and rewriting them
    /// meant a `file_read` of this repository returned different text than the
    /// file holds, so a `file_edit` anchor copied from that view could never
    /// match. The broad pass stays on text whose author is not the operator.
    private static let localFileSubstitutions: [(String, String)] = [
        ("<|im_start|>", "[im_start]"),
        ("<|im_end|>", "[im_end]"),
        ("<tool_call>", "[tool_call]"),
        ("</tool_call>", "[/tool_call]"),
        ("<tool_response>", "[tool_response]"),
        ("</tool_response>", "[/tool_response]"),
    ]

    /// Replaces any chat-template control marker so interpolated text cannot
    /// open, close or forge a turn.
    ///
    /// For text written by somebody other than the operator: a web page, an
    /// MCP server's result, a skill body, a subagent's report, an error body.
    public static func sanitizeControlMarkers(_ text: String) -> String {
        applying(substitutions, to: text)
    }

    /// The narrow pass for content the operator owns and may be editing.
    /// See `localFileSubstitutions`.
    public static func sanitizeLocalFileMarkers(_ text: String) -> String {
        applying(localFileSubstitutions, to: text)
    }

    private static func applying(_ table: [(String, String)], to text: String) -> String {
        var sanitized = text
        for (marker, replacement) in table {
            sanitized = sanitized.replacingOccurrences(of: marker, with: replacement)
        }
        return sanitized
    }
}
