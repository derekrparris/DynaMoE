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
        modelConfig: ModelConfig?
    ) -> String {
        let thinkSuffix: String
        if thinkingEnabled {
            thinkSuffix = "<think>\n"
        } else if modelSupportsThinking {
            thinkSuffix = "<think>\n\n</think>\n\n"
        } else {
            thinkSuffix = ""
        }

        let isLing = (modelConfig?.isLingModel == true)
        let formattedPrompt: String
        if isLing {
            if prompt.contains("<role>") {
                formattedPrompt = prompt
            } else {
                let lingThinking = thinkingEnabled ? "detailed thinking on" : "detailed thinking off"
                let sysPart = cleanSystem.isEmpty ? lingThinking : "\(cleanSystem)\n\(lingThinking)"
                let thinkTag = thinkingEnabled ? "\n<think>" : "\n<think></think>"
                formattedPrompt = "<role>SYSTEM</role>\(sysPart)<|role_end|><role>HUMAN</role>\(prompt)<|role_end|><role>ASSISTANT</role>\(thinkTag)"
            }
        } else if modelConfig?.isSparkModel == true {
            // Spark 2.5 DeepSeek-style dialect
            if prompt.contains("<|Bot|>") || prompt.contains("<|User|>") || prompt.contains("<|System|>") {
                formattedPrompt = prompt
            } else {
                var p = ""
                if cleanSystem.isEmpty {
                    p += "<｜start▁of▁sentence｜><|System|>\nYou are a helpful assistant.<｜end▁of▁sentence｜>"
                } else {
                    p += "<｜start▁of▁sentence｜><|System|>\n\(cleanSystem)<｜end▁of▁sentence｜>"
                }
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
                    p += "<|turn>model\n<|channel>thought\n<channel|>"
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
}
