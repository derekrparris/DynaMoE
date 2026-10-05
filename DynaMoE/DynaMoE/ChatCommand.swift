//
//  ChatCommand.swift
//  DynaMoE
//
//  Created by Derek Parris on 10/5/26.
//

import Foundation

/// Composer slash commands. A message whose first word is a known command name
/// (case-insensitive) is dispatched as an app action instead of sent as chat;
/// anything else — including unknown slash-words — passes through as ordinary
/// text, so pasted shell lines or file paths that begin with "/" can never
/// trigger a command by accident.
nonisolated public enum ChatCommand: Equatable {
    /// Summarize and compact the conversation with the Apple Foundation Model:
    /// everything older than the recent verbatim tail is folded into a rolling
    /// general summary plus a detailed recap of the most recently evicted work.
    /// `focus` carries the user's optional extra instructions for the pass.
    case compact(focus: String?)

    /// Every recognized command name, lowercase — the composer can render
    /// these as hint UI later without touching the parser.
    nonisolated static let names = ["/compact"]

    /// Parses `text` as a slash command. The command word is the first
    /// whitespace-delimited token; whatever follows is the command's argument
    /// text. Returns nil when the text is not a recognized command.
    nonisolated public static func parse(_ text: String) -> ChatCommand? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/") else { return nil }
        let parts = trimmed.split(
            separator: " ",
            maxSplits: 1,
            omittingEmptySubsequences: true
        )
        guard let first = parts.first else { return nil }
        let argument = parts.count > 1 ? parts[1] : ""
        let focus = String(argument).trimmingCharacters(in: .whitespacesAndNewlines)
        switch first.lowercased() {
        case "/compact":
            return .compact(focus: focus.isEmpty ? nil : focus)
        default:
            return nil
        }
    }
}
