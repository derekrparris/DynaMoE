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
    /// the entire history is folded into a rolling general summary plus a
    /// detailed recap of the most recent work, the on-screen chat becomes a
    /// blank slate with just the compaction marker, and both digests ride in
    /// every later prompt on either backend. `focus` carries the user's
    /// optional extra instructions for the pass.
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
        // Every whitespace kind (spaces, tabs, newlines) delimits the command
        // word, not just the literal space: a newline- or tab-separated
        // composer draft must dispatch the same as a space-separated one.
        let parts = trimmed.split(
            maxSplits: 1,
            omittingEmptySubsequences: true,
            whereSeparator: { $0.isWhitespace }
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
