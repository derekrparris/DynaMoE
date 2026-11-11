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
    /// detailed recap of the most recent work, the on-screen history becomes a
    /// digest card showing both plus the compaction marker, and both digests
    /// ride in every later prompt on either backend. `focus` carries the
    /// user's optional extra instructions for the pass.
    case compact(focus: String?)

    /// Switch the conversation between the coder and the read-only planner.
    /// `enabled` is nil for a bare `/plan` (turn plan mode on), true for
    /// `/plan on`, false for `/plan off`.
    case plan(enabled: Bool?)

    /// Every recognized command name, lowercase — the composer can render
    /// these as hint UI later without touching the parser.
    nonisolated static let names = ["/compact", "/plan"]

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
        let rest = String(argument).trimmingCharacters(in: .whitespacesAndNewlines)
        switch first.lowercased() {
        case "/compact":
            return .compact(focus: rest.isEmpty ? nil : rest)
        case "/plan":
            // A bare `/plan` is the short way to enter plan mode; `on`/`off`
            // let a conversation state the mode explicitly, which is what a
            // user who cannot tell which mode they are in needs. An argument
            // that is neither is NOT a command: `/plan off.` used to enter the
            // mode, which does the opposite of what was typed, and returning nil
            // sends the text as an ordinary message where the typo is visible.
            switch rest.lowercased() {
            case "", "on", "enter", "start": return .plan(enabled: true)
            case "off", "exit", "leave", "code": return .plan(enabled: false)
            default: return nil
            }
        default:
            return nil
        }
    }
}
