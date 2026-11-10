//
//  StatusNotice.swift
//  DynaMoE
//
//  One transient message surface for the app: the composer's attachment
//  feedback and the chat window's refusals, failures and turn outcomes.
//
//  A notice is deliberately not a layout child of anything. The composer's
//  height is what sizes the transcript above it and the chat column's height is
//  the window, so a message that could resize either would move the very thing
//  it is reporting on. Both call sites therefore draw these as overlays, and the
//  rules that a layout cannot express live here: which kinds clear themselves and
//  when a repeated message is allowed to re-run its fade.
//

import SwiftUI

/// What a surface is momentarily saying, and whether it clears itself.
nonisolated struct StatusNotice: Identifiable, Equatable {
    enum Kind: Equatable {
        case info
        case success
        case warning
        /// Work that is still running: a timer would clear the message while the
        /// thing it describes is still happening.
        case progress

        var autoDismisses: Bool { self != .progress }
    }

    let id: UInt64
    let kind: Kind
    let text: String
}

nonisolated enum StatusNoticePolicy {
    /// Six seconds: long enough to read a two-line refusal while the pointer is
    /// still where the action landed, short enough not to linger over the chat.
    static let autoDismissNanoseconds: UInt64 = 6_000_000_000
    /// The composer's copy while an attachment decodes.
    static let preparingText = "Preparing image…"

    /// The two writers that describe work still running rather than an event that
    /// happened: compaction folds a long history in several passes and the
    /// on-device model spools up before its first token. Their copy lives here so
    /// it can be pinned beside the rule it obeys — `.progress` is the tint and the
    /// pill draws its own wait indicator, so neither may lead with a severity
    /// glyph that would announce an event while the work is still going.
    static func compactionText(messageCount: Int) -> String {
        "🗂 Compacting \(messageCount) messages via Apple Foundation Model…"
    }

    static let appleFoundationModelGeneratingText = "🍎 Apple Foundation Model — generating…"

    /// True when `text` is already on screen, so a repeated message restarts its
    /// dismissal timer without re-running the fade.
    static func isRepeat(
        text: String,
        kind: StatusNotice.Kind,
        current: StatusNotice?
    ) -> Bool {
        guard let current else { return false }
        return current.text == text && current.kind == kind
    }

    /// The severity each leading glyph states. The call sites and this table are
    /// two owners of the same decision, so the mapping lives in one array: a
    /// message's tint and the glyph in front of it cannot come from different
    /// places, and `statesContradictorySeverity` catches a call site that passes a
    /// kind beside a glyph that says otherwise.
    private static let severityByGlyph: [(glyph: String, kind: StatusNotice.Kind)] = [
        ("✅", .success),
        ("✨", .success),
        // A stop the user asked for is an outcome, not a fault: it reads as
        // neutral, while a guard or a failure is the thing worth an orange pill.
        ("⚠️", .warning),
        ("❌", .warning),
        ("🛡️", .warning)
    ]

    /// Classifies a message authored with a leading glyph. The writers of these
    /// messages already marked severity in the first character, so the tint is
    /// derived from it rather than from a second table that could disagree.
    ///
    /// Nothing here maps to `.progress`: a glyph announces an event, and an event
    /// has happened, so it clears itself like any other outcome. `.progress` is
    /// for a call site that knows it is waiting on unfinished work.
    static func kind(for text: String) -> StatusNotice.Kind {
        for entry in severityByGlyph where text.hasPrefix(entry.glyph) {
            return entry.kind
        }
        return .info
    }

    /// True when the message leads with a glyph that states one severity while the
    /// call site passed another. Unglyphed messages are not a disagreement: a
    /// writer that names no severity in the text is free to pass one, and that is
    /// the only way to tint copy that has no glyph to read.
    static func statesContradictorySeverity(text: String, kind: StatusNotice.Kind) -> Bool {
        guard severityByGlyph.contains(where: { text.hasPrefix($0.glyph) }) else { return false }
        return self.kind(for: text) != kind
    }

    /// The notices one floating layer shows, furthest from the composer first.
    /// The window and the composer can hold the same sentence at once (a model
    /// switch retracts a refusal and then reports the staged images it stranded),
    /// and a fact repeated on two pills reads as two problems, so the composer's
    /// copy is the one kept.
    static func stacked(window: StatusNotice?, composer: StatusNotice?) -> [StatusNotice] {
        var notices: [StatusNotice] = []
        if let window, window.text != composer?.text { notices.append(window) }
        if let composer { notices.append(composer) }
        return notices
    }
}

/// A notice as a floating pill. Nothing in the layout reads its height, and it
/// never takes a click: whatever is under it stays live.
struct StatusNoticeBubble: View {
    let notice: StatusNotice
    let zoomScale: CGFloat

    private var tint: Color {
        switch notice.kind {
        case .warning: return .orange
        case .success: return .green
        case .info: return .secondary
        case .progress: return .purple
        }
    }

    private var glyph: String {
        switch notice.kind {
        case .warning: return "exclamationmark.triangle.fill"
        case .success: return "checkmark.circle.fill"
        case .info, .progress: return "info.circle.fill"
        }
    }

    var body: some View {
        HStack(spacing: 6) {
            if notice.kind == .progress {
                ProgressView()
                    .controlSize(.small)
                    .scaleEffect(0.7)
            } else {
                Image(systemName: glyph)
                    .font(.system(size: max(9, 10.5 * zoomScale), weight: .semibold))
                    .foregroundColor(tint)
            }
            Text(notice.text)
                .font(.system(size: max(9, 10.5 * zoomScale), weight: .medium))
                .foregroundColor(notice.kind == .warning || notice.kind == .success ? tint : .primary)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(NSColor.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 9))
        .overlay(
            RoundedRectangle(cornerRadius: 9)
                .stroke(tint.opacity(0.45), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.12), radius: 8, x: 0, y: 3)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(notice.text)
    }
}
