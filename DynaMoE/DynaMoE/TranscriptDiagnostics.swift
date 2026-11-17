//
//  TranscriptDiagnostics.swift
//  DynaMoE
//
//  Instrumentation for the transcript whiteout (Workstream 0).
//
//  The symptom — a long agent turn renders blank until something changes the
//  content height — has been blamed on three different defects, and each prior
//  fix addressed one of them without confirming which one crosses the limit.
//  This emits the numbers that make the hypothesis falsifiable: the scroll
//  content height, the viewport height, and the height of the tallest realized
//  row, with a one-shot warning the first time a row crosses the compositor's
//  ceiling.
//
//  The content height is reported and deliberately NOT warned on. It is the
//  scroll view's total, which a lazy stack never paints as one layer: a
//  conversation about twenty viewports tall crosses 16,384 pt routinely and
//  paints fine, so a warning there fires on ordinary history and says nothing
//  about the defect. The ceiling applies to a REALIZED row, which is what the
//  row warning measures.
//
//  Metadata only, per `DiagLog`'s contract: counts and heights, never message
//  content.
//

import Foundation

/// Collects transcript geometry and reports it once per turn, plus a one-shot
/// warning on threshold crossing.
///
/// A reference type owned by the transcript view: the measurements arrive from
/// geometry callbacks on every layout pass, and none of them should schedule a
/// SwiftUI update.
final class TranscriptDiagnostics {
    static let category = "transcript"

    /// One realized row's measurement, with the label that says what it is. A
    /// bare height cannot be acted on: whether the tall row is an answer, a
    /// reasoning accordion or a tool-call list decides where the fix belongs.
    struct RowMeasurement {
        var height: CGFloat
        var label: String
    }

    /// Rows measured in the current turn, keyed by row id.
    private var rowHeights: [String: RowMeasurement] = [:]
    private var contentHeight: CGFloat = 0
    private var viewportHeight: CGFloat = 0
    /// The last turn whose numbers were emitted. Cleared when a turn begins so a
    /// later turn whose key collides with it is still reported; the key is built
    /// from the message count, which repeats across conversations of one length.
    private(set) var lastReportedTurnKey: String?
    /// The last turn the row-height warning fired for.
    ///
    /// Keyed on the turn rather than a bare `Bool`, because `beginTurn()` only
    /// runs on the generating transition: a conversation RESTORED from disk
    /// reports without one, so a process that had warned once stayed silent for
    /// every restored long conversation afterwards — the other reproduction this
    /// instrumentation exists for.
    private(set) var warnedTurnKey: String?

    /// Records one realized row's height. Called from the row's own geometry
    /// callback, so it only ever sees rows the lazy stack actually built.
    func record(rowId: String, label: String, height: CGFloat) {
        rowHeights[rowId] = RowMeasurement(height: height, label: label)
    }

    /// Records the scroll view's own geometry.
    func record(contentHeight: CGFloat, viewportHeight: CGFloat) {
        self.contentHeight = contentHeight
        self.viewportHeight = viewportHeight
    }

    /// Forgets per-turn row measurements. Called when a new turn starts, so the
    /// report describes the turn that is running rather than the whole session.
    func beginTurn() {
        rowHeights.removeAll(keepingCapacity: true)
        lastReportedTurnKey = nil
        warnedTurnKey = nil
    }

    /// Emits the turn's numbers once, when the turn settles.
    ///
    /// `turnKey` identifies the turn (the streaming message's id plus its
    /// length bucket), so a settle that fires repeatedly for the same turn logs
    /// once. `messages` supplies the per-message block and character counts,
    /// which are what the segmenter's budget is expressed in.
    func reportIfNeeded(
        turnKey: String,
        messages: [ChatMessage],
        segmentsByMessage: [UUID: [TranscriptSegment]]
    ) {
        guard turnKey != lastReportedTurnKey else { return }
        lastReportedTurnKey = turnKey

        let tallest = rowHeights.values.max { $0.height < $1.height }
        let tallestHeight = tallest?.height ?? 0
        let tallestLabel = tallest?.label ?? "none"
        let realized = rowHeights.count
        var blockCount = 0
        var characters = 0
        var segmentCount = 0
        for message in messages {
            let segments = segmentsByMessage[message.id] ?? []
            segmentCount += segments.count
            for segment in segments {
                blockCount += segment.blocks.count
            }
            characters += message.content.count + (message.thinkingContent?.count ?? 0)
        }

        DiagLog.info(
            Self.category,
            "turn=\(turnKey) contentHeight=\(Self.points(contentHeight))pt "
                + "viewport=\(Self.points(viewportHeight))pt realizedRows=\(realized) "
                + "tallestRow=\(Self.points(tallestHeight))pt tallestRowKind=\(tallestLabel) "
                + "rows=\(segmentCount) "
                + "blocks=\(blockCount) "
                + "characters=\(characters) messages=\(messages.count)"
        )

        if tallestHeight > TranscriptSegmenter.rowHeightWarning, warnedTurnKey != turnKey {
            warnedTurnKey = turnKey
            DiagLog.warning(
                Self.category,
                "row exceeds \(Self.points(TranscriptSegmenter.rowHeightWarning))pt: "
                    + "tallestRow=\(Self.points(tallestHeight))pt kind=\(tallestLabel) turn=\(turnKey) "
                    + "a row this tall is at risk of the compositor ceiling "
                    + "(\(Self.points(TranscriptSegmenter.compositorHeightCeiling))pt)"
            )
        }
    }

    /// One height, formatted for a log line.
    ///
    /// `Int(CGFloat)` TRAPS on a non-finite value, and geometry can report one:
    /// an unlaid-out scroll view publishes `.infinity`, which was enough to kill
    /// the process from this instrumentation. Observing a rendering bug must
    /// never become a worse bug, so a value that cannot be printed is printed as
    /// `unknown` rather than converted.
    private static func points(_ value: CGFloat) -> String {
        guard value.isFinite else { return "unknown" }
        // Clamped before the cast for the same reason: a finite value too large
        // for `Int` traps exactly as hard.
        let clamped = min(max(value, 0), CGFloat(Int.max - 1))
        return String(Int(clamped.rounded()))
    }
}
