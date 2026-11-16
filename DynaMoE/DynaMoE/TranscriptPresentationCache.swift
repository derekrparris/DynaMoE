//
//  TranscriptPresentationCache.swift
//  DynaMoE
//
//  Derives and caches what one message needs to render: its display text and
//  its transcript segments.
//
//  Segmentation must not multiply parse cost. The markdown parse happens ONCE
//  per message content change, and every row of that message renders a slice of
//  the same result. The cache is owned by the transcript view rather than by
//  each row, because a row is created and destroyed as the lazy window moves
//  and would otherwise re-parse the whole message on every realization — which
//  is exactly the cost segmentation exists to avoid.
//

import Foundation

final class TranscriptPresentationCache {
    struct Presentation {
        var displayContent: String
        var segments: [TranscriptSegment]
        /// True when the message is a plan-mode deliverable, which the transcript
        /// draws as `PlanCard` instead of `ChatMessageView`.
        ///
        /// Classified here, once per content change, rather than in the row loop:
        /// the loop runs on every body pass, and the probe is two substring scans
        /// of the whole answer plus a line split of it when they hit. On a long
        /// conversation that was tens of milliseconds per frame, on the main
        /// actor, during a turn. A plan card is also the longest message in a
        /// conversation, so it skips the markdown parse entirely.
        var isPlanCard: Bool = false
    }

    private struct Entry {
        var content: String
        var isGemmaDialect: Bool
        var repairs: Bool
        var presentation: Presentation
    }

    private var entries: [UUID: Entry] = [:]

    /// The presentation for `message`, reusing the cached parse when nothing
    /// that affects it has changed.
    ///
    /// The content comparison is cheap in the common case: the stored value is a
    /// copy of the same `String` buffer, and `String` equality short-circuits on
    /// identical storage before comparing bytes.
    func presentation(
        for message: ChatMessage,
        isGemmaDialect: Bool,
        isStreaming: Bool
    ) -> Presentation {
        // While an answer streams, the malformed-markdown repair passes are
        // skipped: they cost more than the rest of the parse together and the
        // growing text is re-parsed on every update. The settled message
        // re-parses once with them enabled, so its rendering is unaffected.
        let repairs = !isStreaming
        if let entry = entries[message.id],
           entry.content == message.content,
           entry.isGemmaDialect == isGemmaDialect,
           entry.repairs == repairs {
            return entry.presentation
        }

        // A plan card is classified before the parse and skips it: the card
        // renders the message's own text through `PlanMarkers.strip`, so a
        // segment list for it would be thrown away.
        if message.role == .assistant,
           PlanMarkers.anyMarkerMentioned(in: message.content),
           PlanMarkers.startPresent(in: message.content) {
            let presentation = Presentation(
                displayContent: message.content,
                segments: [],
                isPlanCard: true
            )
            entries[message.id] = Entry(
                content: message.content,
                isGemmaDialect: isGemmaDialect,
                repairs: repairs,
                presentation: presentation
            )
            return presentation
        }

        let display = ChatMessageView.responseDisplayContent(
            message.content,
            isGemmaDialect: isGemmaDialect
        )
        var blocks = parseMarkdownBlocks(display, repairs: repairs)
        // A non-empty answer that parses to no blocks must still render its
        // text: dropping it would turn a parser edge case into a blank row.
        if blocks.isEmpty, !display.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            blocks = [.paragraph(text: display)]
        }
        let presentation = Presentation(
            displayContent: display,
            segments: TranscriptSegmenter.segments(for: blocks)
        )
        entries[message.id] = Entry(
            content: message.content,
            isGemmaDialect: isGemmaDialect,
            repairs: repairs,
            presentation: presentation
        )
        return presentation
    }

    /// Drops entries for messages that are no longer in the transcript, so a
    /// long-lived window does not accumulate every message ever shown.
    func prune(keeping ids: Set<UUID>) {
        guard entries.count > ids.count else { return }
        entries = entries.filter { ids.contains($0.key) }
    }
}
