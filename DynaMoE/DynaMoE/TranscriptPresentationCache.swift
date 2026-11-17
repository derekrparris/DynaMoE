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
//  It is also where the parse is AVOIDED. Most messages in a long conversation
//  are far under the row budget, and those are never parsed here at all: the
//  transcript emits one row for each and the row parses when it is realized.
//  What the row loop still needs before it can decide that is the plan-card
//  classification, which is why that answer is cached on its own.
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

    /// A plan-card classification, kept apart from `Entry` because it is asked
    /// BEFORE the decision to parse at all.
    private struct PlanEntry {
        var content: String
        var isPlanCard: Bool
    }

    private var entries: [UUID: Entry] = [:]
    private var planEntries: [UUID: PlanEntry] = [:]
    private var plainTextEntries: [UUID: (content: String, segments: [TranscriptSegment])] = [:]

    /// Whether `message` is a plan-mode deliverable, answered WITHOUT parsing.
    ///
    /// The transcript asks this first: a plan card renders its own text, and a
    /// message that fits one row is parsed by that row when it is realized, so
    /// neither may be parsed here. It is cached on the content like the parse
    /// is, because the probe reads every character of the answer and the row
    /// loop runs on every body pass, i.e. on every streamed chunk.
    func isPlanCard(for message: ChatMessage) -> Bool {
        guard message.role == .assistant else { return false }
        if let entry = planEntries[message.id], entry.content == message.content {
            return entry.isPlanCard
        }
        let isPlanCard = PlanMarkers.anyMarkerMentioned(in: message.content)
            && PlanMarkers.startPresent(in: message.content)
        planEntries[message.id] = PlanEntry(content: message.content, isPlanCard: isPlanCard)
        return isPlanCard
    }

    /// Segments for a user turn, memoized per content change.
    ///
    /// A user turn draws its text verbatim, so there is no markdown parse to
    /// cache — but the chunking is O(characters) and the row loop asks for it on
    /// every body pass. A conversation with a few pasted turns was re-chunking
    /// all of them per streamed chunk.
    func plainTextSegments(for message: ChatMessage) -> [TranscriptSegment] {
        if let entry = plainTextEntries[message.id], entry.content == message.content {
            return entry.segments
        }
        let segments = TranscriptSegmenter.plainTextSegments(for: message.content)
        plainTextEntries[message.id] = (message.content, segments)
        return segments
    }

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
        if isPlanCard(for: message) {
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

    /// The presentation already held for `message`, or nil. Never parses.
    ///
    /// The geometry report asks for the shape of every message in a conversation
    /// it may not have drawn at all — a restored one — and forcing a parse there
    /// would put back, 1.5 s after the window opens, precisely the stall the
    /// lazy path removes. The repair flag is deliberately not part of the match:
    /// this answers "how many rows and blocks", which the two variants agree on.
    func cachedPresentation(for message: ChatMessage, isGemmaDialect: Bool) -> Presentation? {
        guard let entry = entries[message.id],
              entry.content == message.content,
              entry.isGemmaDialect == isGemmaDialect else { return nil }
        return entry.presentation
    }

    /// Drops entries for messages that are no longer in the transcript, so a
    /// long-lived window does not accumulate every message ever shown.
    ///
    /// Always filters. A previous `entries.count > ids.count` short-circuit
    /// skipped the prune whenever the cache held exactly one entry per message in
    /// the current transcript, which is precisely what switching to a session of
    /// the SAME length leaves: every entry belonged to the other session, the
    /// counts matched, and nothing was dropped until some later turn happened to
    /// change them.
    func prune(keeping ids: Set<UUID>) {
        entries = entries.filter { ids.contains($0.key) }
        planEntries = planEntries.filter { ids.contains($0.key) }
        plainTextEntries = plainTextEntries.filter { ids.contains($0.key) }
    }

    /// How many messages' parses are resident. Exposed so the prune above is
    /// observable in a test; nothing in the view reads it.
    var cachedMessageCount: Int { entries.count }

    /// Resident plan-card classifications. Separate from the parse count on
    /// purpose: the row loop classifies every answer on every body pass, so this
    /// is the number that says that work is cached rather than repeated.
    var cachedPlanClassificationCount: Int { planEntries.count }

    /// Resident user-turn chunkings.
    var cachedPlainTextCount: Int { plainTextEntries.count }
}
