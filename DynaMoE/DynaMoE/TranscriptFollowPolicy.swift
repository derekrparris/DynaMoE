//
//  TranscriptFollowPolicy.swift
//  DynaMoE
//
//  The transcript's pin-to-tail decision, as a pure function of two scroll
//  geometries.
//
//  Split out of `ChatDetailView` so the rule the follow depends on can be tested
//  without a window. Every pin defect since the explicit follow replaced
//  `defaultScrollAnchor(.bottom)` has been a disagreement about what an offset
//  change MEANS, and that question has exactly one answer: only the reader may
//  unpin. A programmatic follow, a lazy stack re-estimating an unrealized row and
//  the clamp that follows a content SHRINK all move the offset up the document
//  with nobody touching it, and reading any of them as a drag is what stopped
//  the follow mid-turn.
//
//  The second half of the rule is that the follow cannot key off text alone. A
//  tool-call card landing under a capped reasoning trace, a status line, or a
//  row re-segmenting grows the transcript without a character of new output, so
//  a text bucket never fires; after a long reasoning trace there may be no new
//  text at all until the tool returns, which leaves the newest content a whole
//  card below the fold for the length of the call.
//

import CoreGraphics

nonisolated enum TranscriptFollowPolicy {
    /// The geometry one scroll update is decided from.
    struct Snapshot: Equatable {
        var offset: CGFloat
        var contentHeight: CGFloat
        var containerHeight: CGFloat

        var distanceFromBottom: CGFloat {
            contentHeight - (offset + containerHeight)
        }
    }

    /// What one geometry update means for the pin.
    struct Outcome: Equatable {
        /// The pin to write, or nil to leave it alone.
        var pin: Bool?
        /// True when the transcript grew under a pinned reader and has to follow.
        var follow: Bool

        static let none = Outcome(pin: nil, follow: false)
    }

    /// How close to the tail the reader must be for the pin to reattach. Small
    /// enough that it is not a snap zone, wide enough to absorb the round-trip
    /// of a follow that lands a point short.
    static let reattachDistance: CGFloat = 48

    /// Content growth one follow covers, in points, about two lines.
    ///
    /// Coarse on purpose: a follow per streamed chunk was the
    /// layout-invalidation storm that made the old character bucket necessary,
    /// so structural growth is followed in height steps rather than on every
    /// change. A tool card is hundreds of points tall and crosses at least one
    /// step in a single update, so it still lands in the same frame.
    static let followHeightStep: CGFloat = 60

    /// Below this a change is measurement noise rather than movement.
    static let epsilon: CGFloat = 1

    static func decide(
        from old: Snapshot,
        to new: Snapshot,
        isPinned: Bool,
        isReaderScrolling: Bool,
        isFollowInFlight: Bool
    ) -> Outcome {
        // A follow reports intermediate offsets for its whole duration, and none
        // of them is a drag.
        guard !isFollowInFlight else { return .none }

        if isReaderScrolling, new.offset < old.offset - epsilon {
            // The reader dragged up the document: they are reading history now.
            return isPinned ? Outcome(pin: false, follow: false) : .none
        }

        var outcome = Outcome.none
        if new.distanceFromBottom <= reattachDistance, !isPinned {
            outcome.pin = true
        }
        if new.contentHeight > old.contentHeight + epsilon,
           isPinned || outcome.pin == true,
           step(of: new) > step(of: old) {
            outcome.follow = true
        }
        return outcome
    }

    private static func step(of snapshot: Snapshot) -> Int {
        Int(max(0, snapshot.contentHeight) / followHeightStep)
    }
}
