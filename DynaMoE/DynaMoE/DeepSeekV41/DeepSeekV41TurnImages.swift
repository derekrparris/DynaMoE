//
//  DeepSeekV41TurnImages.swift
//  DynaMoE
//
//  Phase: DeepSeek V4.1 vision, the composer's image path (slice #38)
//
//  The two halves of carrying an attachment into a turn, split so the engine can
//  provision its cache before the tower runs: `price` answers "how many prompt
//  positions will this become" from the resize plan alone, and `bind` runs the
//  tower and puts its rows at the placeholder positions. Both fail closed, because
//  every way they can go wrong (no tower, a placeholder with no image, an image
//  with no placeholder, a row list that does not match its grid) would otherwise
//  reach the model as a plausible but wrong prompt.
//

import Foundation

nonisolated public enum DeepSeekV41TurnImageError: LocalizedError, Equatable {
    case noTower(images: Int)
    case malformedImage(index: Int)
    case placeholderCountMismatch(images: Int, placeholders: Int)
    case bindingFailed(unusedImages: Int, unfilledPlaceholders: Int)
    /// A prompt that already carries its spans, whose recorded layout does not
    /// describe the ids it holds. Named rather than tolerated: the alternative is
    /// rows landing on one picture's span from another's encode.
    case carriedSpanMismatch(reason: String)
    case carriedSpanCountMismatch(spans: Int, images: Int)

    public var errorDescription: String? {
        switch self {
        case .noTower(let images):
            let noun = images == 1 ? "the attached image" : "the \(images) attached images"
            return "This checkpoint publishes no vision tower, so it cannot see \(noun)."
        case .malformedImage(let index):
            return "Image \(index + 1) decoded to a buffer that does not match its size."
        case .placeholderCountMismatch(let images, let placeholders):
            return "The prompt has \(placeholders) image placeholder(s) for \(images) image(s), "
                + "so the rows would land at the wrong positions."
        case .bindingFailed(let unusedImages, let unfilledPlaceholders):
            return "The image rows could not be bound: \(unusedImages) image(s) unused, "
                + "\(unfilledPlaceholders) placeholder(s) unfilled."
        case .carriedSpanMismatch(let reason):
            return "This continuation's image layout does not match its prompt: \(reason)."
        case .carriedSpanCountMismatch(let spans, let images):
            return "This continuation records \(spans) image span(s) for \(images) attached image(s)."
        }
    }
}

nonisolated public enum DeepSeekV41TurnImages {

    /// What the attachments will cost the prompt, before any weight is read.
    public struct Price: Sendable, Equatable {
        /// The prompt length once every placeholder is replaced by its span.
        public let promptTokens: Int
        /// Positions the spans add beyond the placeholders they replace.
        public let extraPositions: Int
        /// Per-image pricing, in prompt order.
        public let estimates: [DeepSeekV41VisionPricing.Estimate]

        public var seconds: Double {
            estimates.reduce(0) { $0 + $1.seconds }
        }
    }

    /// Prices the attachments and checks that the prompt agrees with them.
    ///
    /// The tower is not needed: `DeepSeekV41VisionLayout.fit` is the same function
    /// the encode uses, so the length it returns is the length the bind produces.
    /// That is what lets the engine provision its KV cache for the expanded prompt
    /// before deciding whether it can afford it.
    public static func price(
        tokenIds: [UInt32],
        images: [DeepSeekV41ImageInput],
        spec: DeepSeekV41VisionSpec?,
        costModel: DeepSeekV41VisionCostModel = .measured
    ) throws -> Price {
        // A model with no tower prices nothing, but it must still refuse pixels:
        // dropping them would answer a question about a picture nobody saw.
        guard let spec else {
            guard images.isEmpty else {
                throw DeepSeekV41TurnImageError.noTower(images: images.count)
            }
            return Price(promptTokens: tokenIds.count, extraPositions: 0, estimates: [])
        }
        // A placeholder with no picture is a prompt-assembly bug, and it stays one
        // even when no picture was supplied at all: expanding it would leave an
        // image token whose row set never existed.
        let marker = UInt32(clamping: spec.imageTokenID)
        let placeholders = tokenIds.filter { $0 == marker }.count
        guard placeholders == images.count else {
            throw DeepSeekV41TurnImageError.placeholderCountMismatch(
                images: images.count, placeholders: placeholders
            )
        }
        guard !images.isEmpty else {
            return Price(promptTokens: tokenIds.count, extraPositions: 0, estimates: [])
        }
        var estimates: [DeepSeekV41VisionPricing.Estimate] = []
        estimates.reserveCapacity(images.count)
        for (index, image) in images.enumerated() {
            guard image.isWellFormed else {
                throw DeepSeekV41TurnImageError.malformedImage(index: index)
            }
            guard let estimate = DeepSeekV41VisionPricing.estimate(
                width: image.width, height: image.height, spec: spec, costModel: costModel
            ) else {
                throw DeepSeekV41TurnImageError.malformedImage(index: index)
            }
            estimates.append(estimate)
        }
        let spanPositions = estimates.reduce(0) { $0 + $1.spanPositions }
        // Every placeholder is already one token and is replaced by its whole span.
        let extra = spanPositions - images.count
        return Price(
            promptTokens: tokenIds.count + extra,
            extraPositions: extra,
            estimates: estimates
        )
    }

    /// Runs the tower for each attachment and binds its rows to the placeholders.
    ///
    /// `encode` is the tower call itself (`DeepSeekV41Runtime.imageSpan`), passed in
    /// so the binding can be tested without 970 MB of weights. The result is what
    /// the turn loop wants: the expanded ids, a position -> residual row map, and
    /// the range each image occupies — the last of which is what a continuation
    /// hands back so the NEXT turn can put the same rows at the same positions.
    public static func bind(
        tokenIds: [UInt32],
        images: [DeepSeekV41ImageInput],
        spec: DeepSeekV41VisionSpec?,
        rowWidth: Int,
        encode: (DeepSeekV41ImageInput) throws -> DeepSeekV41ImagePrompt.ImageSpan
    ) throws -> (tokenIds: [UInt32], rows: [Int: [Float]], spans: [Range<Int>]) {
        guard !images.isEmpty else { return (tokenIds, [:], []) }
        guard let spec else { throw DeepSeekV41TurnImageError.noTower(images: images.count) }
        var spans: [DeepSeekV41ImagePrompt.ImageSpan] = []
        spans.reserveCapacity(images.count)
        for (index, image) in images.enumerated() {
            guard image.isWellFormed else {
                throw DeepSeekV41TurnImageError.malformedImage(index: index)
            }
            spans.append(try encode(image))
        }
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: tokenIds,
            spans: spans,
            imageTokenID: spec.imageTokenID,
            rowWidth: rowWidth
        )
        guard plan.report.unusedImageCount == 0,
              plan.report.unfilledPlaceholderCount == 0 else {
            throw DeepSeekV41TurnImageError.bindingFailed(
                unusedImages: plan.report.unusedImageCount,
                unfilledPlaceholders: plan.report.unfilledPlaceholderCount
            )
        }
        return (plan.tokenIds, plan.rows, plan.spans)
    }

    /// Checks a prompt whose spans a previous turn already expanded, without
    /// reading a weight.
    ///
    /// A continuation reuses the previous turn's prompt ids verbatim — that is what
    /// keeps the pinned prefix byte-identical — so it carries the expanded spans
    /// with them. Nothing about those ids says how many pictures produced them, so
    /// the checks that `price` makes from a marker count have to be made from the
    /// recorded layout instead: every span in bounds and on the checkpoint's image
    /// token, and the layout accounting for EVERY image token the prompt holds. A
    /// prompt with an image token the layout does not claim is the one failure that
    /// reads as "the model answered about something else", so it is refused here,
    /// before a runtime is built.
    ///
    /// Returns the prompt length, which is what the cache is provisioned for.
    public static func priceCarried(
        tokenIds: [UInt32],
        ranges: [Range<Int>],
        spec: DeepSeekV41VisionSpec?
    ) throws -> Int {
        guard !ranges.isEmpty else { return tokenIds.count }
        guard let spec else {
            throw DeepSeekV41TurnImageError.noTower(images: ranges.count)
        }
        _ = try validateCarried(tokenIds: tokenIds, ranges: ranges, spec: spec)
        return tokenIds.count
    }

    /// Puts already-encoded spans back at the positions a previous turn recorded.
    ///
    /// The ids are not rewritten: they already hold the span, and replacing them
    /// with the same tokens is only a way to lose the rest of the prompt.
    public static func place(
        tokenIds: [UInt32],
        spans: [DeepSeekV41ImagePrompt.ImageSpan],
        at ranges: [Range<Int>],
        spec: DeepSeekV41VisionSpec?,
        rowWidth: Int
    ) throws -> (tokenIds: [UInt32], rows: [Int: [Float]]) {
        guard !ranges.isEmpty else { return (tokenIds, [:]) }
        guard let spec else {
            throw DeepSeekV41TurnImageError.noTower(images: ranges.count)
        }
        guard spans.count == ranges.count else {
            throw DeepSeekV41TurnImageError.carriedSpanCountMismatch(
                spans: ranges.count, images: spans.count
            )
        }
        let counts = try validateCarried(tokenIds: tokenIds, ranges: ranges, spec: spec)
        var rows: [Int: [Float]] = [:]
        for (index, span) in spans.enumerated() {
            guard span.rows.count == span.tokenCount else {
                throw DeepSeekV41ImagePromptError.spanLengthMismatch(
                    image: index, expected: span.tokenCount, actual: span.rows.count
                )
            }
            // The recorded length and the encoded one have to agree: a stale span
            // (an encode from a different grid) would shift every row after it.
            guard span.tokenCount == counts[index] else {
                throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                    reason: "image \(index + 1) encodes \(span.tokenCount) positions but the prompt's "
                        + "span is \(counts[index]) long"
                )
            }
            for (position, row) in span.rows.enumerated() {
                guard row.count == rowWidth else {
                    throw DeepSeekV41ImagePromptError.rowWidthMismatch(
                        image: index, position: position, expected: rowWidth, actual: row.count
                    )
                }
            }
            let start = ranges[index].lowerBound
            for (offset, row) in span.rows.enumerated() {
                rows[start + offset] = row
            }
        }
        return (tokenIds, rows)
    }

    /// The shared half of the two carried-prompt entry points: every span is in
    /// bounds and holds the image token, and the prompt carries no image token the
    /// layout does not claim.
    private static func validateCarried(
        tokenIds: [UInt32],
        ranges: [Range<Int>],
        spec: DeepSeekV41VisionSpec
    ) throws -> [Int] {
        let marker = UInt32(clamping: spec.imageTokenID)
        var counts: [Int] = []
        counts.reserveCapacity(ranges.count)
        var claimed = 0
        var previousUpper = 0
        for (index, range) in ranges.enumerated() {
            guard !range.isEmpty else {
                throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                    reason: "image \(index + 1) has an empty span"
                )
            }
            guard range.lowerBound >= 0, range.upperBound <= tokenIds.count else {
                throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                    reason: "image \(index + 1) claims positions \(range.lowerBound)..<"
                        + "\(range.upperBound) of a \(tokenIds.count)-token prompt"
                )
            }
            guard range.lowerBound >= previousUpper else {
                throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                    reason: "image \(index + 1) starts at \(range.lowerBound), before the previous "
                        + "span ended at \(previousUpper)"
                )
            }
            previousUpper = range.upperBound
            // A span must sit on the checkpoint's own image token; anything else
            // means the recorded range describes a different prompt.
            if let stray = range.first(where: { tokenIds[$0] != marker }) {
                throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                    reason: "image \(index + 1)'s span holds token \(tokenIds[stray]) at position "
                        + "\(stray), which is not the checkpoint's image token \(marker)"
                )
            }
            counts.append(range.count)
            claimed += range.count
        }
        let present = tokenIds.reduce(into: 0) { count, id in if id == marker { count += 1 } }
        guard present == claimed else {
            throw DeepSeekV41TurnImageError.carriedSpanMismatch(
                reason: "the prompt carries \(present) image token(s) but its recorded spans claim "
                    + "\(claimed)"
            )
        }
        return counts
    }
}
