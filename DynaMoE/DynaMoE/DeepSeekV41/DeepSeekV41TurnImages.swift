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
    /// the turn loop wants: the expanded ids and a position -> residual row map.
    public static func bind(
        tokenIds: [UInt32],
        images: [DeepSeekV41ImageInput],
        spec: DeepSeekV41VisionSpec?,
        rowWidth: Int,
        encode: (DeepSeekV41ImageInput) throws -> DeepSeekV41ImagePrompt.ImageSpan
    ) throws -> (tokenIds: [UInt32], rows: [Int: [Float]]) {
        guard !images.isEmpty else { return (tokenIds, [:]) }
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
        return (plan.tokenIds, plan.rows)
    }
}
