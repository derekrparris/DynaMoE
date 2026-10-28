//
//  DeepSeekV41Vision.swift
//  DynaMoE
//
//  The vision tower: how an image becomes tokens, and what the prompt looks like
//  around them.
//
//  Two halves, and only the first is arithmetic:
//
//    * the PATCH GRID and the prompt layout. How many tokens an image of a given
//      size occupies, how it is rescaled to stay inside the checkpoint's token
//      budget, where the tokens sit inside the conversation, and what the model
//      expects around them. All pure, all tested.
//    * the TOWER itself: patch embedding, transformer blocks, pixel-unshuffle
//      downsampling, and the projector into the language model's hidden width.
//      `DeepSeekV41VisionTowerPlan` describes that dispatch; the kernels are the
//      engine's existing bf16 gemv/attention ones plus the DSV41 linears.
//
//  The token count rule is the one that bites. `max_image_tokens` is a hard cap on
//  the number of tokens ONE image may occupy, and `min_pixels` a floor, so an
//  image is rescaled until it lands in between rather than being tokenized at its
//  native size. Getting this wrong on a large photo overflows the context before
//  the first turn and, because the prompt is then truncated, looks like a model
//  that ignores images instead of an engine that dropped them.
//

import Foundation

// MARK: - Spec

/// The `vision_config` block, decoded.
nonisolated public struct DeepSeekV41VisionSpec: Sendable, Equatable {
    public let patchSize: Int
    /// Pixel-unshuffle factor applied before the projector: one output token covers
    /// `patchSize * downsampleRatio` source pixels per side.
    public let downsampleRatio: Int
    public let minPixels: Int
    public let maxImageTokens: Int
    public let numHiddenLayers: Int
    public let hiddenSize: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    /// Token id the language model expects at each image position.
    public let imageTokenID: Int

    /// Pixels one output token covers, per side.
    public var tokenStride: Int { max(1, patchSize) * max(1, downsampleRatio) }

    public init(
        patchSize: Int,
        downsampleRatio: Int,
        minPixels: Int,
        maxImageTokens: Int,
        numHiddenLayers: Int,
        hiddenSize: Int,
        numAttentionHeads: Int,
        intermediateSize: Int,
        imageTokenID: Int
    ) {
        self.patchSize = patchSize
        self.downsampleRatio = downsampleRatio
        self.minPixels = minPixels
        self.maxImageTokens = maxImageTokens
        self.numHiddenLayers = numHiddenLayers
        self.hiddenSize = hiddenSize
        self.numAttentionHeads = numAttentionHeads
        self.intermediateSize = intermediateSize
        self.imageTokenID = imageTokenID
    }

    /// Decodes a `vision_config` object. Returns nil when the block is absent, which
    /// is how a text-only checkpoint says it has no tower.
    public static func decode(_ object: [String: Any]?) -> DeepSeekV41VisionSpec? {
        guard let object else { return nil }
        func int(_ keys: String...) -> Int? {
            for key in keys {
                if let value = object[key] as? Int { return value }
                if let value = object[key] as? Double { return Int(value) }
                if let value = object[key] as? NSNumber { return value.intValue }
            }
            return nil
        }
        guard let layers = int("num_hidden_layers", "depth"), layers > 0 else { return nil }
        return DeepSeekV41VisionSpec(
            patchSize: int("patch_size") ?? 16,
            downsampleRatio: int("downsample_ratio") ?? 1,
            minPixels: int("min_pixels") ?? 0,
            maxImageTokens: int("max_image_tokens") ?? 1024,
            numHiddenLayers: layers,
            hiddenSize: int("hidden_size", "dim") ?? 0,
            numAttentionHeads: int("num_attention_heads", "n_heads") ?? 0,
            intermediateSize: int("intermediate_size", "mlp_ratio") ?? 0,
            imageTokenID: int("image_token_id") ?? -1
        )
    }
}

// MARK: - Layout

nonisolated public enum DeepSeekV41VisionLayout {

    /// Output tokens an image occupies at its native size, before any budget
    /// clamping.
    public static func tokenCount(width: Int, height: Int, stride: Int) -> Int {
        guard width > 0, height > 0, stride > 0 else { return 0 }
        let columns = (width + stride - 1) / stride
        let rows = (height + stride - 1) / stride
        return columns * rows
    }

    /// Rescales an image so its token count lands inside the spec's budget,
    /// preserving aspect ratio.
    ///
    /// The result is expressed in whole TOKENS, not pixels: the returned
    /// `width`/`height` are the grid size times `tokenStride`, so the caller
    /// resizes to exactly what the tokenizer will consume. Working in token space
    /// is what makes the budget exact — scaling pixels and rounding afterwards
    /// overshoots, because `ceil` on each side can push the product past the cap it
    /// was just fitted to.
    ///
    /// `maxImageTokens` of 0 means no cap and `minPixels` of 0 means no floor.
    public static func fit(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> (width: Int, height: Int, tokens: Int) {
        let stride = spec.tokenStride
        guard width > 0, height > 0, stride > 0 else { return (0, 0, 0) }
        let (startColumns, startRows) = grid(width: width, height: height, spec: spec)
        var columns = startColumns
        var rows = startRows

        if spec.maxImageTokens > 0, columns * rows > spec.maxImageTokens {
            let factor = (Double(spec.maxImageTokens) / Double(columns * rows)).squareRoot()
            columns = max(1, Int((Double(columns) * factor).rounded(.down)))
            rows = max(1, Int((Double(rows) * factor).rounded(.down)))
            // Shrink the longer side until the product is inside the cap; the floor
            // above can still leave the product one row over.
            while columns * rows > spec.maxImageTokens {
                if columns >= rows {
                    columns -= 1
                } else {
                    rows -= 1
                }
                if columns < 1 || rows < 1 { break }
            }
        }

        let minTokens = spec.minPixels > 0
            ? (spec.minPixels + stride * stride - 1) / (stride * stride)
            : 0
        if minTokens > 0, columns * rows < minTokens {
            let factor = (Double(minTokens) / Double(columns * rows)).squareRoot()
            columns = max(1, Int((Double(columns) * factor).rounded(.up)))
            rows = max(1, Int((Double(rows) * factor).rounded(.up)))
            // Never exceed a cap that is smaller than the floor: the cap wins,
            // because a prompt that overflows the context is unusable while a
            // slightly small image is merely low resolution.
            if spec.maxImageTokens > 0, columns * rows > spec.maxImageTokens {
                columns = max(1, min(columns, spec.maxImageTokens))
                rows = max(1, spec.maxImageTokens / columns)
            }
        }

        let tokens = columns * rows
        return (columns * stride, rows * stride, tokens)
    }

    /// Pixel grid a fitted image maps onto: `(columns, rows)` of output tokens.
    public static func grid(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> (columns: Int, rows: Int) {
        let stride = spec.tokenStride
        guard width > 0, height > 0, stride > 0 else { return (0, 0) }
        return ((width + stride - 1) / stride, (height + stride - 1) / stride)
    }

    /// The placeholder run one image occupies in the token stream: the image token
    /// repeated once per output patch.
    public static func placeholder(tokenCount: Int, marker: DeepSeekV41VisionMarkers) -> String {
        guard tokenCount > 0 else { return "" }
        return marker.imageOpen + String(repeating: marker.imageToken, count: tokenCount) + marker.imageClose
    }

    /// What an expansion could not place. Text with no marker is returned
    /// untouched, so an image that has nowhere to go is reported here rather than
    /// silently dropped or appended at the end of the prompt.
    public struct ExpansionReport: Sendable, Equatable {
        /// Images left over because the prompt had fewer placeholders than images.
        public let unusedImageCount: Int
        /// Placeholders left over because there were fewer images than markers. They
        /// are passed through verbatim, so nothing is deleted from the prompt.
        public let unfilledPlaceholderCount: Int

        public var isBalanced: Bool { unusedImageCount == 0 && unfilledPlaceholderCount == 0 }
    }

    /// Expands every image marker in `text` (one per attached image) into its
    /// placeholder run, discarding the report.
    public static func expand(
        text: String,
        images: [(width: Int, height: Int)],
        spec: DeepSeekV41VisionSpec,
        markers: DeepSeekV41VisionMarkers = .familyDefault
    ) -> String {
        expandReporting(text: text, images: images, spec: spec, markers: markers).text
    }

    /// Expands the markers and reports any image/marker mismatch.
    ///
    /// A marker with no image left is passed through verbatim rather than deleted.
    /// An image with no marker is left out of the prompt (appending it would put an
    /// image run where the caller never asked for one) but is counted in the report,
    /// which is what keeps the mismatch from being silent.
    public static func expandReporting(
        text: String,
        images: [(width: Int, height: Int)],
        spec: DeepSeekV41VisionSpec,
        markers: DeepSeekV41VisionMarkers = .familyDefault
    ) -> (text: String, report: ExpansionReport) {
        guard !images.isEmpty else {
            return (text, ExpansionReport(unusedImageCount: 0, unfilledPlaceholderCount: 0))
        }
        // An empty needle matches at every position and never advances the
        // remainder, so the loop below would never terminate.
        guard !markers.imagePlaceholder.isEmpty else {
            return (text, ExpansionReport(unusedImageCount: images.count, unfilledPlaceholderCount: 0))
        }
        var iterator = images.makeIterator()
        var out = ""
        var remainder = Substring(text)
        var unfilled = 0
        while let range = remainder.range(of: markers.imagePlaceholder) {
            out += remainder[remainder.startIndex..<range.lowerBound]
            if let image = iterator.next() {
                let fitted = fit(width: image.width, height: image.height, spec: spec)
                out += placeholder(tokenCount: fitted.tokens, marker: markers)
            } else {
                out += markers.imagePlaceholder
                unfilled += 1
            }
            remainder = remainder[range.upperBound...]
        }
        out += remainder
        var unused = 0
        while iterator.next() != nil { unused += 1 }
        return (out, ExpansionReport(unusedImageCount: unused, unfilledPlaceholderCount: unfilled))
    }

    /// Pixel normalization: RGB bytes to the `[-1, 1]` range the tower expects.
    ///
    /// Written out as a reference so the preprocessing has one definition; the
    /// device path applies the same affine transform through the existing
    /// normalize kernel.
    public static func normalize(r: UInt8, g: UInt8, b: UInt8, mean: Float = 0.5, std: Float = 0.5) -> (Float, Float, Float) {
        func apply(_ value: UInt8) -> Float {
            guard std > 0 else { return 0 }
            return (Float(value) / 255 - mean) / std
        }
        return (apply(r), apply(g), apply(b))
    }
}

// MARK: - Markers

/// How an image occupies the prompt. The reference substitutes a run of image
/// tokens between two wrapper markers; the wrapper spelling is what
/// `DeepSeekV41Conformance.visionPatching` tracks until the probe confirms it.
nonisolated public struct DeepSeekV41VisionMarkers: Sendable, Equatable {
    /// What the chat UI/API writes where an image belongs.
    public let imagePlaceholder: String
    /// What the placeholder expands to, around the repeated image token.
    public let imageOpen: String
    public let imageClose: String
    /// The token text repeated once per output patch.
    public let imageToken: String

    public init(imagePlaceholder: String, imageOpen: String, imageClose: String, imageToken: String) {
        self.imagePlaceholder = imagePlaceholder
        self.imageOpen = imageOpen
        self.imageClose = imageClose
        self.imageToken = imageToken
    }

    /// DeepSeek's published multimodal form: begin/end image markers around a run
    /// of the image token. Built from unicode escapes rather than retyped
    /// (AGENTS.md: never retype tag-bearing strings).
    public static let familyDefault = DeepSeekV41VisionMarkers(
        imagePlaceholder: "<" + "\u{FF5C}" + "image" + "\u{2581}" + "placeholder" + "\u{FF5C}" + ">",
        imageOpen: "<" + "\u{FF5C}" + "begin" + "\u{2581}" + "of" + "\u{2581}" + "image" + "\u{FF5C}" + ">",
        imageClose: "<" + "\u{FF5C}" + "end" + "\u{2581}" + "of" + "\u{2581}" + "image" + "\u{FF5C}" + ">",
        imageToken: "<" + "\u{FF5C}" + "image" + "\u{FF5C}" + ">"
    )
}

// MARK: - Tower plan

/// The tower's forward as a list of stages, with the shapes each one binds.
///
/// Produced once at load so the dispatch loop is a table walk rather than a chain
/// of conditionals, and so the tests can assert the geometry the tower will use
/// without a GPU.
nonisolated public struct DeepSeekV41VisionTowerPlan: Sendable, Equatable {
    public struct Stage: Sendable, Equatable {
        public enum Kind: String, Sendable, Equatable {
            case patchEmbed
            case blockAttention
            case blockMLP
            case pixelUnshuffle
            case projector
        }

        public let kind: Kind
        /// Layer index for `block*` stages, -1 otherwise.
        public let layer: Int
        public let inputWidth: Int
        public let outputWidth: Int

        public init(kind: Kind, layer: Int, inputWidth: Int, outputWidth: Int) {
            self.kind = kind
            self.layer = layer
            self.inputWidth = inputWidth
            self.outputWidth = outputWidth
        }
    }

    public let spec: DeepSeekV41VisionSpec
    public let stages: [Stage]
    /// Width of the vector the projector emits, i.e. the language model's hidden
    /// size. Zero when the spec does not record one.
    public let projectorOutputWidth: Int

    public init(spec: DeepSeekV41VisionSpec, projectorOutputWidth: Int) {
        self.spec = spec
        var stages: [Stage] = []
        stages.append(
            Stage(
                kind: .patchEmbed,
                layer: -1,
                inputWidth: spec.patchSize * spec.patchSize * 3,
                outputWidth: spec.hiddenSize
            )
        )
        for layer in 0..<max(0, spec.numHiddenLayers) {
            stages.append(
                Stage(kind: .blockAttention, layer: layer, inputWidth: spec.hiddenSize, outputWidth: spec.hiddenSize)
            )
            stages.append(
                Stage(kind: .blockMLP, layer: layer, inputWidth: spec.hiddenSize, outputWidth: spec.hiddenSize)
            )
        }
        if spec.downsampleRatio > 1 {
            stages.append(
                Stage(
                    kind: .pixelUnshuffle,
                    layer: -1,
                    inputWidth: spec.hiddenSize,
                    outputWidth: spec.hiddenSize * spec.downsampleRatio * spec.downsampleRatio
                )
            )
        }
        stages.append(
            Stage(
                kind: .projector,
                layer: -1,
                inputWidth: spec.hiddenSize * max(1, spec.downsampleRatio * spec.downsampleRatio),
                outputWidth: projectorOutputWidth
            )
        )
        self.stages = stages
        self.projectorOutputWidth = projectorOutputWidth
    }

    /// Tokens one image of this size contributes, which is what the prompt expands
    /// to and what the KV budget has to afford.
    public func tokenCount(width: Int, height: Int) -> Int {
        DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec).tokens
    }

    public func describe() -> String {
        "vision: patch \(spec.patchSize) downsample \(spec.downsampleRatio) "
            + "layers \(spec.numHiddenLayers) hidden \(spec.hiddenSize) "
            + "max \(spec.maxImageTokens) tokens/image"
    }
}
