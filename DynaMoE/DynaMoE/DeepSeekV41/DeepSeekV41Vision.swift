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
    /// `max_wh_ratio`: an image wider than `maxWidthHeightRatio` times its height is
    /// clamped before any other resizing. Nil means no clamp.
    public let maxWidthHeightRatio: Double?
    /// `rope_theta` for the tower's 2D rotary table, which is a different theta from
    /// the language model's.
    public let ropeTheta: Double

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
        imageTokenID: Int,
        maxWidthHeightRatio: Double? = nil,
        ropeTheta: Double = 10_000
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
        self.maxWidthHeightRatio = maxWidthHeightRatio
        self.ropeTheta = ropeTheta
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
        func number(_ keys: String...) -> Double? {
            for key in keys {
                if let value = object[key] as? Double { return value }
                if let value = object[key] as? Int { return Double(value) }
                if let value = object[key] as? NSNumber { return value.doubleValue }
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
            imageTokenID: int("image_token_id") ?? -1,
            maxWidthHeightRatio: number("max_wh_ratio"),
            ropeTheta: number("rope_theta", "vision_rope_theta") ?? 10_000
        )
    }
}

// MARK: - Layout

nonisolated public enum DeepSeekV41VisionLayout {

    /// One position of an image's token span.
    ///
    /// The checkpoint writes the SAME `image_token_id` at every one of these and
    /// keeps the distinction in a parallel type array (`image_token_types`), which
    /// is what decides which embedding overwrites the position afterwards.
    public enum SpanPosition: String, Sendable, Equatable {
        case start
        case image
        case newLine
        case end
    }

    /// `num_image_tokens(n_llm_h, n_llm_w)`: the aligner grid in reading order,
    /// one newline position per row, plus the two delimiters.
    public static func imageTokenCount(columns: Int, rows: Int) -> Int {
        guard columns > 0, rows > 0 else { return 0 }
        return rows * (columns + 1) + 2
    }

    /// `image_token_types` for a grid: `start`, then per row `columns` image
    /// positions and one newline, then `end`.
    public static func span(columns: Int, rows: Int) -> [SpanPosition] {
        let count = imageTokenCount(columns: columns, rows: rows)
        guard count > 0 else { return [] }
        var positions: [SpanPosition] = [.start]
        positions.reserveCapacity(count)
        for _ in 0..<rows {
            positions.append(contentsOf: Array(repeating: .image, count: columns))
            positions.append(.newLine)
        }
        positions.append(.end)
        return positions
    }

    /// Rescales an image so its token span lands inside the spec's budget,
    /// preserving aspect ratio: `plan_image_grid` followed by `safe_resize`.
    ///
    /// The result is the pixel size to resize to (a whole number of patches), the
    /// aligner grid that size produces, and what the prompt costs. The pixel floor
    /// is applied in PIXELS and the cap is solved in TOKEN space, exactly as the
    /// checkpoint does; doing both in one space overshoots, because each side rounds
    /// up to a patch before the grid is counted.
    ///
    /// `maxImageTokens` of 0 means no cap and `minPixels` of 0 means no floor.
    public static func fit(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> (width: Int, height: Int, columns: Int, rows: Int, tokens: Int) {
        let patch = max(1, spec.patchSize)
        let stride = spec.tokenStride
        guard width > 0, height > 0, stride > 0 else { return (0, 0, 0, 0, 0) }

        var scaledWidth = width
        var scaledHeight = height
        if let cap = spec.maxWidthHeightRatio, cap > 0,
           Double(scaledWidth) > Double(scaledHeight) * cap {
            scaledWidth = max(1, Int(Double(scaledHeight) * cap))
        }
        if spec.minPixels > 0, scaledWidth * scaledHeight < spec.minPixels {
            let factor = (Double(spec.minPixels) / Double(scaledWidth * scaledHeight)).squareRoot()
            scaledWidth = max(1, Int(Double(scaledWidth) * factor))
            scaledHeight = max(1, Int(Double(scaledHeight) * factor))
        }

        // The ViT reads complete patches, so both sides round up before the grid is
        // counted.
        var bestWidth = (scaledWidth + patch - 1) / patch * patch
        var bestHeight = (scaledHeight + patch - 1) / patch * patch
        var (columns, rows) = grid(width: bestWidth, height: bestHeight, spec: spec)

        if spec.maxImageTokens > 0, imageTokenCount(columns: columns, rows: rows) > spec.maxImageTokens {
            let solved = solvedSize(width: scaledWidth, height: scaledHeight, spec: spec)
            bestWidth = solved.width
            bestHeight = solved.height
            (columns, rows) = grid(width: bestWidth, height: bestHeight, spec: spec)
        }

        // The solved size fits by construction; a tiny budget can still land a row
        // over after the patch rounding, so trim to the cap and stop at the smallest
        // grid that exists rather than at zero.
        var trimmed = false
        while spec.maxImageTokens > 0,
              imageTokenCount(columns: columns, rows: rows) > spec.maxImageTokens,
              columns > 1 || rows > 1 {
            if columns >= rows {
                columns -= 1
            } else {
                rows -= 1
            }
            trimmed = true
        }
        if trimmed {
            // The pixel size has to follow the grid that was kept, or the caller
            // resizes to a box whose aligner grid is the one just trimmed away.
            bestWidth = columns * stride
            bestHeight = rows * stride
        }

        return (bestWidth, bestHeight, columns, rows, imageTokenCount(columns: columns, rows: rows))
    }

    /// `llm_grid`: the aligner's token grid for a pixel size.
    ///
    /// One token covers `patchSize * downsampleRatio` pixels per side, and the
    /// aligner pads the patch grid up to whole downsample blocks, hence the round up.
    public static func grid(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> (columns: Int, rows: Int) {
        let patch = max(1, spec.patchSize)
        let ratio = max(1, spec.downsampleRatio)
        guard width > 0, height > 0 else { return (0, 0) }
        let patchColumns = width / patch
        let patchRows = height / patch
        return ((patchColumns + ratio - 1) / ratio, (patchRows + ratio - 1) / ratio)
    }

    /// `solve_resize_ratio`: the largest aspect-preserving pixel size whose span
    /// still fits `maxImageTokens`.
    ///
    /// The width that fits is solved from the budget and the aspect ratio directly
    /// rather than by repeated shrinking, so a very wide or very tall image
    /// collapses to one row or one column instead of to a one-pixel sliver.
    private static func solvedSize(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> (width: Int, height: Int) {
        let patch = max(1, spec.patchSize)
        let cell = spec.tokenStride
        // The two delimiters plus one row's newline cost three positions before any
        // image position exists, so a smaller budget has no solution to solve for;
        // the trim loop in `fit` is what reduces it to the smallest grid.
        let budget = max(spec.maxImageTokens, 3)
        let aspect = Double(height) / Double(width)
        let columns = ((Double(budget) - 2) / aspect + 0.25).squareRoot() - 0.5
        let rows = columns * aspect
        if columns < 1 {
            // Very tall: one column, as many rows as the budget holds.
            return (cell, max(1, (budget - 2) / 2) * cell)
        }
        if rows < 1 {
            // Very wide: one row.
            return (max(1, budget - 3) * cell, cell)
        }
        let beta = min(
            columns.rounded(.down) * Double(cell) / Double(width),
            rows.rounded(.down) * Double(cell) / Double(height)
        )
        let solvedWidth = Int((Double(width) * beta / Double(patch)).rounded(.down)) * patch
        let solvedHeight = Int((Double(height) * beta / Double(patch)).rounded(.down)) * patch
        return (max(patch, solvedWidth), max(patch, solvedHeight))
    }

    /// The span rendered as text: `imageToken` once per position, because every
    /// position of an image span carries the same token id. `span(columns:rows:)`
    /// is what tells the positions apart.
    public static func placeholder(columns: Int, rows: Int, marker: DeepSeekV41VisionMarkers) -> String {
        let count = imageTokenCount(columns: columns, rows: rows)
        guard count > 0, !marker.imageToken.isEmpty else { return "" }
        return String(repeating: marker.imageToken, count: count)
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
            // A prompt with placeholders but no attached images leaves those
            // placeholders in place; reporting zero would hide the mismatch.
            let unfilled = markers.imagePlaceholder.isEmpty
                ? 0
                : text.components(separatedBy: markers.imagePlaceholder).count - 1
            return (
                text,
                ExpansionReport(unusedImageCount: 0, unfilledPlaceholderCount: max(0, unfilled))
            )
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
                out += placeholder(columns: fitted.columns, rows: fitted.rows, marker: markers)
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

// MARK: - Preprocessing

/// Image bytes to the rows the patch-embed projection reads, following
/// `image_processor.load_image`.
///
/// The geometry is the part that can be wrong silently: a grid one row off shifts
/// every aligner row, and a patch whose channels are interleaved differently hands
/// the projection the wrong columns. Both are pure functions of the image size and
/// the spec, so they are written out and tested here rather than left to the
/// device path to rediscover.
///
/// The resample is nearest-neighbour. The checkpoint uses PIL's bicubic, which
/// changes the pixels but not the geometry, and resampling quality belongs to the
/// device path; what this reference fixes is where each output pixel comes from and
/// how the patches are ordered.
nonisolated public enum DeepSeekV41VisionPreprocess {

    /// How an image of `sourceWidth` x `sourceHeight` lands inside the fitted pixel
    /// box the patch grid reads.
    public struct Placement: Sendable, Equatable {
        public let sourceWidth: Int
        public let sourceHeight: Int
        /// The fitted size: a whole number of patches on both sides.
        public let targetWidth: Int
        public let targetHeight: Int
        /// Pixels the image is drawn into. Equal to the target when it fills the box.
        public let drawnWidth: Int
        public let drawnHeight: Int
        public let left: Int
        public let top: Int
        /// `load_image` stretches rather than pads when the width/height ratio cap
        /// already equalised the sides.
        public let stretches: Bool

        /// The padding colour the checkpoint uses for the unpainted margin.
        public static let fill: UInt8 = 127

        /// Patch grid the tower reads, which is the target size in patches.
        public func patchGrid(patchSize: Int) -> (columns: Int, rows: Int) {
            let patch = max(1, patchSize)
            return (targetWidth / patch, targetHeight / patch)
        }
    }

    public static func placement(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> Placement {
        let fitted = DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec)
        guard width > 0, height > 0, fitted.width > 0, fitted.height > 0 else {
            return Placement(
                sourceWidth: max(0, width),
                sourceHeight: max(0, height),
                targetWidth: 0,
                targetHeight: 0,
                drawnWidth: 0,
                drawnHeight: 0,
                left: 0,
                top: 0,
                stretches: false
            )
        }
        let stretches = (spec.maxWidthHeightRatio ?? 0) > 0
            && Double(width) >= spec.maxWidthHeightRatio! * Double(height)
        if stretches {
            return Placement(
                sourceWidth: width,
                sourceHeight: height,
                targetWidth: fitted.width,
                targetHeight: fitted.height,
                drawnWidth: fitted.width,
                drawnHeight: fitted.height,
                left: 0,
                top: 0,
                stretches: true
            )
        }
        // Aspect preserved, centred, integer margins: `ImageOps.pad`.
        let scale = min(
            Double(fitted.width) / Double(width),
            Double(fitted.height) / Double(height)
        )
        let drawnWidth = max(1, min(fitted.width, Int((Double(width) * scale).rounded())))
        let drawnHeight = max(1, min(fitted.height, Int((Double(height) * scale).rounded())))
        return Placement(
            sourceWidth: width,
            sourceHeight: height,
            targetWidth: fitted.width,
            targetHeight: fitted.height,
            drawnWidth: drawnWidth,
            drawnHeight: drawnHeight,
            left: (fitted.width - drawnWidth) / 2,
            top: (fitted.height - drawnHeight) / 2,
            stretches: false
        )
    }

    /// One row per patch of `3 * patchSize * patchSize` normalized values, in the
    /// grid's reading order: `x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4)`,
    /// so each patch is channel-major as the projection expects.
    ///
    /// `rgb` is row-major 8-bit RGB, `width * height * 3` bytes.
    public static func patches(
        rgb: [UInt8],
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec
    ) -> [[Float]] {
        let patch = max(1, spec.patchSize)
        let layout = placement(width: width, height: height, spec: spec)
        let (columns, rows) = layout.patchGrid(patchSize: patch)
        guard columns > 0, rows > 0, rgb.count >= width * height * 3 else { return [] }

        var result: [[Float]] = []
        result.reserveCapacity(columns * rows)
        let patchVolume = patch * patch
        for row in 0..<rows {
            for column in 0..<columns {
                var values = [Float](repeating: 0, count: 3 * patchVolume)
                for y in 0..<patch {
                    for x in 0..<patch {
                        let targetX = column * patch + x
                        let targetY = row * patch + y
                        let offset = y * patch + x
                        var red = Placement.fill
                        var green = Placement.fill
                        var blue = Placement.fill
                        if targetX >= layout.left, targetX < layout.left + layout.drawnWidth,
                           targetY >= layout.top, targetY < layout.top + layout.drawnHeight {
                            // Nearest neighbour inside the drawn box.
                            let sourceX = min(
                                width - 1,
                                (targetX - layout.left) * width / max(1, layout.drawnWidth)
                            )
                            let sourceY = min(
                                height - 1,
                                (targetY - layout.top) * height / max(1, layout.drawnHeight)
                            )
                            let base = (sourceY * width + sourceX) * 3
                            red = rgb[base]
                            green = rgb[base + 1]
                            blue = rgb[base + 2]
                        }
                        let normalized = DeepSeekV41VisionLayout.normalize(r: red, g: green, b: blue)
                        values[offset] = normalized.0
                        values[patchVolume + offset] = normalized.1
                        values[2 * patchVolume + offset] = normalized.2
                    }
                }
                result.append(values)
            }
        }
        return result
    }
}

// MARK: - Markers

/// How an image occupies the prompt. An image becomes one run of the image token,
/// once per span position; there are no distinct wrapper tokens, because the
/// checkpoint writes `image_token_id` everywhere in the span and keeps the
/// start/image/newline/end distinction in a parallel type array. The spelling of
/// the token is what `DeepSeekV41Conformance.visionPatching` tracks until a probe
/// confirms it against the checkpoint's tokenizer.
nonisolated public struct DeepSeekV41VisionMarkers: Sendable, Equatable {
    /// What the chat UI/API writes where an image belongs.
    public let imagePlaceholder: String
    /// The token text repeated once per span position.
    public let imageToken: String

    public init(imagePlaceholder: String, imageToken: String) {
        self.imagePlaceholder = imagePlaceholder
        self.imageToken = imageToken
    }

    /// DeepSeek's published multimodal form. Built from unicode escapes rather than
    /// retyped (AGENTS.md: never retype tag-bearing strings).
    public static let familyDefault = DeepSeekV41VisionMarkers(
        imagePlaceholder: "<" + "\u{FF5C}" + "image" + "\u{2581}" + "placeholder" + "\u{FF5C}" + ">",
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

    /// Span positions one image of this size contributes, which is what the prompt
    /// expands to and what the KV budget has to afford.
    public func tokenCount(width: Int, height: Int) -> Int {
        DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec).tokens
    }

    public func describe() -> String {
        "vision: patch \(spec.patchSize) downsample \(spec.downsampleRatio) "
            + "layers \(spec.numHiddenLayers) hidden \(spec.hiddenSize) "
            + "max \(spec.maxImageTokens) tokens/image"
    }
}
