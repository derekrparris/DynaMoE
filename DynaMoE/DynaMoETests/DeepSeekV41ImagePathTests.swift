//
//  DeepSeekV41ImagePathTests.swift
//  DynaMoETests
//
//  Phase: DeepSeek V4.1 vision, the composer's image path (slice #38)
//
//  Pure logic for the two halves that carry an attachment into a turn: pricing a
//  span from the resize plan before anything is allocated, and binding the tower's
//  rows to the placeholders. Both are tested with a fake encoder, so a mismatch is
//  caught here instead of becoming a prompt where every position after the picture
//  is shifted by one.
//

import AppKit
import XCTest
@testable import DynaMoE

final class DeepSeekV41TurnImagesTests: XCTestCase {

    /// The checkpoint's own geometry, at the published numbers.
    private let spec = DeepSeekV41VisionSpec(
        patchSize: 14,
        downsampleRatio: 3,
        minPixels: 295_936,
        maxImageTokens: 1_024,
        numHiddenLayers: 32,
        hiddenSize: 1_024,
        numAttentionHeads: 16,
        intermediateSize: 2_816,
        imageTokenID: 129_264,
        ropeTheta: 10_000
    )

    private func image(_ side: Int) -> DeepSeekV41ImageInput {
        let rgb = [UInt8](repeating: 7, count: side * side * 3)
        return DeepSeekV41ImageInput(rgb: rgb, width: side, height: side, sha256: "sha-\(side)")
    }

    /// A stand-in for the tower: one row per span position, each recognisable.
    private func span(cells: Int, columns: Int, gridRows: Int, tag: Float)
        -> DeepSeekV41ImagePrompt.ImageSpan {
        let count = DeepSeekV41VisionLayout.imageTokenCount(columns: columns, rows: gridRows)
        let rows = (0..<count).map { _ in [tag, Float(cells), Float(columns)] }
        return DeepSeekV41ImagePrompt.ImageSpan(
            rows: rows, columns: columns, gridRows: gridRows
        )
    }

    func testPriceCountsTheSpanNotThePlaceholder() throws {
        let marker = UInt32(spec.imageTokenID)
        let tokenIds: [UInt32] = [1, 2, marker, 3, 4]
        let price = try DeepSeekV41TurnImages.price(
            tokenIds: tokenIds, images: [image(56)], spec: spec
        )
        let estimate = try XCTUnwrap(price.estimates.first)
        XCTAssertEqual(estimate.spanPositions, DeepSeekV41VisionLayout.imageTokenCount(
            columns: estimate.columns, rows: estimate.rows
        ))
        // One placeholder token becomes the whole span.
        XCTAssertEqual(price.promptTokens, tokenIds.count - 1 + estimate.spanPositions)
        XCTAssertEqual(price.extraPositions, estimate.spanPositions - 1)
    }

    func testPriceWithoutImagesIsThePromptVerbatim() throws {
        let tokenIds: [UInt32] = [9, 8, 7]
        let price = try DeepSeekV41TurnImages.price(tokenIds: tokenIds, images: [], spec: spec)
        XCTAssertEqual(price.promptTokens, tokenIds.count)
        XCTAssertEqual(price.extraPositions, 0)
        XCTAssertTrue(price.estimates.isEmpty)
        XCTAssertEqual(price.seconds, 0)
    }

    func testPriceRefusesAnImageWithNoPlaceholder() {
        XCTAssertThrowsError(
            try DeepSeekV41TurnImages.price(tokenIds: [1, 2, 3], images: [image(56)], spec: spec)
        ) { error in
            XCTAssertEqual(
                error as? DeepSeekV41TurnImageError,
                .placeholderCountMismatch(images: 1, placeholders: 0)
            )
        }
    }

    func testPriceRefusesAPlaceholderWithNoImage() {
        let tokenIds: [UInt32] = [1, UInt32(spec.imageTokenID), 2]
        XCTAssertThrowsError(
            try DeepSeekV41TurnImages.price(tokenIds: tokenIds, images: [], spec: spec)
        ) { error in
            XCTAssertEqual(
                error as? DeepSeekV41TurnImageError,
                .placeholderCountMismatch(images: 0, placeholders: 1)
            )
        }
    }

    func testPriceRefusesAMalformedBuffer() {
        let broken = DeepSeekV41ImageInput(rgb: [1, 2, 3], width: 4, height: 4)
        XCTAssertThrowsError(
            try DeepSeekV41TurnImages.price(
                tokenIds: [UInt32(spec.imageTokenID)], images: [broken], spec: spec
            )
        ) { error in
            XCTAssertEqual(error as? DeepSeekV41TurnImageError, .malformedImage(index: 0))
        }
    }

    func testPriceRefusesImagesWhenTheModelHasNoTower() {
        XCTAssertThrowsError(
            try DeepSeekV41TurnImages.price(
                tokenIds: [UInt32(spec.imageTokenID)], images: [image(56)], spec: nil
            )
        ) { error in
            XCTAssertEqual(error as? DeepSeekV41TurnImageError, .noTower(images: 1))
        }
    }

    func testBindPutsEachSpansRowsAtItsPlaceholder() throws {
        let marker = UInt32(spec.imageTokenID)
        // Two pictures, one between them, so a bind that ignored order would put
        // picture 2 where picture 1 belongs.
        let tokenIds: [UInt32] = [11, marker, 12, marker, 13]
        let images = [image(56), image(112)]
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: tokenIds,
            images: images,
            spec: spec,
            rowWidth: 3,
            encode: { input in
                // The tower's own fit decides the grid; the fake only has to be
                // consistent with it, which is what makes the row count checkable.
                let fitted = DeepSeekV41VisionLayout.fit(
                    width: input.width, height: input.height, spec: spec
                )
                let tag = Float(input.width)
                let rows = (0..<fitted.tokens).map { _ in [tag, 0, 0] }
                return DeepSeekV41ImagePrompt.ImageSpan(
                    rows: rows, columns: fitted.columns, gridRows: fitted.rows
                )
            }
        )
        // The placeholder is replaced by its whole span, so the ids grow by the
        // span positions and every one of them is bound to a row.
        XCTAssertEqual(marker, UInt32(spec.imageTokenID))
        XCTAssertEqual(bound.tokenIds.count, 3 + bound.rows.count)
        let positions = bound.rows.keys.sorted()
        XCTAssertEqual(positions.first, 1)
        let first = try XCTUnwrap(bound.rows[1])
        let second = try XCTUnwrap(bound.rows[positions.last!])
        XCTAssertEqual(first[0], 56, "the first picture must be bound first")
        XCTAssertEqual(second[0], 112, "the second picture must not land on the first's rows")
        for position in positions {
            XCTAssertEqual(bound.rows[position]?.count, 3, "every row is the residual width")
        }
    }

    func testBindRefusesASpanThatDisagreesWithItsGrid() {
        // A span of four rows for a 2x2 cell grid needs six; `expand` refuses it and
        // the bind must pass that through rather than shortening the span.
        let tokenIds: [UInt32] = [1, UInt32(spec.imageTokenID)]
        XCTAssertThrowsError(
            try DeepSeekV41TurnImages.bind(
                tokenIds: tokenIds,
                images: [image(56)],
                spec: spec,
                rowWidth: 3,
                encode: { _ in
                    DeepSeekV41ImagePrompt.ImageSpan(
                        rows: Array(repeating: [1, 2, 3], count: 4),
                        columns: 2,
                        gridRows: 2
                    )
                }
            )
        )
    }

    func testBindWithoutImagesReturnsThePromptUnchanged() throws {
        let tokenIds: [UInt32] = [1, 2, 3]
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: tokenIds,
            images: [],
            spec: spec,
            rowWidth: 3,
            encode: { _ in
                XCTFail("an empty attachment list must not run the tower")
                return DeepSeekV41ImagePrompt.ImageSpan(rows: [], columns: 0, gridRows: 0)
            }
        )
        XCTAssertEqual(bound.tokenIds, tokenIds)
        XCTAssertTrue(bound.rows.isEmpty)
    }
}

final class DeepSeekV41VisionCostTests: XCTestCase {

    private let spec = DeepSeekV41VisionSpec(
        patchSize: 14,
        downsampleRatio: 3,
        minPixels: 295_936,
        maxImageTokens: 1_024,
        numHiddenLayers: 32,
        hiddenSize: 1_024,
        numAttentionHeads: 16,
        intermediateSize: 2_816,
        imageTokenID: 129_264,
        ropeTheta: 10_000
    )

    /// The three measured points on the SHIPPED device encode, from the published
    /// checkpoint on an optimized build (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #43").
    func testTheModelReproducesTheDevicePoints() {
        let model = DeepSeekV41VisionCostModel.measured
        XCTAssertEqual(model.seconds(patchTokens: 1_521), 0.32, accuracy: 0.05)
        XCTAssertEqual(model.seconds(patchTokens: 3_969), 1.70, accuracy: 0.15)
        XCTAssertEqual(model.seconds(patchTokens: 8_649), 6.95, accuracy: 0.4)
        // And it is monotonic, which the composer's label relies on being true.
        XCTAssertLessThan(
            model.seconds(patchTokens: 1_521), model.seconds(patchTokens: 3_969)
        )
        XCTAssertLessThan(
            model.seconds(patchTokens: 3_969), model.seconds(patchTokens: 8_649)
        )
    }

    /// The host tower's model, kept as the oracle's cost (`#37`).
    func testTheHostReferenceReproducesTheHostPoints() {
        let model = DeepSeekV41VisionCostModel.hostReference
        XCTAssertEqual(model.seconds(patchTokens: 0), 0.22, accuracy: 1e-9)
        XCTAssertEqual(model.seconds(patchTokens: 1_521), 2.74, accuracy: 0.25)
        XCTAssertEqual(model.seconds(patchTokens: 8_649), 41.71, accuracy: 1.0)
    }

    func testPricingUsesTheCheckpointsOwnResizePlan() throws {
        let estimate = try XCTUnwrap(
            DeepSeekV41VisionPricing.estimate(width: 546, height: 546, spec: spec)
        )
        let fitted = DeepSeekV41VisionLayout.fit(width: 546, height: 546, spec: spec)
        XCTAssertEqual(estimate.columns, fitted.columns)
        XCTAssertEqual(estimate.rows, fitted.rows)
        XCTAssertEqual(estimate.spanPositions, fitted.tokens)
        XCTAssertEqual(
            estimate.patchTokens, (fitted.width / spec.patchSize) * (fitted.height / spec.patchSize)
        )
        XCTAssertEqual(estimate.spanPositions, 184, "the smallest image the checkpoint accepts")
        XCTAssertEqual(estimate.patchTokens, 1_521)
    }

    /// The cap is the checkpoint's, not a constant of ours: a 20,000 pixel square
    /// collapses to the largest grid `max_image_tokens` affords.
    func testTheLargestAdmissibleSquareIsThirtyOneCells() throws {
        let capped = DeepSeekV41VisionLayout.fit(width: 20_000, height: 20_000, spec: spec)
        let estimate = try XCTUnwrap(
            DeepSeekV41VisionPricing.estimate(width: 20_000, height: 20_000, spec: spec)
        )
        XCTAssertEqual(capped.columns, 31)
        XCTAssertEqual(capped.rows, 31)
        XCTAssertEqual(estimate.patchTokens, 8_649)
        XCTAssertEqual(estimate.spanPositions, DeepSeekV41VisionLayout.imageTokenCount(
            columns: 31, rows: 31
        ))
        XCTAssertLessThanOrEqual(capped.tokens, spec.maxImageTokens)
    }

    func testAnImageCostsMoreThanASmallerOne() throws {
        let small = try XCTUnwrap(
            DeepSeekV41VisionPricing.estimate(width: 546, height: 546, spec: spec)
        )
        let large = try XCTUnwrap(
            DeepSeekV41VisionPricing.estimate(width: 2_000, height: 2_000, spec: spec)
        )
        XCTAssertLessThan(small.seconds, large.seconds)
        XCTAssertLessThan(small.patchTokens, large.patchTokens)
    }

    func testADegenerateSizePricesNothing() {
        XCTAssertNil(DeepSeekV41VisionPricing.estimate(width: 0, height: 10, spec: spec))
        XCTAssertNil(DeepSeekV41VisionPricing.estimate(width: 10, height: 0, spec: spec))
    }

    /// The bounds a client is told, against the plan that produces them: the floor
    /// is a one-pixel image raised to `min_pixels`, and the ceiling is the cap
    /// itself, because `fit` trims until the SPAN fits and the span is what a
    /// prompt pays for.
    func testTheAdvertisedBoundsComeFromThePlansOwnFloorAndCap() throws {
        let bounds = try XCTUnwrap(DeepSeekV41VisionPricing.bounds(spec: spec))
        XCTAssertEqual(bounds.minPatchTokens, 1_521)
        XCTAssertEqual(bounds.minSpanPositions, 184)
        XCTAssertEqual(bounds.minSeconds, 0.32, accuracy: 0.01)
        XCTAssertEqual(bounds.maxSpanPositions, spec.maxImageTokens)
        // Patches per aligner cell is `downsample_ratio^2`, so the widest span the
        // cap allows cannot carry more than this many of them.
        XCTAssertEqual(bounds.maxPatchTokens, 9 * spec.maxImageTokens)
        XCTAssertEqual(
            bounds.maxSeconds,
            DeepSeekV41VisionCostModel.measured.seconds(patchTokens: bounds.maxPatchTokens),
            accuracy: 1e-9
        )
    }

    /// The bounds are a claim about EVERY image, so check them against images rather
    /// than restating the arithmetic above: a square, both thin extremes, a portrait,
    /// a landscape and an image wide enough to reach the cap.
    ///
    /// An aspect ratio, not a pixel count, is what the plan prices, which is why a
    /// single `estimate` cannot stand in for this.
    func testNoAdmissibleImageEscapesTheAdvertisedBounds() throws {
        let bounds = try XCTUnwrap(DeepSeekV41VisionPricing.bounds(spec: spec))
        let shapes = [
            (1, 1), (2, 2), (546, 546), (1_000, 1_000), (2_000, 2_000),
            (20_000, 20_000), (5_000, 100), (100, 5_000), (42_882, 14), (42_882, 42),
        ]
        var reachedTheCap = false
        for (width, height) in shapes {
            let label = "\(width)x\(height)"
            let estimate = try XCTUnwrap(
                DeepSeekV41VisionPricing.estimate(width: width, height: height, spec: spec),
                label
            )
            XCTAssertGreaterThanOrEqual(estimate.patchTokens, bounds.minPatchTokens, label)
            XCTAssertLessThanOrEqual(estimate.patchTokens, bounds.maxPatchTokens, label)
            XCTAssertGreaterThanOrEqual(estimate.spanPositions, bounds.minSpanPositions, label)
            XCTAssertLessThanOrEqual(estimate.spanPositions, bounds.maxSpanPositions, label)
            XCTAssertGreaterThanOrEqual(estimate.seconds, bounds.minSeconds - 1e-9, label)
            XCTAssertLessThanOrEqual(estimate.seconds, bounds.maxSeconds + 1e-9, label)
            if estimate.spanPositions == bounds.maxSpanPositions { reachedTheCap = true }
        }
        // A ceiling nothing can reach is not a bound a client can size against.
        XCTAssertTrue(reachedTheCap, "a one-row span must reach the cap exactly")
    }

    /// An uncapped plan has no upper bound, and reporting a large one would be a lie
    /// in the same shape as a right answer. The published checkpoint caps at 1,024,
    /// so this is the path a future checkpoint would take.
    func testAnUncappedPlanAdvertisesNothing() {
        let uncapped = DeepSeekV41VisionSpec(
            patchSize: 14,
            downsampleRatio: 3,
            minPixels: 295_936,
            maxImageTokens: 0,
            numHiddenLayers: 32,
            hiddenSize: 1_024,
            numAttentionHeads: 16,
            intermediateSize: 2_816,
            imageTokenID: 129_264
        )
        XCTAssertNil(DeepSeekV41VisionPricing.bounds(spec: uncapped))
    }
}

final class DeepSeekV41ImagePixelsTests: XCTestCase {

    /// A CGImage whose top half is red and bottom half blue, as raw bytes.
    private func halfAndHalf(width: Int, height: Int) -> CGImage? {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                // CG bitmap rows run top-down in memory for our purposes here; the
                // test asserts the decoder agrees with an independent reader
                // (`NSBitmapImageRep`), not which end is which.
                let red = y < height / 2
                pixels[offset + 0] = red ? 255 : 0
                pixels[offset + 1] = 0
                pixels[offset + 2] = red ? 0 : 255
                pixels[offset + 3] = 255
            }
        }
        return pixels.withUnsafeBytes { raw -> CGImage? in
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
            return CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        }
    }

    /// One white pixel on black, anywhere, so any flip or transpose of the decode
    /// moves it to a different index.
    private func cornerFixture(width: Int, height: Int, whiteAt: (x: Int, y: Int)) -> CGImage? {
        let bytesPerRow = width * 4
        var pixels = [UInt8](repeating: 0, count: bytesPerRow * height)
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let white = (x == whiteAt.x && y == whiteAt.y)
                let value: UInt8 = white ? 255 : 0
                pixels[offset + 0] = value
                pixels[offset + 1] = value
                pixels[offset + 2] = value
                pixels[offset + 3] = 255
            }
        }
        return pixels.withUnsafeBytes { raw -> CGImage? in
            guard let space = CGColorSpace(name: CGColorSpace.sRGB),
                  let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
            return CGImage(
                width: width,
                height: height,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bytesPerRow,
                space: space,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: false,
                intent: .defaultIntent
            )
        }
    }

    /// The buffer the tower samples must start at the image's TOP-left pixel and
    /// run top-down, or every row of patches reads a different part of the picture
    /// upside down and the model answers about the wrong thing.
    ///
    /// Pinned with an off-diagonal single pixel rather than a symmetric fixture: a
    /// vertical flip moves (2,0) to (2,2), a horizontal flip to (0,0), and a
    /// transpose to (0,2), so each lands on its own index.
    func testTheDecodedBufferStartsAtTheImagesTopLeft() throws {
        let source = try XCTUnwrap(cornerFixture(width: 3, height: 3, whiteAt: (2, 0)))
        let decoded = try DeepSeekV41ImagePixels.decode(source)
        XCTAssertTrue(decoded.isWellFormed)
        XCTAssertEqual(decoded.width, 3)
        XCTAssertEqual(decoded.height, 3)
        let brightPixels = (0..<(3 * 3)).filter { decoded.rgb[$0 * 3] > 200 }
        XCTAssertEqual(brightPixels, [2], "the lit pixel must stay at (2,0), i.e. index 2")
        // Everything else is black, so the check cannot pass on a buffer that was
        // never written.
        for index in 0..<(3 * 3) where index != 2 {
            XCTAssertLessThan(decoded.rgb[index * 3], 40, "pixel \(index) must stay dark")
        }
    }

    /// Channel order: a BGR decode would swap the halves' red and blue, which the
    /// orientation check above cannot see.
    ///
    /// This is asserted against the fixture rather than against `NSColor` on
    /// purpose: `NSBitmapImageRep.colorAt` round-trips through colour management,
    /// which reports the fixture's (255, 0, 0) as roughly (255, 38, 0), so an
    /// equality check there measures the display profile instead of the decoder.
    func testChannelsAreRGBNotBGR() throws {
        let source = try XCTUnwrap(halfAndHalf(width: 4, height: 4))
        let decoded = try DeepSeekV41ImagePixels.decode(source)
        let topRow = Array(decoded.rgb[0..<3])
        let bottomRow = Array(decoded.rgb[(3 * 4) * 3..<((3 * 4) + 1) * 3])
        XCTAssertGreaterThan(topRow[0], 200, "the first decoded row is the image's top, which is red")
        XCTAssertLessThan(topRow[2], 40)
        XCTAssertGreaterThan(bottomRow[2], 200, "the last decoded row is the image's bottom, which is blue")
        XCTAssertLessThan(bottomRow[0], 40)
        // An interior pixel of the top half, so a decoder that only wrote the
        // first and last rows would still be caught.
        let interior = 5 * 3
        XCTAssertGreaterThan(decoded.rgb[interior], 200, "pixel 5 is in row 1, which is red")
        XCTAssertLessThan(decoded.rgb[interior + 2], 40)
    }

    func testASolidColourSurvivesAPNGRoundTrip() throws {
        let source = try XCTUnwrap(halfAndHalf(width: 8, height: 8))
        let data = NSBitmapImageRep(cgImage: source).representation(using: .png, properties: [:])
        let decoded = try DeepSeekV41ImagePixels.decode(try XCTUnwrap(data))
        XCTAssertEqual(decoded.width, 8)
        XCTAssertEqual(decoded.height, 8)
        XCTAssertTrue(decoded.isWellFormed)
    }

    func testGarbageIsRefusedRatherThanGuessed() {
        XCTAssertThrowsError(try DeepSeekV41ImagePixels.decode(Data([0, 1, 2, 3, 4]))) { error in
            XCTAssertEqual(error as? DeepSeekV41ImagePixelsError, .undecodable)
        }
    }

    func testAZeroSizedImageIsRefused() {
        XCTAssertFalse(DeepSeekV41ImagePixels(rgb: [], width: 0, height: 0).isWellFormed)
        XCTAssertFalse(DeepSeekV41ImagePixels(rgb: [1, 2, 3], width: 2, height: 2).isWellFormed)
    }
}

final class DeepSeekV41VisionDirectoryTests: XCTestCase {

    /// The composer's attachment button is driven by this check, so it must key off
    /// the snapshot rather than the config: a checkpoint that advertises a tower
    /// but never downloaded it must keep the composer closed.
    func testASnapshotIsOnlyVisionCapableWhenItsShardsHaveTheTower() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("dsv41-vision-dir-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertFalse(
            DeepSeekV41VisionWeights.directoryHasTower(root),
            "an empty directory has no tower"
        )

        let weightMap: [String: Any] = [
            "weight_map": [
                "vision.blocks.0.attn.wqkv.weight": "model-00001-of-00002.safetensors",
                "vision.blocks.0.mlp.w2.weight": "model-00001-of-00002.safetensors",
                "aligner.w2.weight": "model-00001-of-00002.safetensors",
                "image_start": "model-00002-of-00002.safetensors"
            ]
        ]
        let index = root.appendingPathComponent("model.safetensors.index.json")
        try JSONSerialization.data(withJSONObject: weightMap).write(to: index)

        XCTAssertFalse(
            DeepSeekV41VisionWeights.directoryHasTower(root),
            "the marker embeddings live in their own shard, and half a tower is not a tower"
        )

        var complete = weightMap
        var map = try XCTUnwrap(complete["weight_map"] as? [String: String])
        map["image_end"] = "model-00002-of-00002.safetensors"
        map["image_newline"] = "model-00002-of-00002.safetensors"
        complete["weight_map"] = map
        try JSONSerialization.data(withJSONObject: complete).write(to: index)

        XCTAssertTrue(DeepSeekV41VisionWeights.directoryHasTower(root))
    }
}
