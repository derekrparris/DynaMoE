//
//  DeepSeekV41DSparkVisionTests.swift
//  DynaMoE
//
//  DSpark drafting and the vision tower's token accounting.
//
//  Both are pure arithmetic over data the checkpoint publishes, so both are
//  testable without weights. The DSpark cases focus on the two things that decide
//  whether speculative decoding is a speedup (which token the chain conditions on,
//  and what a verification round emits); the vision cases focus on the token
//  budget, which is the rule that silently overflows a context when it is wrong.
//

import XCTest
@testable import DynaMoE

// MARK: - DSpark

final class DeepSeekV41DSparkTests: XCTestCase {

    private func plan(blockSize: Int = 3, targets: [Int] = [7, 15]) -> DeepSeekV41DSparkPlan {
        DeepSeekV41DSparkPlan(
            blockSize: blockSize,
            noiseTokenID: 100,
            targetLayerIDs: targets,
            draftBlockCount: 1
        )
    }

    func testPlanComesFromTheConfig() throws {
        let config = try DeepSeekV41Config.decode(jsonData: Data("""
        {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
         "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
           "vocab_size": 16, "n_routed_experts": 2, "n_activated_experts": 1,
           "num_nextn_predict_layers": 1, "dspark_block_size": 4,
           "dspark_noise_token_id": 7, "dspark_target_layer_ids": [3, 1]}}
        """.utf8))
        let plan = DeepSeekV41DSparkPlan.make(config: config)
        XCTAssertTrue(plan.isEnabled)
        XCTAssertEqual(plan.blockSize, 4)
        XCTAssertEqual(plan.noiseTokenID, 7)
        // Target layers are sorted, so conditioning order is deterministic.
        XCTAssertEqual(plan.targetLayerIDs, [1, 3])
    }

    func testPlanWithoutABlockSizeIsDisabled() {
        XCTAssertFalse(plan(blockSize: 0).isEnabled)
    }

    func testVerificationWindowCoversTheDraftsPlusTheEmittedToken() {
        XCTAssertEqual(plan(blockSize: 3).verificationWindow(verifiedCount: 5), 5..<9)
        XCTAssertEqual(plan(blockSize: 0).verificationWindow(verifiedCount: 5), 0..<0)
    }

    func testTheMarkovHeadIsAnAdditiveBiasNotAMixture() {
        // `DSparkBlock.forward_head` does `logits[:, i].add_(logits_bias)`. A
        // convex blend is the tempting reading and it is wrong: it scales the draft
        // block's own logits by (1 - w), which reorders tokens the bias never meant
        // to touch.
        let draft: [Float] = [2, 0, -1]
        let bias: [Float] = [0, 4, 3]
        XCTAssertEqual(DeepSeekV41DSpark.applyMarkovBias(draft, bias: bias), [2, 4, 2])
    }

    func testMismatchedWidthsLeaveTheDraftHeadAlone() {
        let draft: [Float] = [1, 2, 3]
        XCTAssertEqual(DeepSeekV41DSpark.applyMarkovBias(draft, bias: [0, 0]), draft)
        XCTAssertTrue(DeepSeekV41DSpark.applyMarkovBias([], bias: []).isEmpty)
    }

    func testPickBreaksTiesTowardTheLowestTokenID() {
        XCTAssertEqual(DeepSeekV41DSpark.pick([0.1, 0.9, 0.9]), 1)
        XCTAssertEqual(DeepSeekV41DSpark.pick([2, 2, 2]), 0)
        XCTAssertNil(DeepSeekV41DSpark.pick([]))
    }

    func testTemperatureZeroDrawsGreedilyWithoutAskingForUniforms() {
        var draws = 0
        let token = DeepSeekV41DSpark.sample(logits: [0.5, 2, 1], temperature: 0) {
            draws += 1
            return 0.5
        }
        XCTAssertEqual(token, 1)
        XCTAssertEqual(draws, 0, "a greedy draw consumes no randomness")
    }

    /// Gumbel-max: each logit is offset by `-log(-log u)`, so a slightly weaker
    /// logit with a much better uniform can still win. The reference's
    /// `probs.div_(exponential_(1)).argmax()` is the same rule, which is what makes
    /// a recorded uniform table replayable off a torch box.
    func testSamplingFollowsTheGumbelDraw() {
        // A poor uniform on the strongest logit (-log(-log 0.1) = -0.83) against a
        // good one on the middle logit (+2.25) is what flips the draw.
        let table: [Float] = [0.1, 0.9, 0.1]
        func draw() -> Int? {
            var cursor = 0
            return DeepSeekV41DSpark.sample(logits: [1.0, 0.9, 0.0], temperature: 1) {
                defer { cursor += 1 }
                return table[cursor]
            }
        }
        XCTAssertEqual(draw(), 1)
        // Deterministic in the uniforms: the same table always gives the same draw.
        XCTAssertEqual(draw(), 1)
        // Greedy would have taken the first logit instead.
        XCTAssertEqual(DeepSeekV41DSpark.pick([1.0, 0.9, 0.0]), 0)
    }

    func testChainConditionsOnItsOwnTokensNotTheTargets() {
        // A step closure that echoes the token it was handed back as the argmax, so
        // the chain's second step can only see the first draft token if the chain
        // really feeds its own output back.
        var seen: [Int] = []
        let draft = DeepSeekV41DSpark.runChain(steps: 3, seedToken: 42) { _, previous in
            seen.append(previous)
            var logits = [Float](repeating: 0, count: 8)
            logits[(previous + 1) % 8] = 5
            return (draftLogits: logits, markovBias: [Float](repeating: 0, count: 8), confidence: 0.9)
        }
        XCTAssertEqual(seen.first, 42)
        XCTAssertEqual(draft.tokens, [3, 4, 5])
        // Each step after the first conditions on the token the chain chose.
        XCTAssertEqual(seen, [42, 3, 4])
    }

    func testChainStopsWhenAStepProducesNothing() {
        var calls = 0
        let draft = DeepSeekV41DSpark.runChain(steps: 5, seedToken: 1) { index, _ in
            calls += 1
            guard index < 2 else { return nil }
            return (draftLogits: [1], markovBias: [1], confidence: 0.9)
        }
        XCTAssertEqual(calls, 3)
        XCTAssertEqual(draft.tokens.count, 2)
    }

    func testSubmitGateLooksAtTheLeadingStep() {
        XCTAssertFalse(DeepSeekV41DSpark.shouldSubmit(.empty))
        XCTAssertTrue(
            DeepSeekV41DSpark.shouldSubmit(.init(tokens: [1, 2], confidences: [0.9, 0.01]))
        )
        XCTAssertFalse(
            DeepSeekV41DSpark.shouldSubmit(.init(tokens: [1, 2], confidences: [0.1, 0.99]))
        )
        XCTAssertFalse(
            DeepSeekV41DSpark.shouldSubmit(.init(tokens: [1], confidences: [0.1]), threshold: 0.5)
        )
    }

    func testAcceptanceStopsAtTheFirstDisagreement() {
        XCTAssertEqual(
            DeepSeekV41DSpark.acceptedCount(draft: [5, 6, 7], targetChoices: [5, 6, 7, 8]),
            3
        )
        XCTAssertEqual(
            DeepSeekV41DSpark.acceptedCount(draft: [5, 6, 7], targetChoices: [5, 9, 7, 8]),
            1
        )
        XCTAssertEqual(DeepSeekV41DSpark.acceptedCount(draft: [], targetChoices: [1]), 0)
        // A target that ran out of positions cannot accept more.
        XCTAssertEqual(DeepSeekV41DSpark.acceptedCount(draft: [1, 2], targetChoices: [1]), 1)
    }

    func testEmissionAlwaysIncludesTheTargetsOwnToken() {
        // Fully accepted: the target's next token is still emitted.
        XCTAssertEqual(DeepSeekV41DSpark.emission(draft: [1, 2], targetChoices: [1, 2, 3]), [1, 2, 3])
        // Rejected at the first token: only the target's token.
        XCTAssertEqual(DeepSeekV41DSpark.emission(draft: [9, 9], targetChoices: [4, 5, 6]), [4])
        // Rejected after one acceptance: the accepted token plus the target's.
        XCTAssertEqual(DeepSeekV41DSpark.emission(draft: [1, 9], targetChoices: [1, 8, 6]), [1, 8])
    }
}

// MARK: - Vision

final class DeepSeekV41VisionLayoutTests: XCTestCase {

    private func spec(
        patch: Int = 16,
        downsample: Int = 3,
        minPixels: Int = 65_536,
        maxTokens: Int = 1024
    ) -> DeepSeekV41VisionSpec {
        DeepSeekV41VisionSpec(
            patchSize: patch,
            downsampleRatio: downsample,
            minPixels: minPixels,
            maxImageTokens: maxTokens,
            numHiddenLayers: 27,
            hiddenSize: 1024,
            numAttentionHeads: 16,
            intermediateSize: 4096,
            imageTokenID: 129_264
        )
    }

    func testTokenStrideMultipliesPatchAndDownsample() {
        XCTAssertEqual(spec().tokenStride, 48)
    }

    /// `num_image_tokens(n_llm_h, n_llm_w) = n_llm_h * (n_llm_w + 1) + 2`: the
    /// aligner grid, a newline position per row, and the two delimiters.
    func testTokenCountIsTheAlignerGridPlusNewlinesAndDelimiters() {
        XCTAssertEqual(DeepSeekV41VisionLayout.imageTokenCount(columns: 1, rows: 1), 4)
        XCTAssertEqual(DeepSeekV41VisionLayout.imageTokenCount(columns: 10, rows: 10), 112)
        XCTAssertEqual(DeepSeekV41VisionLayout.imageTokenCount(columns: 0, rows: 10), 0)
    }

    func testSpanIsTheGridInReadingOrderWrappedByDelimiters() {
        XCTAssertEqual(
            DeepSeekV41VisionLayout.span(columns: 2, rows: 2),
            [.start, .image, .image, .newLine, .image, .image, .newLine, .end]
        )
        XCTAssertEqual(DeepSeekV41VisionLayout.span(columns: 0, rows: 2), [])
    }

    /// `llm_grid` rounds the patch grid up to whole downsample blocks, because the
    /// aligner pads before it unfolds. The size is expected to be patch-aligned,
    /// which is what `fit` returns.
    func testGridCountsWholeDownsampleBlocks() {
        let spec = spec()
        // 3 patches -> one block; 6 patches -> two.
        XCTAssertEqual(DeepSeekV41VisionLayout.grid(width: 48, height: 48, spec: spec).columns, 1)
        XCTAssertEqual(DeepSeekV41VisionLayout.grid(width: 96, height: 48, spec: spec).columns, 2)
        XCTAssertEqual(DeepSeekV41VisionLayout.grid(width: 48, height: 144, spec: spec).rows, 3)
        XCTAssertEqual(DeepSeekV41VisionLayout.grid(width: 0, height: 10, spec: spec).columns, 0)
    }

    func testOversizedImageIsShrunkInsideTheTokenCap() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 4000, height: 3000, spec: spec())
        XCTAssertEqual(fitted.columns, 36)
        XCTAssertEqual(fitted.rows, 27)
        XCTAssertEqual(fitted.tokens, 1001)
        XCTAssertLessThanOrEqual(fitted.tokens, 1024)
        // Aspect ratio is preserved to within a block.
        let aspect = Double(fitted.width) / Double(fitted.height)
        XCTAssertEqual(aspect, 4.0 / 3.0, accuracy: 0.05)
        // The returned pixel size reads back as the grid that was priced.
        let grid = DeepSeekV41VisionLayout.grid(width: fitted.width, height: fitted.height, spec: spec())
        XCTAssertEqual(grid.columns, fitted.columns)
        XCTAssertEqual(grid.rows, fitted.rows)
    }

    func testImageInsideTheBudgetIsLeftAlone() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 480, height: 480, spec: spec())
        XCTAssertEqual(fitted.width, 480)
        XCTAssertEqual(fitted.height, 480)
        // 480 / 48 = 10 blocks on each side: 10 rows of 10 images plus a newline,
        // wrapped by the two delimiters.
        XCTAssertEqual(fitted.columns, 10)
        XCTAssertEqual(fitted.rows, 10)
        XCTAssertEqual(fitted.tokens, 112)
    }

    /// The floor is applied in PIXELS, not in tokens: below it the image is upscaled
    /// until it covers the floor, rounded up to whole patches.
    func testTinyImageIsUpscaledToThePixelFloor() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 48, height: 48, spec: spec())
        XCTAssertGreaterThanOrEqual(fitted.width * fitted.height, 65_536)
        XCTAssertEqual(fitted.width, 256)
        XCTAssertEqual(fitted.height, 256)
        XCTAssertEqual(fitted.tokens, 44)
    }

    /// A budget smaller than the delimiters alone costs still yields the smallest
    /// grid that exists rather than an empty one.
    func testCapSmallerThanTheSmallestSpanYieldsTheSmallestSpan() {
        let fitted = DeepSeekV41VisionLayout.fit(
            width: 4000,
            height: 3000,
            spec: spec(minPixels: 65_536, maxTokens: 1)
        )
        XCTAssertEqual(fitted.columns, 1)
        XCTAssertEqual(fitted.rows, 1)
        XCTAssertEqual(fitted.tokens, 4)
    }

    func testNoCapAndNoFloorMeansNativeSize() {
        let fitted = DeepSeekV41VisionLayout.fit(
            width: 960,
            height: 480,
            spec: spec(minPixels: 0, maxTokens: 0)
        )
        XCTAssertEqual(fitted.width, 960)
        XCTAssertEqual(fitted.height, 480)
        // 960 / 48 = 20 columns, 480 / 48 = 10 rows.
        XCTAssertEqual(fitted.tokens, 10 * 21 + 2)
    }

    /// The span repeats one token, not a wrapped run: every position of an image
    /// span carries the same token id and `span(columns:rows:)` is what tells the
    /// positions apart.
    func testPlaceholderRepeatsTheImageTokenOncePerSpanPosition() {
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let placeholder = DeepSeekV41VisionLayout.placeholder(columns: 2, rows: 2, marker: markers)
        XCTAssertEqual(placeholder, String(repeating: markers.imageToken, count: 8))
        XCTAssertEqual(DeepSeekV41VisionLayout.placeholder(columns: 0, rows: 0, marker: markers), "")
    }

    func testExpandReplacesOnePlaceholderPerAttachedImage() {
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let text = "look " + markers.imagePlaceholder + " then " + markers.imagePlaceholder
        let expanded = DeepSeekV41VisionLayout.expand(
            text: text,
            images: [(width: 480, height: 480), (width: 48, height: 48)],
            spec: spec(),
            markers: markers
        )
        // The marker and the image token are the same string, because the checkpoint
        // writes one token id at every span position. So the expansion is asserted by
        // its shape: the two markers became the two spans, in reading order, and
        // nothing between them moved.
        let first = DeepSeekV41VisionLayout.fit(width: 480, height: 480, spec: spec()).tokens
        let second = DeepSeekV41VisionLayout.fit(width: 48, height: 48, spec: spec()).tokens
        XCTAssertEqual(
            expanded,
            "look " + String(repeating: markers.imageToken, count: first)
                + " then " + String(repeating: markers.imageToken, count: second)
        )
    }

    func testExpandLeavesTextOnlyPromptsUntouched() {
        let text = "no images here"
        XCTAssertEqual(DeepSeekV41VisionLayout.expand(text: text, images: [], spec: spec()), text)
        XCTAssertEqual(
            DeepSeekV41VisionLayout.expand(
                text: text,
                images: [(width: 48, height: 48)],
                spec: spec()
            ),
            text
        )
    }

    func testNormalizeMapsBytesIntoTheTowerRange() {
        let black = DeepSeekV41VisionLayout.normalize(r: 0, g: 0, b: 0)
        XCTAssertEqual(black.0, -1, accuracy: 1e-5)
        let white = DeepSeekV41VisionLayout.normalize(r: 255, g: 255, b: 255)
        XCTAssertEqual(white.0, 1, accuracy: 1e-5)
        let mid = DeepSeekV41VisionLayout.normalize(r: 128, g: 128, b: 128)
        XCTAssertEqual(mid.0, 128.0 / 255 * 2 - 1, accuracy: 1e-5)
    }
}

final class DeepSeekV41VisionPreprocessTests: XCTestCase {

    private func spec(patch: Int = 16, downsample: Int = 1) -> DeepSeekV41VisionSpec {
        DeepSeekV41VisionSpec(
            patchSize: patch,
            downsampleRatio: downsample,
            minPixels: 0,
            maxImageTokens: 0,
            numHiddenLayers: 1,
            hiddenSize: 8,
            numAttentionHeads: 1,
            intermediateSize: 8,
            imageTokenID: 7
        )
    }

    /// R: `4 * x`, G: `4 * y`, B: 200, so every channel and every pixel is
    /// distinguishable after normalization.
    private func gradient(width: Int, height: Int) -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                bytes.append(UInt8(x * 4 % 256))
                bytes.append(UInt8(y * 4 % 256))
                bytes.append(200)
            }
        }
        return bytes
    }

    /// `x.reshape(3, n_h, p, n_w, p).permute(1, 3, 0, 2, 4)`: one row per patch, in
    /// reading order, channel-major inside the patch.
    func testPatchesAreChannelMajorAndInReadingOrder() {
        let patches = DeepSeekV41VisionPreprocess.patches(
            rgb: gradient(width: 32, height: 32),
            width: 32,
            height: 32,
            spec: spec()
        )
        XCTAssertEqual(patches.count, 4)
        XCTAssertTrue(patches.allSatisfy { $0.count == 3 * 16 * 16 })

        let topLeft = patches[0]
        // Channel-major: R plane first, then G, then B.
        XCTAssertEqual(topLeft[0], DeepSeekV41VisionLayout.normalize(r: 0, g: 0, b: 200).0, accuracy: 1e-6)
        XCTAssertEqual(topLeft[16 * 16], DeepSeekV41VisionLayout.normalize(r: 0, g: 0, b: 200).1, accuracy: 1e-6)
        XCTAssertEqual(topLeft[2 * 16 * 16], DeepSeekV41VisionLayout.normalize(r: 0, g: 0, b: 200).2, accuracy: 1e-6)
        // Pixel (1, 0) of the patch reads source x = 1.
        XCTAssertEqual(topLeft[1], DeepSeekV41VisionLayout.normalize(r: 4, g: 0, b: 0).0, accuracy: 1e-6)
        // The patch's second image row is 16 values into the G plane, and reads y = 1.
        XCTAssertEqual(
            topLeft[16 * 16 + 16],
            DeepSeekV41VisionLayout.normalize(r: 0, g: 4, b: 0).1,
            accuracy: 1e-6
        )

        // The second patch is the top-RIGHT one, so it starts at source x = 16.
        XCTAssertEqual(patches[1][0], DeepSeekV41VisionLayout.normalize(r: 64, g: 0, b: 0).0, accuracy: 1e-6)
    }

    /// A tall image is padded on both sides with the checkpoint's gray (127) rather
    /// than stretched, so the margin columns are gray and the drawn columns are not.
    func testPaddedImageCentresWithTheCheckpointGrey() {
        let placement = DeepSeekV41VisionPreprocess.placement(
            width: 10,
            height: 40,
            spec: spec()
        )
        XCTAssertEqual(placement.targetWidth, 16)
        XCTAssertEqual(placement.targetHeight, 48)
        XCTAssertEqual(placement.drawnWidth, 12)
        XCTAssertEqual(placement.drawnHeight, 48)
        XCTAssertEqual(placement.left, 2)
        XCTAssertEqual(placement.top, 0)
        XCTAssertFalse(placement.stretches)

        let patches = DeepSeekV41VisionPreprocess.patches(
            rgb: gradient(width: 10, height: 40),
            width: 10,
            height: 40,
            spec: spec()
        )
        XCTAssertEqual(patches.count, 3)
        let first = patches[0]
        // Columns 0 and 1 are the pad; column 2 reads source x = 0.
        XCTAssertEqual(first[0], DeepSeekV41VisionLayout.normalize(r: 127, g: 127, b: 127).0, accuracy: 1e-6)
        XCTAssertEqual(first[1], DeepSeekV41VisionLayout.normalize(r: 127, g: 127, b: 127).0, accuracy: 1e-6)
        XCTAssertEqual(first[2], DeepSeekV41VisionLayout.normalize(r: 0, g: 0, b: 0).0, accuracy: 1e-6)
    }

    /// The ratio cap equalises the sides first, so the image fills the box instead of
    /// being padded into it.
    func testRatioCapStretchesRatherThanPads() {
        let spec = DeepSeekV41VisionSpec(
            patchSize: 8,
            downsampleRatio: 1,
            minPixels: 0,
            maxImageTokens: 0,
            numHiddenLayers: 1,
            hiddenSize: 8,
            numAttentionHeads: 1,
            intermediateSize: 8,
            imageTokenID: 7,
            maxWidthHeightRatio: 1
        )
        let placement = DeepSeekV41VisionPreprocess.placement(width: 80, height: 10, spec: spec)
        XCTAssertTrue(placement.stretches)
        XCTAssertEqual(placement.left, 0)
        XCTAssertEqual(placement.top, 0)
        XCTAssertEqual(placement.drawnWidth, placement.targetWidth)
    }

    func testShortBufferYieldsNoPatches() {
        XCTAssertTrue(
            DeepSeekV41VisionPreprocess.patches(rgb: [0, 0, 0], width: 32, height: 32, spec: spec()).isEmpty
        )
    }
}

final class DeepSeekV41VisionPlanTests: XCTestCase {

    private func spec() -> DeepSeekV41VisionSpec {
        DeepSeekV41VisionSpec(
            patchSize: 16,
            downsampleRatio: 3,
            minPixels: 0,
            maxImageTokens: 1024,
            numHiddenLayers: 2,
            hiddenSize: 64,
            numAttentionHeads: 4,
            intermediateSize: 128,
            imageTokenID: 129_264
        )
    }

    func testStagesCoverPatchEmbedBlocksUnshuffleAndProjector() {
        let plan = DeepSeekV41VisionTowerPlan(spec: spec(), projectorOutputWidth: 7168)
        XCTAssertEqual(plan.stages.first?.kind, .patchEmbed)
        XCTAssertEqual(plan.stages.last?.kind, .projector)
        XCTAssertEqual(plan.stages.filter { $0.kind == .blockAttention }.count, 2)
        XCTAssertEqual(plan.stages.filter { $0.kind == .blockMLP }.count, 2)
        XCTAssertEqual(plan.stages.filter { $0.kind == .pixelUnshuffle }.count, 1)
        XCTAssertEqual(plan.projectorOutputWidth, 7168)
    }

    func testPatchEmbedInputMatchesThePatchVolume() {
        let plan = DeepSeekV41VisionTowerPlan(spec: spec(), projectorOutputWidth: 7168)
        XCTAssertEqual(plan.stages.first?.inputWidth, 16 * 16 * 3)
        XCTAssertEqual(plan.stages.first?.outputWidth, 64)
    }

    func testPixelUnshuffleWidensByTheSquareOfTheRatio() {
        let plan = DeepSeekV41VisionTowerPlan(spec: spec(), projectorOutputWidth: 7168)
        let unshuffle = try? XCTUnwrap(plan.stages.first { $0.kind == .pixelUnshuffle })
        XCTAssertEqual(unshuffle?.outputWidth, 64 * 9)
        let projector = try? XCTUnwrap(plan.stages.first { $0.kind == .projector })
        XCTAssertEqual(projector?.inputWidth, 64 * 9)
    }

    func testNoUnshuffleStageWhenTheRatioIsOne() {
        let flat = DeepSeekV41VisionSpec(
            patchSize: 16, downsampleRatio: 1, minPixels: 0, maxImageTokens: 256,
            numHiddenLayers: 2, hiddenSize: 64, numAttentionHeads: 4,
            intermediateSize: 128, imageTokenID: 1
        )
        let plan = DeepSeekV41VisionTowerPlan(spec: flat, projectorOutputWidth: 7168)
        XCTAssertTrue(plan.stages.allSatisfy { $0.kind != .pixelUnshuffle })
    }

    func testTokenCountMatchesTheLayoutRule() {
        let plan = DeepSeekV41VisionTowerPlan(spec: spec(), projectorOutputWidth: 7168)
        // 480 / 48 = 10 blocks on each side: 10 rows of (10 images + newline) plus
        // the two delimiters. Pinned as a literal: comparing against the helper
        // under test would pass even if the helper were wrong.
        XCTAssertEqual(plan.tokenCount(width: 480, height: 480), 112)
        // 96 x 144 is 2 x 3 blocks: 3 * (2 + 1) + 2.
        XCTAssertEqual(plan.tokenCount(width: 96, height: 144), 11)
    }

    /// A square image whose floor exceeds a very small cap must stay square, and
    /// the cap must still hold: the solved size is 32 x 32 at patch 8, which is a
    /// 4 x 4 grid costing 22 positions.
    func testSquareImageStaysSquareWhenTheFloorExceedsTheCap() {
        let tight = DeepSeekV41VisionSpec(
            patchSize: 8, downsampleRatio: 1, minPixels: 36 * 8 * 8, maxImageTokens: 25,
            numHiddenLayers: 2, hiddenSize: 16, numAttentionHeads: 2,
            intermediateSize: 32, imageTokenID: 1
        )
        let fitted = DeepSeekV41VisionLayout.fit(width: 48, height: 48, spec: tight)
        XCTAssertEqual(fitted.tokens, 22)
        XCTAssertLessThanOrEqual(fitted.tokens, 25)
        XCTAssertEqual(fitted.columns, fitted.rows)
        XCTAssertEqual(fitted.width, fitted.height)
        XCTAssertEqual(fitted.width / tight.tokenStride, 4)
    }
}

// MARK: - Config wiring

final class DeepSeekV41VisionConfigTests: XCTestCase {

    func testVisionConfigIsDecodedFromTheWrapper() throws {
        let json = """
        {
          "architectures": ["DeepseekV41ForCausalLM"],
          "model_type": "deepseek_v41",
          "image_token_id": 129264,
          "vision_config": {"num_hidden_layers": 27, "patch_size": 16, "downsample_ratio": 3,
                            "min_pixels": 65536, "max_image_tokens": 1024,
                            "hidden_size": 1152, "num_attention_heads": 16,
                            "intermediate_size": 4304, "image_token_id": 129264},
          "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
                          "vocab_size": 16, "n_routed_experts": 2, "n_activated_experts": 1}
        }
        """
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        let vision = try XCTUnwrap(config.vision)
        XCTAssertTrue(config.hasVisionTower)
        XCTAssertEqual(vision.numHiddenLayers, 27)
        XCTAssertEqual(vision.patchSize, 16)
        XCTAssertEqual(vision.downsampleRatio, 3)
        XCTAssertEqual(vision.minPixels, 65_536)
        XCTAssertEqual(vision.maxImageTokens, 1024)
        XCTAssertEqual(vision.hiddenSize, 1152)
        XCTAssertEqual(config.imageTokenID, 129_264)

        let plan = try XCTUnwrap(config.visionTowerPlan)
        XCTAssertEqual(plan.projectorOutputWidth, config.hiddenSize)
    }

    func testTextOnlyCheckpointHasNoTower() throws {
        let json = """
        {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
         "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
                         "vocab_size": 16, "n_routed_experts": 2, "n_activated_experts": 1}}
        """
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        XCTAssertFalse(config.hasVisionTower)
        XCTAssertNil(config.vision)
        XCTAssertNil(config.visionTowerPlan)
    }

    func testImageTokenFallsBackToTheVisionBlock() throws {
        let json = """
        {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
         "vision_config": {"num_hidden_layers": 4, "patch_size": 14, "image_token_id": 77},
         "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
                         "vocab_size": 16, "n_routed_experts": 2, "n_activated_experts": 1}}
        """
        let config = try DeepSeekV41Config.decode(jsonData: Data(json.utf8))
        XCTAssertEqual(config.imageTokenID, 77)
    }

    func testTowerWithoutATokenIDIsRefused() throws {
        let json = """
        {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"],
         "vision_config": {"num_hidden_layers": 4, "patch_size": 14},
         "text_config": {"num_hidden_layers": 2, "hidden_size": 8, "num_attention_heads": 2,
                         "vocab_size": 16, "n_routed_experts": 2, "n_activated_experts": 1}}
        """
        XCTAssertThrowsError(try DeepSeekV41Config.decode(jsonData: Data(json.utf8))) { error in
            guard case DeepSeekV41ConfigError.missingField(let field) = error else {
                return XCTFail("unexpected error \(error)")
            }
            XCTAssertEqual(field, "image_token_id")
        }
    }
}
