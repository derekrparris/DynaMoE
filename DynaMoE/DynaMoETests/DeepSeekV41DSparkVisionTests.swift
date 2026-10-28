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

    func testCombineLogitsInterpolatesBetweenTheTwoHeads() {
        let draft: [Float] = [2, 0]
        let markov: [Float] = [0, 4]
        XCTAssertEqual(DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: markov, weight: 0), draft)
        XCTAssertEqual(DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: markov, weight: 1), markov)
        XCTAssertEqual(
            DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: markov, weight: 0.5),
            [1, 2]
        )
        // Out-of-range weights are clamped, not trusted.
        XCTAssertEqual(DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: markov, weight: 9), markov)
        XCTAssertEqual(DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: markov, weight: -9), draft)
    }

    func testMismatchedWidthsLeaveTheDraftHeadAlone() {
        let draft: [Float] = [1, 2, 3]
        XCTAssertEqual(
            DeepSeekV41DSpark.combineLogits(draftLogits: draft, markovLogits: [0, 0], weight: 1),
            draft
        )
        XCTAssertTrue(DeepSeekV41DSpark.combineLogits(draftLogits: [], markovLogits: [], weight: 1).isEmpty)
    }

    func testPickBreaksTiesTowardTheLowestTokenID() {
        XCTAssertEqual(DeepSeekV41DSpark.pick([0.1, 0.9, 0.9]), 1)
        XCTAssertEqual(DeepSeekV41DSpark.pick([2, 2, 2]), 0)
        XCTAssertNil(DeepSeekV41DSpark.pick([]))
    }

    func testChainConditionsOnItsOwnTokensNotTheTargets() {
        // A step closure that echoes the token it was handed back as the argmax, so
        // the chain's second step can only see the first draft token if the chain
        // really feeds its own output back.
        var seen: [Int] = []
        let draft = DeepSeekV41DSpark.runChain(steps: 3, seedToken: 42, markovWeight: 0) { _, previous in
            seen.append(previous)
            var logits = [Float](repeating: 0, count: 8)
            logits[(previous + 1) % 8] = 5
            return (draftLogits: logits, markovLogits: logits, confidence: 0.9)
        }
        XCTAssertEqual(seen.first, 42)
        XCTAssertEqual(draft.tokens, [3, 4, 5])
        // Each step after the first conditions on the token the chain chose.
        XCTAssertEqual(seen, [42, 3, 4])
    }

    func testChainStopsWhenAStepProducesNothing() {
        var calls = 0
        let draft = DeepSeekV41DSpark.runChain(steps: 5, seedToken: 1, markovWeight: 0) { index, _ in
            calls += 1
            guard index < 2 else { return nil }
            return (draftLogits: [1], markovLogits: [1], confidence: 0.9)
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

    func testTokenCountRoundsUpToWholeStrides() {
        XCTAssertEqual(DeepSeekV41VisionLayout.tokenCount(width: 48, height: 48, stride: 48), 1)
        XCTAssertEqual(DeepSeekV41VisionLayout.tokenCount(width: 49, height: 48, stride: 48), 2)
        XCTAssertEqual(DeepSeekV41VisionLayout.tokenCount(width: 96, height: 144, stride: 48), 6)
        XCTAssertEqual(DeepSeekV41VisionLayout.tokenCount(width: 0, height: 10, stride: 48), 0)
    }

    func testOversizedImageIsShrunkInsideTheTokenCap() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 4000, height: 3000, spec: spec())
        XCTAssertLessThanOrEqual(fitted.tokens, 1024)
        XCTAssertGreaterThan(fitted.tokens, 0)
        // The result is a whole number of tokens on both sides.
        XCTAssertEqual(fitted.width % 48, 0)
        XCTAssertEqual(fitted.height % 48, 0)
        XCTAssertEqual(fitted.width / 48 * (fitted.height / 48), fitted.tokens)
        // Aspect ratio is preserved to within a token.
        let aspect = Double(fitted.width) / Double(fitted.height)
        XCTAssertEqual(aspect, 4.0 / 3.0, accuracy: 0.08)
    }

    func testImageInsideTheBudgetIsLeftAlone() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 480, height: 480, spec: spec())
        XCTAssertEqual(fitted.width, 480)
        XCTAssertEqual(fitted.height, 480)
        // 480 / 48 = 10 on each side.
        XCTAssertEqual(fitted.tokens, 100)
    }

    func testTinyImageIsUpscaledToThePixelFloor() {
        let fitted = DeepSeekV41VisionLayout.fit(width: 48, height: 48, spec: spec())
        XCTAssertGreaterThanOrEqual(fitted.tokens * 48 * 48, 65_536)
    }

    func testCapWinsOverFloorWhenTheyConflict() {
        // A cap of one token cannot also satisfy a 65k pixel floor.
        let fitted = DeepSeekV41VisionLayout.fit(
            width: 4000,
            height: 3000,
            spec: spec(minPixels: 65_536, maxTokens: 1)
        )
        XCTAssertEqual(fitted.tokens, 1)
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
        XCTAssertEqual(fitted.tokens, 200)
    }

    func testPlaceholderRepeatsTheImageTokenOncePerPatch() {
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let placeholder = DeepSeekV41VisionLayout.placeholder(tokenCount: 3, marker: markers)
        XCTAssertTrue(placeholder.hasPrefix(markers.imageOpen))
        XCTAssertTrue(placeholder.hasSuffix(markers.imageClose))
        XCTAssertEqual(
            placeholder.components(separatedBy: markers.imageToken).count - 1,
            3
        )
        XCTAssertEqual(DeepSeekV41VisionLayout.placeholder(tokenCount: 0, marker: markers), "")
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
        XCTAssertEqual(expanded.components(separatedBy: markers.imageOpen).count - 1, 2)
        XCTAssertFalse(expanded.contains(markers.imagePlaceholder))
        XCTAssertTrue(expanded.hasPrefix("look "))
        XCTAssertTrue(expanded.contains(" then "))
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
        XCTAssertEqual(
            plan.tokenCount(width: 480, height: 480),
            DeepSeekV41VisionLayout.fit(width: 480, height: 480, spec: spec()).tokens
        )
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
