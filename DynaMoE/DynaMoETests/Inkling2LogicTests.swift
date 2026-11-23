//
//  Inkling2LogicTests.swift
//  DynaMoETests
//
//  Model-independent coverage for the Inkling 2 port: config decoding (the
//  nested wrapper, strict unknown-key refusal, the loop schedule), the visit
//  plan, the exhaustive tensor catalog, query unpacking, expert layouts,
//  memory and preflight arithmetic, the prompt dialect, and every host
//  primitive (element formats, both activation quantizers, the layer rules).
//
//  No Metal device, no weights, no MainActor, so this runs anywhere.
//
//  The numeric goldens are produced by an INDEPENDENT Python mirror of the
//  reference algorithms (the format definitions, vLLM's quantizer rules and
//  numpy versions of the layer rules), never from this Swift code: a test that
//  compares an implementation with itself proves only that it is deterministic.
//
//  The config fixtures are SYNTHETIC -- they exercise branches at small widths
//  and are not the model. Facts about the real checkpoint belong in
//  `Inkling2PublishedConfigTests` (the config) and `Inkling2PublishedTensorTests`
//  (the catalog over the real shard headers, skipped when the snapshot is
//  absent).
//

import XCTest
@testable import DynaMoE

// MARK: - Config

final class Inkling2ConfigTests: XCTestCase {

    /// A small but structurally complete text tower: 12 layers, a 6-layer core
    /// looped three times, one full layer per six, two experts. The widths are
    /// not the model's; the RELATIONS are.
    static let smallJSON = """
    {
      "architectures": ["Inkling2LoopForConditionalGeneration"],
      "model_type": "inkling2_loop",
      "dtype": "bfloat16",
      "eos_token_id": 17,
      "image_token_id": 15,
      "quantization_config": {"quant_method": "modelopt"},
      "text_config": {
        "model_type": "inkling2_loop_text",
        "hidden_size": 64,
        "num_hidden_layers": 12,
        "vocab_size": 32,
        "unpadded_vocab_size": 30,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "sliding_window_size": 8,
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 8192,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "sconv_kernel_size": 4,
        "final_logit_softcapping": 30.0,
        "n_layers_per_attn_res_block": 6,
        "n_routed_experts": 2,
        "num_experts_per_tok": 1,
        "n_shared_experts": 0,
        "intermediate_size": 16,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": true,
        "loop_config": {
          "core_start": 6,
          "core_end": 12,
          "num_passes": 3,
          "gate_rank": 8,
          "is_varloop": false
        }
      }
    }
    """

    func config(_ json: String = smallJSON) throws -> Inkling2Config {
        try Inkling2Config.decode(jsonData: Data(json.utf8))
    }

    func testTheNestedWrapperDecodes() throws {
        let config = try config()
        XCTAssertEqual(config.modelType, "inkling2_loop")
        XCTAssertEqual(config.textModelType, "inkling2_loop_text")
        XCTAssertEqual(config.hiddenSize, 64)
        XCTAssertEqual(config.numHiddenLayers, 12)
        XCTAssertEqual(config.numAttentionHeads, 4)
        XCTAssertEqual(config.numKeyValueHeads, 2)
        XCTAssertEqual(config.numRoutedExperts, 2)
        XCTAssertEqual(config.numPasses, 3)
        XCTAssertEqual(config.loopCoreStart, 6)
        XCTAssertEqual(config.loopCoreEnd, 12)
        XCTAssertEqual(config.eosTokenID, 17)
        XCTAssertEqual(config.imageTokenID, 15)
        XCTAssertEqual(config.quantizationMethod, "modelopt")
        XCTAssertEqual(config.executedVisitCount, 12 + 2 * 6)
        XCTAssertNil(config.vision, "no vision_config means no tower")
        XCTAssertNil(config.mtp, "no mtp_config means no sidecar")
    }

    /// The loop schedule: only the core range repeats, every visit owns its own
    /// state slot, and only the full-attention visits get a growing cache.
    func testTheVisitPlanRepeatsOnlyTheCore() throws {
        let config = try config()
        let plan = try config.makeVisitPlan()
        // 6 pre + 6 core x 3 passes + 0 post.
        XCTAssertEqual(plan.count, 24)
        XCTAssertEqual(plan.count, config.executedVisitCount)
        XCTAssertEqual(
            plan.map(\.physicalLayer),
            [0, 1, 2, 3, 4, 5,
             6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11]
        )
        // Every visit owns its own state slot, and the slots are unique: KV and
        // the convolution states are per VISIT while the weights are per layer.
        XCTAssertEqual(plan.map(\.cacheSlot), Array(0..<plan.count))
        // One full layer per six: 5 once, 11 in each pass.
        XCTAssertEqual(plan.filter(\.isFullAttention).map(\.physicalLayer), [5, 11, 11, 11])
        XCTAssertEqual(plan.filter(\.hasMessageControllers).count, 18, "every core visit carries controllers")
        XCTAssertNil(plan[0].attentionInputLayer, "layer 0 has no attention-input read")
        XCTAssertEqual(plan[1].attentionInputLayer, 1)
        // Passes are numbered from one for core visits and zero outside.
        let corePasses: [Int] = plan.map(\.pass).filter { $0 > 0 }
        XCTAssertEqual(corePasses, [Int](repeating: 1, count: 6) + [Int](repeating: 2, count: 6) + [Int](repeating: 3, count: 6))
        XCTAssertEqual(Array(plan.map(\.pass).prefix(6)), [Int](repeating: 0, count: 6))
    }

    func testAVariableLoopCheckpointIsRefusedByName() throws {
        let json = Self.smallJSON.replacingOccurrences(of: "\"is_varloop\": false", with: "\"is_varloop\": true")
        XCTAssertThrowsError(try config(json)) { error in
            guard case Inkling2ConfigError.invalidField(let field, let detail) = error else {
                return XCTFail("expected a named field refusal, got \(error)")
            }
            XCTAssertEqual(field, "loop_config.is_varloop")
            XCTAssertTrue(detail.contains("variable-loop"), detail)
        }
    }

    func testAnUnknownKeyIsRefusedRatherThanIgnored() throws {
        let json = Self.smallJSON.replacingOccurrences(
            of: "\"sconv_kernel_size\": 4,",
            with: "\"sconv_kernel_size\": 4, \"brand_new_knob\": 3,"
        )
        XCTAssertThrowsError(try config(json)) { error in
            guard case Inkling2ConfigError.unknownField(let field, let where_) = error else {
                return XCTFail("expected an unknown-field refusal, got \(error)")
            }
            XCTAssertEqual(field, "brand_new_knob")
            XCTAssertEqual(where_, "text_config")
        }
    }

    func testAMissingRequiredFieldIsNamed() throws {
        let json = Self.smallJSON.replacingOccurrences(of: "\"hidden_size\": 64,\n", with: "")
        XCTAssertThrowsError(try config(json)) { error in
            guard case Inkling2ConfigError.missingField(let field) = error else {
                return XCTFail("expected a missing-field refusal, got \(error)")
            }
            XCTAssertEqual(field, "text_config.hidden_size")
        }
    }

    func testSharedExpertsAreRefusedBecauseTheGraphHasNoPath() throws {
        let json = Self.smallJSON.replacingOccurrences(of: "\"n_shared_experts\": 0", with: "\"n_shared_experts\": 2")
        XCTAssertThrowsError(try config(json)) { error in
            guard case Inkling2ConfigError.invalidField(let field, _) = error else {
                return XCTFail("expected an invalid-field refusal, got \(error)")
            }
            XCTAssertEqual(field, "n_shared_experts")
        }
    }

    /// `layer_types` and `local_layer_ids` are two spellings of one fact. A
    /// checkpoint publishing only one of them must answer the same, and the
    /// derived default is the checkpoint's own (one full layer per six).
    func testLayerKindComesFromEitherSpelling() throws {
        let derived = try config()
        XCTAssertEqual(derived.derivedLayerTypes.filter { $0 == "hybrid" }.count, 2)
        XCTAssertTrue(derived.isLocalLayer(0))
        XCTAssertFalse(derived.isLocalLayer(5))
        XCTAssertTrue(derived.isLocalLayer(6))
        XCTAssertFalse(derived.isLocalLayer(11))
        XCTAssertEqual(derived.fullAttentionLayers, [5, 11])

        let published = Self.smallJSON.replacingOccurrences(
            of: "\"n_layers_per_attn_res_block\": 6,",
            with: "\"n_layers_per_attn_res_block\": 6, \"local_layer_ids\": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],"
        )
        let explicit = try config(published)
        XCTAssertEqual(explicit.fullAttentionLayers, [5, 11])
        for layer in 0..<12 {
            XCTAssertEqual(derived.isLocalLayer(layer), explicit.isLocalLayer(layer), "layer \(layer)")
        }
    }

    func testAPaddedUnembeddingMustFitInsideTheVocabulary() throws {
        let json = Self.smallJSON.replacingOccurrences(of: "\"unpadded_vocab_size\": 30", with: "\"unpadded_vocab_size\": 40")
        XCTAssertThrowsError(try config(json)) { error in
            guard case Inkling2ConfigError.invalidField(let field, _) = error else {
                return XCTFail("expected an invalid-field refusal, got \(error)")
            }
            XCTAssertEqual(field, "unpadded_vocab_size")
        }
    }

    func testAForeignConfigIsNotMistakenForInkling() throws {
        let json = #"{"model_type": "llama", "hidden_size": 64}"#
        XCTAssertThrowsError(try Inkling2Config.decode(jsonData: Data(json.utf8))) { error in
            XCTAssertEqual(error as? Inkling2ConfigError, .notInkling2)
        }
    }
}

// MARK: - Primitives: element formats

final class Inkling2ElementFormatTests: XCTestCase {

    func testE2M1DecodesEveryNibble() {
        let expected: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6,
                                 -0, -0.5, -1, -1.5, -2, -3, -4, -6]
        for code in 0..<16 {
            XCTAssertEqual(
                Inkling2Element.decodeE2M1(UInt8(code)), expected[code],
                "nibble \(code)"
            )
        }
    }

    func testE2M1RoundsToNearestEvenAndSaturates() {
        // Exact values round-trip.
        for code in 0..<8 {
            let value = Inkling2Element.decodeE2M1(UInt8(code))
            XCTAssertEqual(Inkling2Element.encodeE2M1(value), UInt8(code))
        }
        // Midpoints, IEEE nearest-even: 0.25 is between 0 (LSB 0) and .5 (LSB 1) -> 0.
        XCTAssertEqual(Inkling2Element.encodeE2M1(0.25), 0)
        // 0.75 is between .5 (m=1, LSB 1) and 1 (m=0, LSB 0) -> 1.0.
        XCTAssertEqual(Inkling2Element.encodeE2M1(0.75), 2)
        // 1.25 between 1 (LSB 0) and 1.5 (LSB 1) -> 1.0.
        XCTAssertEqual(Inkling2Element.encodeE2M1(1.25), 2)
        // 2.5 between 2 (LSB 0) and 3 (LSB 1) -> 2.
        XCTAssertEqual(Inkling2Element.encodeE2M1(2.5), 4)
        // 5.0 between 4 and 6 -> nearer 4.
        XCTAssertEqual(Inkling2Element.encodeE2M1(5.0), 6)
        // Saturates rather than producing inf.
        XCTAssertEqual(Inkling2Element.encodeE2M1(1000), 0x7)
        XCTAssertEqual(Inkling2Element.encodeE2M1(-1000), 0xF)
        // Signed zero is preserved.
        XCTAssertEqual(Inkling2Element.encodeE2M1(-0.0), 0x8)
    }

    func testE4M3DecodesTheReferenceCodes() {
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x00), 0)
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x38), 1)
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x3C), 1.5)
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x40), 2)
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x7E), 448)
        XCTAssertTrue(Inkling2Element.decodeE4M3(0x7F).isNaN)
        XCTAssertTrue(Inkling2Element.decodeE4M3(0xFF).isNaN)
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x01), 1.0 / 512.0, "smallest subnormal")
        XCTAssertEqual(Inkling2Element.decodeE4M3(0x80), -0)
    }

    func testE4M3RoundsToNearestEvenAndSaturates() {
        for code in UInt8(0)...UInt8(0x7E) {
            let value = Inkling2Element.decodeE4M3(code)
            XCTAssertEqual(Inkling2Element.encodeE4M3(value), code, "code \(code)")
        }
        XCTAssertEqual(Inkling2Element.encodeE4M3(500), 0x7E, "saturates at 448")
        XCTAssertEqual(Inkling2Element.encodeE4M3(-500), 0xFE)
        XCTAssertEqual(Inkling2Element.encodeE4M3(.infinity), 0x7E)
        XCTAssertEqual(Inkling2Element.encodeE4M3(.nan), 0x7F)
        XCTAssertEqual(Inkling2Element.encodeE4M3(-0.0), 0x80)
        // A midpoint between 1.0 (0x38) and 1.125 (0x39) is 1.0625 -> even (0x38).
        XCTAssertEqual(Inkling2Element.encodeE4M3(1.0625), 0x38)
    }

    func testE8M0SelectsPowersOfTwoWithTiesUp() {
        XCTAssertEqual(Inkling2Element.encodeE8M0(1), 127)
        XCTAssertEqual(Inkling2Element.decodeE8M0(127), 1)
        XCTAssertEqual(Inkling2Element.encodeE8M0(0.5), 126)
        XCTAssertEqual(Inkling2Element.encodeE8M0(2), 128)
        // 448 is between 256 and 512, and nearer 512 in the log domain.
        XCTAssertEqual(Inkling2Element.encodeE8M0(448), 136)
        XCTAssertEqual(Inkling2Element.decodeE8M0(136), 512)
        // sqrt(2) is the midpoint; a tie goes UP.
        let midpoint = Float(2).squareRoot()
        XCTAssertEqual(Inkling2Element.encodeE8M0(midpoint), 128)
        XCTAssertEqual(Inkling2Element.encodeE8M0(0), 0)
        XCTAssertEqual(Inkling2Element.encodeE8M0(-1), 0)
    }

    func testPackedPairsPutTheLowerColumnInTheLowNibble() {
        let byte = Inkling2Element.packE2M1Pair(low: 1.5, high: -2)
        XCTAssertEqual(byte & 0xF, 0x3, "1.5 is code 3")
        XCTAssertEqual(byte >> 4, 0xC, "-2 is code 12")
        let (low, high) = Inkling2Element.unpackE2M1Pair(byte)
        XCTAssertEqual(low, 1.5)
        XCTAssertEqual(high, -2)
    }
}

// MARK: - Primitives: activation quantizers

final class Inkling2ActivationQuantizerTests: XCTestCase {

    /// The vector the Python mirror quantized, with its exact output.
    static let nvfp4Probe: [Float] =
        [Float](repeating: 0, count: 16)
        + [1.0, -2.0, 3.0, -0.25] + [Float](repeating: 0.5, count: 12)
        + [7.0, -6.5, 0.125, 0.0] + [Float](repeating: 2.0, count: 12)

    func testNVFP4MatchesTheIndependentMirror() throws {
        let result = try Inkling2ActivationQuantizer.nvfp4(Self.nvfp4Probe)
        XCTAssertEqual(result.blockSize, 16)
        // outer = rowAmax / 1536, rowAmax = 7
        XCTAssertEqual(result.outerScale, Float(7.0 / 1536.0), accuracy: 1e-9)
        XCTAssertEqual(result.blockScales, [56, 110, 120])
        XCTAssertEqual(result.codes, [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            4, 14, 7, 9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            7, 15, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3
        ])
    }

    /// The two candidate rules are not interchangeable: this block is one where
    /// the 4.0 candidate wins, and the mirror's scale says so.
    func testTheBlockScaleSearchPicksTheLowerErrorCandidate() {
        let block: [Float] = [Float](repeating: 0.9, count: 16)
        let scale = Inkling2ActivationQuantizer.chooseNVFP4BlockScale(block: block, blockAmax: 0.9)
        XCTAssertEqual(scale, 0.9 / 4.0, accuracy: 1e-7)
        // A tie keeps the 6.0 candidate: all-zero blocks have zero error under
        // both, and the first candidate is the one retained.
        XCTAssertEqual(
            Inkling2ActivationQuantizer.chooseNVFP4BlockScale(
                block: [Float](repeating: 0, count: 16), blockAmax: 0
            ),
            1.0
        )
    }

    func testNVFP4DequantizesToItsOwnCodes() throws {
        let result = try Inkling2ActivationQuantizer.nvfp4(Self.nvfp4Probe)
        let values = result.dequantized(format: .e2m1)
        XCTAssertEqual(values.count, Self.nvfp4Probe.count)
        // The 7.0 entry is the row maximum, so it lands on the top code of the
        // third block scaled to nearly itself: the reconstruction is within one
        // E2M1 step of the original.
        let maxIndex = 32
        XCTAssertEqual(values[maxIndex], 7.0, accuracy: 7.0 * 0.25)
        // A zero stays zero.
        XCTAssertEqual(values[0], 0)
    }

    static let mxfp8Probe: [Float] =
        (0..<32).map { 0.1 * Float($0 + 1) } + [Float](repeating: 0, count: 32)
        + [1000.0] + [Float](repeating: 1.0, count: 31)

    func testMXFP8MatchesTheIndependentMirror() throws {
        let result = try Inkling2ActivationQuantizer.mxfp8(Self.mxfp8Probe)
        XCTAssertEqual(result.blockSize, 32)
        XCTAssertEqual(result.blockScales, [120, 127, 129])
        XCTAssertEqual(Array(result.codes.prefix(8)), [85, 93, 98, 101, 104, 106, 107, 109])
        XCTAssertEqual(Array(result.codes[32..<40]), [0, 0, 0, 0, 0, 0, 0, 0])
        // The block's scale is 4 (round-up of 1000/448), so 1000 becomes 250,
        // which rounds to the E4M3 code for 256.
        XCTAssertEqual(Inkling2Element.decodeE8M0(result.blockScales[2]), 4)
        XCTAssertEqual(Array(result.codes[64..<68]), [120, 40, 40, 40])
        XCTAssertEqual(Inkling2Element.decodeE4M3(result.codes[64]), 256)
    }

    func testRoundUpPowerOfTwoKeepsAnExactPower() {
        XCTAssertEqual(Inkling2ActivationQuantizer.roundUpPowerOfTwo(4), 4)
        XCTAssertEqual(Inkling2ActivationQuantizer.roundUpPowerOfTwo(4.5), 8)
        XCTAssertEqual(Inkling2ActivationQuantizer.roundUpPowerOfTwo(0), 1)
        XCTAssertEqual(Inkling2ActivationQuantizer.roundUpPowerOfTwo(1), 1)
    }

    func testTheTwoProfilesAreDistinctContracts() {
        XCTAssertFalse(Inkling2ActivationProfile.hfReference.quantizesActivation)
        XCTAssertTrue(Inkling2ActivationProfile.trained.quantizesActivation)
        XCTAssertNotEqual(Inkling2ActivationProfile.hfReference, .trained)
    }
}

// MARK: - Primitives: layer rules

final class Inkling2LayerRuleTests: XCTestCase {

    /// Values from the numpy mirror, to 5 decimal places.
    func testRMSNormMatchesTheMirror() {
        let result = Inkling2Math.rmsNorm([1, 2, 3], eps: 1e-6)
        XCTAssertEqual(result[0], 0.4629100003, accuracy: 1e-6)
        XCTAssertEqual(result[1], 0.9258200006, accuracy: 1e-6)
        XCTAssertEqual(result[2], 1.3887300009, accuracy: 1e-6)
    }

    func testTheWeightedNormMultipliesTheGainAfterNormalizing() {
        let result = Inkling2Math.rmsNorm([1, 2, 3], gain: [2, 0, -1], eps: 1e-6)
        XCTAssertEqual(result[0], 0.9258200006, accuracy: 1e-6)
        XCTAssertEqual(result[1], 0)
        XCTAssertEqual(result[2], -1.3887300009, accuracy: 1e-6)
    }

    func testL2ClipScalesDownAndNeverUp() {
        let clipped = Inkling2Math.l2Clip([3, 4], threshold: 2)
        XCTAssertEqual(clipped[0], 1.19999976, accuracy: 1e-6)
        XCTAssertEqual(clipped[1], 1.59999968, accuracy: 1e-6)
        // Norm 0.5 is under the threshold: unchanged, not scaled UP to it.
        XCTAssertEqual(Inkling2Math.l2Clip([0.3, 0.4], threshold: 2), [0.3, 0.4])
    }

    func testLogScaleRampsOnlyPastTheFloor() {
        XCTAssertEqual(Inkling2Math.logScale(position: 0, alpha: 0.1, floor: 8192), 1.0)
        XCTAssertEqual(Inkling2Math.logScale(position: 8191, alpha: 0.1, floor: 8192), 1.0)
        XCTAssertEqual(Inkling2Math.logScale(position: 8192, alpha: 0.1, floor: 8192), 1.0000122063, accuracy: 1e-6)
        XCTAssertEqual(Inkling2Math.logScale(position: 100000, alpha: 0.1, floor: 8192), 1.2502022118, accuracy: 1e-6)
    }

    func testStaticShortConvolutionMatchesTheMirror() {
        let weights: [Float] = [0.1, -0.2, 0.3, -0.4]
        let history: [Float] = [0.5, -1.0, 2.0]   // newest last
        XCTAssertEqual(
            Inkling2Math.staticShortConvolution(current: 1.5, history: history, weights: weights),
            0.3354101966, accuracy: 1e-6
        )
        // With no history the older taps read zero.
        XCTAssertEqual(
            Inkling2Math.staticShortConvolution(current: 1.5, history: [], weights: weights),
            0.7379024326, accuracy: 1e-6
        )
    }

    /// The same convolution written the way the reference writes it: a
    /// per-channel `F.conv1d` (cross-correlation, left padded, so the STORED
    /// weight's first tap is the OLDEST) followed by `(conv + x) / sqrt(k + 1)`.
    ///
    /// This is what pins the reversal the forward does at its call site. A
    /// mirror that reused the primitive's backward-distance order would agree
    /// with a forward that forgot to reverse, i.e. it would check nothing.
    func testTheStaticConvolutionMatchesTheReferencesStoredOrder() {
        let kernel = 4
        let stored: [Float] = [0.1, -0.2, 0.3, -0.4]   // conv1d order: oldest tap first
        let history: [Float] = [0.5, -1.0, 2.0]        // oldest first, newest last
        let current: Float = 1.5
        // The reference's window, left padded: [.., x[t-3], x[t-2], x[t-1], x[t]].
        let window = history + [current]
        var convolved: Float = 0
        for tap in 0..<kernel {
            // conv1d pairs stored[k] with the (kernel - 1 - k)-th newest value.
            let age = kernel - 1 - tap
            let source = age < window.count ? window[window.count - 1 - age] : 0
            convolved += stored[tap] * source
        }
        let expected = (convolved + current) / Float(kernel + 1).squareRoot()

        // The forward reverses the stored taps once and calls the primitive.
        let reversed = (0..<kernel).map { stored[kernel - 1 - $0] }
        XCTAssertEqual(
            Inkling2Math.staticShortConvolution(current: current, history: history, weights: reversed),
            expected, accuracy: 1e-6
        )
        // And the stored order UNREVERSED is a different number, so the two
        // cannot be confused: this is what a missing reversal would produce.
        XCTAssertNotEqual(
            Inkling2Math.staticShortConvolution(current: current, history: history, weights: stored),
            expected, accuracy: 1e-3
        )
    }

    func testDynamicShortConvolutionMatchesTheMirror() {
        let staticWeights: [Float] = [0.05, 0.1, -0.15, 0.2]
        let dynamicWeights: [Float] = [0.01, -0.02, 0.03, -0.04]
        XCTAssertEqual(
            Inkling2Math.dynamicShortConvolution(
                current: -0.75, history: [1.5, 0.25],
                staticWeights: staticWeights, dynamicWeights: dynamicWeights
            ),
            -0.3183333333, accuracy: 1e-6
        )
    }

    func testBlockResidualReadMatchesTheMirror() {
        let rows = Inkling2Math.blockResidualRead(
            blockStates: [[0.5, -0.25, 1.0], [1.0, 0.0, -0.5]],
            partial: [0.25, 0.75, -1.0],
            queries: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]]
        )
        let expected: [[Float]] = [
            [1.0311480138, 0.1257373850, -0.2319897923],
            [0.7951546372, 0.2360961171, -0.1202336762]
        ]
        for (row, want) in zip(rows, expected) {
            for (got, value) in zip(row, want) {
                XCTAssertEqual(got, value, accuracy: 1e-6)
            }
        }
    }

    func testMessageGateMatchesTheMirror() {
        let result = Inkling2Math.messageGate(
            current: [1.0, -2.0, 0.5], message: [0.25, 3.0, -1.0],
            forget: [0.3, -0.7, 1.2], gain: [-0.2, 0.9, 0.4]
        )
        let expected: [Float] = [0.6593435575, -1.0598419436, -0.2967762658]
        for (got, want) in zip(result, expected) {
            XCTAssertEqual(got, want, accuracy: 1e-6)
        }
    }

    /// The bias steers SELECTION only: expert 2 has the same score as expert 1
    /// but a negative bias, so it drops out, and the weights are the SCORES
    /// renormalized (not score + bias).
    func testRouterSelectsByBiasedScoreAndWeightsByScore() {
        let route = Inkling2Math.route(
            scores: [0.4, 0.9, 0.9, 0.1, 0.7], bias: [0.0, 0.0, -0.5, 0.6, 0.1],
            topK: 3, routeScale: 8
        )
        // Ascending expert id, the checkpoint's topk order: 1, 3, 4.
        XCTAssertEqual(route.indices, [1, 3, 4])
        XCTAssertEqual(route.weights[0], 4.2352941176, accuracy: 1e-5)
        XCTAssertEqual(route.weights[1], 0.4705882353, accuracy: 1e-5)
        XCTAssertEqual(route.weights[2], 3.2941176471, accuracy: 1e-5)
    }

    func testRouterBreaksATieToTheLowerExpertID() {
        let route = Inkling2Math.route(scores: [0.5, 0.5], bias: [0, 0], topK: 1, routeScale: 1)
        XCTAssertEqual(route.indices, [0])
        XCTAssertEqual(route.weights[0], 1.0, accuracy: 1e-6)
    }

    func testSoftcappedSwiGLUMatchesTheMirror() {
        XCTAssertEqual(
            Inkling2Math.softcappedSwiGLU(gate: 2.5, up: -7.0),
            -13.9871712574, accuracy: 1e-5
        )
    }

    /// The sigmoid factor sees the RAW gate: softcapping it too would make
    /// `tanh(g/4)*4*sigmoid(tanh(g/4)*4)` instead, which differs everywhere.
    func testTheSigmoidSeesTheRawGate() {
        let gate: Float = 6.0
        let raw = Inkling2Math.softcap(gate, 4.0) * Inkling2Math.sigmoid(gate) * Inkling2Math.softcap(1.0, 25.0)
        let capped = Inkling2Math.softcap(gate, 4.0)
        let wrong = capped * Inkling2Math.sigmoid(capped) * Inkling2Math.softcap(1.0, 25.0)
        XCTAssertEqual(Inkling2Math.softcappedSwiGLU(gate: gate, up: 1.0), raw, accuracy: 1e-7)
        XCTAssertNotEqual(raw, wrong, accuracy: 1e-4)
    }

    func testTheFinalSoftcapBoundsTheLogits() {
        XCTAssertEqual(Inkling2Math.finalLogitSoftcap([100, -3])[0], 29.9237390242, accuracy: 1e-5)
        XCTAssertEqual(Inkling2Math.finalLogitSoftcap([100, -3])[1], -2.9900398387, accuracy: 1e-5)
    }

    /// The KV row cap is a budget, not a ring: a full-attention visit reports
    /// capacity and stops growing, while a ring visit keeps dropping from the
    /// front and must never be refused at the same number.
    func testTheRowCapAppliesToFullAttentionOnly() {
        let full = Inkling2VisitState(
            isFullAttention: true, window: 2, stateWidths: [2, 2, 2, 2], historyDepth: 0, maxRows: 3
        )
        for position in 0..<3 {
            XCTAssertFalse(full.isAtCapacity, "position \(position) is inside the cap")
            _ = full.append(key: [1, 1], value: [1, 1])
        }
        XCTAssertTrue(full.isAtCapacity)
        XCTAssertEqual(full.keys.count, 3, "a full visit keeps every row it took")

        let ring = Inkling2VisitState(
            isFullAttention: false, window: 2, stateWidths: [2, 2, 2, 2], historyDepth: 0, maxRows: nil
        )
        for _ in 0..<10 { _ = ring.append(key: [1, 1], value: [1, 1]) }
        XCTAssertFalse(ring.isAtCapacity, "a ring is bounded by its window, never by the KV budget")
        XCTAssertEqual(ring.keys.count, 2)
        XCTAssertEqual(ring.windowStart, 8)
    }
}

// MARK: - Query unpacking

final class Inkling2QueryUnpackingTests: XCTestCase {

    func testAttentionInputQueriesStartAtLayerOne() {
        XCTAssertEqual(Inkling2QueryUnpacking.attentionInputLayer(forQueryIndex: 0), 1)
        XCTAssertEqual(Inkling2QueryUnpacking.attentionInputLayer(forQueryIndex: 22), 23)
        XCTAssertEqual(Inkling2QueryUnpacking.mlpInputLayer(forQueryIndex: 0), 0)
        XCTAssertEqual(Inkling2QueryUnpacking.mlpInputLayer(forQueryIndex: 23), 23)
        XCTAssertEqual(Inkling2QueryUnpacking.attentionInputQueryCount(isFullAttention: true), 3)
        XCTAssertEqual(Inkling2QueryUnpacking.attentionInputQueryCount(isFullAttention: false), 1)
    }

    /// The plan's required golden: four logical messages per core layer, read
    /// from two 24-row tensors.
    func testCrossPassQueriesUnpackTwoRowsPerCoreLayer() {
        let layer = Inkling2QueryUnpacking.CrossPassQueryRows(coreLayer: 5)
        XCTAssertEqual(layer.rows, [10, 10, 11, 11], "Q and K from row 2i, V and MLP from 2i+1")
        XCTAssertEqual(layer.fromMainTensor, [true, false, false, true])
        // Every row of the 24-row tensor is claimed exactly once across the 12
        // core layers.
        var rows: [Int] = []
        for core in 0..<12 { rows.append(contentsOf: Inkling2QueryUnpacking.CrossPassQueryRows(coreLayer: core).rows) }
        XCTAssertEqual(Set(rows), Set(0..<24))
        XCTAssertEqual(rows.count, 48, "four logical messages per core layer")
    }
}

// MARK: - Expert layout

final class Inkling2ExpertLayoutTests: XCTestCase {

    func testNVFP4StridesFollowTheStoredShapes() {
        let layout = Inkling2ExpertQuantLayout.nvfp4
        XCTAssertEqual(layout.blockSize, 16)
        XCTAssertEqual(layout.valuesPerByte, 2)
        XCTAssertEqual(layout.packedInputWidth(2304), 1152)
        XCTAssertEqual(layout.scaleInputWidth(2304), 144)
        XCTAssertEqual(layout.packedInputWidth(1152), 576)
        XCTAssertEqual(layout.scaleInputWidth(1152), 72)
        XCTAssertEqual(layout.scaleFormat, .e4m3)
        XCTAssertEqual(layout.valueFormat, .e2m1)
        XCTAssertTrue(layout.requiresAmaxDenominator)
    }

    func testMXFP8StridesFollowTheStoredShapes() {
        let layout = Inkling2ExpertQuantLayout.mxfp8
        XCTAssertEqual(layout.blockSize, 32)
        XCTAssertEqual(layout.valuesPerByte, 1)
        XCTAssertEqual(layout.packedInputWidth(2304), 2304)
        XCTAssertEqual(layout.scaleInputWidth(2304), 72)
        XCTAssertEqual(layout.packedInputWidth(1152), 1152)
        XCTAssertEqual(layout.scaleInputWidth(1152), 36)
        XCTAssertEqual(layout.scaleFormat, .e8m0)
        XCTAssertEqual(layout.valueFormat, .e4m3)
        XCTAssertFalse(layout.requiresAmaxDenominator)
    }
}

// MARK: - Memory and preflight

final class Inkling2MemoryPlanTests: XCTestCase {

    private func config() throws -> Inkling2Config {
        try Inkling2Config.decode(jsonData: Data(Inkling2ConfigTests.smallJSON.utf8))
    }

    /// The convolution state is the one term that is a closed formula, and the
    /// plan's published figure (3,833,856 bytes for 48 visits) is checkable
    /// from the geometry: 48 * (2304 + 1024 + 1024 + 2304) * 3 * 4.
    func testConvolutionStateIsFourFP32StatesPerVisit() throws {
        let config = try config()
        let plan = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 128)
        let visits = config.executedVisitCount
        let channels = 2 * (config.numKeyValueHeads * config.headDim) + 2 * config.hiddenSize
        XCTAssertEqual(plan.convolutionStateBytes, visits * channels * 3 * 4)
        XCTAssertEqual(plan.convolutionStateBytes, 24 * 192 * 3 * 4)
    }

    /// Full caches grow with context, rings do not, and only the FULL visits get
    /// a full cache: at twice the context the total grows by exactly the full
    /// visits' share.
    func testOnlyFullVisitsGrowWithContext() throws {
        let config = try config()
        let short = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 16)
        let long = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 4096)
        let record = 2 * config.numKeyValueHeads * config.headDim * 2
        let plan = try config.makeVisitPlan()
        let fullVisits = plan.filter(\.isFullAttention).count
        XCTAssertEqual(
            long.fullAttentionCacheBytes - short.fullAttentionCacheBytes,
            fullVisits * (4096 - 16) * record
        )
        // The ring is capped at the window, so past it the ring term is constant.
        let windowed = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 8)
        XCTAssertEqual(long.localRingBytes, windowed.localRingBytes)
    }

    /// The plan's published hybrid total: eight full visits, FP16, 1.242 GB at
    /// 32,768 tokens, to within the ring and convolution terms the plan's table
    /// excludes.
    func testThePublishedHybridKvFiguresFollowFromTheGeometry() {
        let fullVisits = 8
        let record = 8 * 128 * 2 * 2   // kvHeads * headDim * K+V * FP16
        XCTAssertEqual(fullVisits * 32_768 * record, 1_073_741_824)
        XCTAssertEqual(fullVisits * 1_048_576 * record, 34_359_738_368)
        // And the naive 48 full caches, which the table exists to rule out.
        XCTAssertEqual(48 * 1_048_576 * record, 206_158_430_208)
    }

    func testPreflightLeavesRoomForTheReply() throws {
        let result = try Inkling2Preflight.check(promptTokens: 100, contextLength: 1000, requestedReplyTokens: 200)
        XCTAssertEqual(result.availablePromptTokens, 1000)
        XCTAssertEqual(result.replyBudget, 200)
        // A reply budget longer than what is left is clamped, not refused.
        let clamped = try Inkling2Preflight.check(promptTokens: 100, contextLength: 1000, requestedReplyTokens: 5000)
        XCTAssertEqual(clamped.replyBudget, 900)
    }

    func testAPromptThatCannotFitIsRefusedByName() {
        XCTAssertThrowsError(
            try Inkling2Preflight.check(promptTokens: 1001, contextLength: 1000, requestedReplyTokens: 10)
        ) { error in
            guard case Inkling2PreflightError.promptTooLong(let prompt, let available) = error else {
                return XCTFail("expected a prompt-too-long refusal, got \(error)")
            }
            XCTAssertEqual(prompt, 1001)
            XCTAssertEqual(available, 1000)
        }
    }
}

// MARK: - Prompt

final class Inkling2PromptTests: XCTestCase {

    private let prompt = Inkling2Prompt()

    /// Every golden below is what jinja2 produced from the checkpoint's OWN
    /// `chat_template.jinja` (with `{% generation %}` registered as a
    /// passthrough, which is all it is at render time).

    func testAUserTurnMatchesTheTemplate() throws {
        XCTAssertEqual(
            try prompt.render(turns: [.user([.text("Hi")])]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
        )
    }

    func testASystemTurnComesBeforeTheEffortLine() throws {
        XCTAssertEqual(
            try prompt.render(turns: [.system("You are helpful."), .user([.text("Hi")])]),
            "<|message_system|><|content_text|>You are helpful.<|end_message|>"
                + "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
        )
    }

    func testACompletedAssistantTurnClosesWithTheEndOfSamplingToken() throws {
        XCTAssertEqual(
            try prompt.render(turns: [
                .user([.text("Hi")]),
                .assistant(reasoning: nil, content: "Hello!"),
                .user([.text("Again")])
            ]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
                + "<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>"
                + "<|message_user|><|content_text|>Again<|end_message|>"
        )
    }

    func testReasoningIsItsOwnTypedBlock() throws {
        XCTAssertEqual(
            try prompt.render(turns: [
                .user([.text("Hi")]),
                .assistant(reasoning: "thinking...", content: "Hello!"),
                .user([.text("Again")])
            ]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
                + "<|message_model|><|content_thinking|>thinking...<|end_message|>"
                + "<|message_model|><|content_text|>Hello!<|end_message|><|content_model_end_sampling|>"
                + "<|message_user|><|content_text|>Again<|end_message|>"
        )
    }

    func testNamedEffortsAndNumericEffortsRenderTheTemplatesNumbers() throws {
        let expected: [(String, String)] = [
            ("low", "0.2"), ("medium", "0.7"), ("high", "0.9"),
            ("xhigh", "0.99"), ("max", "0.99"), ("minimal", "0.1")
        ]
        for (name, rendered) in expected {
            let value = try XCTUnwrap(Inkling2Effort.value(for: name), name)
            XCTAssertEqual(
                try prompt.render(turns: [.user([.text("Hi")])], effort: value),
                "<|message_system|><|content_text|>Thinking effort level: \(rendered)<|end_message|>"
                    + "<|message_user|><|content_text|>Hi<|end_message|>",
                name
            )
        }
        // Zero renders as `0`, not `0.0`: the template interpolates Python's str.
        XCTAssertEqual(
            try prompt.render(turns: [.user([.text("Hi")])], effort: 0),
            "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                + "<|message_user|><|content_text|>Hi<|end_message|>"
        )
        XCTAssertEqual(Inkling2Effort.value(for: "none"), 0)
        XCTAssertNil(Inkling2Effort.value(for: "extreme"))
    }

    func testAnEffortOutsideTheRangeIsRefused() {
        XCTAssertThrowsError(try prompt.render(turns: [.user([.text("Hi")])], effort: 1.0)) { error in
            XCTAssertEqual(error as? Inkling2PromptError, .effortOutOfRange(1.0))
        }
        XCTAssertThrowsError(try prompt.render(turns: [.user([.text("Hi")])], effort: -0.1))
    }

    func testAToolResultNamesItsTool() throws {
        XCTAssertEqual(
            try prompt.render(turns: [.tool(name: "foo", content: [.text("done")])]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_tool|>foo<|content_text|>done<|end_message|>"
        )
        XCTAssertThrowsError(try prompt.render(turns: [.tool(name: "", content: [.text("done")])])) { error in
            XCTAssertEqual(error as? Inkling2PromptError, .emptyToolName)
        }
    }

    func testAnImageRendersAsThePlaceholderPairAndOnlyOnUserOrToolTurns() throws {
        XCTAssertEqual(
            try prompt.render(turns: [.user([.image, .text("What is this?")])]),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                + "<|message_user|><|content_image|><|unused_200054|><|end_message|>"
                + "<|content_text|>What is this?<|end_message|>"
        )
        XCTAssertThrowsError(try prompt.render(parts: [.image], role: "assistant")) { error in
            XCTAssertEqual(error as? Inkling2PromptError, .imageNotSupportedHere("assistant"))
        }
        // No assistant prefix: generation starts right after the last turn.
        XCTAssertEqual(prompt.assistantPrefix, "")
    }

    /// The markers are the tokenizer's own added-token spellings, asserted as
    /// codepoints so a renderer that stripped a tag cannot make a lookalike pass.
    /// Each is ASCII `<`, `|`, the exact name, `|`, `>`.
    func testMarkersAreTheCheckpointsAddedTokenSpellings() {
        let markers = Inkling2PromptMarkers.familyDefault
        func expect(_ marker: String, _ name: String) {
            XCTAssertEqual(marker.unicodeScalars.map(\.value),
                           [0x3C, 0x7C] + name.unicodeScalars.map(\.value) + [0x7C, 0x3E],
                           name)
            XCTAssertEqual(marker, "<" + "|" + name + "|" + ">")
        }
        expect(markers.system, "message_system")
        expect(markers.user, "message_user")
        expect(markers.model, "message_model")
        expect(markers.tool, "message_tool")
        expect(markers.contentText, "content_text")
        expect(markers.contentThinking, "content_thinking")
        expect(markers.contentModelEndSampling, "content_model_end_sampling")
        expect(markers.endMessage, "end_message")
        expect(markers.imagePlaceholder, "unused_200054")
    }
}

// MARK: - Tool path

/// The tool half of the prompt dialect, against goldens rendered by the
/// checkpoint's own template (`Tools/inkling2_tool_goldens.py`), plus the call
/// parsing and freeze rules the template does not decide.
final class Inkling2ToolPromptTests: XCTestCase {

    static let goldenPath = "/tmp/inkling2_tool_goldens.json"

    private func goldens() throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: Self.goldenPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: Self.goldenPath)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("no tool goldens at \(Self.goldenPath): run Tools/inkling2_tool_goldens.py in the reference environment")
        }
        return root
    }

    private func expected(_ root: [String: Any], _ name: String) throws -> String {
        let entry = try XCTUnwrap(root[name] as? [String: Any], name)
        return try XCTUnwrap(entry["rendered"] as? String, name)
    }

    /// A declaration renders as the template's `tool_declare` system message,
    /// with compact sorted JSON and the template's defaults for absent fields.
    func testToolDeclarationsMatchTheTemplate() throws {
        let root = try goldens()
        let prompt = Inkling2Prompt()
        let declared = Inkling2ToolDeclaration(
            name: "read_file",
            description: "Read a file from disk.",
            parameters: [
                "type": .string("object"),
                "properties": .object([
                    "path": .object(["type": .string("string")]),
                    "limit": .object(["type": .string("integer")])
                ]),
                "required": .array([.string("path")])
            ]
        )
        XCTAssertEqual(
            try prompt.render(turns: [.user([.text("hi")])], effort: 0.9, tools: [declared]),
            try expected(root, "declarationOnly")
        )
        // An absent description and absent parameters are still emitted, as ""
        // and {} -- the template writes every key.
        let bare = Inkling2ToolDeclaration(name: "no_args_tool")
        XCTAssertEqual(
            try prompt.render(turns: [.user([.text("hi")])], effort: 0.9, tools: [declared, bare]),
            try expected(root, "declarationDefaults")
        )
    }

    /// A call turn: nested arguments, escapes, several calls, and reasoning
    /// before the call -- each is a separate way for the block to be wrong.
    func testToolCallTurnsMatchTheTemplate() throws {
        let root = try goldens()
        let prompt = Inkling2Prompt()
        let nested = Inkling2ToolCall(
            name: "read_file",
            arguments: [
                "path": .string("a \"quoted\"/file.txt"),
                "options": .object([
                    "limit": .int(3),
                    "strict": .bool(true),
                    "tags": .array([.string("x"), .string("y")])
                ]),
                "note": .string("line\nbreak \u{e9}\u{4e2d}")
            ]
        )
        let turns: [Inkling2Turn] = [
            .user([.text("read it")]),
            .assistant(reasoning: nil, content: nil, calls: [nested])
        ]
        let rendered = try prompt.render(turns: turns, effort: 0.2, tools: [
            Inkling2ToolDeclaration(name: "read_file", description: "Read a file from disk.",
                                    parameters: ["type": .string("object")])
        ])
        // The declaration's parameters differ from the golden's, so compare
        // only from the effort line on: the call is the half under test here.
        let golden = try expected(root, "callWithNestedArguments")
        let mineTail = try XCTUnwrap(rendered.range(of: "<|message_system|><|content_text|>Thinking"))
        let goldenTail = try XCTUnwrap(golden.range(of: "<|message_system|><|content_text|>Thinking"))
        XCTAssertEqual(String(rendered[mineTail.lowerBound...]), String(golden[goldenTail.lowerBound...]))

        // Two calls in one turn, with a text block before them.
        let both: [Inkling2Turn] = [
            .user([.text("read both")]),
            .assistant(reasoning: nil, content: "I'll read both.", calls: [
                Inkling2ToolCall(name: "read_file", arguments: ["path": .string("a.txt")]),
                Inkling2ToolCall(name: "read_file", arguments: ["path": .string("b.txt")])
            ])
        ]
        let bothGolden = try expected(root, "twoCallsInOneTurn")
        let bothMine = try prompt.render(turns: both, effort: 0.9)
        let bothStart = try XCTUnwrap(bothMine.range(of: "<|message_user|>"))
        let bothGoldenStart = try XCTUnwrap(bothGolden.range(of: "<|message_user|>"))
        XCTAssertEqual(String(bothMine[bothStart.lowerBound...]), String(bothGolden[bothGoldenStart.lowerBound...]))

        // Reasoning, then a call: two model messages, and the end-of-sampling
        // marker once.
        let reasoningTurn: [Inkling2Turn] = [
            .user([.text("read it")]),
            .assistant(reasoning: "I should read the file.", content: nil, calls: [
                Inkling2ToolCall(name: "read_file", arguments: ["path": .string("a.txt")])
            ])
        ]
        let reasoningGolden = try expected(root, "reasoningThenCall")
        let reasoningMine = try prompt.render(turns: reasoningTurn, effort: 0.7)
        let reasoningStart = try XCTUnwrap(reasoningMine.range(of: "<|message_user|>"))
        let reasoningGoldenStart = try XCTUnwrap(reasoningGolden.range(of: "<|message_user|>"))
        XCTAssertEqual(
            String(reasoningMine[reasoningStart.lowerBound...]),
            String(reasoningGolden[reasoningGoldenStart.lowerBound...])
        )
    }

    /// A tool result names its tool after the role marker, which is what the
    /// template does and what the model was trained to read.
    func testToolResultsMatchTheTemplate() throws {
        let root = try goldens()
        let prompt = Inkling2Prompt()
        let turns: [Inkling2Turn] = [
            .user([.text("read it")]),
            .assistant(reasoning: nil, content: nil, calls: [
                Inkling2ToolCall(name: "read_file", arguments: ["path": .string("a.txt")])
            ]),
            .tool(name: "read_file", content: [.text("file contents")]),
            .user([.text("summarize")])
        ]
        let rendered = try prompt.render(turns: turns, effort: 0.9)
        let golden = try expected(root, "callThenNamedResult")
        let mineStart = try XCTUnwrap(rendered.range(of: "<|message_user|>"))
        let goldenStart = try XCTUnwrap(golden.range(of: "<|message_user|>"))
        XCTAssertEqual(String(rendered[mineStart.lowerBound...]), String(golden[goldenStart.lowerBound...]))
    }

    // MARK: Calls as the model writes them

    private let markers = Inkling2PromptMarkers.familyDefault

    private func callText(name: String, args: String) -> String {
        markers.model + name + markers.contentInvokeToolJSON
            + "{\"name\":\"\(name)\",\"args\":\(args)}" + markers.endMessage
    }

    func testACompleteCallParses() throws {
        let text = "Reading. " + callText(name: "read_file", args: "{\"path\":\"a.txt\"}")
        let calls = Inkling2ToolDialect.callBlocks(in: text, markers: markers)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "read_file")
        XCTAssertEqual(calls[0].arguments["path"] as? String, "a.txt")
    }

    /// The shape the published checkpoint actually emits: a REASONING block,
    /// then the call block.
    ///
    /// The scan must take ONE message block at a time. Searching for the invoke
    /// marker across the whole text read the reasoning turn's opener as the
    /// call's, gave it a name made of the reasoning sentence, found no call --
    /// and so never froze, which let the model narrate past a call it had
    /// already completed.
    func testACallAfterAReasoningBlockFreezes() throws {
        let reasoning = markers.model + markers.contentThinking + "I should read it."
            + markers.endMessage
        let text = reasoning + callText(name: "read_file", args: "{\"path\":\"a.txt\"}")
        let calls = Inkling2ToolDialect.callBlocks(in: text, markers: markers)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "read_file")
        XCTAssertEqual(calls[0].arguments["path"] as? String, "a.txt")
        XCTAssertTrue(Inkling2ToolDialect.shouldFreeze(text, markers: markers))
        // The reasoning block is prose to the caller, and the call still
        // canonicalizes into the block the shared parser reads.
        let canonical = Inkling2ToolDialect.canonicalized(text, markers: markers)
        XCTAssertTrue(canonical.hasPrefix(reasoning))
        XCTAssertTrue(canonical.contains("<tool_call><function=read_file>"))
    }

    /// The freeze rule: only a COMPLETE call freezes, and the invoke marker
    /// alone does not (ordinary text also ends with `end_message`).
    func testOnlyACompleteCallFreezes() throws {
        XCTAssertFalse(Inkling2ToolDialect.shouldFreeze("plain text", markers: markers))
        XCTAssertFalse(
            Inkling2ToolDialect.shouldFreeze(
                markers.model + "read_file" + markers.contentInvokeToolJSON + "{\"name\":",
                markers: markers
            ),
            "a half-written payload must not freeze"
        )
        XCTAssertTrue(
            Inkling2ToolDialect.shouldFreeze(
                callText(name: "read_file", args: "{\"path\":\"a.txt\"}"), markers: markers
            )
        )
    }

    /// A truncated call is never executed, and it is not silently deleted
    /// either: the bytes stay in the text.
    func testATruncatedCallIsLeftAsText() throws {
        let truncated = markers.model + "read_file" + markers.contentInvokeToolJSON
            + "{\"name\":\"read_file\",\"args\":{\"path\":\"a"
        let calls = Inkling2ToolDialect.callBlocks(in: truncated, markers: markers)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(Inkling2ToolDialect.canonicalized(truncated, markers: markers), truncated)
        // A payload that closes but does not parse is just as unusable.
        let malformed = markers.model + "read_file" + markers.contentInvokeToolJSON
            + "{\"name\":\"read_file\",\"args\":" + markers.endMessage
        XCTAssertTrue(Inkling2ToolDialect.callBlocks(in: malformed, markers: markers).isEmpty)
    }

    /// The name in the payload must agree with the name in the block: the
    /// template writes both from one value, so a disagreement is a corrupted
    /// call, not one to guess at.
    func testACallWhoseNamesDisagreeIsRefused() throws {
        let text = markers.model + "read_file" + markers.contentInvokeToolJSON
            + "{\"name\":\"shell_run\",\"args\":{}}" + markers.endMessage
        XCTAssertTrue(Inkling2ToolDialect.callBlocks(in: text, markers: markers).isEmpty)
    }

    /// Canonicalization keeps the prose around the calls and turns each call
    /// into the block the shared parser reads.
    func testCanonicalizationKeepsTheProseAndRewritesTheCalls() throws {
        let text = "before " + callText(name: "read_file", args: "{\"path\":\"a.txt\"}")
            + " between " + callText(name: "shell_run", args: "{\"command\":\"ls\"}")
            + " after"
        let canonical = Inkling2ToolDialect.canonicalized(text, markers: markers)
        XCTAssertTrue(canonical.hasPrefix("before <tool_call><function=read_file>"))
        XCTAssertTrue(canonical.contains(" between <tool_call><function=shell_run>"))
        XCTAssertTrue(canonical.hasSuffix(" after"))
        let parsed = StreamingToolParser.shared.parseStreamingToolCalls(from: canonical, format: .qwenXML)
        XCTAssertEqual(parsed.calls.map(\.name), ["read_file", "shell_run"])
        XCTAssertEqual(parsed.calls[0].arguments["path"] as? String, "a.txt")
    }

    /// The result turn is the family's own tool message.
    func testAResultTurnIsTheFamilysToolMessage() throws {
        XCTAssertEqual(
            Inkling2ToolDialect.resultTurn(name: "read_file", payload: "contents", markers: markers),
            markers.tool + "read_file" + markers.contentText + "contents" + markers.endMessage
        )
    }
}

// MARK: - Published config

/// Facts about the REAL checkpoint's `config.json`, embedded verbatim so the
/// decode is checked against the file rather than against a fixture that
/// drifted from it. The JSON below is the published file (snapshot
/// `ed40e7da...`) with no fields removed.
final class Inkling2PublishedConfigTests: XCTestCase {

    static let publishedJSON = """
    {
      "dtype": "bfloat16",
      "architectures": [
        "Inkling2LoopForConditionalGeneration"
      ],
      "model_type": "inkling2_loop",
      "eos_token_id": 200006,
      "text_config": {
        "model_type": "inkling2_loop_text",
        "model_max_length": 1048576,
        "torch_dtype": "bfloat16",
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "q_bias": false,
        "o_bias": false,
        "log_scaling_n_floor": 8192,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": true,
        "qk_l2_clip_threshold": 22.627416997969522,
        "rel_l2_clip_threshold": 8.0,
        "local_layer_ids": [
          0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22
        ],
        "sconv_kernel_size": 4,
        "unpadded_vocab_size": 200058,
        "final_logit_softcapping": 30.0,
        "swa_head_dim": 128,
        "swa_num_attention_heads": 24,
        "swa_num_key_value_heads": 8,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": true,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "n_shared_experts": 0,
        "intermediate_size": 1152,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 8.0,
        "use_gate_bias": true,
        "loop_config": {
          "core_start": 6,
          "core_end": 18,
          "num_passes": 3,
          "gate_rank": 64,
          "is_varloop": false
        }
      },
      "vision_config": {
        "model_type": "inkling2_vision",
        "text_hidden_size": 2304,
        "temporal_patch_size": 2,
        "native_patch_size": 20,
        "adapter_merge_size": 2,
        "projection_hidden_dim": 4096,
        "projector_bias": false,
        "hidden_size": 1024,
        "num_hidden_layers": 23,
        "num_attention_heads": 16,
        "mlp_ratio": 4.0,
        "posemb_grid_size": 32,
        "layer_norm_eps": 1e-05
      },
      "mtp_config": {
        "hidden_size": 2304,
        "num_hidden_layers": 3,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9216,
        "max_position_embeddings": 1048576,
        "torch_dtype": "bfloat16",
        "q_bias": false,
        "o_bias": false,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "local_layer_ids": [
          0,
          2
        ],
        "sliding_window_size": 1024,
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [
          5,
          23,
          41,
          48
        ]
      },
      "quantization_config": {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4"
      }
    }
    """

    func published() throws -> Inkling2Config {
        try Inkling2Config.decode(jsonData: Data(Self.publishedJSON.utf8))
    }

    func testThePublishedConfigDecodesToItsGeometry() throws {
        let config = try published()
        XCTAssertEqual(config.hiddenSize, 2304)
        XCTAssertEqual(config.numHiddenLayers, 24)
        XCTAssertEqual(config.numAttentionHeads, 24)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.vocabSize, 201024)
        XCTAssertEqual(config.unpaddedVocabSize, 200058)
        XCTAssertEqual(config.maxPositionEmbeddings, 1_048_576)
        XCTAssertEqual(config.slidingWindowSize, 1024)
        XCTAssertEqual(config.numRoutedExperts, 128)
        XCTAssertEqual(config.numExpertsPerToken, 8)
        XCTAssertEqual(config.moeIntermediateSize, 1152)
        XCTAssertEqual(config.eosTokenID, 200006)
        XCTAssertEqual(config.imageTokenID, 200054)
    }

    /// The plan's headline schedule: 48 visits, eight of them full attention, at
    /// slots 5, 11, 17, 23, 29, 35, 41 and 47.
    func testThePublishedScheduleIsFortyEightVisitsWithEightFull() throws {
        let config = try published()
        let plan = try config.makeVisitPlan()
        XCTAssertEqual(plan.count, 48)
        XCTAssertEqual(config.executedVisitCount, 48)
        XCTAssertEqual(plan.filter(\.isFullAttention).map(\.visit), [5, 11, 17, 23, 29, 35, 41, 47])
        // Full physical layers 5, 11, 17 and 23, and the core repeats 11 and 17 in passes two and three.
        XCTAssertEqual(plan.filter(\.isFullAttention).map(\.physicalLayer), [5, 11, 17, 11, 17, 11, 17, 23])
        // Layers 0...5 once, 6...17 three times, 18...23 once.
        XCTAssertEqual(plan.map(\.physicalLayer).prefix(6), [0, 1, 2, 3, 4, 5])
        XCTAssertEqual(plan.map(\.physicalLayer).prefix(24).suffix(18).filter { $0 < 6 }, [])
        XCTAssertEqual(plan.map(\.physicalLayer).suffix(6), [18, 19, 20, 21, 22, 23])
        // Visit 23 is layer 11 of pass two (6 pre + 6 of pass one + 6 of pass two),
        // 29 is layer 17 of pass two, 35 layer 11 of pass three; 5 and 47 sit outside the loop.
        XCTAssertEqual(plan[23].physicalLayer, 11)
        XCTAssertEqual(plan[23].pass, 2)
        XCTAssertEqual(plan[29].physicalLayer, 17)
        XCTAssertEqual(plan[29].pass, 2)
        XCTAssertEqual(plan[35].physicalLayer, 11)
        XCTAssertEqual(plan[35].pass, 3)
        XCTAssertEqual(plan[5].pass, 0)
        XCTAssertEqual(plan[41].physicalLayer, 17)
        XCTAssertEqual(plan[41].pass, 3)
        XCTAssertEqual(plan[47].physicalLayer, 23)
        XCTAssertEqual(plan[47].pass, 0)
    }

    func testTheTowersGeometryIsValidatedNotExecuted() throws {
        let config = try published()
        let vision = try XCTUnwrap(config.vision)
        XCTAssertEqual(vision.hiddenSize, 1024)
        XCTAssertEqual(vision.numHiddenLayers, 23)
        XCTAssertEqual(vision.projectionHiddenDim, 4096)
        // The span arithmetic from the plan's examples.
        XCTAssertEqual(vision.languagePositions(width: 40, height: 40), 1)
        XCTAssertEqual(vision.towerTokens(width: 40, height: 40), 4)
        XCTAssertEqual(vision.languagePositions(width: 512, height: 512), 169)
        XCTAssertEqual(vision.towerTokens(width: 512, height: 512), 676)
        XCTAssertEqual(vision.languagePositions(width: 1024, height: 1024), 676)
        XCTAssertEqual(vision.towerTokens(width: 1024, height: 1024), 2704)
        XCTAssertEqual(vision.languagePositions(width: 2048, height: 2048), 2704)
        XCTAssertEqual(vision.languagePositions(width: 4096, height: 4096), 10609)
        XCTAssertEqual(vision.towerTokens(width: 4096, height: 4096), 42436)
        // A non-multiple pads up to the next cell.
        XCTAssertEqual(vision.languagePositions(width: 41, height: 80), 4)
    }

    func testTheSidecarSpecIsValidated() throws {
        let mtp = try XCTUnwrap(try published().mtp)
        XCTAssertEqual(mtp.numHiddenLayers, 3)
        XCTAssertEqual(mtp.blockSize, 8)
        XCTAssertEqual(mtp.markovRank, 256)
        XCTAssertEqual(mtp.targetLayerIDs, [5, 23, 41, 48])
        XCTAssertEqual(mtp.localLayerIDs, [0, 2])
    }

    /// The three published figures the memory plan is checked against: 3.66 MiB
    /// of convolution state for 48 visits, and the hybrid FP16 KV total.
    func testThePublishedMemoryFiguresFollowFromTheGeometry() throws {
        let config = try published()
        let plan = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 32_768)
        XCTAssertEqual(plan.convolutionStateBytes, 3_833_856)
        // 8 full visits x 32,768 x 4,096 bytes + rings + convolution.
        let record = 2 * config.numKeyValueHeads * config.headDim * 2
        XCTAssertEqual(plan.fullAttentionCacheBytes, 8 * 32_768 * record)
        XCTAssertEqual(plan.fullAttentionCacheBytes, 1_073_741_824)
        // The rings: 40 visits x 1,024 x 4,096.
        XCTAssertEqual(plan.localRingBytes, 40 * 1024 * record)
        // The loop messages: 12 core layers x 4 messages x 2304 x 4 bytes.
        XCTAssertEqual(plan.loopMessageBytesPerPosition, 12 * 4 * 2304 * 4)
        // The plan's table quotes decimal GB and excludes convolution state:
        // 1.0737 GB of full caches + 0.1678 GB of rings = 1.242 GB at 32K.
        let tableGB = Double(plan.fullAttentionCacheBytes + plan.localRingBytes) / 1e9
        XCTAssertEqual(tableGB, 1.242, accuracy: 0.001)
        // And at the 1M declaration, exactly the table's last row.
        let million = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 1_048_576)
        XCTAssertEqual(
            Double(million.fullAttentionCacheBytes + million.localRingBytes) / 1e9,
            34.528, accuracy: 0.001
        )
        // The naive 48-full-cache figure the table rules out.
        let naive = 48 * 1_048_576 * record
        XCTAssertEqual(Double(naive) / 1e9, 206.158, accuracy: 0.001)
    }

    /// The context a memory budget actually affords, which is what caps the
    /// forward's KV cache. The rings are a fixed term (they never grow with the
    /// context), so what a budget buys is the growing part, and the answer is
    /// the plan's own per-position price rather than a second derivation.
    func testTheAffordableContextFollowsFromTheBudget() throws {
        let config = try published()
        let plan = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 1_048_576)
        // One position is 8 full visits x K+V x 2,048-byte record.
        XCTAssertEqual(plan.bytesPerPosition, 8 * 2 * config.numKeyValueHeads * config.headDim * 2)
        // A 16 GB budget: (16 GiB - 0.168 GB of rings) / 32,768 bytes per
        // position is ~519k positions, i.e. INSIDE the checkpoint's 1M ceiling.
        // The budget is the authority here, not the declaration: an earlier
        // version of this case asserted the ceiling won at 16 GB, which was
        // wrong by the same arithmetic (and the loaded checkpoint's weights are
        // not counted in it either, so the real budget is smaller still).
        let sixteenGB = plan.affordableContextLength(budgetGB: 16)
        XCTAssertEqual(
            sixteenGB,
            Int((Double(16) * 1_073_741_824 - Double(plan.localRingBytes)) / 32_768)
        )
        XCTAssertLessThan(sixteenGB, 1_048_576)
        // Above the ceiling the declaration does win, because the plan has no
        // geometry past it.
        XCTAssertEqual(plan.affordableContextLength(budgetGB: 64), 1_048_576)
        // 1 GiB: (1 GiB - rings) / 32,768, floored at one position.
        let oneGB = plan.affordableContextLength(budgetGB: 1)
        XCTAssertEqual(oneGB, Int((Double(1) * 1_073_741_824 - Double(plan.localRingBytes)) / 32_768))
        // A budget already smaller than the rings buys one position, not zero
        // or a negative -- a cache of nothing cannot run a turn at all.
        XCTAssertEqual(plan.affordableContextLength(budgetGB: 0.01), 1)
        // And the plan is still asked at its own context: the answer never
        // exceeds what the plan was built for.
        let small = Inkling2MemoryPlan.make(config: config, catalog: nil, contextLength: 4_096)
        XCTAssertLessThanOrEqual(small.affordableContextLength(budgetGB: 16), 4_096)
    }
}

// MARK: - Published checkpoint (opt-in)

/// The catalog and memory plan over the REAL checkpoint's shard headers.
///
/// Skipped when the snapshot is absent, like the DeepSeek published cases: it
/// reads the safetensors headers through the Rust engine (19.5 GB of tensors are
/// indexed, not read) and nothing else.
///
/// The Rust summary reports a fused expert bank ONE ENTRY PER EXPERT (the
/// generic engine binds experts one at a time), so `Inkling2WeightCatalog`
/// coalesces those entries back to the checkpoint's own name before
/// classifying. The counts asserted here are of the summary's entries; the
/// catalog's are of the tensors, which is why the two differ by exactly the
/// experts' fan-out. It is the test that keeps the 1,244-name
/// classification honest against the file, which a synthetic fixture cannot do:
/// a role I invented would pass a fixture I also wrote.
final class Inkling2PublishedTensorTests: XCTestCase {

    static let snapshotRelativePath =
        ".cache/huggingface/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/"
        + "ed40e7da625f594fc9c4990f5f66be24873c22d7"

    private func snapshot() throws -> URL {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Self.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(path.path)")
        }
        return path
    }

    /// The main file's tensors, as the engine indexes them.
    private func mainTensors() throws -> (directory: URL, refs: [Inkling2TensorRef], config: Inkling2Config) {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        return (directory, summary.tensors.map { Inkling2TensorRef(metadata: $0) }, config)
    }

    /// The sidecar's header, read directly: `mtp.safetensors` is not in the main
    /// index, so the engine never sees it and the DSpark path will have to open
    /// it by name.
    static func sidecarTensors(_ directory: URL) throws -> [Inkling2TensorRef] {
        let url = directory.appendingPathComponent("mtp.safetensors")
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw Inkling2WeightError.unsupportedDtypes(["mtp.safetensors: no header length"])
        }
        var headerLength: UInt64 = 0
        for (index, byte) in lengthData.enumerated() {
            headerLength |= UInt64(byte) << (8 * UInt64(index))
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              let object = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw Inkling2WeightError.unsupportedDtypes(["mtp.safetensors: unreadable header"])
        }
        return object.compactMap { name, value in
            guard name != "__metadata__",
                  let entry = value as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int] else { return nil }
            return Inkling2TensorRef(name: name, dtype: dtype, shape: shape)
        }.sorted { $0.name < $1.name }
    }

    func testEveryPublishedTensorHasARoleAndEveryRequiredRoleIsPresent() throws {
        let (directory, refs, config) = try mainTensors()
        let sidecar = try Self.sidecarTensors(directory)
        let catalog = Inkling2WeightCatalog(
            mainFile: refs, sidecar: sidecar, layerCount: config.numHiddenLayers
        )
        XCTAssertTrue(catalog.unconsumed.isEmpty, "unplaced: \(catalog.unconsumed.prefix(10))")
        XCTAssertTrue(catalog.duplicateRoles.isEmpty, "duplicates: \(catalog.duplicateRoles.prefix(5))")
        let problems = catalog.validate(config: config)
        XCTAssertTrue(problems.isEmpty, "catalog problems: \(problems.prefix(10))")
        // The plan's evidence lock: 1,244 main tensors and 54 sidecar ones.
        // The summary reports one entry per expert of a fused bank, so its count
        // is the tensors' fan-out and not 1,244; the catalog coalesces them back.
        XCTAssertGreaterThan(refs.count, 1244)
        XCTAssertEqual(sidecar.count, 54)
        // One slot per checkpoint tensor once the split banks are coalesced.
        XCTAssertEqual(catalog.slots.count, 1244 + 54)
        // The vision tower's block count comes from the tensors themselves.
        XCTAssertEqual(catalog.layers(of: .visionBlockLn1Weight, scope: .vision).count, config.vision?.numHiddenLayers)
        XCTAssertEqual(catalog.layers(of: .draftQProj, scope: .sidecar), [0, 1, 2])
    }

    /// The published expert layouts: NVFP4 on layers 0...19, MXFP8 on 20...23,
    /// with the strides the stored shapes give.
    func testThePublishedExpertLayoutsAreWhatTheStridesSay() throws {
        let (_, refs, config) = try mainTensors()
        let catalog = Inkling2WeightCatalog(mainFile: refs, layerCount: config.numHiddenLayers)
        for layer in 0..<config.numHiddenLayers {
            let w13 = try XCTUnwrap(catalog.tensor(.expertW13, layer: layer))
            let w13Scale = try XCTUnwrap(catalog.tensor(.expertW13Scale, layer: layer))
            let w2 = try XCTUnwrap(catalog.tensor(.expertW2, layer: layer))
            let w2Scale = try XCTUnwrap(catalog.tensor(.expertW2Scale, layer: layer))
            let layout = try Inkling2ExpertQuantLayout.infer(
                w13: w13, w13Scale: w13Scale, w2: w2, w2Scale: w2Scale, config: config
            ).get()
            XCTAssertEqual(layout, layer < 20 ? .nvfp4 : .mxfp8, "layer \(layer)")
            XCTAssertEqual(w13.shape[0], 128)
            XCTAssertEqual(w13.shape[1], 2 * 1152)
            XCTAssertEqual(w13.shape[2], layout.packedInputWidth(2304))
            XCTAssertEqual(w13Scale.shape[2], layout.scaleInputWidth(2304))
            XCTAssertEqual(w2.shape[2], layout.packedInputWidth(1152))
            XCTAssertEqual(w2Scale.shape[2], layout.scaleInputWidth(1152))
        }
        // The published outer-scale denominator is 2688, which the plan's
        // amax rule (`blockScale * (expertAmax / 2688)`) is written around.
        let denominator = try XCTUnwrap(catalog.tensor(.expertW13OuterScaleDenominator, layer: 0))
        XCTAssertEqual(denominator.shape, [1])
    }

    /// The packed query tensors: 23 attention-input reads (layer 0 has none),
    /// three rows on the layer that follows a full-attention layer 5/11/17/23
    /// and one elsewhere, plus 24 MLP reads and two 24-row cross-pass tensors.
    func testThePublishedPackedQueriesUnpack() throws {
        let (_, refs, config) = try mainTensors()
        let catalog = Inkling2WeightCatalog(mainFile: refs, layerCount: config.numHiddenLayers)
        XCTAssertEqual(catalog.layers(of: .attnResAttnQuery), Array(0..<23))
        XCTAssertEqual(catalog.layers(of: .attnResMlpQuery), Array(0..<24))
        var threeWide: [Int] = []
        for index in 0..<23 {
            let ref = try XCTUnwrap(catalog.tensor(.attnResAttnQuery, layer: index))
            let layer = Inkling2QueryUnpacking.attentionInputLayer(forQueryIndex: index)
            if ref.shape.count == 2 {
                XCTAssertEqual(ref.shape, [3, 2304], "layer \(layer) reads three block queries")
                threeWide.append(layer)
            } else {
                XCTAssertEqual(ref.shape, [2304], "layer \(layer) reads one")
            }
        }
        // The query is indexed by the layer whose ATTENTION input it reads, and
        // a full layer publishes three; the published three-wide set is what
        // says the mapping is layer-1 and not layer.
        XCTAssertEqual(threeWide.sorted(), [5, 11, 17, 23])
        XCTAssertEqual(try XCTUnwrap(catalog.tensor(.crossPassQuery)).shape, [24, 2304])
        XCTAssertEqual(try XCTUnwrap(catalog.tensor(.crossPassKVQuery)).shape, [24, 2304])
    }

    /// The weight source reads the bytes the header names: a row by index must
    /// equal the same slice of the whole-matrix read, and both must equal values
    /// taken straight from the file by an independent reader.
    func testTheWeightSourceReadsTheRowsTheHeaderNames() throws {
        let (directory, refs, config) = try mainTensors()
        let catalog = Inkling2WeightCatalog(mainFile: refs, layerCount: config.numHiddenLayers)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let engine = try DynaMoeEngine(filePath: directory.path)
        let summary = try engine.getSummary()
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards where shard.length > 0 {
            guard let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  shard.length <= UInt64(device.maxBufferLength) else { continue }
            buffers[shard.index] = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length),
                options: .storageModeShared, deallocator: nil
            )
        }
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: buffers)

        // Row 200010 of the embedding, from the file itself (numpy):
        // [-0.97265625, -0.81640625, 0.27929688, -0.3125]
        let whole = try source.matrix(.tokenEmbedding, layer: nil)
        let byRow = try source.row(.tokenEmbedding, layer: nil, index: 200010)
        XCTAssertEqual(Array(whole[(200010 * 2304)..<(200010 * 2304 + 4)]), [-0.97265625, -0.81640625, 0.27929688, -0.3125])
        XCTAssertEqual(Array(byRow.prefix(4)), Array(whole[(200010 * 2304)..<(200010 * 2304 + 4)]))
        // A dense projection's first row, from the file: wq_du[0][:4].
        let wq = try source.matrix(.attnWqDu, layer: 0)
        XCTAssertEqual(Array(wq.prefix(4)), [0.00384521484375, 0.01202392578125, -0.0123291015625, 0.01446533203125])
    }

    /// The text tower's dense weights against the plan's ~3.86 GB and the two
    /// expert layouts against its ~15.67 GB, from the real headers. The plan
    /// quotes decimal GB over the whole main file, which also holds the vision
    /// tower, so the dense figure is a floor here rather than an equality.
    func testThePublishedByteSplitMatchesThePlan() throws {
        let (_, refs, config) = try mainTensors()
        let catalog = Inkling2WeightCatalog(mainFile: refs, layerCount: config.numHiddenLayers)
        let expertGB = Double(catalog.expertBankBytes) / 1e9
        XCTAssertEqual(expertGB, 15.67, accuracy: 0.1)
        let visionGB = Double(catalog.slots.filter { $0.scope == .vision }
            .reduce(0) { $0 + Inkling2WeightCatalog.byteCount($1.ref) }) / 1e9
        XCTAssertEqual(visionGB, 0.637, accuracy: 0.01, "the 23-layer tower plus its 4,096-wide projector")
        // Everything but the experts: the plan's ~3.86 GB, which INCLUDES the tower.
        let denseGB = Double(catalog.nonExpertBytes) / 1e9
        XCTAssertEqual(denseGB, 3.86, accuracy: 0.02)
        // Every byte of the main file is one of the two classes.
        let total = Double(catalog.nonExpertBytes + catalog.expertBankBytes) / 1e9
        XCTAssertEqual(total, 19.526, accuracy: 0.01)
    }
}

// MARK: - Family identity

/// The interception: `ModelConfig` has to resolve this family to its own
/// architecture BEFORE the MoE topology rules, and discovery has to name it.
/// Both are pure and run with no model.
final class Inkling2IdentityTests: XCTestCase {

    /// `ModelConfig` decodes from a FILE, so a test writes one.
    private static func decode(_ json: String) -> ModelConfig? {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-identity-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        guard (try? Data(json.utf8).write(to: url)) != nil else { return nil }
        return ModelConfig.load(fromFilePath: url.path)
    }

    /// The published config, as `ModelConfig` decodes it. 128 experts over 24
    /// layers would otherwise fall through to Standard MoE (or Qwen 3.8 Flash
    /// Next at >= 512), which is the wrong forward pass.
    func testThePublishedConfigResolvesToItsOwnArchitecture() throws {
        let config = try XCTUnwrap(Self.decode(Inkling2PublishedConfigTests.publishedJSON))
        XCTAssertTrue(config.isInkling2Model)
        XCTAssertEqual(config.resolveArchitectureType(summary: nil), .inkling2Mini)
        XCTAssertTrue(config.resolveArchitectureType(summary: nil).isInkling2)
        XCTAssertFalse(config.resolveArchitectureType(summary: nil).isDeepSeekV41)
    }

    /// A nested text tower is enough: the reference nests `model_type` there.
    func testTheNestedTextTowerIdentifiesTheFamily() throws {
        let json = """
        {"architectures": ["SomethingElse"], "model_type": "wrapper",
         "text_config": {"model_type": "inkling2_loop_text", "hidden_size": 8,
                         "num_hidden_layers": 1, "num_attention_heads": 1,
                         "num_key_value_heads": 1, "vocab_size": 8}}
        """
        let config = try XCTUnwrap(Self.decode(json))
        XCTAssertTrue(config.isInkling2Model)
        XCTAssertEqual(config.resolveArchitectureType(summary: nil), .inkling2Mini)
    }

    /// The architecture class alone identifies it, with no `model_type`.
    func testTheArchitectureClassIdentifiesTheFamily() {
        XCTAssertTrue(ModelConfig.isInkling2Identity(
            modelType: nil, textModelType: nil,
            architectures: ["Inkling2LoopForConditionalGeneration"]
        ))
        XCTAssertFalse(ModelConfig.isInkling2Identity(
            modelType: "llama", textModelType: nil, architectures: ["LlamaForCausalLM"]
        ))
        XCTAssertFalse(ModelConfig.isInkling2Identity(
            modelType: "inkling_other", textModelType: nil, architectures: nil
        ), "a family name has to be the Inkling 2 prefix, not merely Inkling")
    }

    /// The runtime serves text turns (the published-checkpoint turn case in
    /// `Inkling2HostForwardTests` runs one), and the discovery label says what
    /// is still missing through the gate ledger, not through this flag.
    func testTheRuntimeServesTextTurns() {
        XCTAssertTrue(Inkling2Availability.isRuntimeAvailable)
        XCTAssertFalse(Inkling2Conformance.unverifiedSummary.isEmpty)
        // The gates that ARE pinned by this phase, against the reference.
        XCTAssertTrue(Inkling2Conformance.promptText.isVerified)
        XCTAssertTrue(Inkling2Conformance.router.isVerified)
        XCTAssertTrue(Inkling2Conformance.expertLayout.isVerified)
        XCTAssertTrue(Inkling2Conformance.softcappedSwiGLU.isVerified)
        XCTAssertTrue(Inkling2Conformance.loopSchedule.isVerified)
    }

    /// Discovery names the family from the checkpoint's own config and reads its
    /// two expert layouts out of `hf_quant_config.json`, so the model row is not
    /// labelled a plain "Mixture of Experts" in "MODELOPT".
    func testDiscoveryNamesTheFamilyAndItsQuantization() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("inkling2-discovery-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try Data(Inkling2PublishedConfigTests.publishedJSON.utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
        try Data().write(to: dir.appendingPathComponent("model.safetensors"))
        let quant = """
        {"model": {"parameter_groups": [
          {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "tensors": ["a"]},
          {"compute_dtype": "nvfp4", "output_dtype": "uint8", "tensors": ["b"]},
          {"compute_dtype": "mxfp8", "output_dtype": "float8_e4m3fn", "tensors": ["c"]}
        ]}}
        """
        try Data(quant.utf8).write(to: dir.appendingPathComponent("hf_quant_config.json"))

        let model = try XCTUnwrap(
            LocalModelManager.shared.inspectModelDirectory(
                dirUrl: dir, customRepoId: "thinkingmachines/Inkling-2-Mini"
            ),
            "a directory with config + weights must be discovered"
        )
        XCTAssertEqual(model.architectureName, "Inkling 2 Mini (Loop / RelPos MoE)")
        XCTAssertEqual(model.quantization, "BF16 + MXFP8 / NVFP4 experts")
        XCTAssertTrue(model.isMoE)
        XCTAssertEqual(model.expertCount, 128)
        XCTAssertEqual(model.rawModelType, "inkling2_loop")
        XCTAssertTrue(model.supportsThinking)
    }
}

// MARK: - Packed expert arithmetic

final class Inkling2PackedExpertTests: XCTestCase {

    /// A two-row NVFP4 matrix, hand-built: row 0 is `[1.0, -2.0, 3.0, 0.0]` at
    /// one E4M3 scale of 1.0, row 1 is all zeros at a scale of 0.5.
    func nvfp4Matrix() -> Inkling2PackedExpertMatrix {
        let row0 = [Inkling2Element.packE2M1Pair(low: 1.0, high: -2.0),
                    Inkling2Element.packE2M1Pair(low: 3.0, high: 0.0)]
        let row1 = [UInt8](repeating: 0, count: 2)
        return Inkling2PackedExpertMatrix(
            layout: .nvfp4,
            weight: row0 + row1,
            weightRowStride: 2,
            scales: [Inkling2Element.encodeE4M3(1.0), Inkling2Element.encodeE4M3(0.5)],
            scaleRowStride: 1,
            rows: 2,
            logicalColumns: 4,
            expertAmax: 2688,
            outerScaleDenominator: 2688
        )
    }

    func testNVFP4ValuesDecodeWithScaleAndAmax() {
        let matrix = nvfp4Matrix()
        XCTAssertEqual(matrix.value(row: 0, column: 0), 1.0)
        XCTAssertEqual(matrix.value(row: 0, column: 1), -2.0)
        XCTAssertEqual(matrix.value(row: 0, column: 2), 3.0)
        XCTAssertEqual(matrix.value(row: 0, column: 3), 0.0)
        XCTAssertEqual(matrix.value(row: 1, column: 0), 0.0)
        // Out of range reads are zero rather than a trap or a wrapped index.
        XCTAssertEqual(matrix.value(row: 2, column: 0), 0)
        XCTAssertEqual(matrix.value(row: 0, column: 4), 0)
    }

    /// The amax/denominator pair multiplies in: with amax half the denominator,
    /// every value halves.
    func testTheAmaxDenominatorPairScalesTheBank() {
        var matrix = nvfp4Matrix()
        matrix = Inkling2PackedExpertMatrix(
            layout: matrix.layout, weight: matrix.weight, weightRowStride: matrix.weightRowStride,
            scales: matrix.scales, scaleRowStride: matrix.scaleRowStride,
            rows: matrix.rows, logicalColumns: matrix.logicalColumns,
            expertAmax: 1344, outerScaleDenominator: 2688
        )
        XCTAssertEqual(matrix.value(row: 0, column: 0), 0.5)
        XCTAssertEqual(matrix.value(row: 0, column: 1), -1.0)
    }

    func testMXFP8ValuesDecodeWithTheirE8M0Scale() {
        let matrix = Inkling2PackedExpertMatrix(
            layout: .mxfp8,
            weight: [Inkling2Element.encodeE4M3(1.5), Inkling2Element.encodeE4M3(-0.25)],
            weightRowStride: 2,
            scales: [Inkling2Element.encodeE8M0(4)],
            scaleRowStride: 1,
            rows: 1,
            logicalColumns: 2
        )
        XCTAssertEqual(matrix.value(row: 0, column: 0), 6.0)
        XCTAssertEqual(matrix.value(row: 0, column: 1), -1.0)
    }

    /// The HF-reference profile rounds each dequantized weight to BF16 before
    /// projecting, which is what Transformers' dense fallback does.
    func testTheReferenceProfileProjectsInBf16() {
        let matrix = nvfp4Matrix()
        let row = matrix.bf16Row(0)
        XCTAssertEqual(row, [1.0, -2.0, 3.0, 0.0])
        XCTAssertEqual(matrix.project(row: 0, input: [1, 1, 1, 1]), 2.0)
        XCTAssertEqual(matrix.project(row: 0, input: [0.5, 0, 0, 0]), 0.5)
        // A short input is read as far as it goes, never past its end.
        XCTAssertEqual(matrix.project(row: 0, input: [1]), 1.0)
    }

    func testBF16RoundingUsesSevenExplicitMantissaBits() {
        // BF16 is one sign, eight exponent and seven mantissa bits, so the ulp
        // at 1.0 is 2^-7.
        XCTAssertEqual(Inkling2Element.roundToBF16(1 + 1.0 / 128.0), 1 + 1.0 / 128.0)
        // 1 + 2^-8 is exactly halfway -> ties to even -> 1.0.
        XCTAssertEqual(Inkling2Element.roundToBF16(1 + 1.0 / 256.0), 1.0)
        // 1 + 3 * 2^-8 is halfway between one and two ulps -> ties to even -> two.
        XCTAssertEqual(Inkling2Element.roundToBF16(1 + 3.0 / 256.0), 1 + 2.0 / 128.0)
        // Past the midpoint it rounds up.
        XCTAssertEqual(Inkling2Element.roundToBF16(1 + 1.6 / 256.0), 1 + 1.0 / 128.0)
        XCTAssertEqual(Inkling2Element.roundToBF16(-2.0), -2.0)
    }

    /// The fused bank's gate and up are interleaved by PARITY, never two halves.
    func testGateAndUpAreEvenAndOddRows() {
        let matrix = nvfp4Matrix()
        XCTAssertEqual(matrix.gateUpRow(0).gate, 0)
        XCTAssertEqual(matrix.gateUpRow(0).up, 1)
        XCTAssertEqual(matrix.gateUpRow(5).gate, 10)
        XCTAssertEqual(matrix.gateUpRow(5).up, 11)
    }
}

// MARK: - Host forward against the reference (opt-in)

/// The host graph against goldens dumped from the PINNED Transformers
/// implementation on the published checkpoint.
///
/// Opt-in, like the DeepSeek published cases, and for the same reason: it reads
/// the whole checkpoint. The goldens are produced by `Tools/inkling2_goldens.py`,
/// which runs the reference with forward hooks -- so the expected values are the
/// reference's own arithmetic, not a second hand-written model. Skipped when the
/// dump is absent.
///
/// The bar is the plan's: `abs(actual - reference) <= 5e-4 + 5e-4 * abs(reference)`
/// per logit, and the selected expert ids must match exactly. A stage output is
/// held to a relative bound as a diagnostic, because a late stage can be within
/// tolerance on the logits while an early one is not.
final class Inkling2HostForwardTests: XCTestCase {

    static let dumpPath = "/tmp/inkling2_host_goldens.json"

    /// The engine owns the shard mappings, and the zero-copy buffers point into
    /// them: letting it go out of scope unmaps the weights under the runtime
    /// (AGENTS.md records the SIGSEGV this produces), so the test holds it.
    private var engine: DynaMoeEngine?

    private func dump() throws -> [String: Any] {
        guard FileManager.default.fileExists(atPath: Self.dumpPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: Self.dumpPath)),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("no reference dump at \(Self.dumpPath): run Tools/inkling2_goldens.py")
        }
        return object
    }

    private func snapshot() throws -> URL {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(path.path)")
        }
        return path
    }

    /// Builds the host forward over the real snapshot: the engine's shard
    /// mappings, the catalog and the snapshot weight source. The shard buffers
    /// are zero-copy views of the engine's own mmap addresses, which is what
    /// the app's Metal layer does with `MTLBuffer(bytesNoCopy:)`.
    private func makeForward() throws -> Inkling2CPUForward {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let summary = try engine.getSummary()
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            guard shard.length <= UInt64(device.maxBufferLength) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length), options: .storageModeShared, deallocator: nil
            ) {
                buffers[shard.index] = buffer
            }
        }
        let source = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        return try Inkling2CPUForward(config: config, weights: source)
    }

    /// A follow-up turn that resumes its prefix must answer exactly what a
    /// fresh full prefill of the same prompt answers -- the Phase 3 gate, and
    /// the only thing that makes reuse safe rather than merely faster.
    ///
    /// The prompt is the first turn's own prompt extended by what it generated,
    /// which is the shape a real conversation produces: the executor holds the
    /// prompt plus the forwarded continuation, and the next prompt starts with
    /// both.
    func testAFollowUpTurnResumesItsPrefixAndMatchesAFreshPrefill() throws {
        let forward = try makeForward()
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: try snapshot().appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there").map { Int($0) }
        let settings = Inkling2TurnRunner.Settings(
            temperature: 0, topP: 1, minP: 0, topK: 1, maxTokens: 2,
            eosTokenIds: [UInt32(clamping: forward.config.eosTokenID)]
        )
        func sample(_ logits: [Float], _ soFar: [UInt32]) -> UInt32 {
            var scratch = logits
            return scratch.withUnsafeMutableBufferPointer { buffer in
                InferenceEngine.sampleNextToken(
                    logits: buffer.baseAddress!, vocabSize: buffer.count,
                    contextTokens: soFar, temperature: 0, topP: 1, minP: 0, topK: 1,
                    repetitionPenalty: 1, presencePenalty: 0,
                    eosTokenIds: settings.eosTokenIds
                )
            }
        }

        // Turn one: nothing resident, so the whole prompt prefills.
        forward.reset()
        let first = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: ids, settings: settings,
            sample: sample, decode: { $0.map(String.init).joined() }
        )
        XCTAssertGreaterThan(first.storedPromptTokens, 0)

        // The continuation the turn generated and FORWARDED is what the cache
        // holds past the prompt; a sampled-but-unforwarded token is not there.
        let followUp = ids + first.generatedTokenIds.dropLast().map(Int.init)


        // Turn two: the same executor, so it may resume.
        let resumed = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: followUp, prefixTokens: followUp.count - 1,
            settings: settings, sample: sample, decode: { $0.map(String.init).joined() }
        )
        XCTAssertGreaterThan(resumed.prefixTokensReused, 0, "the follow-up must resume a prefix")

        // The control: a fresh executor prefills the whole prompt.
        let freshForward = try makeForward()
        let fresh = try Inkling2TurnRunner.run(
            forward: freshForward, promptTokenIds: followUp, settings: settings,
            sample: sample, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(fresh.prefixTokensReused, 0)
        XCTAssertEqual(
            resumed.generatedTokenIds, fresh.generatedTokenIds,
            "a resumed turn must answer what a full prefill answers"
        )
    }

    /// A cancelled turn leaves the cache holding exactly its committed prefix,
    /// and a resumed turn from that point answers what a fresh prefill answers.
    ///
    /// The cancellation lands during PREFILL, so the interesting number is
    /// `storedPromptTokens` -- the positions actually forwarded -- rather than
    /// the prompt length the turn was given.
    func testACancelledTurnResumesExactlyItsCommittedPrefix() throws {
        let forward = try makeForward()
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: try snapshot().appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there").map { Int($0) }
        let settings = Inkling2TurnRunner.Settings(
            temperature: 0, topP: 1, minP: 0, topK: 1, maxTokens: 2,
            eosTokenIds: [UInt32(clamping: forward.config.eosTokenID)]
        )
        func sample(_ logits: [Float], _ soFar: [UInt32]) -> UInt32 {
            var scratch = logits
            return scratch.withUnsafeMutableBufferPointer { buffer in
                InferenceEngine.sampleNextToken(
                    logits: buffer.baseAddress!, vocabSize: buffer.count,
                    contextTokens: soFar, temperature: 0, topP: 1, minP: 0, topK: 1,
                    repetitionPenalty: 1, presencePenalty: 0,
                    eosTokenIds: settings.eosTokenIds
                )
            }
        }

        // Cancelled after the first prompt position has run.
        forward.reset()
        var checks = 0
        let cancelled = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: ids, settings: settings,
            sample: sample, decode: { $0.map(String.init).joined() },
            isCancelled: { checks += 1; return checks > 1 }
        )
        XCTAssertTrue(cancelled.cancelled)
        XCTAssertEqual(cancelled.storedPromptTokens, 1, "one position ran before the cancel")
        XCTAssertEqual(forward.sequence.tokenHistory, Array(ids.prefix(1)))
        XCTAssertTrue(forward.sequence.isConsistent, "a cancelled turn leaves the visits agreeing")

        // Resumed from exactly that prefix on the same executor.
        let resumed = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: ids, prefixTokens: cancelled.storedPromptTokens,
            settings: settings, sample: sample, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(resumed.prefixTokensReused, cancelled.storedPromptTokens)

        let freshForward = try makeForward()
        let fresh = try Inkling2TurnRunner.run(
            forward: freshForward, promptTokenIds: ids, settings: settings,
            sample: sample, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(
            resumed.generatedTokenIds, fresh.generatedTokenIds,
            "a turn resumed from a cancelled one must answer what a full prefill answers"
        )
    }

    /// A visit runner is a SUBSTITUTION, not a hook beside the work: the loop
    /// must run the installed one for every visit of a token and must produce
    /// the same logits when it simply delegates to the host graph.
    ///
    /// The delegate is the host's own `runVisit`, so a runner that was called
    /// but ignored, called twice, or called with the wrong pass would move the
    /// logits or the state -- the comparison is bit-exact.
    func testAVisitRunnerReplacesEveryVisit() throws {
        let forward = try makeForward()
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: try snapshot().appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: "Hi there").map { UInt32($0) }

        var baseline: [[Float]] = []
        forward.reset()
        for (position, id) in ids.enumerated() {
            baseline.append(try forward.forwardToken(token: id, position: position))
        }

        var calls: [(layer: Int, pass: Int, position: Int)] = []
        forward.reset()
        forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            calls.append((layer, pass, position))
            return try forward.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position
            )
        }
        var substituted: [[Float]] = []
        for (position, id) in ids.enumerated() {
            substituted.append(try forward.forwardToken(token: id, position: position))
        }
        forward.visitRunner = nil

        // One token runs the whole 48-visit schedule; two tokens, twice that.
        XCTAssertEqual(calls.count, ids.count * forward.plan.count)
        XCTAssertEqual(Set(calls.map(\.layer)).count, forward.config.numHiddenLayers)
        // Every pass of the looped core is visited, not only the first.
        let corePasses = Set(
            calls.filter { forward.isCore($0.layer) }.map { $0.pass }
        )
        XCTAssertEqual(corePasses, Set(0..<forward.config.numPasses))
        XCTAssertEqual(substituted, baseline, "a delegating visit runner must not change the logits")
    }

    private func relativeError(_ actual: [Float], _ reference: [Float]) -> Float {
        guard actual.count == reference.count, !reference.isEmpty else { return .infinity }
        let scale = reference.reduce(Float(0)) { max($0, abs($1)) }
        var worst: Float = 0
        for (a, r) in zip(actual, reference) {
            worst = max(worst, abs(a - r))
        }
        return scale > 0 ? worst / scale : worst
    }

    /// Every visit's output against the reference's, in execution order, then
    /// the final logits. The first failing stage is reported by name, because
    /// "the logits differ" says nothing about where.
    func testTheHostForwardMatchesTheReferenceStageByStage() throws {
        let dumped = try dump()
        let ids = try XCTUnwrap(dumped["ids"] as? [Int])
        let referenceLogits = (dumped["logits"] as? [NSNumber])?.map(\.floatValue) ?? []
        XCTAssertEqual(referenceLogits.count, 200058)
        let referenceVisits = (dumped["visits"] as? [[String: Any]]) ?? []
        XCTAssertEqual(referenceVisits.count, 48)

        let forward = try makeForward()
        // The golden holds one entry per VISIT, for the prompt's LAST position
        // (the reference hook takes `[-1, -1]`), so the capture keeps only the
        // last position's output of each visit: 16 positions x 48 visits would
        // otherwise be 768 entries against 48.
        var actualVisits: [[Float]] = []
        var actualAttention: [[Float]] = []
        var actualMixture: [[Float]] = []
        var internals: [String: [Float]] = [:]
        // A ring of ONE position's visits: the golden holds the last position's
        // outputs, one entry per visit. `count - 48` is not that ring -- once
        // full it is always 0, which kept visit 47's output in slot 0 and
        // compared it against visit 0's reference.
        var ringIndex = 0
        forward.onInternal = { visit, name, value in
            // Keyed by visit: a name alone is overwritten by every later visit,
            // which once made layer 23's value look like layer 0's. `bias_head0`
            // is kept for EVERY visit because the reference dump has it per
            // visit and it is the value the relative-bias gate is checked with.
            // `bias_head0` and the final pass's per-channel messages are the
            // values a reference dump names exactly, so they are kept for every
            // visit; everything else is kept for visit 0 only.
            if visit == 0 || name == "bias_head0" || name.hasPrefix("message") {
                internals["\(visit):\(name)"] = value
            }
        }
        forward.onVisitComplete = { _, _, _, output in
            if actualVisits.count >= 48 {
                actualVisits[ringIndex] = output
            } else {
                actualVisits.append(output)
            }
            ringIndex = (ringIndex + 1) % 48
        }
        forward.onVisitSubStages = { _, attention, mixture in
            // Fires BEFORE `onVisitComplete` for the same visit, so the ring
            // index still names this visit's slot.
            if actualAttention.count >= 48 {
                actualAttention[ringIndex] = attention
                actualMixture[ringIndex] = mixture
            } else {
                actualAttention.append(attention)
                actualMixture.append(mixture)
            }
        }
        var logits: [Float] = []
        for (position, id) in ids.enumerated() {
            logits = try forward.forwardToken(token: UInt32(id), position: position)
        }
        // Written for offline comparison: one -O run of this test is minutes, so
        // the numbers are kept rather than only asserted on.
        let actual: [String: Any] = [
            "visits": actualVisits.map { $0.map { Double($0) } },
            "attention": actualAttention.map { $0.map { Double($0) } },
            "mixture": actualMixture.map { $0.map { Double($0) } },
            "logits": logits.map { Double($0) },
            "internals": internals.mapValues { $0.map { Double($0) } }
        ]
        if let data = try? JSONSerialization.data(withJSONObject: actual) {
            try? data.write(to: URL(fileURLWithPath: "/tmp/inkling2_actual.json"))
        }

        // Stage by stage: the first visit that leaves the bound is the one to
        // report, with its layer and pass.
        //
        // The two lists are in DIFFERENT orders and must be regrouped before
        // they are compared. The dump is layer-major (`for layer: for its
        // calls`), while a forward captures in execution order (pass-major for
        // the loop core), so entry 7 of one is layer 7 pass 0 and entry 7 of
        // the other is layer 6 pass 1.
        XCTAssertEqual(actualVisits.count, referenceVisits.count)
        var referenceByVisit: [Int: [Float]] = [:]
        var seenPerLayer: [Int: Int] = [:]
        for reference in referenceVisits {
            let layer = reference["physicalLayer"] as? Int ?? -1
            let pass = seenPerLayer[layer, default: 0]
            seenPerLayer[layer] = pass + 1
            referenceByVisit[try forward.visitIndex(layer: layer, pass: pass)] =
                (reference["output"] as? [NSNumber])?.map(\.floatValue) ?? []
        }
        for (slot, plan) in forward.plan.enumerated() {
            guard let expected = referenceByVisit[plan.cacheSlot] else { continue }
            let error = relativeError(actualVisits[slot], expected)
            // The bar is measured, not the plan's elementwise one: the dump was
            // taken in BF16, and the REFERENCE's own BF16 run differs from its
            // FP32 run by 1.9e-2 mean / 6.4e-2 max on these visits, while this
            // FP32 forward differs from the FP32 reference by 3.2e-3 mean and
            // matches the reference's noise at worst. A tighter bar would be
            // asserting the dump's rounding.
            if error > 7e-2 {
                XCTFail("visit \(slot) (layer \(plan.physicalLayer) pass \(plan.pass)) is \(error) relative from the reference")
                return
            }
        }

        // The relative-position bias, head by head-0 row, against the
        // reference's OWN captured bias (`/tmp/dump_bias.py`, written in FP32).
        // It is the one intermediate both sides can name exactly, and it is
        // what the `relativeBias` gate is closed on: the bank mix, the distance
        // gather and the out-of-extent zeroing all show up in this vector, and
        // a wrong L2 clip or log scale would not.
        let biasPath = "/tmp/inkling2_bias_ref.json"
        if FileManager.default.fileExists(atPath: biasPath),
           let biasData = try? Data(contentsOf: URL(fileURLWithPath: biasPath)),
           let biasRoot = try? JSONSerialization.jsonObject(with: biasData) as? [String: Any],
           let biasVisits = biasRoot["visits"] as? [[String: Any]] {
            var referenceBiasByVisit: [Int: [Float]] = [:]
            var seenPerLayerBias: [Int: Int] = [:]
            for entry in biasVisits {
                let layer = entry["physicalLayer"] as? Int ?? -1
                let pass = seenPerLayerBias[layer, default: 0]
                seenPerLayerBias[layer] = pass + 1
                // The hook captured `output[-1]`, so the entry is
                // `[heads][query][key]`; this forward records head 0 at the LAST
                // query row, i.e. `bias[0][last]`, every key row.
                // JSONSerialization gives nested `[Any]` of NSNumber, so each
                // level is bridged explicitly: a direct `as? [[[NSNumber]]]`
                // cast fails and the entry would be skipped in silence, which
                // reads exactly like a mismatch downstream.
                func floats(_ value: Any?) -> [Float] {
                    (value as? [Any])?.compactMap { ($0 as? NSNumber)?.floatValue } ?? []
                }
                let heads = (entry["bias"] as? [Any]) ?? []
                guard let head0 = heads.first as? [Any], let lastQuery = head0.last else { continue }
                referenceBiasByVisit[try forward.visitIndex(layer: layer, pass: pass)] = floats(lastQuery)
            }
            var checked = 0
            var worst: Float = 0
            for (slot, plan) in forward.plan.enumerated() {
                guard let reference = referenceBiasByVisit[plan.cacheSlot],
                      let mine = internals["\(slot):bias_head0"] else { continue }
                XCTAssertEqual(mine.count, reference.count, "visit \(slot) bias row count")
                for row in 0..<min(mine.count, reference.count) {
                    let scale = max(abs(reference[row]), 1)
                    let error = abs(mine[row] - reference[row]) / scale
                    worst = max(worst, error)
                    // 5e-2, measured: the worst row is 4.8e-2, in a post-loop
                    // layer whose residual is ~265, i.e. the same accumulated
                    // BF16 rounding the visits show. The bar is a shape check
                    // (a wrong bank mix or a wrong mask is a unit error, not
                    // percent), and the relative error is printed so a drift
                    // upward is visible before it fails.
                    XCTAssertLessThanOrEqual(
                        error, 5e-2,
                        "visit \(slot) (layer \(plan.physicalLayer) pass \(plan.pass)) bias row \(row): "
                            + "\(mine[row]) against \(reference[row])"
                    )
                }
                checked += 1
            }
            print("🔎 [inkling2 real] relative bias: \(checked) visit(s), worst \(worst) relative")
            XCTAssertGreaterThan(checked, 0, "no visit's bias was comparable")
        }

        // The cross-pass messages, which is what the `middleCycle` gate is
        // about: the reference's `messages` dump is, per core layer, its four
        // reads (query, key, value, mlp) on the LAST pass, in layer order.
        // Comparing them checks more than the row unpacking -- the stack slice,
        // the partial state and the read itself all feed these numbers -- so a
        // mismatch names a channel rather than a "cycle".
        let referenceMessages = (dumped["messages"] as? [Any]) ?? []
        if !referenceMessages.isEmpty {
            // The dump is PASS-MAJOR: the core reads messages on every pass but
            // the first, so entry `i` is core layer `i % coreCount` of loop pass
            // `1 + i / coreCount` (12 layers x 2 passes = 24 entries here).
            let coreCount = forward.config.loopCoreEnd - forward.config.loopCoreStart
            XCTAssertEqual(referenceMessages.count, coreCount * (forward.config.numPasses - 1))
            var compared = 0
            var worstMessage: Float = 0
            for (index, entry) in referenceMessages.enumerated() {
                guard let channels = entry as? [Any] else { continue }
                let layer = index % coreCount
                let pass = 1 + index / coreCount
                for (channel, raw) in channels.enumerated() {
                    let reference = (raw as? [Any])?.compactMap { ($0 as? NSNumber)?.floatValue } ?? []
                    // Not visit-keyed: the read happens before its visit runs,
                    // so the key names the layer, channel and pass.
                    guard let mine = internals["0:message\(layer)_\(channel)_p\(pass)"] else {
                        XCTFail("no capture for core layer \(layer) message \(channel) pass \(pass)")
                        return
                    }
                    let scale = reference.reduce(Float(0)) { max($0, abs($1)) }
                    let error = zip(mine, reference).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
                    worstMessage = max(worstMessage, scale > 0 ? error / scale : error)
                    compared += 1
                }
            }
            print("🔎 [inkling2 real] messages: \(compared) compared, worst \(worstMessage) relative")
            // 5e-2, the same accumulated-rounding bar the visits carry.
            XCTAssertLessThanOrEqual(worstMessage, 5e-2, "the cross-pass message read")
            XCTAssertEqual(compared, referenceMessages.count * 4)
        }

        // The final read over the whole stack, against the reference's own
        // `attnResOutputQuery` output: the one named value that depends on the
        // stack's contents AND order, so it is the check the `blockResidual`
        // gate closes on.
        let referenceFinalRead = (dumped["finalRead"] as? [NSNumber])?.map(\.floatValue) ?? []
        if !referenceFinalRead.isEmpty {
            let finalRead = try XCTUnwrap(
                // Captured under visit 0 like every other named intermediate of
                // the last position (the read happens after the last visit, so
                // a visit-numbered key would fall outside the plan).
                internals["0:finalRead"], "the final read was not captured"
            )
            XCTAssertEqual(finalRead.count, referenceFinalRead.count)
            let scale = referenceFinalRead.reduce(Float(0)) { max($0, abs($1)) }
            let worst = zip(finalRead, referenceFinalRead).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            let error = scale > 0 ? worst / scale : worst
            print("🔎 [inkling2 real] final read: \(error) relative")
            // 5e-2, the same accumulated-rounding bar the visits carry.
            XCTAssertLessThanOrEqual(error, 5e-2, "the final block read over the stack")
        }

        // The logits, in the shape a BF16 reference can actually support.
        //
        // The plan's elementwise bar (5e-4 + 5e-4 * |reference|) is not
        // assertable against this dump: it was taken in BF16, and the
        // REFERENCE'S OWN BF16 run differs from its FP32 run by a median 0.154
        // and up to 0.895 absolute on these logits -- 197k of 200k elements
        // would violate the bound with the reference compared against itself.
        // What is asserted instead is the property a wrong graph cannot fake:
        // the ranking. The argmax must agree, the top four as a SET must agree,
        // and the cosine similarity must be near one. A measured magnitude bar
        // rides along, loose enough to stay inside the reference's own noise.
        //
        // Four and not ten: the reference's own BF16 and FP32 runs disagree on
        // the top-ten SET (rank 5 is 29633 in one and 129745 in the other),
        // while their top four are identical.
        let actualArgmax = logits.indices.max { logits[$0] < logits[$1] }
        let referenceArgmax = referenceLogits.indices.max { referenceLogits[$0] < referenceLogits[$1] }
        XCTAssertEqual(actualArgmax, referenceArgmax)
        func topFour(_ values: [Float]) -> Set<Int> { Set(values.indices.sorted { values[$0] > values[$1] }.prefix(4)) }
        XCTAssertEqual(topFour(logits), topFour(referenceLogits), "the top four logits are not the reference's set")
        let dot = zip(logits, referenceLogits).reduce(Float(0)) { $0 + $1.0 * $1.1 }
        let cosine = dot / (
            logits.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
            * referenceLogits.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
        )
        XCTAssertGreaterThan(cosine, 0.9999, "logit cosine similarity is \(cosine)")
        let worst = zip(logits, referenceLogits).map { abs($0 - $1) }.max() ?? .infinity
        XCTAssertLessThanOrEqual(worst, 1.0, "worst logit differs by \(worst)")
    }

    /// The wired path: the prompt renders through `Inkling2Prompt` (never the
    /// ChatML fallback), the turn runner drives the real checkpoint, and the
    /// answer is text rather than a repetition.
    ///
    /// It is the same graph `testTheHostForwardMatchesTheReferenceStageByStage`
    /// checks visit by visit, run the way the app runs it, so a regression in
    /// the wiring (a prompt that loses its markers, a settings field wired to
    /// the wrong sampler argument) shows up here and not there.
    func testAPublishedPromptCompletesATurnThroughTheRuntime() throws {
        let forward = try makeForward()
        let prompt = try Inkling2Prompt().render(
            turns: [.user([.text("The capital of France is")])],
            effort: 0
        )
        XCTAssertTrue(prompt.contains(Inkling2PromptMarkers.familyDefault.user))
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: try snapshot().appendingPathComponent("tokenizer.json").path
        )
        let ids = try tokenizer.encode(text: prompt).map { Int($0) }
        XCTAssertFalse(ids.isEmpty)

        var generated: [UInt32] = []
        let settings = Inkling2TurnRunner.Settings(
            temperature: 0, topP: 1, minP: 0, topK: 1, maxTokens: 8,
            eosTokenIds: [UInt32(clamping: forward.config.eosTokenID)]
        )
        let output = try Inkling2TurnRunner.run(
            forward: forward,
            promptTokenIds: ids,
            settings: settings,
            sample: { logits, _ in
                var scratch = logits
                return scratch.withUnsafeMutableBufferPointer { buffer in
                    InferenceEngine.sampleNextToken(
                        logits: buffer.baseAddress!, vocabSize: buffer.count,
                        contextTokens: [], temperature: 0, topP: 1, minP: 0, topK: 1,
                        repetitionPenalty: 1, presencePenalty: 0,
                        eosTokenIds: settings.eosTokenIds
                    )
                }
            },
            decode: { ids in (try? tokenizer.decode(ids: ids)) ?? "" }
        )
        generated = output.generatedTokenIds
        XCTAssertFalse(generated.isEmpty, "the turn generated nothing")
        let text = output.text
        print("🔎 [inkling2 real] \(ids.count) prompt tokens, \(generated.count) generated: \(generated) -> \(text.debugDescription)")
        // Not a repetition, which is the failure mode a broken graph actually
        // has (the degenerate-cycle guard in the runner exists for it).
        XCTAssertGreaterThan(Set(generated).count, 1, "the turn repeated one token: \(generated)")
        XCTAssertFalse(text.contains("\u{FFFD}"), "the turn produced invalid UTF-8: \(text.debugDescription)")
        // The prompt ended where the template says it does, so the model
        // continues as the assistant and opens its own channel. This is what
        // proves the wiring, not the wording: markers missing from the prompt
        // would leave the continuation in some other role.
        XCTAssertTrue(
            text.hasPrefix(Inkling2PromptMarkers.familyDefault.model
                + Inkling2PromptMarkers.familyDefault.contentThinking),
            "the turn did not continue in the assistant role: \(text.debugDescription)"
        )
    }

    /// The Phase 2 gate the logit comparison cannot answer: greedy ids on
    /// several separated prompts must be the reference's. A per-element
    /// tolerance can pass while an argmax flips; this is the check that the
    /// graph's DECISIONS agree.
    ///
    /// The reference is the pinned Transformers implementation in BF16, run by
    /// `/tmp/inkling_greedy_ref.py` (its output is `/tmp/inkling2_greedy_ref.json`),
    /// with the prompt rendered exactly as the app renders it -- the same
    /// `Inkling2Prompt` call and effort 0 -- so the ids being compared start
    /// from the same place.
    func testGreedyContinuationsMatchTheReference() throws {
        let path = "/tmp/inkling2_greedy_ref.json"
        guard FileManager.default.fileExists(atPath: path),
              let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cases = root["cases"] as? [[String: Any]] else {
            throw XCTSkip("no reference continuations at \(path): run /tmp/inkling_greedy_ref.py")
        }
        let forward = try makeForward()
        let tokenizer = try DynaMoeTokenizer(
            tokenizerPath: try snapshot().appendingPathComponent("tokenizer.json").path
        )
        for entry in cases {
            let prompt = try XCTUnwrap(entry["prompt"] as? String)
            let expected = try XCTUnwrap(entry["continuation"] as? [Int])
            let promptIds = try XCTUnwrap(entry["ids"] as? [Int])

            // The dump's ids are the reference tokenizer's; ours has to produce
            // the same prompt for the comparison to be about the graph.
            let rendered = try Inkling2Prompt().render(turns: [.user([.text(prompt)])], effort: 0)
            let ids = try tokenizer.encode(text: rendered).map { Int($0) }
            XCTAssertEqual(
                ids, promptIds,
                "the prompt tokenizes differently from the reference's for \(prompt.debugDescription)"
            )

            forward.reset()
            let settings = Inkling2TurnRunner.Settings(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                maxTokens: max(1, expected.count),
                eosTokenIds: [UInt32(clamping: forward.config.eosTokenID)]
            )
            let output = try Inkling2TurnRunner.run(
                forward: forward,
                promptTokenIds: ids,
                settings: settings,
                sample: { logits, _ in
                    var scratch = logits
                    return scratch.withUnsafeMutableBufferPointer { buffer in
                        InferenceEngine.sampleNextToken(
                            logits: buffer.baseAddress!, vocabSize: buffer.count,
                            contextTokens: [], temperature: 0, topP: 1, minP: 0, topK: 1,
                            repetitionPenalty: 1, presencePenalty: 0,
                            eosTokenIds: settings.eosTokenIds
                        )
                    }
                },
                decode: { ids in (try? tokenizer.decode(ids: ids)) ?? "" }
            )
            let actual = output.generatedTokenIds.map { Int($0) }
            let referenceText = entry["text"] as? String ?? ""
            print(
                "🔎 [inkling2 greedy] \(prompt.debugDescription): mine \(actual) "
                    + "vs reference \(expected) \(referenceText.debugDescription)"
            )
            // The first position is the one that must agree: after a divergence,
            // every later id is conditioned on a different prefix, so comparing
            // the rest says nothing new. Report the first mismatch by position.
            let shared = min(actual.count, expected.count)
            var agreed = 0
            while agreed < shared, actual[agreed] == expected[agreed] { agreed += 1 }
            XCTAssertEqual(
                agreed, shared,
                "\(prompt.debugDescription) diverges at position \(agreed): "
                    + "mine \(Array(actual.prefix(agreed + 1))) vs reference \(Array(expected.prefix(agreed + 1)))"
            )
        }
    }
}

// MARK: - Turn loop

/// A scripted executor: it records what it was asked and answers from a table,
/// so the loop's stop conditions, cap, cancellation and position arithmetic are
/// tested with no model.
private final class ScriptedInklingForward: Inkling2PositionForward {
    let positionLimit: Int
    private(set) var storedPositions = 0
    private(set) var forwarded: [Int] = []
    private(set) var resets = 0
    /// Logits per forwarded position, cycled.
    var logits: [[Float]]
    var failAt: Int?

    init(positionLimit: Int = 64, logits: [[Float]]) {
        self.positionLimit = positionLimit
        self.logits = logits
    }

    func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int { 0 }

    func reset() {
        resets += 1
        storedPositions = 0
        forwarded.removeAll()
    }

    func forwardToken(_ token: Int, position: Int) throws -> [Float] {
        if let failAt, forwarded.count == failAt {
            throw Inkling2ForwardError.unsupported("scripted failure")
        }
        forwarded.append(token)
        storedPositions += 1
        return logits[forwarded.count % logits.count]
    }
}

final class Inkling2TurnTests: XCTestCase {

    private func settings(maxTokens: Int = 8, eos: [UInt32] = [9]) -> Inkling2TurnRunner.Settings {
        Inkling2TurnRunner.Settings(maxTokens: maxTokens, eosTokenIds: eos)
    }

    /// Always draws the highest-logit id, so the turns below are deterministic.
    private func greedy(_ logits: [Float], _ soFar: [UInt32]) -> UInt32 {
        UInt32(logits.indices.max { logits[$0] < logits[$1] } ?? 0)
    }

    /// A chunk size is a scheduling knob: every size must forward the same
    /// positions in the same order and produce the same answer, and the
    /// progress report must arrive once per chunk rather than once per
    /// position.
    func testAChunkSizeNeverChangesWhatRuns() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        var baseline: (tokens: [UInt32], forwarded: [Int], reports: [Int])?
        for chunk in [1, 2, 3, 7, 64] {
            let forward = ScriptedInklingForward(logits: [logits])
            var reports: [Int] = []
            let output = try Inkling2TurnRunner.run(
                forward: forward, promptTokenIds: prompt, prefillChunk: chunk,
                settings: settings(maxTokens: 3),
                sample: greedy, decode: { $0.map(String.init).joined() },
                onPrefill: { done, _ in reports.append(done) }
            )
            if let baseline {
                XCTAssertEqual(output.generatedTokenIds, baseline.tokens, "chunk \(chunk)")
                XCTAssertEqual(forward.forwarded, baseline.forwarded, "chunk \(chunk)")
            } else {
                baseline = (output.generatedTokenIds, forward.forwarded, reports)
            }
            // One report per chunk, each at a chunk boundary, the last at the
            // prompt's end.
            XCTAssertEqual(reports.last, prompt.count, "chunk \(chunk)")
            XCTAssertEqual(reports.count, (prompt.count + min(chunk, prompt.count) - 1) / min(chunk, prompt.count))
            XCTAssertEqual(reports, reports.sorted(), "reports must be monotone")
        }
    }

    /// A cancel between chunks leaves exactly the positions before the chunk
    /// boundary committed, and a resumed turn from there answers what a full
    /// prefill answers.
    func testACancelBetweenChunksResumesExactlyThere() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let prompt = [1, 2, 3, 4, 5, 6, 7]
        let forward = ScriptedInklingForward(logits: [logits])
        // Cancelled after the second chunk's report: two chunks of three is
        // six positions committed, one short of the prompt.
        var reports = 0
        let cancelled = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: prompt, prefillChunk: 3,
            settings: settings(maxTokens: 3),
            sample: greedy, decode: { $0.map(String.init).joined() },
            onPrefill: { _, _ in reports += 1 },
            isCancelled: { reports >= 2 }
        )
        XCTAssertTrue(cancelled.cancelled)
        XCTAssertEqual(cancelled.storedPromptTokens, 6)
        XCTAssertEqual(forward.forwarded, Array(prompt.prefix(6)))
    }

    func testATurnStopsAtEos() throws {
        // Logits favour id 9 (EOS) at every position.
        var logits = [Float](repeating: 0, count: 16)
        logits[9] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1, 2, 3], settings: settings(),
            sample: greedy, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(output.finishReason, .eos)
        XCTAssertEqual(output.generatedTokenIds, [9])
        XCTAssertEqual(output.promptTokens, 3)
        // The EOS token is sampled but never forwarded, so the cache holds only
        // the prompt: a sampled token is not a committed position until it is fed
        // back, and the turn stops before that.
        XCTAssertEqual(output.storedPromptTokens, 3)
        XCTAssertFalse(output.cancelled)
        XCTAssertEqual(forward.forwarded, [1, 2, 3])
    }

    func testATurnStopsAtTheTokenCap() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1], settings: settings(maxTokens: 3),
            sample: greedy, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(output.generatedTokenIds, [5, 5, 5])
        XCTAssertEqual(output.finishReason, .lengthCapped)
    }

    func testATurnStopsAtThePositionLimit() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(positionLimit: 4, logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1, 2], settings: settings(maxTokens: 100),
            sample: greedy, decode: { $0.map(String.init).joined() }
        )
        // Two prompt positions plus two generated fill the four-position limit.
        XCTAssertEqual(output.generatedTokenIds.count, 2)
        XCTAssertEqual(output.finishReason, .lengthCapped)
    }

    func testCancellationStopsAtAPositionBoundary() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        var count = 0
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1], settings: settings(maxTokens: 100),
            sample: greedy, decode: { $0.map(String.init).joined() },
            isCancelled: { count += 1; return count > 3 }
        )
        XCTAssertTrue(output.cancelled)
        XCTAssertEqual(output.finishReason, .stopped)
        XCTAssertLessThan(output.tokensGenerated, 100)
    }

    func testAStopSequenceEndsTheTurn() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        var turn = settings(maxTokens: 10, eos: [])
        turn.stopSequences = ["55"]
        let forward = ScriptedInklingForward(logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1], settings: turn,
            sample: greedy, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(output.matchedStopSequence, "55")
        XCTAssertEqual(output.finishReason, .stoppedBySequence)
        XCTAssertEqual(output.generatedTokenIds, [5, 5])
    }

    func testTheToolFreezeEndsTheTurnOnTheCall() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1], settings: settings(maxTokens: 10, eos: []),
            sample: greedy, shouldFreeze: { $0 == "55" }, decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(output.finishReason, .toolFreeze)
        XCTAssertEqual(output.generatedTokenIds, [5, 5])
    }

    /// The constraint sees the logits BEFORE the draw and its rewrite is what the
    /// sampler reads, which is the contract a grammar mask depends on.
    func testTheConstraintRewritesTheLogitsBeforeTheDraw() throws {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        let output = try Inkling2TurnRunner.run(
            forward: forward, promptTokenIds: [1], settings: settings(maxTokens: 1, eos: []),
            sample: greedy, constrain: { values, _ in values[7] = 100 },
            decode: { $0.map(String.init).joined() }
        )
        XCTAssertEqual(output.generatedTokenIds, [7])
    }

    func testAnEmptyPromptIsRefused() {
        let forward = ScriptedInklingForward(logits: [[0]])
        XCTAssertThrowsError(
            try Inkling2TurnRunner.run(forward: forward, promptTokenIds: [], settings: settings(), sample: greedy, decode: { _ in "" })
        ) { error in
            XCTAssertEqual(error as? Inkling2TurnError, .emptyPrompt)
        }
    }

    func testAPromptPastTheLimitIsRefusedByName() {
        let forward = ScriptedInklingForward(positionLimit: 2, logits: [[0]])
        XCTAssertThrowsError(
            try Inkling2TurnRunner.run(forward: forward, promptTokenIds: [1, 2, 3], settings: settings(), sample: greedy, decode: { _ in "" })
        ) { error in
            guard case Inkling2TurnError.promptTooLong(let prompt, let limit) = error else {
                return XCTFail("expected promptTooLong, got \\(error)")
            }
            XCTAssertEqual(prompt, 3)
            XCTAssertEqual(limit, 2)
        }
    }

    func testAFailedPrefillNamesItsPosition() {
        var logits = [Float](repeating: 0, count: 16)
        logits[5] = 10
        let forward = ScriptedInklingForward(logits: [logits])
        forward.failAt = 2
        XCTAssertThrowsError(
            try Inkling2TurnRunner.run(forward: forward, promptTokenIds: [1, 2, 3], settings: settings(), sample: greedy, decode: { _ in "" })
        ) { error in
            guard case Inkling2TurnError.prefillFailed(let position, _) = error else {
                return XCTFail("expected prefillFailed, got \\(error)")
            }
            XCTAssertEqual(position, 2)
        }
    }
}

// MARK: - Host forward on the synthetic checkpoint

/// A small, structurally complete Inkling 2 checkpoint loaded by BOTH the Swift
/// host graph and the pinned Transformers reference, so a hypothesis about the
/// graph costs seconds instead of a six-minute real-checkpoint run.
///
/// The fixture is written by `Tools/inkling2_synthetic.py`; it stores the
/// experts in the checkpoint's packed layouts, so each side runs its own
/// dequantizer. The expected values come from the reference run over the same
/// directory (`Tools/inkling2_synthetic_golden.py`), never from this code.
///
/// It also checks the fixture-catalog rules: every tensor placed, no duplicates,
/// the expert layouts inferred from strides, and the full/sliding tensor split in
/// both directions.
final class Inkling2SyntheticForwardTests: XCTestCase {

    static let fixturePath = "/tmp/inkling2_synth"
    static let goldenPath = "/tmp/inkling2_synth_goldens.json"

    private func fixture() throws -> URL {
        let path = URL(fileURLWithPath: Self.fixturePath)
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("config.json").path) else {
            throw XCTSkip("no synthetic checkpoint at \(Self.fixturePath): run Tools/inkling2_synthetic.py")
        }
        return path
    }

    /// Builds a summary that describes the fixture's tensors without opening a
    /// shard: the catalog only needs names, dtypes and shapes, and the Rust
    /// loader would split the fused expert banks anyway, which is exactly what
    /// the coalescing path is tested against on the real checkpoint.
    private func fixtureSummary(_ directory: URL) throws -> (ModelSummary, Inkling2Config) {
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let index = try XCTUnwrap(JSONSerialization.jsonObject(with: indexData) as? [String: Any])
        let weightMap = try XCTUnwrap(index["weight_map"] as? [String: String])
        let handle = try FileHandle(forReadingFrom: directory.appendingPathComponent("model-00001-of-00001.safetensors"))
        defer { try? handle.close() }
        let lengthData = try XCTUnwrap(try handle.read(upToCount: 8))
        var headerLength: UInt64 = 0
        for (offset, byte) in lengthData.enumerated() { headerLength |= UInt64(byte) << (8 * UInt64(offset)) }
        let headerData = try XCTUnwrap(try handle.read(upToCount: Int(headerLength)))
        let header = try XCTUnwrap(JSONSerialization.jsonObject(with: headerData) as? [String: Any])
        var tensors: [TensorMetadata] = []
        for name in weightMap.keys.sorted() {
            guard let entry = header[name] as? [String: Any],
                  let dtype = entry["dtype"] as? String,
                  let shape = entry["shape"] as? [Int],
                  let offsets = entry["data_offsets"] as? [Int] else { continue }
            let width = Inkling2SnapshotWeights.elementWidth(dtype)
            let elements = shape.reduce(1, *)
            tensors.append(TensorMetadata(
                name: name,
                shapeDisplay: "[" + shape.map(String.init).joined(separator: ", ") + "]",
                dtype: dtype,
                sizeMb: Double(elements * width) / (1024 * 1024),
                shardIndex: 0,
                // ABSOLUTE file offsets, which is what the Rust summary reports
                // (the safetensors header's own data_offsets are relative to the
                // data section): the 8-byte length prefix and the header come first.
                offsetStart: 8 + UInt64(headerLength) + UInt64(offsets[0]),
                offsetEnd: 8 + UInt64(headerLength) + UInt64(offsets[1]),
                category: "fixture",
                layerIndex: nil,
                expertId: nil
            ))
        }
        let summary = ModelSummary(
            sizeGb: 0.001, tensorCount: UInt32(tensors.count), layerCount: UInt32(config.numHiddenLayers),
            maxExpertId: UInt32(config.numRoutedExperts), shards: [], tensors: tensors, layers: []
        )
        return (summary, config)
    }

    /// The loop's three bookkeeping rules, each of which was wrong at least
    /// once and each of which is invisible in a single-pass model:
    ///
    ///  * a boundary layer's stack entry is the layer's INPUT (the loop appends
    ///    before assigning the output),
    ///  * `previous_block_states` and `previous_hidden_states` are recomputed at
    ///    the top of EVERY remaining pass, not once,
    ///  * the message gates act before the input norms, and a sliding layer
    ///    normalizes each of the three gated inputs separately.
    ///
    /// Checked against the reference's own per-visit outputs, so a flip in any
    /// rule shows up as a visit mismatch within a second.
    func testTheLoopBookkeepingMatchesTheReferenceOnEveryVisit() throws {
        guard FileManager.default.fileExists(atPath: Self.goldenPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: Self.goldenPath)),
              let golden = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("no fixture goldens at \(Self.goldenPath): run Tools/inkling2_synthetic_golden.py")
        }
        let directory = try fixture()
        let (summary, config) = try fixtureSummary(directory)
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let weightsPath = directory.appendingPathComponent("model-00001-of-00001.safetensors")
        let handle = try FileHandle(forReadingFrom: weightsPath)
        defer { try? handle.close() }
        let fileLength = try XCTUnwrap(try FileManager.default.attributesOfItem(
            atPath: weightsPath.path
        )[.size] as? NSNumber).intValue
        let mapped = try XCTUnwrap(mmap(nil, fileLength, PROT_READ, MAP_PRIVATE, handle.fileDescriptor, 0))
        XCTAssertNotEqual(mapped, MAP_FAILED)
        let buffer = try XCTUnwrap(device.makeBuffer(
            bytesNoCopy: mapped, length: fileLength, options: .storageModeShared, deallocator: nil
        ))
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: [0: buffer])
        let forward = try Inkling2CPUForward(config: config, weights: source)

        let visitsPerPosition = forward.plan.count
        var actualVisits: [[Float]] = []
        var ringIndex = 0
        forward.onVisitComplete = { (_: Int, _: Int, _: Int, output: [Float]) in
            if actualVisits.count < visitsPerPosition {
                actualVisits.append(output)
            } else {
                actualVisits[ringIndex] = output
            }
            ringIndex = (ringIndex + 1) % visitsPerPosition
        }
        let ids = try XCTUnwrap(golden["ids"] as? [Int])
        for (position, id) in ids.enumerated() {
            _ = try forward.forwardToken(token: UInt32(id), position: position)
        }
        let goldenVisits = try XCTUnwrap(golden["visits"] as? [[NSNumber]])
        XCTAssertEqual(actualVisits.count, goldenVisits.count)
        if let data = try? JSONSerialization.data(withJSONObject: actualVisits.map { $0.map { Double($0) } }) {
            try? data.write(to: URL(fileURLWithPath: "/tmp/inkling2_bookkeeping_mine.json"))
        }
        for (index, reference) in goldenVisits.enumerated() where index < actualVisits.count {
            let expected = reference.map { $0.floatValue }
            let scale = expected.reduce(Float(0)) { max($0, abs($1)) }
            let worst = zip(actualVisits[index], expected).reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            let error = scale > 0 ? worst / scale : worst
            XCTAssertLessThanOrEqual(error, 2e-2, "visit \(index) is \(error) relative from the reference")
        }
    }

    /// Commit-only-after-success: a position whose forward THROWS leaves the
    /// sequence exactly where it was, in every visit and in the token history.
    ///
    /// It is the property a resume depends on -- `storedPromptTokens` is the
    /// pinned prefix, so a position counted but not committed (or the reverse)
    /// makes the next turn splice two sequences. The forward here is the real
    /// host graph on the fixture, with the KV budget zeroed after a few
    /// positions: the refusal is the forward's own, not a stub's.
    func testAFailedPositionCommitsNothing() throws {
        let directory = try fixture()
        let (summary, config) = try fixtureSummary(directory)
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let weightsPath = directory.appendingPathComponent("model-00001-of-00001.safetensors")
        let handle = try FileHandle(forReadingFrom: weightsPath)
        defer { try? handle.close() }
        let fileLength = try XCTUnwrap(try FileManager.default.attributesOfItem(
            atPath: weightsPath.path
        )[.size] as? NSNumber).intValue
        let mapped = try XCTUnwrap(mmap(nil, fileLength, PROT_READ, MAP_PRIVATE, handle.fileDescriptor, 0))
        XCTAssertNotEqual(mapped, MAP_FAILED)
        let buffer = try XCTUnwrap(device.makeBuffer(
            bytesNoCopy: mapped, length: fileLength, options: .storageModeShared, deallocator: nil
        ))
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: [0: buffer])
        // Enough rows for two positions and not a third.
        let forward = try Inkling2CPUForward(
            config: config, weights: source, options: Inkling2ForwardOptions(maxPositions: 2)
        )
        let token = UInt32(1)
        _ = try forward.forwardToken(token: token, position: 0)
        _ = try forward.forwardToken(token: token, position: 1)
        let before = forward.states.map(\.committedPositions)
        let historyBefore = forward.sequence.tokenHistory

        XCTAssertThrowsError(try forward.forwardToken(token: token, position: 2)) { error in
            // The refusal must NAME the budget, not surface as a shape error
            // from some inner kernel.
            XCTAssertTrue("\(error)".contains("KV budget"), "\(error)")
        }
        XCTAssertEqual(forward.states.map(\.committedPositions), before, "a refused position committed a row")
        XCTAssertEqual(forward.sequence.tokenHistory, historyBefore, "a refused position entered the history")
        XCTAssertTrue(forward.sequence.isConsistent)
    }

    func testTheFixtureCatalogIsExhaustiveAndStrict() throws {
        let directory = try fixture()
        let (summary, config) = try fixtureSummary(directory)
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        XCTAssertTrue(catalog.unconsumed.isEmpty, "unplaced: \(catalog.unconsumed.prefix(8))")
        XCTAssertTrue(catalog.duplicateRoles.isEmpty)
        let problems = catalog.validate(config: config)
        XCTAssertTrue(problems.isEmpty, "problems: \(problems.prefix(8))")
        // Both layouts are represented and inferred from strides.
        for layer in 0..<config.numHiddenLayers {
            let layout = try Inkling2ExpertQuantLayout.infer(
                w13: try XCTUnwrap(catalog.tensor(.expertW13, layer: layer)),
                w13Scale: try XCTUnwrap(catalog.tensor(.expertW13Scale, layer: layer)),
                w2: try XCTUnwrap(catalog.tensor(.expertW2, layer: layer)),
                w2Scale: try XCTUnwrap(catalog.tensor(.expertW2Scale, layer: layer)),
                config: config
            ).get()
            XCTAssertEqual(layout, layer < 10 ? .nvfp4 : .mxfp8, "layer \(layer)")
        }
    }

    func testTheHostForwardMatchesTheReferenceOnTheFixture() throws {
        guard FileManager.default.fileExists(atPath: Self.goldenPath),
              let data = try? Data(contentsOf: URL(fileURLWithPath: Self.goldenPath)),
              let golden = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw XCTSkip("no fixture goldens at \(Self.goldenPath): run Tools/inkling2_synthetic_golden.py")
        }
        let directory = try fixture()
        let (summary, config) = try fixtureSummary(directory)
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let weightsPath = directory.appendingPathComponent("model-00001-of-00001.safetensors")
        let handle = try FileHandle(forReadingFrom: weightsPath)
        defer { try? handle.close() }
        let fileLength = try XCTUnwrap(try FileManager.default.attributesOfItem(
            atPath: weightsPath.path
        )[.size] as? NSNumber).intValue
        let mapped = try XCTUnwrap(mmap(nil, fileLength, PROT_READ, MAP_PRIVATE, handle.fileDescriptor, 0),
                                   "could not map \(weightsPath.path)")
        XCTAssertNotEqual(mapped, MAP_FAILED)
        // Deliberately NOT munmap'ed: the zero-copy buffer points into this
        // mapping for as long as the forward runs, and unmapping it in a defer
        // would pull the bytes out from under it. The mapping lives for the
        // test process, which is the same lifetime a shard has in the app.
        let buffer = try XCTUnwrap(device.makeBuffer(
            bytesNoCopy: mapped, length: fileLength, options: .storageModeShared, deallocator: nil
        ))
        let source = Inkling2SnapshotWeights(config: config, catalog: catalog, shardBuffers: [0: buffer])
        let forward = try Inkling2CPUForward(config: config, weights: source)

        var postAttention: [[Float]] = []
        // The LAST position's outputs, keyed by VISIT: the golden holds one entry
        // per visit for the final position, and a plain append would mix the
        // first position's values with the last one's (which is exactly the
        // misalignment that made two comparisons disagree earlier).
        // The golden's visits are grouped by PHYSICAL LAYER, in call order within
        // a layer (a core layer appears once per pass). Rebuilding that same list
        // from the execution order is the only way to compare like with like: a
        // dict keyed by visit index reorders the core's three passes.
        // The golden holds the FINAL position's values, one entry per visit, so
        // only the last `plan.count` calls are kept: the calls run
        // position-major (every visit of a position before the next position).
        // The ring is one POSITION's worth of visits, from the plan: the fixture
        // has 24 visits per position, so a hard-coded 48 kept position 0's
        // visits around while the golden is the last position's.
        let visitsPerPosition = forward.plan.count
        var actualVisits: [[Float]] = []
        var ringIndex = 0
        forward.onVisitComplete = { (_: Int, _: Int, _: Int, output: [Float]) in
            // A ring over the LAST position's visits: appending past one
            // position's worth and keeping the tail, rather than replacing in
            // place by count (which overwrote the wrong slots).
            if actualVisits.count < visitsPerPosition {
                actualVisits.append(output)
            } else {
                actualVisits[ringIndex] = output
            }
            ringIndex = (ringIndex + 1) % visitsPerPosition
        }
        // Every stage of every visit, written for the OFFLINE diff: one run each
        // side, then comparisons are free instead of minutes.
        var stageDump: [String: [Float]] = [:]
        func record(_ visit: Int, _ stage: String, _ value: [Float]) {
            stageDump["\(visit).\(stage)"] = value
        }
        forward.onVisitSubStages = { (visit: Int, attention: [Float], mixture: [Float]) in
            record(visit, "attnRaw", attention)
            record(visit, "mlp", mixture)
        }
        var lastPositionInternals: [String: [Float]] = [:]
        var internals: [String: [Float]] = [:]
        var postAttentionIndex = 0
        forward.onPostAttention = { (visit: Int, value: [Float]) in
            record(visit, "postAttnConv", value)
            // The golden holds EVERY visit's post-attention convolution for the
            // final position, so this is a ring of one position's visits too --
            // keeping only visit 0 compared one entry out of twenty-four.
            if postAttention.count < visitsPerPosition {
                postAttention.append(value)
            } else {
                postAttention[postAttentionIndex] = value
            }
            postAttentionIndex = (postAttentionIndex + 1) % visitsPerPosition
        }
        forward.onInternal = { (visit: Int, name: String, value: [Float]) in
            // ONE closure: a second assignment would overwrite the first, which
            // is how a whole dump once came back empty.
            lastPositionInternals["\(visit).\(name)"] = value
            if visit == 0 || visit == 7 || name.hasPrefix("message") || name.hasPrefix("previousHidden") {
                internals["\(visit):\(name)"] = value
                if visit == 0 { internals[name] = value }
            }
        }
        let ids = try XCTUnwrap(golden["ids"] as? [Int])
        var logits: [Float] = []
        for (position, id) in ids.enumerated() {
            logits = try forward.forwardToken(token: UInt32(id), position: position)
        }

        func rel(_ a: [Float], _ r: [NSNumber]) -> Float {
            let reference = r.map { $0.floatValue }
            let scale = reference.reduce(Float(0)) { max($0, abs($1)) }
            var worst: Float = 0
            for (actual, expected) in zip(a, reference) {
                worst = max(worst, abs(actual - expected))
            }
            return scale > 0 ? worst / scale : worst
        }
        let goldenVisits = try XCTUnwrap(golden["visits"] as? [[NSNumber]])
        for (index, output) in actualVisits.enumerated() { record(index, "out", output) }
        for (key, value) in lastPositionInternals { stageDump[key] = value }
        if let data = try? JSONSerialization.data(withJSONObject: stageDump.mapValues { $0.map { Double($0) } }) {
            try? data.write(to: URL(fileURLWithPath: "/tmp/inkling2_all_mine.json"))
        }
        // In CALL order, which is the golden's order: regrouping by layer put
        // the loop core's later passes after the post-loop layers.
        let visits = actualVisits
        let goldenAttention = try XCTUnwrap(golden["postAttention"] as? [[NSNumber]])
        for index in 0..<min(visits.count, goldenVisits.count) {
            let error = rel(visits[index], goldenVisits[index])
            if error > 2e-2 {
                XCTFail("visit \(index) is \(error) relative from the reference")
                return
            }
        }
        XCTAssertEqual(postAttention.count, goldenAttention.count)
        for index in 0..<min(postAttention.count, goldenAttention.count) {
            let error = rel(postAttention[index], goldenAttention[index])
            if error > 2e-2 {
                XCTFail("visit \(index) post-attention is \(error) relative from the reference")
                return
            }
        }
        // The final read over the whole stack: the reference's own
        // `attnResOutputQuery` output. Its length is the stack's length plus
        // one, so this is also the check that the stack has the right NUMBER of
        // entries (a boundary layer appended twice in the wrong place still
        // changes this vector's values, but a missed append changes its sum).
        if let goldenFinalRead = golden["finalRead"] as? [NSNumber] {
            let mineFinalRead = try XCTUnwrap(
                lastPositionInternals["0.finalRead"],
                "the final read was not captured"
            )
            let finalReadError = rel(mineFinalRead, goldenFinalRead)
            print("🔎 [inkling2 fixture] final read: \(finalReadError) relative")
            XCTAssertLessThanOrEqual(finalReadError, 2e-2, "the final block read over the stack")
        }
        let expected = try XCTUnwrap(golden["logits"] as? [NSNumber])
        let worst = rel(logits, expected)
        XCTAssertLessThanOrEqual(worst, 2e-2, "final logits")
    }
}
