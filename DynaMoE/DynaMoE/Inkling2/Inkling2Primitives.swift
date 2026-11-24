//
//  Inkling2Primitives.swift
//  DynaMoE
//
//  Inkling 2 Mini's numerical vocabulary, as host code.
//
//  Phase 1 of the Inkling 2 port: every conversion and rule the graph will be
//  built from, isolated so each one can be checked against a reference on its
//  own before a single kernel exists. Nothing here touches Metal, and nothing
//  reads a checkpoint: these are pure functions of their arguments.
//
//  The three families:
//
//    * Element formats. E2M1 (NVFP4 values), E4M3 (NVFP4 block scales, MXFP8
//      values) and E8M0 (MXFP8 block scales) each decode to Float and encode
//      back with the reference's rounding. The encoders are exhaustive where
//      they can be (E4M3 and E8M0 over their whole 8-bit code space) because a
//      lookup table is also the reference the GPU kernel is checked against.
//    * The two activation quantizers. They are different contracts, not two
//      spellings of one: NVFP4 is a 16-value block with an E4M3 scale chosen by
//      reconstruction ERROR from two candidate rules, MXFP8 a 32-value block
//      with a power-of-two E8M0 scale. Inkling2ActivationProfile names them so a
//      session can never silently switch.
//    * The scalar layer rules: the softcapped SwiGLU, the FP32 sigmoid router,
//      the static and dynamic short convolutions, the residual read, the message
//      gate, L2 clipping, log scaling and the final softcap.
//
//  Every rule below is transcribed from the pinned Transformers reference
//  (inkling2_loop/modeling_inkling2_loop.py, revision 046bdb8) and the vLLM
//  native quantizers, and the tests carry vectors produced by those sources.
//

import Accelerate
import Foundation

// MARK: - Element formats

/// The element formats Inkling 2's experts and scales are stored in.
nonisolated public enum Inkling2ElementFormat: String, Sendable, Equatable, CaseIterable {
    /// 4-bit float, 1 sign + 2 exponent + 1 mantissa, bias 1, max 6.
    case e2m1
    /// 8-bit float, 1 sign + 4 exponent + 3 mantissa, bias 7, max 448.
    case e4m3
    /// 8-bit power-of-two scale, bias 127, no sign and no mantissa.
    case e8m0

    public var bitWidth: Int {
        switch self {
        case .e2m1: return 4
        case .e4m3, .e8m0: return 8
        }
    }

    public var maximumFiniteValue: Float {
        switch self {
        case .e2m1: return 6.0
        case .e4m3: return 448.0
        case .e8m0: return Float(sign: .plus, exponent: 127, significand: 1.0)
        }
    }
}

nonisolated public enum Inkling2QuantError: Error, Equatable, CustomStringConvertible {
    case notRepresentable(String)
    case invalidScale(String)

    public var description: String {
        switch self {
        case .notRepresentable(let detail):
            return "value is not representable in the target format: \(detail)"
        case .invalidScale(let detail):
            return "invalid quantization scale: \(detail)"
        }
    }
}

/// Conversions between the storage formats and `Float`.
///
/// The DECODERS are exact: each format has a finite code space, so decode is a
/// table lookup by definition. The ENCODERS are where a mistake hides, and the
/// two families round differently on purpose:
///
///   * E2M1 rounds to nearest with ties to EVEN (the IEEE rule, and what
///     `torch.float4_e2m1fn_x2` does), saturating at 6.0 rather than producing
///     inf or NaN.
///   * E4M3 rounds to nearest with ties to even, saturating at 448 (the
///     `float8_e4m3fn` convention: the top code is the largest finite value,
///     there is no infinity), and its subnormals are real values (2^-9 step).
///   * E8M0 is exact power-of-two selection: nearest in the log domain, i.e.
///     round the exponent to nearest with ties to the LARGER exponent, which is
///     what `torch.float8_e8m0fnu` does. It has no zero, so 0 encodes as the
///     smallest exponent.
nonisolated public enum Inkling2Element {

    // MARK: Decode

    public static func decode(_ code: UInt8, as format: Inkling2ElementFormat) -> Float {
        switch format {
        case .e2m1:
            return decodeE2M1(code & 0xF)
        case .e4m3:
            return decodeE4M3(code)
        case .e8m0:
            return decodeE8M0(code)
        }
    }

    /// One E2M1 nibble. 0x0/0x8 are +0/-0; there is no inf or NaN.
    public static func decodeE2M1(_ nibble: UInt8) -> Float {
        let code = nibble & 0xF
        let sign: Float = (code & 0x8) != 0 ? -1 : 1
        let exponent = Int((code >> 1) & 0x3)
        let mantissa = Int(code & 0x1)
        // exponent 0 is the subnormal range: 0 or 0.5
        let magnitude: Float
        if exponent == 0 {
            magnitude = Float(mantissa) * 0.5
        } else {
            magnitude = (1.0 + Float(mantissa) * 0.5) * pow(2, Float(exponent - 1))
        }
        return sign * magnitude
    }

    /// One E4M3 byte in the `float8_e4m3fn` convention: no inf, NaN is 0x7F.
    public static func decodeE4M3(_ byte: UInt8) -> Float {
        let sign: Float = (byte & 0x80) != 0 ? -1 : 1
        let exponent = Int((byte >> 3) & 0xF)
        let mantissa = Int(byte & 0x7)
        if exponent == 0xF && mantissa == 0x7 { return .nan }
        let magnitude: Float
        if exponent == 0 {
            magnitude = Float(mantissa) * pow(2, -9)
        } else {
            magnitude = (1.0 + Float(mantissa) / 8.0) * pow(2, Float(exponent - 7))
        }
        return sign * magnitude
    }

    /// One E8M0 byte: pure exponent, bias 127. 0xFF is NaN.
    public static func decodeE8M0(_ byte: UInt8) -> Float {
        if byte == 0xFF { return .nan }
        return Float(sign: .plus, exponent: Int(byte) - 127, significand: 1.0)
    }

    // MARK: Encode

    /// E2M1 with nearest-even rounding and saturation at 6.0.
    public static func encodeE2M1(_ value: Float) -> UInt8 {
        guard !value.isNaN else { return 0 }
        let sign: UInt8 = value < 0 || (value == 0 && value.sign == .minus) ? 0x8 : 0x0
        let magnitude = abs(value)
        if magnitude >= 6.0 { return sign | 0x7 }   // no tie above the largest value
        // The E2M1 magnitudes are 0, .5, 1, 1.5, 2, 3, 4, 6 -- nearest with ties
        // to even means the tie goes to the code whose mantissa bit is 0.
        let table: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        var best: UInt8 = 0
        var bestDistance = Float.infinity
        for (code, candidate) in table.enumerated() {
            let distance = abs(magnitude - candidate)
            if distance < bestDistance || (distance == bestDistance && (UInt8(code) & 0x1) == 0) {
                bestDistance = distance
                best = UInt8(code)
            }
        }
        return sign | best
    }

    /// E4M3 (`float8_e4m3fn`) with nearest-even rounding and saturation at 448.
    public static func encodeE4M3(_ value: Float) -> UInt8 {
        let signBit: UInt8 = value < 0 || (value == 0 && value.sign == .minus) ? 0x80 : 0x00
        if value.isNaN { return 0x7F }
        let magnitude = abs(value)
        if magnitude >= 448.0 { return signBit | 0x7E }
        // Exhaustive over the positive code space: 128 candidates, exact.
        var best: UInt8 = 0
        var bestDistance = Float.infinity
        for code in UInt8(0)...UInt8(0x7E) {
            let candidate = decodeE4M3(code)
            let distance = abs(magnitude - candidate)
            if distance < bestDistance || (distance == bestDistance && (code & 0x1) == 0) {
                bestDistance = distance
                best = code
            }
        }
        return signBit | best
    }

    /// E8M0 with the exponent rounded to nearest, ties to the LARGER exponent.
    /// Zero and negative values have no code; they map to the smallest exponent.
    public static func encodeE8M0(_ value: Float) -> UInt8 {
        guard value > 0, value.isFinite else { return 0 }
        let exponent = Int(value.exponent)   // floor(log2(value)) for a normal Float
        // value = m * 2^exponent with m in [1, 2). The power-of-two candidates are
        // 2^exponent and 2^(exponent+1); compare in the log domain, where the
        // midpoint is sqrt(2).
        let mantissa = value / pow(2, Float(exponent))
        let chosen = mantissa >= Float(2).squareRoot() ? exponent + 1 : exponent
        let biased = chosen + 127
        if biased <= 0 { return 0 }
        if biased >= 0xFE { return 0xFE }
        return UInt8(biased)
    }

    /// Rounds to the nearest BF16 value (eight explicit mantissa bits, ties to
    /// even), which is what Transformers' dequantize-then-BF16 projection does.
    /// Kept here rather than using `Float16`-style conversions because BF16 is
    /// not a Swift type: it is the top 16 bits of a Float32.
    public static func roundToBF16(_ value: Float) -> Float {
        guard value.isFinite else { return value }
        var bits = value.bitPattern
        // Round to nearest even on the 16 discarded bits.
        let lsb = (bits >> 16) & 1
        bits += 0x7FFF + lsb
        bits &= 0xFFFF_0000
        return Float(bitPattern: bits)
    }

    // MARK: Packed pairs

    /// Two E2M1 values per byte, the LOWER input column in the LOW nibble --
    /// the layout both NVFP4 experts and the reference's packing use.
    public static func packE2M1Pair(low: Float, high: Float) -> UInt8 {
        (encodeE2M1(low) & 0xF) | ((encodeE2M1(high) & 0xF) << 4)
    }

    public static func unpackE2M1Pair(_ byte: UInt8) -> (low: Float, high: Float) {
        (decodeE2M1(byte & 0xF), decodeE2M1(byte >> 4))
    }
}

// MARK: - Activation profiles

/// Which arithmetic the expert activations are quantized in.
///
/// The two are separate CONTRACTS, not a quality setting. The HF reference
/// dequantizes each weight and runs an ordinary BF16 activation; vLLM's native
/// recipe also quantizes the ACTIVATION, and the two disagree by far more than
/// rounding. A session picks one and stays on it: switching mid-conversation
/// would mix two rounding contracts inside one prefix.
nonisolated public enum Inkling2ActivationProfile: String, Sendable, Equatable, CaseIterable {
    /// Dequantize the logical weight to BF16, multiply by an unquantized
    /// activation. Matches Transformers' dense-execution fallback.
    case hfReference = "HF reference (W4A16 / W8A16)"
    /// Quantize the activation to E2M1 with E4M3 block scales (NVFP4) or to
    /// E4M3 with E8M0 block scales (MXFP8), as the checkpoint was trained.
    case trained = "Trained activation quantization (W4A4 / W8A8)"

    public var quantizesActivation: Bool { self == .trained }
}

/// One quantized activation block: the values, their block scales and the
/// per-row outer scale the values were divided by.
nonisolated public struct Inkling2QuantizedActivation: Sendable, Equatable {
    /// Values in the target format, one per input element (E2M1 is stored one
    /// per element here and packed only at the buffer boundary).
    public let codes: [UInt8]
    /// One scale per block.
    public let blockScales: [UInt8]
    /// The per-row outer scale. Kept separate from `blockScales` so a caller can
    /// reproduce the reference's exact multiply order
    /// (`code * blockScale * outerScale`).
    public let outerScale: Float
    public let blockSize: Int

    /// Reconstructs the dequantized values, for a test or a host reference.
    public func dequantized(format: Inkling2ElementFormat) -> [Float] {
        codes.enumerated().map { offset, code in
            let scale = Inkling2Element.decode(blockScales[offset / blockSize], as: format == .e2m1 ? .e4m3 : .e8m0)
            return Inkling2Element.decode(code, as: format) * scale * outerScale
        }
    }
}

nonisolated public enum Inkling2ActivationQuantizer {

    /// NVFP4: 16-value blocks. The outer scale is `rowAmax / 1536` (the E4M3
    /// maximum times the E2M1 maximum), each block's scale is chosen from two
    /// candidate rules -- `blockAmax / 6` and `blockAmax / 4` -- by rounding
    /// each to E4M3, reconstructing, and keeping the candidate with the lower
    /// summed absolute error. Ties choose the 6.0 candidate.
    public static func nvfp4(_ values: [Float]) throws -> Inkling2QuantizedActivation {
        let blockSize = 16
        guard !values.isEmpty else {
            return Inkling2QuantizedActivation(codes: [], blockScales: [], outerScale: 1, blockSize: blockSize)
        }
        let rowAmax = values.reduce(Float(0)) { max($0, abs($1)) }
        let outerScale = rowAmax > 0 ? rowAmax / 1536.0 : 1.0
        guard outerScale.isFinite, outerScale > 0 else {
            throw Inkling2QuantError.invalidScale("row amax \(rowAmax) gives outer scale \(outerScale)")
        }
        var blockScales: [UInt8] = []
        var codes: [UInt8] = []
        blockScales.reserveCapacity((values.count + blockSize - 1) / blockSize)
        codes.reserveCapacity(values.count)
        var start = 0
        while start < values.count {
            let end = min(start + blockSize, values.count)
            let block = values[start..<end].map { $0 / outerScale }
            let blockAmax = block.reduce(Float(0)) { max($0, abs($1)) }
            let scale = chooseNVFP4BlockScale(block: Array(block), blockAmax: blockAmax)
            blockScales.append(Inkling2Element.encodeE4M3(scale))
            let decodedScale = Inkling2Element.decodeE4M3(Inkling2Element.encodeE4M3(scale))
            for value in block {
                codes.append(Inkling2Element.encodeE2M1(decodedScale > 0 ? value / decodedScale : 0))
            }
            start = end
        }
        return Inkling2QuantizedActivation(
            codes: codes, blockScales: blockScales, outerScale: outerScale, blockSize: blockSize
        )
    }

    /// The two-candidate scale search, in the reference's order: candidate 6.0
    /// first, and a candidate only replaces it on a STRICTLY lower error.
    static func chooseNVFP4BlockScale(block: [Float], blockAmax: Float) -> Float {
        guard blockAmax > 0 else { return 1.0 }
        var bestScale = blockAmax / 6.0
        var bestError = reconstructionError(block: block, scale: bestScale)
        let alternative = blockAmax / 4.0
        let alternativeError = reconstructionError(block: block, scale: alternative)
        if alternativeError < bestError {
            bestScale = alternative
            bestError = alternativeError
        }
        return bestScale
    }

    /// Summed absolute error of the block after E4M3-encoding the scale and
    /// E2M1-encoding each value, exactly as the reference reconstructs them.
    static func reconstructionError(block: [Float], scale: Float) -> Float {
        let stored = Inkling2Element.decodeE4M3(Inkling2Element.encodeE4M3(scale))
        guard stored > 0 else { return Float.infinity }
        return block.reduce(Float(0)) { total, value in
            let code = Inkling2Element.encodeE2M1(value / stored)
            let reconstructed = Inkling2Element.decodeE2M1(code) * stored
            return total + abs(value - reconstructed)
        }
    }

    /// MXFP8: 32-value blocks. `scale = round_up_power_of_two(blockAmax / 448)`,
    /// values clamped to +/-448 and rounded to E4M3. "Round up" is the
    /// reference's: a block whose amax already sits on a power of two keeps it.
    public static func mxfp8(_ values: [Float]) throws -> Inkling2QuantizedActivation {
        let blockSize = 32
        guard !values.isEmpty else {
            return Inkling2QuantizedActivation(codes: [], blockScales: [], outerScale: 1, blockSize: blockSize)
        }
        var blockScales: [UInt8] = []
        var codes: [UInt8] = []
        blockScales.reserveCapacity((values.count + blockSize - 1) / blockSize)
        codes.reserveCapacity(values.count)
        var start = 0
        while start < values.count {
            let end = min(start + blockSize, values.count)
            let block = Array(values[start..<end])
            let blockAmax = block.reduce(Float(0)) { max($0, abs($1)) }
            let scale = roundUpPowerOfTwo(blockAmax / 448.0)
            let scaleCode = Inkling2Element.encodeE8M0(scale)
            let decodedScale = Inkling2Element.decodeE8M0(scaleCode)
            blockScales.append(scaleCode)
            for value in block {
                let scaled = decodedScale > 0 ? value / decodedScale : 0
                let clamped = max(-448.0, min(448.0, scaled))
                codes.append(Inkling2Element.encodeE4M3(clamped))
            }
            start = end
        }
        return Inkling2QuantizedActivation(
            codes: codes, blockScales: blockScales, outerScale: 1, blockSize: blockSize
        )
    }

    /// Smallest power of two at or above `value`. Zero maps to the smallest
    /// normal exponent so a block of zeros still encodes (as zeros).
    public static func roundUpPowerOfTwo(_ value: Float) -> Float {
        guard value > 0, value.isFinite else { return 1.0 }
        let exponent = Int(value.exponent)
        let power = pow(2, Float(exponent))
        return power >= value ? power : power * 2
    }
}

// MARK: - Layer rules

nonisolated public enum Inkling2Math {

    /// `tanh(x / softcap) * softcap`.
    public static func softcap(_ value: Float, _ softcap: Float) -> Float {
        tanh(value / softcap) * softcap
    }

    /// The checkpoint's SwiGLU: the sigmoid sees the RAW gate while the
    /// multiplicative factors are softcapped. Computed in FP32.
    public static func softcappedSwiGLU(
        gate: Float,
        up: Float,
        gateSoftcap: Float = 4.0,
        upSoftcap: Float = 25.0
    ) -> Float {
        let gated = softcap(gate, gateSoftcap)
        let activated = gated * sigmoid(gate)
        return activated * softcap(up, upSoftcap)
    }

    public static func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    /// Unweighted RMS normalization: `x * rsqrt(mean(x^2) + eps)`. FP32.
    public static func rmsNorm(_ values: [Float], eps: Float) -> [Float] {
        guard !values.isEmpty else { return values }
        var squares: Float = 0
        vDSP_svesq(values, 1, &squares, vDSP_Length(values.count))
        var inverse = 1 / (squares / Float(values.count) + eps).squareRoot()
        var output = values
        vDSP_vsmul(values, 1, &inverse, &output, 1, vDSP_Length(values.count))
        return output
    }

    /// Weighted RMS normalization, the checkpoint's `Inkling2LoopRMSNorm`.
    public static func rmsNorm(_ values: [Float], gain: [Float], eps: Float) -> [Float] {
        let normalized = rmsNorm(values, eps: eps)
        var output = normalized
        let count = min(normalized.count, gain.count)
        vDSP_vmul(normalized, 1, gain, 1, &output, 1, vDSP_Length(count))
        return output
    }

    /// Scales a head vector down to `threshold` when its L2 norm is larger.
    /// `(threshold / (norm + 1e-6)).clamp(max: 1)`, so a shorter vector is
    /// unchanged rather than scaled up. FP32 in, caller's dtype out.
    public static func l2Clip(_ values: [Float], threshold: Float) -> [Float] {
        var squares: Float = 0
        vDSP_svesq(values, 1, &squares, vDSP_Length(values.count))
        let factor = min(threshold / (squares.squareRoot() + 1e-6), 1.0)
        var scale = factor
        var output = values
        vDSP_vsmul(values, 1, &scale, &output, 1, vDSP_Length(values.count))
        return output
    }

    /// `tau = 1 + alpha * log(max((position + 1) / floor, 1))`, applied on
    /// full-attention layers only.
    public static func logScale(
        position: Int,
        alpha: Float,
        floor: Int
    ) -> Float {
        let ratio = Float(position + 1) / Float(max(1, floor))
        return 1 + alpha * log(max(ratio, 1))
    }

    /// One static short-convolution tap set at a position:
    /// `(sum_k w[k] * x[t-k] + x[t]) / sqrt(kernel + 1)`.
    ///
    /// `history` is newest-last and holds at most `kernel - 1` earlier values.
    /// `weights` is indexed by BACKWARD DISTANCE (`weights[0]` weights the current
    /// token), which is NOT the order the checkpoint stores a `conv1d` weight in:
    /// `F.conv1d` is a cross-correlation, so with left padding the stored `w[k]`
    /// weights the token `kernel - 1 - k` steps back. The reference's own
    /// `Inkling2LoopShortConvolution` therefore reads its weight REVERSED
    /// relative to this function's contract, and a caller must reverse it. The
    /// dynamic convolution is the opposite: the reference builds it with
    /// `unfold(...).flip(-1)`, so its `static_weight[0]` is the current token,
    /// which is this function's order already.
    public static func staticShortConvolution(
        current: Float,
        history: [Float],
        weights: [Float]
    ) -> Float {
        let kernel = weights.count
        var convolved: Float = 0
        for tap in 0..<kernel {
            // tap index is the backward distance: weights[0] weights the current token
            let source: Float
            if tap == 0 {
                source = current
            } else {
                let back = tap - 1
                source = back < history.count ? history[history.count - 1 - back] : 0
            }
            convolved += weights[tap] * source
        }
        return (convolved + current) / Float(kernel + 1).squareRoot()
    }

    /// The dynamic form used on full-attention K/V: the taps are the static
    /// weights plus per-token weights projected from the layer input, shared
    /// within each key/value head. Normalized by `sqrt(2 * kernel + 1)`.
    public static func dynamicShortConvolution(
        current: Float,
        history: [Float],
        staticWeights: [Float],
        dynamicWeights: [Float]
    ) -> Float {
        let kernel = staticWeights.count
        var convolved: Float = 0
        for tap in 0..<kernel {
            let source: Float
            if tap == 0 {
                source = current
            } else {
                let back = tap - 1
                source = back < history.count ? history[history.count - 1 - back] : 0
            }
            convolved += (staticWeights[tap] + dynamicWeights[tap]) * source
        }
        return (convolved + current) / Float(2 * kernel + 1).squareRoot()
    }

    /// The block-attention residual read: one FP32 softmax attention of the
    /// block queries over the stacked block states, scaled by `1/sqrt(width)`.
    /// Returns one row per query.
    ///
    /// This is the rule as written, kept as the definition the vectorized form
    /// is asserted against. `blockResidualReadVectorized` computes the same
    /// expression through Accelerate; the two differ only in FP32 summation
    /// order (measured ~1e-06 relative at the published width).
    public static func blockResidualRead(
        blockStates: [[Float]],
        partial: [Float],
        queries: [[Float]]
    ) -> [[Float]] {
        var stacked = blockStates
        stacked.append(partial)
        let normalized = stacked.map { rmsNorm($0, eps: 1e-6) }
        let width = normalized.first?.count ?? 0
        guard width > 0, !queries.isEmpty else { return queries.map { _ in [] } }
        let scaling = 1 / Float(width).squareRoot()
        return queries.map { query in
            let scores = normalized.map { row -> Float in
                var dot: Float = 0
                for index in 0..<width { dot += query[index] * row[index] }
                return dot * scaling
            }
            let maximum = scores.max() ?? 0
            let exponentials = scores.map { exp($0 - maximum) }
            let denominator = exponentials.reduce(0, +)
            var output = [Float](repeating: 0, count: width)
            for (row, weight) in zip(normalized, exponentials) {
                let normalizedWeight = weight / denominator
                for index in 0..<width { output[index] += normalizedWeight * row[index] }
            }
            return output
        }
    }

    /// The block read the graph runs, with the same expression as
    /// `blockResidualRead` but each loop handed to vDSP.
    ///
    /// Nothing about the rule moves: the norm is the same sum of squares with
    /// the same `eps` folded in, the score is the same dot times the same
    /// `1/sqrt(width)`, and the softmax is the same max/shift/exp/normalize.
    /// What changes is that each of those is one vector call instead of a
    /// bounds-checked scalar loop, so the reduction tree (and therefore the
    /// last bits) is Accelerate's rather than the loop's.
    ///
    /// Measured at the published width (2,304) against the scalar form: 2.5-3.7x
    /// faster depending on the stack depth and query count, agreeing to
    /// 4e-07..1.8e-06 absolute.
    public static func blockResidualReadVectorized(
        blockStates: [[Float]],
        partial: [Float],
        queries: [[Float]]
    ) -> [[Float]] {
        var stacked = blockStates
        stacked.append(partial)
        let width = stacked.first?.count ?? 0
        guard width > 0, !queries.isEmpty else { return queries.map { _ in [] } }
        // A ragged stack is what the scalar rule would trap on, so this form
        // hands it back rather than reading past a short row.
        guard stacked.allSatisfy({ $0.count >= width }),
              queries.allSatisfy({ $0.count >= width }) else {
            return blockResidualRead(blockStates: blockStates, partial: partial, queries: queries)
        }
        let length = vDSP_Length(width)
        // The same row normalization: sum of squares, then one scaled copy.
        var normalized = stacked
        for row in normalized.indices {
            var squares: Float = 0
            vDSP_svesq(normalized[row], 1, &squares, length)
            var inverse = 1 / (squares / Float(width) + 1e-6).squareRoot()
            vDSP_vsmul(normalized[row], 1, &inverse, &normalized[row], 1, length)
        }
        let scaling = 1 / Float(width).squareRoot()
        return queries.map { query in
            var scores = [Float](repeating: 0, count: normalized.count)
            for (row, values) in normalized.enumerated() {
                vDSP_dotpr(query, 1, values, 1, &scores[row], length)
            }
            var scale = scaling
            vDSP_vsmul(scores, 1, &scale, &scores, 1, vDSP_Length(scores.count))
            var maximum: Float = 0
            vDSP_maxv(scores, 1, &maximum, vDSP_Length(scores.count))
            var shift = -maximum
            vDSP_vsadd(scores, 1, &shift, &scores, 1, vDSP_Length(scores.count))
            var exponentialCount = Int32(scores.count)
            vvexpf(&scores, scores, &exponentialCount)
            var denominator: Float = 0
            vDSP_sve(scores, 1, &denominator, vDSP_Length(scores.count))
            var inverse = denominator > 0 ? 1 / denominator : 0
            vDSP_vsmul(scores, 1, &inverse, &scores, 1, vDSP_Length(scores.count))
            var output = [Float](repeating: 0, count: width)
            var scaled = [Float](repeating: 0, count: width)
            for (row, weight) in scores.enumerated() {
                var rowWeight = weight
                vDSP_vsmul(normalized[row], 1, &rowWeight, &scaled, 1, length)
                vDSP_vadd(output, 1, scaled, 1, &output, 1, length)
            }
            return output
        }
    }

    /// `y = W x` for a `[rows][input.count]` matrix, one vDSP dot product per row.
    ///
    /// The scalar form is `linearScalar`, kept as the definition this is
    /// asserted against. The two differ only in FP32 summation order: measured
    /// on the shapes this graph asks for (a 64-wide bottleneck, a 384-wide
    /// relative projection, a 2,304-wide sink projection) they agree to ~1e-06
    /// relative and vDSP is 2.8-10.8x faster. Below 16 columns vDSP is not
    /// faster (measured 0.9-1.0x), so a short row stays scalar.
    public static func linear(_ matrix: [Float], _ input: [Float], rows: Int) -> [Float] {
        guard input.count >= 16 else { return linearScalar(matrix, input, rows: rows) }
        var output = [Float](repeating: 0, count: rows)
        let length = vDSP_Length(input.count)
        matrix.withUnsafeBufferPointer { matrixPointer in
            input.withUnsafeBufferPointer { inputPointer in
                for row in 0..<rows {
                    vDSP_dotpr(
                        matrixPointer.baseAddress! + row * input.count, 1,
                        inputPointer.baseAddress!, 1,
                        &output[row], length
                    )
                }
            }
        }
        return output
    }

    /// `y = W x` as written: one scalar accumulation chain per row.
    public static func linearScalar(_ matrix: [Float], _ input: [Float], rows: Int) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            var total: Float = 0
            let base = row * input.count
            for column in 0..<input.count { total += matrix[base + column] * input[column] }
            output[row] = total
        }
        return output
    }

    /// One cross-pass message gate: `(1 - tanh(forget)) * current + tanh(gain) * message`.
    /// `down`, `forget` and `gain` are the projections of the concatenated
    /// normalized pair, already applied.
    public static func messageGate(
        current: [Float],
        message: [Float],
        forget: [Float],
        gain: [Float]
    ) -> [Float] {
        zip(zip(current, message), zip(forget, gain)).map { pair, gates in
            let (currentValue, messageValue) = pair
            let (forgetValue, gainValue) = gates
            return (1 - tanh(forgetValue)) * currentValue + tanh(gainValue) * messageValue
        }
    }

    /// The mixture weight rule: FP32 sigmoid scores, the bias steers SELECTION
    /// only, the selected scores are renormalized and scaled by `routeScale`.
    ///
    /// The order is the reference's `torch.topk(..., sorted=False)`: the top-k
    /// come back in ascending EXPERT ID, not by score. The pair of index and
    /// weight is what a caller consumes, so a score order would still be
    /// self-consistent -- but it pairs each expert with its own weight only if
    /// the caller preserves it, and the checkpoint's own MoE loop does not
    /// assume any order. Ascending id is the one order that is also the
    /// checkpoint's, which is what a golden comparison can see.
    public static func route(
        scores: [Float],
        bias: [Float],
        topK: Int,
        routeScale: Float
    ) -> (indices: [Int], weights: [Float]) {
        let count = min(max(0, topK), scores.count)
        struct Candidate {
            let index: Int
            let rank: Float
        }
        var candidates: [Candidate] = []
        candidates.reserveCapacity(scores.count)
        for index in scores.indices {
            let steer = index < bias.count ? bias[index] : 0
            candidates.append(Candidate(index: index, rank: scores[index] + steer))
        }
        candidates.sort { left, right in
            if left.rank == right.rank { return left.index < right.index }
            return left.rank > right.rank
        }
        let selected = candidates.prefix(count).sorted { $0.index < $1.index }
        let indices: [Int] = selected.map { $0.index }
        let chosen: [Float] = indices.map { scores[$0] }
        let total = chosen.reduce(Float(0), +)
        let weights: [Float] = chosen.map { total > 0 ? $0 / total * routeScale : 0 }
        return (indices, weights)
    }

    /// Final logits: `softcap(logits, 30)` then the unpadded slice. Slice first
    /// when the caller knows the unpadded width -- this only does the softcap.
    public static func finalLogitSoftcap(_ logits: [Float], softcap: Float = 30) -> [Float] {
        logits.map { self.softcap($0, softcap) }
    }
}
