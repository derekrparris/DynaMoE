//
//  DeepSeekV41VisionCost.swift
//  DynaMoE
//
//  Phase: DeepSeek V4.1 vision, the composer's image path (slice #38)
//
//  What an attachment will cost, priced from the resize plan BEFORE the tower
//  runs. The measured range across images the checkpoint accepts is wide enough
//  to matter: 0.32 s for the smallest and 6.90-6.95 s for the largest on the shipped
//  device encode, and 2.82-42.20 s on the host tower it replaced (`#43`), so a
//  composer that encodes silently turns a large photo into what looks like a hang.
//

import Foundation

/// The tower's measured cost as a function of the patch-token count.
///
/// Solves `fixed + perToken * t + perTokenSquared * t^2`. On the shipped device
/// encode the linear term is the projections, norms and glue, and the quadratic
/// term is attention plus softmax; the constants come from the published
/// checkpoint on the app's own optimized build (`Benchmarks/PERF_FINDINGS.md`,
/// "DSV4.1 #43"). They are a model of this machine class, not a promise, which is
/// why the estimate is shown as an approximation rather than used to refuse
/// anything. `hostReference` keeps the CPU tower's own model (`#37`) beside it.
nonisolated public struct DeepSeekV41VisionCostModel: Sendable, Equatable {
    public let fixed: Double
    public let perToken: Double
    public let perTokenSquared: Double

    public init(fixed: Double, perToken: Double, perTokenSquared: Double) {
        self.fixed = fixed
        self.perToken = perToken
        self.perTokenSquared = perTokenSquared
    }

    /// The shipped path: the device-resident encode.
    ///
    /// Measured at three published sizes: 0.32 s at 1,521 patch tokens, 1.70 s at
    /// 3,969 and 6.90-6.95 s at 8,649. Fitted with no constant term (all three
    /// points land within 4%), because the tower uploads its weights once per
    /// lifetime and the span cache reuses an encode, so a per-image constant would
    /// price an upload a picture usually does not pay. That one-time upload is
    /// ~0.35 s.
    public static let measured = DeepSeekV41VisionCostModel(
        fixed: 0,
        perToken: 8.4e-5,
        perTokenSquared: 8.32e-8
    )

    /// The CPU tower's model, measured before the device encode (`#37`):
    /// 0.22 s + 1.126 ms per patch token + 4.245e-07 s per token^2, i.e. 2.74 s at
    /// the smallest image and 41.71 s at the largest. Kept as the oracle's cost.
    public static let hostReference = DeepSeekV41VisionCostModel(
        fixed: 0.22,
        perToken: 1.126e-3,
        perTokenSquared: 4.245e-7
    )

    public func seconds(patchTokens: Int) -> Double {
        let tokens = Double(max(0, patchTokens))
        return fixed + perToken * tokens + perTokenSquared * tokens * tokens
    }
}

nonisolated public enum DeepSeekV41VisionPricing {

    /// What one image of this size is expected to cost.
    public struct Estimate: Sendable, Equatable {
        /// Patches the tower runs, i.e. its token count.
        public let patchTokens: Int
        /// The aligner grid, which is what the prompt pays for.
        public let columns: Int
        public let rows: Int
        /// Prompt positions the span takes, newlines and delimiters included.
        public let spanPositions: Int
        /// Expected tower time, on the measured model above.
        public let seconds: Double
    }

    /// Prices an image from the checkpoint's own resize plan, without decoding a
    /// single weight: `fit` is the same function the tower calls, so the grid here
    /// is the grid the encode will use.
    public static func estimate(
        width: Int,
        height: Int,
        spec: DeepSeekV41VisionSpec,
        costModel: DeepSeekV41VisionCostModel = .measured
    ) -> Estimate? {
        guard width > 0, height > 0, spec.patchSize > 0 else { return nil }
        let fitted = DeepSeekV41VisionLayout.fit(width: width, height: height, spec: spec)
        guard fitted.columns > 0, fitted.rows > 0 else { return nil }
        let patchTokens = (fitted.width / spec.patchSize) * (fitted.height / spec.patchSize)
        guard patchTokens > 0 else { return nil }
        return Estimate(
            patchTokens: patchTokens,
            columns: fitted.columns,
            rows: fitted.rows,
            spanPositions: fitted.tokens,
            seconds: costModel.seconds(patchTokens: patchTokens)
        )
    }
}
