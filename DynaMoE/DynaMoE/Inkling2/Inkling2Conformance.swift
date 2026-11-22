//
//  Inkling2Conformance.swift
//  DynaMoE
//
//  Which parts of the Inkling 2 port have been matched against the
//  checkpoint's own reference implementation, and which are still inferred
//  from the architecture description.
//
//  The distinction exists because the two failure modes are not symmetric. A
//  wrong tensor name or a wrong dtype fails closed at load. A wrong NUMERIC rule
//  (a convolution's normalization, the router's renormalization, a query row
//  order, a prompt marker) runs at full speed and produces plausible text,
//  which is far worse: the model looks like it is working.
//
//  So each such rule carries a gate here. The load log prints the unverified
//  ones, and flipping a gate to `.verifiedAgainstReference` is a one-line change
//  that comes WITH a test carrying vectors from a real reference run. The gates
//  are diagnostics, not switches: a stage the port implements runs whether or
//  not its rule has been matched.
//

import Foundation

/// Whether the family can serve a turn.
///
/// TRUE for TEXT turns: the host graph runs the published checkpoint through
/// `Inkling2TurnRunner`, which `CompletionEngine.launchTurn` calls, and
/// `testAPublishedPromptCompletesATurnThroughTheRuntime` runs exactly that.
/// What is still missing is named here rather than hidden behind the flag: no
/// image path (the vision tower is not wired into this family's turns), no
/// device (Metal) graph, no speculative verify, and no tool dialect -- so an
/// agent turn ends on EOS or the cap instead of freezing on a call.
nonisolated public enum Inkling2Availability {
    public static let isRuntimeAvailable = true
}

nonisolated public enum Inkling2Conformance {

    public enum Gate: String, Sendable, Equatable {
        /// Inferred from the architecture description and the tensor shapes.
        case unverified
        /// Matched against the checkpoint's own reference implementation, with
        /// the reference vectors checked into the test suite.
        case verifiedAgainstReference

        public var isVerified: Bool { self == .verifiedAgainstReference }
    }

    /// The six-layer loop schedule and the 48 visits it produces.
    ///
    /// Derived from `loop_config` and the reference's own `total_num_hidden_layers`
    /// property, and pinned by the visit plan's tests (eight full-attention
    /// visits at slots 5, 11, 17, 23, 29, 35, 41, 47). The reference runs the
    /// core layers `num_passes` times and restarts each pass from the same
    /// pre-loop state, which is what makes a visit's KV distinct from its
    /// physical layer's.
    public static let loopSchedule: Gate = .verifiedAgainstReference

    /// Where the attention-residual block stack is read and written.
    ///
    /// The reference appends at `layer_idx % n_layers_per_attn_res_block == 0`
    /// and reads a query over the stack plus the current partial. The residual
    /// read is pinned by tests against a hand-computed stack; the queries'
    /// unpacking (which checkpoint row belongs to which layer) is pinned
    /// separately by `middle_cycle` below.
    public static let blockResidual: Gate = .unverified

    /// The four logical cross-pass queries and how the checkpoint's two
    /// 24-row tensors unpack into them.
    ///
    /// READ FROM THE REFERENCE MODULE STRUCTURE: `middle_cycle_cross_pass_queries`
    /// rows `2i`/`2i+1` are core layer `i`'s Q and MLP queries, and the `_kv_`
    /// tensor's same rows are its K and V. The plan requires this pinned by a
    /// golden before any turn runs, and Phase 1 has no reference run to pin it
    /// with, so it stays open here rather than being asserted from a guess.
    public static let middleCycle: Gate = .unverified

    /// The short convolutions' normalization constants.
    ///
    /// `(conv(x) + x) / sqrt(kernel + 1)` for the static form and
    /// `(conv(x) + x) / sqrt(2 * kernel + 1)` for the dynamic one, transcribed
    /// from `Inkling2LoopShortConvolution.forward` and its dynamic sibling. The
    /// host primitive is unit-tested against a hand-computed window, but no
    /// reference run has produced a vector yet.
    public static let shortConvolution: Gate = .unverified

    /// Relative-attention bias: `R @ bank`, gathered by backward distance and
    /// zeroed outside `0 <= distance < rel_extent`.
    public static let relativeBias: Gate = .unverified

    /// The NVFP4 activation quantizer's two-candidate scale search.
    ///
    /// Transcribed from the vLLM `nvfp4.py` rule (candidates `amax/6` and
    /// `amax/4`, E4M3-rounded, lowest summed absolute reconstruction error wins,
    /// ties to 6.0) and pinned by a test against an independent Python mirror.
    /// No reference run of the checkpoint's own quantized experts has been
    /// compared yet.
    public static let nvfp4Activation: Gate = .unverified

    /// The MXFP8 activation quantizer: power-of-two E8M0 scales.
    public static let mxfp8Activation: Gate = .unverified

    /// The router: FP32 sigmoid scores, the bias steering selection only,
    /// selected scores renormalized and scaled by `route_scale`.
    ///
    /// MATCHED AGAINST `Inkling2LoopTopKRouter.forward`, which is eight lines:
    /// `scores = sigmoid(logits)`, `topk(scores + bias)`, `weights = scores`
    /// at those indices, normalized by their sum, times `routed_scaling_factor`.
    /// The host rule is pinned by a test using the published geometry.
    public static let router: Gate = .verifiedAgainstReference

    /// The softcapped SwiGLU:
    /// `tanh(gate/4)*4 * sigmoid(gate) * tanh(up/25)*25`.
    public static let softcappedSwiGLU: Gate = .verifiedAgainstReference

    /// The prompt dialect, against the checkpoint's `chat_template.jinja`.
    ///
    /// Rendered by jinja2 itself (a pinned Python environment) for the text
    /// cases the Phase 1 renderer covers, and pinned by exact-string goldens.
    /// Tool declarations are rendered by the same template but are Phase 4.
    public static let promptText: Gate = .verifiedAgainstReference

    /// The expert layouts, inferred from stored shapes and dtypes.
    ///
    /// The plan's rule: shapes and quant metadata are authoritative, not the
    /// parameter-group labels. The two layouts are told apart by what only one
    /// of them has (a U8 value bank with E4M3 scales is NVFP4; the reverse is
    /// MXFP8), and the input strides are checked against that.
    public static let expertLayout: Gate = .verifiedAgainstReference

    /// Every gate in the ledger, so the summary cannot miss one.
    public static let all: [(name: String, gate: Gate)] = [
        ("loopSchedule", loopSchedule),
        ("blockResidual", blockResidual),
        ("middleCycle", middleCycle),
        ("shortConvolution", shortConvolution),
        ("relativeBias", relativeBias),
        ("nvfp4Activation", nvfp4Activation),
        ("mxfp8Activation", mxfp8Activation),
        ("router", router),
        ("softcappedSwiGLU", softcappedSwiGLU),
        ("promptText", promptText),
        ("expertLayout", expertLayout)
    ]

    public static var unverifiedSummary: [String] {
        all.filter { !$0.gate.isVerified }.map(\.name)
    }
}
