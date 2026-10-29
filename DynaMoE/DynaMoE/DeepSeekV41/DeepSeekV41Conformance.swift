//
//  DeepSeekV41Conformance.swift
//  DynaMoE
//
//  Which parts of the port have been matched against the checkpoint's own
//  reference implementation, and which are still inferred from the architecture
//  description.
//
//  The distinction exists because the two failure modes are not symmetric. A wrong
//  tensor name or a wrong dtype fails closed at load. A wrong *numeric rule* (the
//  Engram hash combination, the compressor's pooling, a prompt marker spelling)
//  runs at full speed and produces plausible text, which is far worse: the model
//  looks like it is working.
//
//  So each such rule carries a gate here. The load log prints the unverified ones,
//  `Tools/dsv41_probe.py` dumps the checkpoint's own files (it does not compute
//  reference outputs), and flipping a gate to `.verifiedAgainstReference` is a
//  one-line change accompanied by a test that carries vectors produced by a real
//  reference run. The gates are diagnostics rather than switches, with
//  one deliberate exception: the device forward refuses an Engram layer while the
//  hash combination is unverified, because reading the wrong rows is silent.
//

import Foundation

/// Whether the family can serve a turn yet. Detection and the clear refusal stay
/// on unconditionally; discovery and the graded-effort UI read this so a user is
/// not invited to download a ~510 GB checkpoint that can only fail closed.
nonisolated public enum DeepSeekV41Availability {
    public static let isRuntimeAvailable = false
}

nonisolated public enum DeepSeekV41Conformance {

    public enum Gate: String, Sendable, Equatable {
        /// Inferred from the architecture description and the tensor shapes.
        case unverified
        /// Matched against the checkpoint's own reference implementation, with the
        /// reference vectors checked into `DeepSeekV41LogicTests`.
        case verifiedAgainstReference

        public var isVerified: Bool { self == .verifiedAgainstReference }
    }

    /// How a run of positions pools into one compressed row.
    public static let compressorPooling: Gate = .unverified

    /// How the compressed-row candidate blocks are scored and truncated.
    ///
    /// Both backends currently read rows through
    /// `DeepSeekV41CacheAddressing.attentionRowPlan`, which is value-free: the
    /// count a position reads is known on the host before any dispatch. The
    /// block-score path stays a recorded diagnostic (the reference keeps
    /// `candidateScores`, the device path does not compute them) because a
    /// value-dependent selection needs a per-layer readback of the selected count,
    /// and paying that before the rule is verified would slow every token down to
    /// reproduce a guess.
    public static let candidateSelection: Gate = .unverified

    /// How retrieved Engram rows are combined into the layer's residual.
    ///
    /// Still an INFERENCE and known to disagree with the reference in two places:
    /// `Engram.forward` forms `kv = wkv(embed(hashes).flatten(-2))`, splits it into a
    /// per-hc-copy key and one shared value, and gates the VALUE by a normalized
    /// dot product against the residual stream; the port instead softmaxes the rows
    /// per head and adds the result to the post-`ffn_norm` input. Rewiring it moves
    /// where the contribution lands in the residual, so it is its own pass.
    public static let engramCombination: Gate = .unverified

    /// The mHC hyper-connection mixing rule.
    public static let hyperConnections: Gate = .unverified

    /// DSpark's draft-chain construction.
    public static let dsparkDrafting: Gate = .unverified

    /// The vision tower's patch grid and image token layout.
    public static let visionPatching: Gate = .unverified

    /// Every gate, paired with the label the load log prints.
    public static var all: [(name: String, gate: Gate)] {
        [
            ("Engram hashing", DeepSeekV41EngramHasher.conformance == .verifiedAgainstReference
                ? .verifiedAgainstReference : .unverified),
            ("compressor pooling", compressorPooling),
            ("Engram combination", engramCombination),
            ("candidate selection", candidateSelection),
            ("hyper-connections", hyperConnections),
            ("DSpark drafting", dsparkDrafting),
            ("vision patching", visionPatching)
        ]
    }

    /// The unverified rules, phrased for the load log. Empty means every rule was
    /// matched against the reference.
    public static var unverifiedSummary: [String] {
        all.filter { !$0.gate.isVerified }.map(\.name)
    }
}
