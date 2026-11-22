//
//  Inkling2Turn.swift
//  DynaMoE
//
//  The turn loop for an Inkling 2 Mini checkpoint: prefill, decode and the stop
//  conditions that end a turn.
//
//  It is a separate type from `CompletionEngine`'s generic loop rather than a
//  branch inside it, for the reason the DeepSeek port records: that loop binds
//  per-head K/V caches, one QKV projection, a `model.`-rooted catalog, JetSpec
//  trees and packed expert files, none of which describes this architecture.
//  Sharing it would turn every one of those decisions into a conditional, and
//  the failure mode of getting one wrong is text that looks fine and is not.
//
//  Everything the loop needs from the rest of the app arrives as a closure, so
//  this type stays `nonisolated` and is testable without a model, a tokenizer or
//  a GPU:
//
//    * `sample` draws one token from a position's logits,
//    * `constrain` may rewrite a position's logits in place before `sample`
//      sees them (a grammar mask belongs there),
//    * `shouldFreeze` ends the turn the moment its text completes a tool call,
//    * `decode` turns ids into text (the UI re-decodes the whole sequence each
//      step so a multi-byte character split across tokens is never frozen
//      half-formed),
//    * `onText` receives the accumulated text for a streaming frame,
//    * `isCancelled` is the cooperative-stop check, consulted at every position.
//
//  Stop conditions: EOS (200006), the per-turn token cap, the checkpoint's
//  context limit, a caller-supplied stop sequence, the tool-call freeze and a
//  degenerate-cycle guard.
//

import Foundation

// MARK: - Errors

nonisolated public enum Inkling2TurnError: Error, Equatable, CustomStringConvertible {
    case emptyPrompt
    case promptTooLong(promptTokens: Int, limit: Int)
    case prefillFailed(position: Int, underlying: String)

    public var description: String {
        switch self {
        case .emptyPrompt:
            return "⚠️ Inkling 2 received a prompt that tokenized to 0 tokens."
        case .promptTooLong(let promptTokens, let limit):
            return "⚠️ Inkling 2 needs \(promptTokens) positions but its cache was provisioned for "
                + "\(limit); raise the max-token budget or shorten the prompt."
        case .prefillFailed(let position, let underlying):
            return "❌ Inkling 2 prefill failed at position \(position): \(underlying)"
        }
    }
}

// MARK: - Position source

/// What the turn loop needs from an executor: reset, position limits, prefix
/// reuse and a per-position forward returning logits.
///
/// A protocol rather than the concrete host or device forward, so the loop's
/// stop conditions, token cap, cancellation and position arithmetic are tested
/// without a checkpoint.
nonisolated public protocol Inkling2PositionForward: AnyObject {
    /// Positions the cache was provisioned for.
    var positionLimit: Int { get }
    /// Positions the sequence already holds.
    var storedPositions: Int { get }
    /// How much of `promptTokenIds` this executor can actually skip.
    ///
    /// The caller's proposal comes from the app-wide prefix pin, which knows the
    /// conversation but not what this cache holds; only the executor can answer.
    /// Zero means "prefill the prompt in full".
    func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int
    /// Drops the sequence so the next forward starts at position 0.
    func reset()
    /// Appends one position and returns its logits.
    func forwardToken(_ token: Int, position: Int) throws -> [Float]
}

public extension Inkling2PositionForward {
    /// Conservative default: nothing is provably resident, so nothing is skipped.
    /// A wrong resume mixes two sequences with no error anywhere, which is worse
    /// than re-prefilling.
    func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int { 0 }
}

// MARK: - The host forward as a position source

extension Inkling2CPUForward: Inkling2PositionForward {
    /// The context ceiling: what the forward was provisioned for when the
    /// engine set a KV budget, the checkpoint's declared maximum otherwise. A
    /// turn past it ends as length-capped rather than as a refusal one position
    /// later.
    public var positionLimit: Int { options.maxPositions ?? config.maxPositionEmbeddings }

    public var storedPositions: Int { states.first?.committedPositions ?? 0 }

    public func forwardToken(_ token: Int, position: Int) throws -> [Float] {
        try forwardToken(token: UInt32(token), position: position)
    }
}

// MARK: - Runner

nonisolated public enum Inkling2TurnRunner {

    /// What one turn needs from the caller's settings.
    public struct Settings: Sendable, Equatable {
        public var temperature: Float
        public var topP: Float
        public var minP: Float
        public var topK: Int
        public var repetitionPenalty: Float
        public var presencePenalty: Float
        public var maxTokens: Int
        /// Every id that ends the turn. The checkpoint's own list, not a guess:
        /// a wrong EOS set either runs to the cap or stops mid-sentence.
        public var eosTokenIds: [UInt32]
        public var stopSequences: [String]

        public init(
            temperature: Float = 0.6,
            topP: Float = 0.95,
            minP: Float = 0,
            topK: Int = 20,
            repetitionPenalty: Float = 1,
            presencePenalty: Float = 0,
            maxTokens: Int = 4096,
            eosTokenIds: [UInt32] = [200006],
            stopSequences: [String] = []
        ) {
            self.temperature = temperature
            self.topP = topP
            self.minP = minP
            self.topK = topK
            self.repetitionPenalty = repetitionPenalty
            self.presencePenalty = presencePenalty
            self.maxTokens = maxTokens
            self.eosTokenIds = eosTokenIds
            self.stopSequences = stopSequences
        }
    }

    /// What the caller needs to build a `TurnResult`.
    struct Output: Sendable, Equatable {
        public var generatedTokenIds: [UInt32]
        public var text: String
        public var finishReason: TurnFinishReason
        public var matchedStopSequence: String?
        public var tokensGenerated: Int
        public var promptTokens: Int
        public var prefixTokensReused: Int
        /// Prompt positions the cache actually holds after this turn: shorter
        /// than the prompt when a cancellation stopped it partway, and exactly
        /// that many ids are pinned by the caller.
        public var storedPromptTokens: Int
        public var cancelled: Bool
    }

    /// Runs one turn.
    ///
    /// - Parameters:
    ///   - forward: the executor. Its cache is reset unless `prefixTokens` says
    ///     part of this prompt is resident.
    ///   - promptTokenIds: the rendered prompt, already tokenized.
    ///   - prefixTokens: leading prompt positions the cache already holds.
    ///   - settings: sampling and the turn's stop conditions.
    ///   - sample: draws one id from a position's logits, given the ids so far.
    ///   - constrain: rewrites a position's logits in place, with the text
    ///     decoded so far. Masked tokens are `-inf`, so it is destructive.
    ///   - shouldFreeze: `(text) -> completes a tool call`. Called after each
    ///     appended token.
    ///   - decode: decodes the generated sequence so far.
    ///   - onPrefill: `(positions done, total)`.
    ///   - onText: `(text, tokens generated)` at most once per token.
    ///   - isCancelled: consulted before every position and every draw.
    static func run(
        forward: any Inkling2PositionForward,
        promptTokenIds: [Int],
        prefixTokens: Int = 0,
        settings: Settings,
        sample: ([Float], [UInt32]) -> UInt32,
        constrain: (inout [Float], String) -> Void = { _, _ in },
        shouldFreeze: (String) -> Bool = { _ in false },
        decode: ([UInt32]) -> String,
        onPrefill: (Int, Int) -> Void = { _, _ in },
        onText: (String, Int) -> Void = { _, _ in },
        isCancelled: () -> Bool = { false }
    ) throws -> Output {
        let limit = forward.positionLimit
        guard !promptTokenIds.isEmpty else { throw Inkling2TurnError.emptyPrompt }
        guard promptTokenIds.count <= limit else {
            throw Inkling2TurnError.promptTooLong(promptTokens: promptTokenIds.count, limit: limit)
        }
        let reusable = max(0, min(prefixTokens, promptTokenIds.count - 1))
        if reusable > 0 {
            _ = forward.reusablePrefix(proposedPrefix: reusable, promptTokenIds: promptTokenIds)
        } else {
            forward.reset()
        }

        // Prefill: every prompt position runs; the last one's logits are the
        // first draw's.
        var logits: [Float] = []
        for (position, token) in promptTokenIds.enumerated() {
            if isCancelled() {
                return Output(
                    generatedTokenIds: [], text: "", finishReason: .stopped,
                    matchedStopSequence: nil, tokensGenerated: 0,
                    promptTokens: promptTokenIds.count, prefixTokensReused: 0,
                    storedPromptTokens: forward.storedPositions, cancelled: true
                )
            }
            do {
                logits = try forward.forwardToken(token, position: position)
            } catch {
                throw Inkling2TurnError.prefillFailed(position: position, underlying: "\(error)")
            }
            onPrefill(position + 1, promptTokenIds.count)
        }

        var generated: [UInt32] = []
        var text = ""
        var finishReason: TurnFinishReason = .lengthCapped
        var matchedStop: String?
        var cancelled = false
        // A cycle guard: the same short window repeating means the turn will not
        // produce anything new, which on a broken graph is a common shape.
        var recent: [UInt32] = []

        while generated.count < settings.maxTokens {
            if isCancelled() { cancelled = true; break }
            let nextPosition = promptTokenIds.count + generated.count
            if nextPosition >= limit { finishReason = .lengthCapped; break }
            var constrained = logits
            constrain(&constrained, text)
            let token = sample(constrained, generated)
            generated.append(token)
            text = decode(generated)
            onText(text, generated.count)

            if settings.eosTokenIds.contains(token) {
                finishReason = .eos
                break
            }
            if let stop = settings.stopSequences.first(where: { !$0.isEmpty && text.hasSuffix($0) }) {
                finishReason = .stoppedBySequence
                matchedStop = stop
                break
            }
            if shouldFreeze(text) {
                finishReason = .toolFreeze
                break
            }
            recent.append(token)
            if recent.count > 24 { recent.removeFirst() }
            if recent.count == 24, Set(recent).count <= 2 {
                finishReason = .degenerateCycle
                break
            }
            do {
                logits = try forward.forwardToken(Int(token), position: nextPosition)
            } catch {
                throw Inkling2TurnError.prefillFailed(position: nextPosition, underlying: "\(error)")
            }
        }

        return Output(
            generatedTokenIds: generated,
            text: text,
            finishReason: cancelled ? .stopped : finishReason,
            matchedStopSequence: matchedStop,
            tokensGenerated: generated.count,
            promptTokens: promptTokenIds.count,
            prefixTokensReused: reusable,
            storedPromptTokens: forward.storedPositions,
            cancelled: cancelled
        )
    }
}
