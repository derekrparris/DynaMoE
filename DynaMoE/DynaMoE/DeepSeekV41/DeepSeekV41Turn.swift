//
//  DeepSeekV41Turn.swift
//  DynaMoE
//
//  The turn loop for a DeepSeek-V4.1 checkpoint: prefill, decode, and the stop
//  conditions that end a turn.
//
//  It is deliberately a separate type from `CompletionEngine`'s generic loop rather
//  than a branch inside it. That loop binds per-head K/V caches, a single QKV
//  projection, a `model.`-rooted catalog, JetSpec trees and the FlashMoE packed
//  expert files; none of that describes this architecture, whose per-position entry
//  point is `DeepSeekV41MetalForward.forwardToken` and whose cache is the CSA2 ring
//  the runtime owns. Sharing the loop would mean every one of those decisions
//  becoming a conditional, and the failure mode of getting one wrong is text that
//  looks fine and is not.
//
//  Everything the loop needs from the rest of the app arrives as a closure, so this
//  type stays `nonisolated` and can be tested without a model, a tokenizer or a GPU:
//
//    * `sample` draws one token from a position's logits,
//    * `decode` turns ids into text (the UI re-decodes the whole sequence each step
//      so a multi-byte character split across tokens is never frozen half-formed),
//    * `onText` receives the accumulated text for a streaming frame,
//    * `isCancelled` is the cooperative-stop check, consulted at every position.
//
//  Stop conditions are the ones that actually terminate a turn here: EOS, the
//  per-turn token cap, the checkpoint's context limit, a caller-supplied stop
//  sequence, and the degenerate-cycle guard. The chat UI's reasoning split and tool
//  parsing happen in the caller, on the text this loop produces.
//

import Foundation

nonisolated public enum DeepSeekV41TurnError: Error, CustomStringConvertible {
    /// The prompt does not fit the provisioned cache.
    case promptTooLong(promptTokens: Int, limit: Int)
    /// Nothing to prefill: sampling from no positions would read empty logits.
    case emptyPrompt
    /// The block forward refused a position.
    case prefillFailed(position: Int, underlying: String)

    public var description: String {
        switch self {
        case let .promptTooLong(promptTokens, limit):
            return "⚠️ DeepSeek V4.1 needs \(promptTokens) positions but its cache was provisioned "
                + "for \(limit); raise the max-token budget or shorten the prompt."
        case .emptyPrompt:
            return "⚠️ DeepSeek V4.1 received a prompt that tokenized to 0 tokens."
        case let .prefillFailed(position, underlying):
            return "❌ DeepSeek V4.1 prefill failed at position \(position): \(underlying)"
        }
    }
}

// MARK: - Position source

/// What the turn loop needs from the executor: reset, a per-position forward that
/// returns logits, and the position limit.
///
/// A protocol rather than the concrete `DeepSeekV41MetalForward` so the loop's stop
/// conditions, token cap, cancellation and position arithmetic can be tested without
/// a 510 GB checkpoint or a GPU. The device executor is the only real conformance.
nonisolated public protocol DeepSeekV41PositionForward: AnyObject {
    /// Positions the cache was provisioned for.
    var positionLimit: Int { get }
    /// Positions the sequence already holds, or 0 when the cache is empty.
    var storedPositions: Int { get }
    /// How much of `promptTokenIds` this executor can actually skip.
    ///
    /// The caller's `proposedPrefix` comes from the app-wide prefix pin, which knows
    /// the conversation but not what this cache holds. Only the executor can answer:
    /// the resident sequence may be shorter than the pin, or be a different
    /// conversation's entirely, and resuming one of those mixes two sequences inside
    /// an n-gram with no error anywhere. Zero means "prefill the prompt in full".
    func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int
    /// Drops the sequence so the next forward starts at position 0.
    func reset()
    /// Keeps the resident sequence and the token history that produced it, so the
    /// next `forwardToken` may continue at `prefixTokenIds.count`.
    ///
    /// Refuses rather than rebuilds. The history an n-gram hashes over includes
    /// flags a replay cannot invent (an image span takes no part in one), so the
    /// executor is asked to keep the state it has: an idle cache or a history that
    /// does not spell out these tokens is an error, not something to guess at.
    func resume(prefixTokenIds: [Int]) throws
    /// Appends one position and returns its logits.
    func forwardToken(_ token: Int, position: Int) throws -> [Float]
}

nonisolated public enum DeepSeekV41TurnRunner {

    /// What one turn needs from the caller's settings.
    struct Settings: Sendable {
        var temperature: Float
        var topP: Float
        var minP: Float
        var topK: Int
        var repetitionPenalty: Float
        var presencePenalty: Float
        var maxTokens: Int
        /// Every id that ends the turn. The checkpoint's own list, not a guess: a
        /// wrong EOS set either runs to the token cap or stops mid-sentence.
        var eosTokenIds: [UInt32]
        var stopSequences: [String]

        init(
            temperature: Float,
            topP: Float,
            minP: Float,
            topK: Int,
            repetitionPenalty: Float,
            presencePenalty: Float = 0,
            maxTokens: Int,
            eosTokenIds: [UInt32],
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
    struct Output: Sendable {
        var generatedTokenIds: [UInt32]
        /// The decoded text of `generatedTokenIds`, without any trailing U+FFFD an
        /// incomplete multi-byte character would have produced.
        var text: String
        var finishReason: TurnFinishReason
        var matchedStopSequence: String?
        var tokensGenerated: Int
        var promptTokens: Int
        /// Prompt positions this turn did not have to run because the cache already
        /// held them.
        var prefixTokensReused: Int
        var prefillSeconds: Double
        var decodeSeconds: Double
        var tokensPerSecond: Double
        var timeToFirstTokenSeconds: Double?
        var cancelled: Bool
    }

    /// Runs one turn.
    ///
    /// - Parameters:
    ///   - forward: the executor for the loaded checkpoint. Its cache is reset here
    ///     unless `prefixTokens` says part of this prompt is already resident.
    ///   - promptTokenIds: the rendered prompt, already tokenized.
    ///   - prefixTokens: leading prompt positions the cache already holds, from
    ///     `DeepSeekV41CacheAddressing.reusablePrefix`. Zero means a turn from
    ///     position 0.
    ///   - settings: sampling and the turn's stop conditions.
    ///   - sample: draws one id from a position's logits.
    ///   - decode: decodes the whole generated sequence so far.
    ///   - onPrefill: `(positions done, positions total)`, once per position.
    ///   - onText: `(text so far, tokens generated, elapsed seconds)` at most once
    ///     per token.
    ///   - isCancelled: consulted before every position and every sample.
    static func run(
        forward: any DeepSeekV41PositionForward,
        promptTokenIds: [UInt32],
        prefixTokens: Int = 0,
        settings: Settings,
        sample: ([Float], [UInt32]) -> UInt32,
        decode: ([UInt32]) -> String,
        onPrefill: (Int, Int) -> Void = { _, _ in },
        onText: (String, Int, Double) -> Void = { _, _, _ in },
        isCancelled: () -> Bool = { false }
    ) throws -> Output {
        let limit = forward.positionLimit
        guard !promptTokenIds.isEmpty else { throw DeepSeekV41TurnError.emptyPrompt }
        guard promptTokenIds.count <= limit else {
            throw DeepSeekV41TurnError.promptTooLong(
                promptTokens: promptTokenIds.count, limit: limit
            )
        }

        // A turn starts at position 0 unless the caller says a prefix of this prompt
        // is already resident. `reset` also drops any partial compressor group, which
        // is what keeps a cancelled turn from leaking a half-filled block into the
        // next one; a resumed turn must not reset, because the half-filled group is
        // this sequence's own and the skip is what the cache was kept for.
        let resumeFrom = max(0, min(prefixTokens, promptTokenIds.count - 1))
        if resumeFrom > 0 {
            try forward.resume(prefixTokenIds: promptTokenIds.prefix(resumeFrom).map { Int($0) })
        } else {
            forward.reset()
        }

        // MARK: Prefill
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let newPositions = promptTokenIds.count - resumeFrom
        var logits: [Float] = []
        for offset in resumeFrom..<promptTokenIds.count {
            if isCancelled() {
                return cancelledOutput(
                    generated: [], text: "", promptTokens: promptTokenIds.count,
                    prefixTokensReused: resumeFrom,
                    prefillSeconds: CFAbsoluteTimeGetCurrent() - prefillStart,
                    decodeSeconds: 0, firstToken: nil
                )
            }
            do {
                logits = try forward.forwardToken(Int(promptTokenIds[offset]), position: offset)
            } catch {
                throw DeepSeekV41TurnError.prefillFailed(
                    position: offset, underlying: "\(error)"
                )
            }
            onPrefill(offset + 1 - resumeFrom, newPositions)
        }
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        // MARK: Decode
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated: [UInt32] = []
        var text = ""
        var firstTokenSeconds: Double?
        var finishReason: TurnFinishReason = .stopped
        var matchedStopSequence: String?
        var cancelled = false

        // A prompt that fills the cache can only be answered by nothing at all, which
        // is a token cap of zero rather than a crash in `forwardToken`.
        let cap = max(0, min(settings.maxTokens, limit - promptTokenIds.count - 1))

        while generated.count < cap {
            if isCancelled() {
                cancelled = true
                break
            }
            let next = sample(logits, generated)
            if settings.eosTokenIds.contains(next) {
                finishReason = .eos
                break
            }
            generated.append(next)
            if firstTokenSeconds == nil {
                firstTokenSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
            }
            // Re-decoded in whole: a BPE token boundary can split a multi-byte
            // character, and a partial sequence decodes to U+FFFD. Holding the
            // placeholder back one frame keeps it from flashing mid-stream.
            let decoded = decode(generated)
            text = decoded.hasSuffix("\u{FFFD}") ? String(decoded.dropLast()) : decoded
            onText(text, generated.count, CFAbsoluteTimeGetCurrent() - decodeStart)

            if let sequence = settings.stopSequences.first(where: { !$0.isEmpty && text.contains($0) }) {
                finishReason = .stoppedBySequence
                matchedStopSequence = sequence
                break
            }
            if TurnTextSplitter.detectDegenerateCycle(
                tokenIds: generated,
                decodeUnit: { decode($0) }
            ) != nil {
                finishReason = .degenerateCycle
                break
            }
            if generated.count >= cap {
                finishReason = .lengthCapped
                break
            }
            // The token just appended has not been through the block yet; it is the
            // input at the position after the prompt.
            let position = promptTokenIds.count + generated.count - 1
            do {
                logits = try forward.forwardToken(Int(next), position: position)
            } catch DeepSeekV41EngineError.contextTooLong {
                finishReason = .lengthCapped
                break
            } catch {
                throw DeepSeekV41TurnError.prefillFailed(
                    position: position, underlying: "\(error)"
                )
            }
        }
        if generated.count >= cap, finishReason == .stopped {
            finishReason = .lengthCapped
        }

        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart
        return Output(
            generatedTokenIds: generated,
            text: text,
            finishReason: finishReason,
            matchedStopSequence: matchedStopSequence,
            tokensGenerated: generated.count,
            promptTokens: promptTokenIds.count,
            prefixTokensReused: resumeFrom,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: decodeSeconds > 0 ? Double(generated.count) / decodeSeconds : 0,
            timeToFirstTokenSeconds: firstTokenSeconds,
            cancelled: cancelled
        )
    }

    private static func cancelledOutput(
        generated: [UInt32],
        text: String,
        promptTokens: Int,
        prefixTokensReused: Int,
        prefillSeconds: Double,
        decodeSeconds: Double,
        firstToken: Double?
    ) -> Output {
        Output(
            generatedTokenIds: generated,
            text: text,
            finishReason: .stopped,
            matchedStopSequence: nil,
            tokensGenerated: generated.count,
            promptTokens: promptTokens,
            prefixTokensReused: prefixTokensReused,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds,
            tokensPerSecond: 0,
            timeToFirstTokenSeconds: firstToken,
            cancelled: true
        )
    }
}
