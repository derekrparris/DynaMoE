//
//  DeepSeekV41TurnTests.swift
//  DynaMoE
//
//  The turn loop's decisions, against a scripted executor: which token ends the
//  turn, how many tokens the cap allows, where each appended token is forwarded,
//  and what a cancelled turn reports.
//
//  The loop owns the arithmetic that a wrong answer hides behind plausible text: a
//  position that is off by one re-reads a cache row the previous token already
//  wrote, and an EOS set that misses an id runs a turn to its cap. Both are cheap to
//  assert here and very expensive to notice in a 510 GB run.
//

import XCTest
@testable import DynaMoE

/// Answers every position with a scripted token, and records what it was asked to
/// forward.
private final class ScriptedForward: DeepSeekV41PositionForward, @unchecked Sendable {
    let positionLimit: Int
    /// Tokens the sampler is scripted to produce, in order. Exhausted means the
    /// fake keeps returning the last value.
    var scripted: [UInt32]
    private(set) var forwardedTokens: [Int] = []
    private(set) var forwardedPositions: [Int] = []
    private(set) var resetCount = 0
    /// The sequence the fake claims is already resident. A resumed turn that names a
    /// different one is refused, the way the device executor refuses it.
    private(set) var residentTokens: [Int] = []
    private(set) var resumedPrefixes: [[Int]] = []
    /// Setting this makes the fake fail at a chosen position, so the loop's error
    /// path is exercised without a GPU.
    var failAtPosition: Int?
    /// The failure the fake raises. A context overrun is not an error to the loop
    /// (it ends the turn as length-capped), so the two are exercised separately.
    var failure: DeepSeekV41EngineError = .missingTensor("test")

    init(positionLimit: Int, scripted: [UInt32], resident: [Int] = []) {
        self.positionLimit = positionLimit
        self.scripted = scripted
        self.residentTokens = resident
    }

    var storedPositions: Int { residentTokens.count }

    func reusablePrefix(proposedPrefix: Int, promptTokenIds: [Int]) -> Int {
        let stored = residentTokens.count
        guard stored > 0, stored <= promptTokenIds.count - 1 else { return 0 }
        let cut = min(proposedPrefix, stored)
        guard cut > 0 else { return 0 }
        guard Array(promptTokenIds.prefix(cut)) == Array(residentTokens.prefix(cut)) else {
            return 0
        }
        return cut
    }

    func reset() {
        resetCount += 1
        forwardedTokens = []
        forwardedPositions = []
        residentTokens = []
    }

    func resume(prefixTokenIds: [Int]) throws {
        guard prefixTokenIds.count == residentTokens.count else {
            throw DeepSeekV41EngineError.outOfOrderPosition(
                prefixTokenIds.count, residentTokens.count
            )
        }
        if let divergence = zip(prefixTokenIds, residentTokens)
            .enumerated()
            .first(where: { $0.element.0 != $0.element.1 })?.offset {
            throw DeepSeekV41EngineError.prefixStateMismatch(position: divergence)
        }
        resumedPrefixes.append(prefixTokenIds)
        forwardedTokens = []
        forwardedPositions = []
    }

    func forwardToken(_ token: Int, position: Int) throws -> [Float] {
        if let failAtPosition, position == failAtPosition {
            throw failure
        }
        forwardedTokens.append(token)
        forwardedPositions.append(position)
        residentTokens.append(token)
        // One logit per scripted token; the loop's sampler is the injectable, so the
        // values only need a length the caller can read.
        return [0, 1, 2, 3]
    }

    /// Records that the loop asked for a chunk, and walks it the way the device
    /// executor does today: the same `forwardToken` path, so a chunk and a walk
    /// are interchangeable by construction.
    private(set) var chunkedPrompts: [[Int]] = []

    func prefillChunk(_ tokens: [Int]) throws -> DeepSeekV41ChunkLogits {
        chunkedPrompts.append(tokens)
        guard !tokens.isEmpty else { return DeepSeekV41ChunkLogits(last: []) }
        var last: [Float] = []
        for (position, token) in tokens.enumerated() {
            last = try forwardToken(token, position: position)
        }
        return DeepSeekV41ChunkLogits(last: last)
    }
}

final class DeepSeekV41TurnLoopTests: XCTestCase {

    private func settings(
        maxTokens: Int,
        eos: [UInt32] = [2],
        stops: [String] = []
    ) -> DeepSeekV41TurnRunner.Settings {
        DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: maxTokens,
            eosTokenIds: eos,
            stopSequences: stops
        )
    }

    private func run(
        forward: ScriptedForward,
        prompt: [UInt32],
        prefixTokens: Int = 0,
        settings: DeepSeekV41TurnRunner.Settings,
        decode: @escaping ([UInt32]) -> String = { $0.map(String.init).joined(separator: " ") },
        prefill: @escaping (Int, Int) -> Void = { _, _ in },
        cancelled: @escaping () -> Bool = { false }
    ) throws -> DeepSeekV41TurnRunner.Output {
        var cursor = 0
        return try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: prompt,
            prefixTokens: prefixTokens,
            settings: settings,
            sample: { _, _ in
                let value = forward.scripted[min(cursor, forward.scripted.count - 1)]
                cursor += 1
                return value
            },
            decode: decode,
            onPrefill: prefill,
            isCancelled: cancelled
        )
    }

    func testAnEOSDrawEndsTheTurnWithoutAppendingIt() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [2])
        let output = try run(
            forward: forward,
            prompt: [10, 11, 12],
            settings: settings(maxTokens: 16)
        )
        XCTAssertEqual(output.finishReason, .eos)
        XCTAssertEqual(output.tokensGenerated, 0)
        XCTAssertTrue(output.generatedTokenIds.isEmpty)
        // The prompt was forwarded, and nothing else: the EOS draw never becomes a
        // position of its own.
        XCTAssertEqual(forward.forwardedPositions, [0, 1, 2])
    }

    func testTheCapBoundsTheTurnAndReportsLengthCapped() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        let output = try run(
            forward: forward,
            prompt: [10],
            settings: settings(maxTokens: 3)
        )
        XCTAssertEqual(output.finishReason, .lengthCapped)
        XCTAssertEqual(output.tokensGenerated, 3)
        XCTAssertEqual(output.generatedTokenIds, [7, 7, 7])
    }

    func testEachAppendedTokenIsForwardedAtTheNextPosition() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        _ = try run(forward: forward, prompt: [10, 11], settings: settings(maxTokens: 3))
        // Positions 0 and 1 are the prompt; the token drawn after position 1 becomes
        // the input at position 2, and so on.
        XCTAssertEqual(forward.forwardedPositions, [0, 1, 2, 3])
        XCTAssertEqual(forward.forwardedTokens, [10, 11, 7, 7])
        XCTAssertEqual(forward.resetCount, 1)
    }

    /// A turn from position 0 is handed over as ONE chunk, which is the shape the
    /// reference's prefill takes and the only one an executor may batch. The chunk
    /// still answers with the logits of its last position, which is where the first
    /// draw comes from.
    func testAWholePromptIsPrefilledAsOneChunk() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        var progress: [(Int, Int)] = []
        let output = try run(
            forward: forward,
            prompt: [10, 11, 12],
            settings: settings(maxTokens: 2),
            prefill: { progress.append(($0, $1)) }
        )
        XCTAssertEqual(forward.chunkedPrompts, [[10, 11, 12]])
        // Positions 0...2 belong to the chunk; position 3 is the first drawn token
        // being fed back. The second draw hits the cap before it is forwarded.
        XCTAssertEqual(forward.forwardedPositions, [0, 1, 2, 3])
        XCTAssertEqual(output.promptTokens, 3)
        XCTAssertEqual(progress.count, 1, "a chunk is one progress step, not three")
        XCTAssertEqual(progress.first?.0, 3)
        XCTAssertEqual(progress.first?.1, 3)
    }

    /// A resumed turn keeps the walk. The reference has no chunked form that starts
    /// mid-sequence, so a prefix skip is never handed over as one.
    func testAResumedTurnWalksInsteadOfChunking() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7], resident: [10, 11])
        let output = try run(
            forward: forward,
            prompt: [10, 11, 12],
            prefixTokens: 2,
            settings: settings(maxTokens: 1)
        )
        XCTAssertEqual(output.prefixTokensReused, 2)
        XCTAssertTrue(forward.chunkedPrompts.isEmpty)
        // Position 2 is the one prompt position the cache did not hold; the token
        // drawn from it hits the cap before it is fed back.
        XCTAssertEqual(forward.forwardedPositions, [2])
    }

    func testAStopSequenceEndsTheTurnAndIsReported() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [5])
        let output = try run(
            forward: forward,
            prompt: [10],
            settings: settings(maxTokens: 16, stops: ["5 5"]),
            decode: { $0.map(String.init).joined(separator: " ") }
        )
        XCTAssertEqual(output.finishReason, .stoppedBySequence)
        XCTAssertEqual(output.matchedStopSequence, "5 5")
        XCTAssertEqual(output.tokensGenerated, 2)
    }

    func testTheCapIsClampedToTheRemainingContext() throws {
        // A prompt that leaves two positions must not ask the cache for a third.
        let forward = ScriptedForward(positionLimit: 5, scripted: [7])
        let output = try run(
            forward: forward,
            prompt: [10, 11, 12],
            settings: settings(maxTokens: 100)
        )
        XCTAssertEqual(output.finishReason, .lengthCapped)
        XCTAssertEqual(output.tokensGenerated, 1)
    }

    func testAPromptThatDoesNotFitIsRefusedBeforeAnyWork() throws {
        let forward = ScriptedForward(positionLimit: 4, scripted: [7])
        XCTAssertThrowsError(
            try run(forward: forward, prompt: [1, 2, 3, 4, 5], settings: settings(maxTokens: 4))
        ) { error in
            guard case DeepSeekV41TurnError.promptTooLong(let tokens, let limit) = error else {
                return XCTFail("expected promptTooLong, got \(error)")
            }
            XCTAssertEqual(tokens, 5)
            XCTAssertEqual(limit, 4)
        }
        XCTAssertEqual(forward.forwardedPositions, [])
    }

    func testCancellationKeepsWhatWasAlreadyGenerated() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        var draws = 0
        let output = try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: [10],
            settings: settings(maxTokens: 16),
            sample: { _, _ in
                draws += 1
                return 7
            },
            decode: { $0.map(String.init).joined(separator: " ") },
            isCancelled: { draws >= 2 }
        )
        XCTAssertTrue(output.cancelled)
        XCTAssertEqual(output.tokensGenerated, 2)
        XCTAssertEqual(output.text, "7 7")
    }

    func testAnIncompleteMultiByteCharacterIsHeldBackFromTheText() throws {
        // A BPE boundary in the middle of a character decodes to U+FFFD; streaming it
        // would flash a placeholder the next token then replaces.
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        let output = try run(
            forward: forward,
            prompt: [10],
            settings: settings(maxTokens: 1),
            decode: { _ in "hello\u{FFFD}" }
        )
        XCTAssertEqual(output.text, "hello")
    }

    func testAContextOverrunEndsTheTurnAsLengthCapped() throws {
        // The loop treats the cache's own refusal as the end of the turn rather than
        // as a failure: the tokens generated so far are real.
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        forward.failAtPosition = 2
        forward.failure = .contextTooLong(3, 64)
        let output = try run(forward: forward, prompt: [10], settings: settings(maxTokens: 8))
        XCTAssertEqual(output.finishReason, .lengthCapped)
        XCTAssertEqual(output.tokensGenerated, 2)
    }

    func testAFailedForwardIsReportedWithItsPosition() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        forward.failAtPosition = 2
        XCTAssertThrowsError(
            try run(forward: forward, prompt: [10], settings: settings(maxTokens: 4))
        ) { error in
            guard case DeepSeekV41TurnError.prefillFailed(let position, _) = error else {
                return XCTFail("expected prefillFailed, got \(error)")
            }
            XCTAssertEqual(position, 2)
        }
    }

    // MARK: Prefix reuse

    /// A resident prefix is continued, not replayed: the turn forwards only the
    /// prompt positions after it, and the cache is left alone.
    func testAResidentPrefixIsSkippedRatherThanReplayed() throws {
        let forward = ScriptedForward(
            positionLimit: 64, scripted: [2], resident: [10, 11]
        )
        var progress: [(done: Int, total: Int)] = []
        let output = try run(
            forward: forward,
            prompt: [10, 11, 12, 13],
            prefixTokens: 2,
            settings: settings(maxTokens: 4),
            prefill: { progress.append(($0, $1)) }
        )
        XCTAssertEqual(forward.forwardedTokens, [12, 13])
        XCTAssertEqual(forward.forwardedPositions, [2, 3])
        XCTAssertEqual(forward.resumedPrefixes, [[10, 11]])
        XCTAssertEqual(forward.resetCount, 0, "a resumed turn must not drop the cache")
        XCTAssertEqual(output.prefixTokensReused, 2)
        // Progress counts the positions this turn actually runs.
        XCTAssertEqual(progress.map(\.total), [2, 2])
        XCTAssertEqual(progress.map(\.done), [1, 2])
        XCTAssertEqual(output.promptTokens, 4)
    }

    /// The last prompt token is never skipped: its logits are what the first sampled
    /// token comes from, so a caller asking for more than the cache can give has the
    /// request cut back rather than the logits invented.
    func testAPrefixCannotCoverTheLastPromptToken() throws {
        let forward = ScriptedForward(
            positionLimit: 64, scripted: [2], resident: [10]
        )
        let output = try run(
            forward: forward,
            prompt: [10, 11],
            prefixTokens: 9,
            settings: settings(maxTokens: 4)
        )
        XCTAssertEqual(forward.forwardedTokens, [11])
        XCTAssertEqual(forward.forwardedPositions, [1])
        XCTAssertEqual(output.prefixTokensReused, 1)
    }

    /// The executor decides whether the state it holds really is this prompt's. The
    /// loop asks and fails rather than decoding from another conversation's state.
    func testAPrefixTheExecutorDoesNotHoldFailsTheTurn() {
        let forward = ScriptedForward(
            positionLimit: 64, scripted: [2], resident: [10, 99]
        )
        XCTAssertThrowsError(
            try run(
                forward: forward,
                prompt: [10, 11, 12],
                prefixTokens: 2,
                settings: settings(maxTokens: 4)
            )
        ) { error in
            guard case DeepSeekV41EngineError.prefixStateMismatch(let position) = error else {
                return XCTFail("expected prefixStateMismatch, got \(error)")
            }
            XCTAssertEqual(position, 1)
        }
    }

    func testNoPrefixResetsTheSequence() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [2], resident: [10])
        _ = try run(forward: forward, prompt: [10, 11], settings: settings(maxTokens: 4))
        XCTAssertEqual(forward.resetCount, 1)
        XCTAssertTrue(forward.resumedPrefixes.isEmpty)
        XCTAssertEqual(forward.forwardedPositions, [0, 1])
    }

    // MARK: Agent constraints

    /// The mask is driven from the text decoded so far, and it is applied before
    /// each draw rather than once per turn: a grammar state machine advances one
    /// token at a time, so a mask computed from stale text either withholds a legal
    /// tag or offers an illegal one.
    func testTheConstrainHookSeesTheTextBeforeEveryDraw() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [7])
        var seen: [String] = []
        let output = try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: [10],
            settings: settings(maxTokens: 3),
            sample: { _, _ in 7 },
            constrain: { _, text in seen.append(text) },
            decode: { $0.map(String.init).joined(separator: " ") }
        )
        // Nothing was decoded before the first draw, and prefill draws nothing.
        XCTAssertEqual(seen, ["", "7", "7 7"])
        XCTAssertEqual(output.tokensGenerated, 3)
    }

    /// The rewrite lands where the sampler reads it. The fake answers `[0, 1, 2, 3]`
    /// and the sampler takes the highest standing logit, so the drawn id follows
    /// whatever the constraint masks — a copy the mask never reached would keep
    /// drawing 3.
    func testAConstraintDecidesTheDrawnToken() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [])
        let output = try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: [10],
            settings: settings(maxTokens: 2, eos: []),
            sample: { logits, _ in
                var best: (index: UInt32, value: Float) = (0, -.infinity)
                for (index, value) in logits.enumerated() where value > best.value {
                    best = (UInt32(index), value)
                }
                return best.index
            },
            constrain: { logits, text in
                if !text.isEmpty { logits[3] = -.infinity }
            },
            decode: { $0.map(String.init).joined(separator: " ") }
        )
        XCTAssertEqual(output.generatedTokenIds, [3, 2])
    }

    /// Pre-execution catching: the turn ends on the token that completes the call,
    /// so the model never writes the result of a call that has not run.
    func testAFrozenToolCallEndsTheTurn() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [5])
        var frozenAt: [String] = []
        let output = try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: [10],
            settings: settings(maxTokens: 16),
            sample: { _, _ in 5 },
            shouldFreeze: { text in
                frozenAt.append(text)
                return text == "5 5"
            },
            decode: { $0.map(String.init).joined(separator: " ") }
        )
        XCTAssertEqual(output.finishReason, .toolFreeze)
        XCTAssertEqual(output.tokensGenerated, 2)
        XCTAssertEqual(output.text, "5 5")
        XCTAssertEqual(frozenAt, ["5", "5 5"])
    }

    /// A freeze and the token cap can land on the same token. The freeze is the
    /// true reason and is what the caller acts on, so it wins.
    func testAFreezeOnTheLastAllowedTokenIsReportedAsAFreeze() throws {
        let forward = ScriptedForward(positionLimit: 64, scripted: [5])
        let output = try DeepSeekV41TurnRunner.run(
            forward: forward,
            promptTokenIds: [10],
            settings: settings(maxTokens: 1),
            sample: { _, _ in 5 },
            shouldFreeze: { !$0.isEmpty },
            decode: { $0.map(String.init).joined(separator: " ") }
        )
        XCTAssertEqual(output.finishReason, .toolFreeze)
        XCTAssertEqual(output.tokensGenerated, 1)
    }
}
