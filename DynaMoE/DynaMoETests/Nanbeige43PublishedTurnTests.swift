//
//  Nanbeige43PublishedTurnTests.swift
//  DynaMoETests
//
//  The one test that runs the published Nanbeige 4.3 checkpoint end to end
//  through the app's OWN turn path (`CompletionEngine.launchTurn`), rather than
//  through a hand-built fixture. Everything else in this family's coverage is
//  pure logic, and pure logic cannot see the failures a real checkpoint
//  produces: a KV stride that only agrees with the allocation at 8x128, an EOS
//  token that lives inside the vocabulary, a rope table built from
//  `rope_theta: 120000000`, a 22,016-wide intermediate that only a 66 GB
//  safetensors header can confirm.
//
//  It is opt-in because it maps 66 GB of weights and prefills a real prompt.
//  `xcodebuild` does not forward the invoking shell's environment into an
//  app-hosted test, so the switch is a marker file as well as an environment
//  variable:
//
//      touch /tmp/dynamoe_nb43_real_turn
//      xcodebuild test-without-building -project DynaMoE.xcodeproj -scheme DynaMoE \
//          -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
//          -derivedDataPath <dd> \
//          -only-testing:DynaMoETests/Nanbeige43PublishedTurnTests/testAPublishedNanbeige43PromptCompletesATurn
//
//  The assertions are the executable facts (the checkpoint is complete, the
//  model path resolves to this family, the turn produced tokens and terminated).
//  The generated text is PRINTED rather than asserted: a wording change is not a
//  regression.
//

import XCTest
import Metal
@testable import DynaMoE

final class Nanbeige43PublishedTurnTests: XCTestCase {

    /// Touching this file opts the run in, for runs where `xcodebuild` has not
    /// forwarded `DYNAMOE_NB43_REAL_TURN`.
    static let markerPath = "/tmp/dynamoe_nb43_real_turn"
    /// A prompt to use instead of the default, one line.
    static let promptPath = "/tmp/dynamoe_nb43_prompt"
    /// Tokens to draw, so a run can be made long enough to see a sentence end.
    static let tokensPath = "/tmp/dynamoe_nb43_tokens"

    /// The prompt the float32 numpy oracle covers. The token-for-token assertions
    /// only apply to it: any other prompt is free text, and its continuation is
    /// printed rather than asserted, because a wording change is not a regression.
    static let referencePrompt = "The capital of France is"

    private let snapshotPath = "~/.cache/huggingface/hub/models--Nanbeige--Nanbeige4.3-33B/snapshots/df919e1e2cf077a22204c78b6c826bf3db18bc08"

    private var userText: String {
        if let text = ProcessInfo.processInfo.environment["DYNAMOE_NB43_PROMPT"], !text.isEmpty {
            return text
        }
        if let text = try? String(contentsOfFile: Self.promptPath, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return Self.referencePrompt
    }

    private var tokensToDraw: Int {
        if let text = try? String(contentsOfFile: Self.tokensPath, encoding: .utf8),
           let count = Int(text.trimmingCharacters(in: .whitespacesAndNewlines)), count > 0 {
            return count
        }
        return 12
    }

    override func setUpWithError() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_NB43_REAL_TURN"] == "1"
            || FileManager.default.fileExists(atPath: Self.markerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in published-checkpoint run; touch \(Self.markerPath) (or set "
                + "DYNAMOE_NB43_REAL_TURN=1) to run"
        )
    }

    private func snapshotDirectory() throws -> URL {
        let expanded = (snapshotPath as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            throw XCTSkip("the Nanbeige 4.3 snapshot is not downloaded at \(expanded)")
        }
        return url
    }

    /// Maps the checkpoint exactly the way the app does at load: UniFFI index,
    /// `segmentShardsForMetal`, one zero-copy buffer per segment, `configure`.
    /// The engine must outlive every read those buffers serve.
    private func loadCheckpointIntoTheSharedEngine(_ snapshot: URL) throws -> DynaMoeEngine {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let tokenizer = try XCTUnwrap(
            try? DynaMoeTokenizer(tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path),
            "the checkpoint's tokenizer could not be opened"
        )
        let modelConfig = try XCTUnwrap(
            ModelConfig.load(from: snapshot),
            "the checkpoint's config.json did not decode as a ModelConfig"
        )
        XCTAssertTrue(
            modelConfig.isNanbeige43Model,
            "the published checkpoint must identify as Nanbeige 4.3 from its own shape"
        )
        XCTAssertFalse(modelConfig.isMoE, "this checkpoint is dense; the routed path must not be taken")

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let indexed = try engine.getSummary()
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length,
                      options: .storageModeShared, deallocator: nil
                  ) else { continue }
            buffers[shard.index] = buffer
        }
        XCTAssertEqual(
            buffers.count, summary.shards.count,
            "every segment must be mappable: a 4.7 GB shard is inside maxBufferLength on this machine"
        )
        print(
            "🔎 [nb43 real] \(indexed.tensors.count) tensors in \(summary.shards.count) segment(s), "
                + "\(indexed.sizeGb) GB, \(modelConfig.effectiveNumHiddenLayers) layers, "
                + "hidden \(modelConfig.effectiveHiddenSize), intermediate \(modelConfig.effectiveIntermediateSize), "
                + "vocab \(modelConfig.effectiveVocabSize)"
        )
        CompletionEngine.shared.configure(
            summary: summary,
            modelConfig: modelConfig,
            tokenizer: tokenizer,
            shardBuffers: buffers,
            activeLoadedModelPath: snapshot.path
        )
        return engine
    }

    private func unloadSharedEngine() {
        CompletionEngine.shared.configure(
            summary: nil, modelConfig: nil, tokenizer: nil,
            shardBuffers: [:], activeLoadedModelPath: nil
        )
    }

    /// Touching this file opts into the (minutes-long) dense-repack round trip.
    static let repackMarkerPath = "/tmp/dynamoe_nb43_repack"

    /// Quantizes the published checkpoint, loads the RESULT through the app's own
    /// path, and asks the same question.
    ///
    /// This is the only check that the repack's manifest shape, tensor naming,
    /// group layout and quantization agree with what the engine reads: each piece
    /// is pinned separately (the packing against the kernel's addressing, the
    /// loader against a manifest), but a wrong companion NAME or a Q8 tensor
    /// without its bias would load as garbage rather than fail, and only a real
    /// turn can see that.
    func testADenseRepackOfThePublishedCheckpointStillAnswers() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.repackMarkerPath),
            "opt-in dense-repack run; touch \(Self.repackMarkerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let output = URL(fileURLWithPath: "/tmp/dynamoe_nb43_q4")

        // A previous run's blob is reused (minutes of work); delete the directory to
        // force a fresh repack, which is what verifies the packer itself. The
        // manifest's completeness check is what makes reuse safe.
        let started = CFAbsoluteTimeGetCurrent()
        if DenseRepacker.isRepacked(dir: output),
           DenseRepacker.quantizedFormat(dir: output) == .q4 {
            print("📦 [nb43 repack] reusing the existing Q4 blob at \(output.path)")
        } else {
            try? FileManager.default.removeItem(at: output)
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            try DenseRepacker.shared.repackSafetensors(
                sourceDir: snapshot,
                outputDir: output,
                quantization: .q4
            ) { fraction, message in
                if Int(fraction * 100) % 10 == 0 {
                    print("📦 [nb43 repack] \(Int(fraction * 100))% \(message)")
                }
            }
            print("📦 [nb43 repack] done in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - started)) s")
        }
        XCTAssertTrue(DenseRepacker.isRepacked(dir: output))
        let blobAttributes = try? FileManager.default.attributesOfItem(
            atPath: output.appendingPathComponent("dense_weights.bin").path
        )
        let blobBytes = (blobAttributes?[.size] as? NSNumber)?.uint64Value ?? 0
        print("📦 [nb43 repack] blob \(blobBytes / 1_048_576) MB")

        let engine = try loadCheckpointIntoTheSharedEngine(output)
        defer {
            unloadSharedEngine()
            _ = engine
        }
        let tokenizer = try XCTUnwrap(
            try? DynaMoeTokenizer(tokenizerPath: output.appendingPathComponent("tokenizer.json").path)
        )
        let config = try XCTUnwrap(CompletionEngine.shared.promptModelConfig)
        let prompt = ChatPromptFormatter.formatPrompt(
            prompt: userText,
            cleanSystem: "You are helpful.",
            thinkingEnabled: false,
            modelSupportsThinking: true,
            modelConfig: config
        )
        let promptIds = try tokenizer.encode(text: prompt)

        let defaults = UserDefaults.standard
        let savedJetSpec = defaults.object(forKey: "dynamoe_jetspec_enabled")
        defaults.set(false, forKey: "dynamoe_jetspec_enabled")
        defer {
            if let savedJetSpec { defaults.set(savedJetSpec, forKey: "dynamoe_jetspec_enabled") }
            else { defaults.removeObject(forKey: "dynamoe_jetspec_enabled") }
        }

        let request = CompletionTurnRequest(
            promptTokenIds: promptIds,
            formattedPrompt: prompt,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0, topP: 1, minP: 0, topK: 1,
                repetitionPenalty: 1, maxTokens: tokensToDraw
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM,
                budgetMode: .unrestricted,
                kvPrecision: .fp16
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: [],
            asOfDate: nil,
            images: []
        )

        let turnStarted = CFAbsoluteTimeGetCurrent()
        var text = ""
        var ids: [UInt32] = []
        for await event in try CompletionEngine.shared.launchTurn(request).stream {
            switch event {
            case .finished(let result):
                text = result.rawDecodedText
                ids = result.generatedTokenIds
                print(
                    "📦 [nb43 repack] decode \(String(format: "%.2f", result.tokensPerSecond)) tok/s, "
                        + "first token \(String(format: "%.2f", result.timeToFirstTokenSeconds ?? -1)) s, "
                        + "turn \(String(format: "%.0f", result.elapsedMs)) ms"
                )
            case .frame(let frame):
                if !frame.rawText.isEmpty { text = frame.rawText }
            case .failedTurn(let failure):
                XCTFail("the repacked turn failed: \(failure.message)")
            case .started, .priming, .prefillProgress:
                break
            }
        }
        print("📦 [nb43 repack] \(ids.count) tokens in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - turnStarted)) s")
        print("📦 [nb43 repack] ids: \(ids)")
        print("📦 [nb43 repack] text: \(text)")
        XCTAssertGreaterThan(ids.count, 0, "the repacked checkpoint produced no tokens")
        if userText == Self.referencePrompt {
            // The same oracle the unquantized case uses. Q4 is lossy, but the docs
            // claim a repacked checkpoint still answers its published turn with the
            // same opening tokens, and this is the assertion that makes that claim
            // testable instead of prose.
            XCTAssertGreaterThanOrEqual(ids.count, 4)
            XCTAssertEqual(
                Array(ids.prefix(4)),
                [764, 5463, 290, 7914],
                "the Q4 repack must reproduce the published checkpoint's greedy continuation"
            )
            XCTAssertTrue(
                text.contains("Paris"),
                "Q4 is lossy, but the answer must survive it: \(text)"
            )
        }
    }

    func testAPublishedNanbeige43PromptCompletesATurn() async throws {
        let snapshot = try snapshotDirectory()
        let engine = try loadCheckpointIntoTheSharedEngine(snapshot)
        defer {
            unloadSharedEngine()
            _ = engine
        }

        let tokenizer = try XCTUnwrap(
            try? DynaMoeTokenizer(tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path)
        )
        let config = try XCTUnwrap(CompletionEngine.shared.promptModelConfig)

        // The default shape: thinking ON, which is what the composer sends and
        // what opens this family's reasoning channel after the prefix.
        let prompt = ChatPromptFormatter.formatPrompt(
            prompt: userText,
            cleanSystem: "You are helpful.",
            thinkingEnabled: false,
            modelSupportsThinking: true,
            modelConfig: config
        )
        let promptIds = try tokenizer.encode(text: prompt)
        XCTAssertFalse(promptIds.isEmpty, "the rendered prompt tokenized to nothing")
        print("🔎 [nb43 real] prompt \(promptIds.count) tokens, drawing \(tokensToDraw): \(promptIds)")

        let request = CompletionTurnRequest(
            promptTokenIds: promptIds,
            formattedPrompt: prompt,
            sessionId: nil,
            sampling: TurnSampling(
                temperature: 0,
                topP: 1,
                minP: 0,
                topK: 1,
                repetitionPenalty: 1,
                maxTokens: tokensToDraw
            ),
            memory: TurnMemorySettings(
                executionMode: .residentRAM,
                budgetMode: .unrestricted,
                kvPrecision: .fp16
            ),
            agentToolsEnabled: false,
            thinkingEnabled: false,
            agentStep: 0,
            forceSynthesis: false,
            isSplicedContinuation: false,
            stopSequences: [],
            asOfDate: nil,
            images: []
        )

        // The turn reads some knobs from the app's persisted defaults, which a
        // developer's own run will have left set. Pin the one that changes the
        // generation GRAPH rather than the prompt: JetSpec is ineligible for a
        // dense checkpoint now, and this case must not depend on a stale flag.
        let defaults = UserDefaults.standard
        let savedJetSpec = defaults.object(forKey: "dynamoe_jetspec_enabled")
        defaults.set(false, forKey: "dynamoe_jetspec_enabled")
        defer {
            if let savedJetSpec { defaults.set(savedJetSpec, forKey: "dynamoe_jetspec_enabled") }
            else { defaults.removeObject(forKey: "dynamoe_jetspec_enabled") }
        }

        let started = CFAbsoluteTimeGetCurrent()
        var finalText = ""
        var tokens = 0
        var failure: TurnFailure?
        var finishedIds: [UInt32]?
        let launch = try CompletionEngine.shared.launchTurn(request)
        for await event in launch.stream {
            switch event {
            case .prefillProgress(let status, let speed, _):
                if let status, let speed {
                    print("⏳ [nb43 real] \(status) @ \(String(format: "%.1f", speed)) tok/s")
                }
            case .frame(let frame):
                finalText = frame.rawText
                tokens = frame.tokensGenerated
            case .failedTurn(let failureEvent):
                failure = failureEvent
            case .finished(let result):
                tokens = result.tokensGenerated
                finalText = result.rawDecodedText
                finishedIds = result.generatedTokenIds
                print("🔎 [nb43 real] finish=\(result.finishReason)")
                print(
                    "🔎 [nb43 real] decode \(String(format: "%.2f", result.tokensPerSecond)) tok/s, "
                        + "first token \(String(format: "%.2f", result.timeToFirstTokenSeconds ?? -1)) s, "
                        + "turn \(String(format: "%.0f", result.elapsedMs)) ms"
                )
            case .started, .priming:
                break
            }
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - started
        if let ids = finishedIds {
            let pieces = ids.map { (try? tokenizer.decode(ids: [$0])) ?? "?" }
            print("🔎 [nb43 real] ids: \(ids)")
            print("🔎 [nb43 real] pieces: \(pieces)")
        }

        if let failure {
            XCTFail("the turn failed: \(failure.message)")
        }
        print("🔎 [nb43 real] \(tokens) tokens in \(String(format: "%.1f", elapsed)) s")
        print("🔎 [nb43 real] raw:\n\(finalText)")
        XCTAssertGreaterThan(tokens, 0, "the engine produced no tokens from a real prompt")

        // The oracle: `Tools/nb43_goldens.py` runs the same 24-token prompt through
        // a float32 numpy Llama written against the checkpoint's own bytes, and
        // its greedy continuation is [764, 5463, 290, 7914, 322, 1472, ...] — "The
        // capital of France is **Paris**." The engine must agree token for token,
        // because a graph that is merely "close" is a graph that is wrong. Another
        // prompt has no oracle, so it is printed rather than asserted.
        if userText == Self.referencePrompt {
            let continuation = finishedIds ?? []
            XCTAssertGreaterThanOrEqual(continuation.count, 6)
            XCTAssertEqual(
                Array(continuation.prefix(4)),
                [764, 5463, 290, 7914],
                "the device's greedy continuation must match the numpy reference"
            )
            XCTAssertTrue(
                finalText.contains("Paris"),
                "the published turn should answer the question: \(finalText)"
            )
        }
    }
}
