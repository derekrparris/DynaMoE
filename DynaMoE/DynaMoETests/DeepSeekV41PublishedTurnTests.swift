//
//  DeepSeekV41PublishedTurnTests.swift
//  DynaMoETests
//
//  The one test that runs the published checkpoint end to end: load the 48 shards,
//  prefill a rendered prompt, draw tokens from the logits the device produced, and
//  continue from the sequence the first turn left resident.
//
//  Everything else in the DeepSeek suite either runs on a synthetic checkpoint or
//  stops at the engine's doorstep. That is deliberate for the default suite, but it
//  also means no test has ever seen a real weight reach a real token: the geometry
//  pins, the parity cases and the goldens all describe a model that has never run.
//  The failures that survive those checks are the ones that only a published tensor
//  shape can produce (a stride the synthetic widths divide evenly, an expert slot
//  sized from a config field a fixture leaves at zero), and they surface here.
//
//  It is opt-in because it reads a few hundred gigabytes: a first turn pulls every
//  routed expert it needs from NVMe, and the second one exercises `resume` rather
//  than paying for a second full prefill. `xcodebuild` does not forward the
//  invoking shell's environment into an app-hosted test, so the switch is a marker
//  file as well as an environment variable. Run it with
//
//      touch /tmp/dynamoe_dsv41_real_turn
//      xcodebuild test-without-building ... \
//          -only-testing:DynaMoETests/DeepSeekV41PublishedTurnTests
//
//  and read the printed lines. The test asserts the executable facts it can (the
//  checkpoint is complete, the runtime is runnable, tokens were produced) and
//  prints the rest, because "the model answered" is not something a threshold can
//  decide.
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41PublishedTurnTests: XCTestCase {

    /// Touching this file opts the run in, for the runs where `xcodebuild` has not
    /// forwarded `DYNAMOE_DSV41_REAL_TURN`.
    static let markerPath = "/tmp/dynamoe_dsv41_real_turn"
    /// Touching this file runs the turn with the Engram layers silenced, which is
    /// how the memory path is ruled in or out as the cause of a bad answer.
    static let engramOffMarkerPath = "/tmp/dynamoe_dsv41_engram_off_test"
    /// A prompt to use instead of the default, one line.
    static let promptPath = "/tmp/dynamoe_dsv41_prompt"
    /// Tokens to draw, for a run long enough to see the arena's steady state.
    static let tokensPath = "/tmp/dynamoe_dsv41_tokens"

    /// The prompt, readable from the environment or from a file beside the marker,
    /// because `xcodebuild` does not forward the invoking shell's environment into
    /// an app-hosted test. Short on purpose: every position costs a full 40-layer
    /// pass, and the first turn pays the expert reads for all of them.
    private var userText: String {
        if let text = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_PROMPT"], !text.isEmpty {
            return text
        }
        if let text = try? String(contentsOfFile: Self.promptPath, encoding: .utf8) {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { return trimmed }
        }
        return "hey"
    }

    /// Tokens the turn may draw. Enough to see a sentence start, small enough that
    /// a wrong graph is not paid for 200 times. Every token is a full 40-layer pass
    /// over freshly streamed experts, so this is the test's whole cost. A number in
    /// the marker file raises it, which is how a run measures the arena once it has
    /// filled instead of during its first few cold tokens.
    private var firstTurnTokens: Int {
        if let text = try? String(contentsOfFile: Self.tokensPath, encoding: .utf8),
           let count = Int(text.trimmingCharacters(in: .whitespacesAndNewlines)), count > 0 {
            return count
        }
        return 8
    }

    override func setUpWithError() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_REAL_TURN"] == "1"
            || FileManager.default.fileExists(atPath: Self.markerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in published-checkpoint run; touch \(Self.markerPath) (or set "
                + "DYNAMOE_DSV41_REAL_TURN=1) to run"
        )
    }

    // MARK: - Harness

    private func snapshotDirectory() throws -> URL {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                ".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots"
            )
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: nil
        ) else {
            throw XCTSkip("the DeepSeek-V4.1 hub directory is missing")
        }
        guard let snapshot = entries.first(where: {
            FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("config.json").path
            ) && FileManager.default.fileExists(
                atPath: $0.appendingPathComponent("model.safetensors.index.json").path
            )
        }) else {
            throw XCTSkip("the DeepSeek-V4.1 snapshot is not downloaded")
        }
        return snapshot
    }

    /// The same loading sequence the app performs: index the checkpoint, split any
    /// shard past `maxBufferLength`, map every segment zero-copy, then hand the
    /// summary to the runtime. A test that built its own summary or its own buffers
    /// would not be running the graph the app runs.
    private func makeRuntime(
        snapshot: URL,
        engine: DynaMoeEngine,
        config: DeepSeekV41Config,
        contextLength: Int,
        allowEngramSkip: Bool
    ) throws -> DeepSeekV41Runtime {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let indexed = try engine.getSummary()
        print(
            "🔎 [dsv41 real] indexed \(indexed.tensors.count) tensors in "
                + "\(indexed.shards.count) shard(s), \(indexed.sizeGb) GB"
        )
        let summary = try InferenceEngine.segmentShardsForMetal(
            indexed, maxBufferLength: UInt64(device.maxBufferLength)
        )
        var buffers: [UInt32: MTLBuffer] = [:]
        var mapped = 0
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer,
                      length: length,
                      options: .storageModeShared,
                      deallocator: nil
                  )
            else {
                print("⚠️ [dsv41 real] shard #\(shard.index) (\(shard.filename)) was not mapped")
                continue
            }
            buffers[shard.index] = buffer
            mapped += length
        }
        print(
            "🔎 [dsv41 real] mapped \(buffers.count)/\(summary.shards.count) segment(s), "
                + String(format: "%.1f GB", Double(mapped) / 1_073_741_824)
        )

        guard let runtime = DeepSeekV41Runtime(
            snapshotDir: snapshot,
            summary: summary,
            config: config,
            shardBuffers: buffers,
            contextLength: contextLength,
            allowEngramSkip: allowEngramSkip
        ) else {
            throw XCTSkip(
                "the runtime did not build: "
                    + (DeepSeekV41Runtime.lastInitializationError ?? "no reason recorded")
            )
        }
        return runtime
    }

    /// The draws the graph made, printed for the first few so a run that produces
    /// no words still says whether the logits were flat or confident.
    ///
    /// It also carries the arena's byte counter, because a decoded token's cost is
    /// the expert bytes it misses on divided by the disk's rate: a token that reads
    /// more than the six experts per layer it routed is an arena problem, and one
    /// that reads exactly that is the hit rate to fix.
    private final class DrawLog: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        private var previousBytes = -1
        private var previousHits = 0
        private var previousAcquisitions = 0
        private var prefillBytes = 0
        private var perTokenBytes: [Int] = []
        private var perTokenHits: [(hits: Int, acquisitions: Int)] = []

        func record(
            logits: [Float],
            drawn: UInt32,
            generated: Int,
            bytesRead: Int,
            hits: Int,
            acquisitions: Int
        ) {
            lock.lock()
            defer { lock.unlock() }
            if count < 3 {
                let top = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
                let shown = top.map { "\($0.offset):\(String(format: "%.2f", $0.element))" }
                print("🔎 [dsv41 real] draw \(generated): \(drawn) top5 \(shown.joined(separator: " "))")
            }
            if previousBytes < 0 {
                prefillBytes = bytesRead
                print(
                    "🔎 [dsv41 real] prefill read "
                        + String(format: "%.2f GB", Double(bytesRead) / 1_073_741_824)
                )
            } else {
                let delta = bytesRead - previousBytes
                perTokenBytes.append(delta)
                perTokenHits.append((hits - previousHits, acquisitions - previousAcquisitions))
                if perTokenBytes.count <= 32 {
                    let ratio = Double(hits - previousHits) / Double(max(1, acquisitions - previousAcquisitions))
                    print(
                        "🔎 [dsv41 real] token \(generated): read "
                            + String(format: "%.1f MB, %.0f%% hit", Double(delta) / 1_048_576, ratio * 100)
                    )
                }
            }
            previousBytes = bytesRead
            previousHits = hits
            previousAcquisitions = acquisitions
            count += 1
        }

        /// The steady-state view: the last half of the draws, which is after the
        /// arena has filled, because a turn's first tokens are always cold.
        func summary() -> String {
            lock.lock()
            defer { lock.unlock() }
            guard !perTokenBytes.isEmpty else { return "no decode draws" }
            let tail = perTokenBytes.suffix(max(1, perTokenBytes.count / 2))
            let tailHits = perTokenHits.suffix(max(1, perTokenHits.count / 2))
            let meanBytes = tail.reduce(0, +) / tail.count
            let acquisitions = tailHits.reduce(0) { $0 + $1.acquisitions }
            let hits = tailHits.reduce(0) { $0 + $1.hits }
            return String(
                format: "prefill %.2f GB, steady-state %.1f MB/token at %.0f%% hit over %d token(s)",
                Double(prefillBytes) / 1_073_741_824,
                Double(meanBytes) / 1_048_576,
                Double(hits) / Double(max(1, acquisitions)) * 100,
                tail.count
            )
        }
    }

    /// Greedy, so two runs of this test are the same run. The app's sampler is
    /// exercised by its own tests; what this one needs is a token the graph
    /// actually ranked first.
    private nonisolated static func argmax(_ logits: [Float]) -> UInt32 {
        var best = 0
        var bestValue = -Float.greatestFiniteMagnitude
        for (index, value) in logits.enumerated() where value > bestValue {
            bestValue = value
            best = index
        }
        return UInt32(clamping: best)
    }

    // MARK: - The turn

    /// Whether the graph's logits depend on the token it was given at all.
    ///
    /// A published checkpoint that answers every prompt with the same first token is
    /// not slow or badly prompted, it is not reading its input: the head, the final
    /// collapse or the embedding is constant. Two single-position forwards are the
    /// cheapest way to ask, and they print the top five logits for each so a flat
    /// distribution and a confident constant are distinguishable.
    func testTwoDifferentTokensProduceDifferentLogits() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let forward = runtime.forward()
        XCTAssertTrue(forward.isRunnable, forward.describe())

        let probe = try await Task.detached { [runtime] () -> ([Float], [Float], [Float]) in
            let executor = runtime.forward()
            executor.reset()
            let a = try executor.forwardToken(47493, position: 0)
            executor.reset()
            let b = try executor.forwardToken(301, position: 0)
            executor.reset()
            let c = try executor.forwardToken(47493, position: 0)
            return (a, b, c)
        }.value
        func top(_ values: [Float]) -> String {
            values.enumerated().sorted { $0.element > $1.element }.prefix(5)
                .map { "\($0.offset):\(String(format: "%.3f", $0.element))" }
                .joined(separator: " ")
        }
        print("🔎 [dsv41 real] token 47493: \(top(probe.0))")
        print("🔎 [dsv41 real] token   301: \(top(probe.1))")
        print("🔎 [dsv41 real] token 47493 again: \(top(probe.2))")
        // What the graph leaves in its own workspace after one position. A collapsed
        // stream far from the head's input scale, or a normalized vector with no
        // spread, says the residual stack is not carrying what the head expects.
        func statistics(_ buffer: MTLBuffer, count: Int) -> String {
            let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
            var mean: Float = 0
            var maxValue = -Float.greatestFiniteMagnitude
            var minValue = Float.greatestFiniteMagnitude
            var square: Float = 0
            for index in 0..<count {
                let value = pointer[index]
                mean += value
                square += value * value
                maxValue = max(maxValue, value)
                minValue = min(minValue, value)
            }
            let n = Float(count)
            let average = mean / n
            let variance = max(0, square / n - average * average)
            return String(
                format: "mean %.4f std %.4f min %.3f max %.3f",
                average, variance.squareRoot(), minValue, maxValue
            )
        }
        print("🔎 [dsv41 real] hidden: \(statistics(runtime.workspace.hidden, count: config.hiddenSize))")
        print("🔎 [dsv41 real] normed: \(statistics(runtime.workspace.output, count: config.hiddenSize))")
        print("🔎 [dsv41 real] logits: \(statistics(runtime.workspace.logits, count: min(config.vocabSize, runtime.workspace.logits.length / 4)))")
        // The last three stages, dumped for a NumPy check: the final stack, the mix
        // that collapses it, the collapsed vector, its norm and the head's output.
        func dump(_ buffer: MTLBuffer, count: Int, to path: String) {
            let data = Data(bytes: buffer.contents(), count: count * MemoryLayout<Float>.stride)
            try? data.write(to: URL(fileURLWithPath: path))
        }
        dump(
            runtime.workspace.streams,
            count: max(1, config.hcMult) * config.hiddenSize,
            to: "/tmp/dsv41_streams.bin"
        )
        dump(
            runtime.workspace.hcMix,
            count: max(1, config.hcMult),
            to: "/tmp/dsv41_mix.bin"
        )
        dump(runtime.workspace.hidden, count: config.hiddenSize, to: "/tmp/dsv41_hidden.bin")
        dump(runtime.workspace.output, count: config.hiddenSize, to: "/tmp/dsv41_normed.bin")
        dump(
            runtime.workspace.logits,
            count: min(config.vocabSize, runtime.workspace.logits.length / 4),
            to: "/tmp/dsv41_logits.bin"
        )
        print("🔎 [dsv41 real] dumped streams/normed/logits to /tmp/dsv41_*.bin")
        if let tokenizer = try? DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        ) {
            let topIds = probe.2.enumerated().sorted { $0.element > $1.element }
                .prefix(5).map { UInt32($0.offset) }
            let words = topIds.map { (try? tokenizer.decode(ids: [$0])) ?? "" }
            print("🔎 [dsv41 real] top ids: \(Array(zip(topIds, words)))")
        }
        // Does a position's output depend on the row BEFORE it? Same last token, same
        // position, only the context row differs. Identical logits would mean the
        // query attends over its own row alone, which is exactly what a decode loop
        // that repeats one token forever looks like from the outside.
        let context = try await Task.detached { [runtime] () -> ([Float], [Float]) in
            let executor = runtime.forward()
            executor.reset()
            _ = try executor.forwardToken(47493, position: 0)
            let a = try executor.forwardToken(47493, position: 1)
            executor.reset()
            _ = try executor.forwardToken(301, position: 0)
            let b = try executor.forwardToken(47493, position: 1)
            return (a, b)
        }.value
        let contextDelta = zip(context.0, context.1).map { abs($0 - $1) }.max() ?? 0
        let contextDifferences = zip(context.0, context.1).filter { abs($0 - $1) > 1e-3 }.count
        print("🔎 [dsv41 real] context row changes \(contextDifferences) logits, max delta \(contextDelta)")
        print("🔎 [dsv41 real]   first row 47493: \(top(context.0))")
        print("🔎 [dsv41 real]   first row   301: \(top(context.1))")
        let differences = zip(probe.0, probe.1).filter { abs($0 - $1) > 1e-3 }.count
        print("🔎 [dsv41 real] logits differing between the two tokens: \(differences)/\(probe.0.count)")
        XCTAssertGreaterThan(
            differences, 0,
            "the head produced the same logits for two different tokens"
        )
        XCTAssertEqual(probe.0, probe.2)
        withExtendedLifetime(engine) {}
    }

    /// The block prefill against the walk, at the checkpoint's real widths.
    ///
    /// The parity suite checks this on a synthetic model, whose widths are chosen to
    /// fit a test. The published tensor shapes are what can break a batched stage:
    /// a stride the fixture divides evenly, a head count that changes the grid
    /// height, an expert set that no longer fits the arena. A turn prefills through
    /// the block, so a disagreement here is a wrong answer on every prompt while the
    /// walk keeps working.
    func testABlockPrefillsLikeTheWalkAtThePublishedWidths() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 16,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let tokens = [0, 128803, 47493]
        let probe = try await Task.detached { [runtime] () -> ([Float], [Float]) in
            let executor = runtime.forward()
            executor.reset()
            var walked: [Float] = []
            for (position, token) in tokens.enumerated() {
                walked = try executor.forwardToken(token, position: position)
            }
            executor.reset()
            let blocked = try executor.prefillChunk(tokens).last
            return (walked, blocked)
        }.value
        func top(_ values: [Float]) -> String {
            values.enumerated().sorted { $0.element > $1.element }.prefix(5)
                .map { "\($0.offset):\(String(format: "%.3f", $0.element))" }
                .joined(separator: " ")
        }
        print("🔎 [dsv41 real] walk  last: \(top(probe.0))")
        print("🔎 [dsv41 real] block last: \(top(probe.1))")
        let largest = zip(probe.0, probe.1).map { abs($0 - $1) }.max() ?? 0
        let differs = zip(probe.0, probe.1).filter { abs($0 - $1) > 1e-2 }.count
        print("🔎 [dsv41 real] block vs walk: \(differs) logits differ, max delta \(largest)")
        XCTAssertEqual(
            Int(Self.argmax(probe.0)), Int(Self.argmax(probe.1)),
            "the block and the walk disagree on the next token"
        )
        withExtendedLifetime(engine) {}
    }

    /// Layer 0 alone, at the published widths, against the NumPy reference.
    ///
    /// `Tools/dsv41_layer_reference.py` computes the same layer from the same shard
    /// bytes in the checkpoint's own arithmetic, so this is the one comparison that
    /// is not the port grading itself. The runtime is built from a summary filtered
    /// to layer 0 and a config with one layer, which is what makes the final
    /// collapse, norm and head read layer 0's output directly.
    func testLayerZeroMatchesTheNumPyReference() async throws {
        let snapshot = try snapshotDirectory()
        var config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        config.numHiddenLayers = 1
        config.compressRatios = [0]
        config.engramLayerIDs = []
        config.engramRowCounts = []
        config.kvSourceLayerIDs = []
        config.indexSourceLayerIDs = []
        config.candidateSourceLayerID = -1
        config.dsparkTargetLayerIDs = []
        config.numNextNPredictLayers = 0

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let indexed = try engine.getSummary()
        let kept = indexed.tensors.filter { tensor in
            ["embed.weight", "norm.weight", "head.weight"].contains(tensor.name)
                || tensor.name.hasPrefix("layers.0.")
        }
        print("🔎 [dsv41 real] one-layer summary keeps \(kept.count)/\(indexed.tensors.count) tensors")
        let summary = ModelSummary(
            sizeGb: indexed.sizeGb,
            tensorCount: UInt32(kept.count),
            layerCount: 1,
            maxExpertId: indexed.maxExpertId,
            shards: indexed.shards,
            tensors: kept,
            layers: []
        )
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let length = Int(shard.length)
            guard length > 0, length <= Int(device.maxBufferLength),
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)),
                  let buffer = device.makeBuffer(
                      bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
                  )
            else { continue }
            buffers[shard.index] = buffer
        }
        guard let runtime = DeepSeekV41Runtime(
            snapshotDir: snapshot,
            summary: summary,
            config: config,
            shardBuffers: buffers,
            contextLength: 4
        ) else {
            throw XCTSkip(
                "the one-layer runtime did not build: "
                    + (DeepSeekV41Runtime.lastInitializationError ?? "no reason recorded")
            )
        }
        print("🔎 [dsv41 real] one-layer: \(runtime.describe())")
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        XCTAssertTrue(executor.isRunnable, executor.describe())

        let probe = try XCTUnwrap(device.makeBuffer(
            length: max(1, config.hcMult) * config.hiddenSize * 4, options: .storageModeShared
        ))
        let mixes = try XCTUnwrap(device.makeBuffer(
            length: (2 + config.hcMult) * config.hcMult * 4, options: .storageModeShared
        ))
        let before = try XCTUnwrap(device.makeBuffer(
            length: max(1, config.hcMult) * config.hiddenSize * 4, options: .storageModeShared
        ))
        executor.parityProbes = DeepSeekV41ParityProbes(
            beforeAttention: before, attentionLogits: mixes, afterAttention: probe
        )
        let logits = try await Task.detached { [runtime] in
            let executor = runtime.forward()
            return try executor.forwardToken(47493, position: 0)
        }.value
        let top = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
        print("🔎 [dsv41 real] one-layer logits: "
            + top.map { "\($0.offset):\(String(format: "%.3f", $0.element))" }.joined(separator: " "))
        // The reference's top five for this token. Only the RANKING is compared:
        // both sides accumulate in fp32 but not in the same order, so the values
        // agree to a few hundredths while the order is exact. A wrong head-group
        // stride changes the ranking, which is how this test earns its keep.
        XCTAssertEqual(top.map { Int($0.offset) }, [62535, 99801, 65820, 100489, 101125])

        func dump(_ buffer: MTLBuffer, count: Int, to path: String) {
            let data = Data(bytes: buffer.contents(), count: count * MemoryLayout<Float>.stride)
            try? data.write(to: URL(fileURLWithPath: path))
        }
        dump(
            probe,
            count: max(1, config.hcMult) * config.hiddenSize,
            to: "/tmp/one_layer_attention.bin"
        )
        dump(runtime.workspace.hidden, count: config.hiddenSize, to: "/tmp/one_layer_hidden.bin")
        dump(runtime.workspace.logits, count: config.vocabSize, to: "/tmp/one_layer_logits.bin")
        dump(
            mixes,
            count: (2 + config.hcMult) * config.hcMult,
            to: "/tmp/one_layer_mixes.bin"
        )
        dump(
            before,
            count: max(1, config.hcMult) * config.hiddenSize,
            to: "/tmp/one_layer_before.bin"
        )
        dump(
            runtime.workspace.reduced,
            count: config.oLoraRank * config.oGroups,
            to: "/tmp/one_layer_reduced.bin"
        )
        dump(
            runtime.workspace.query,
            count: config.numAttentionHeads * config.qkHeadDim,
            to: "/tmp/one_layer_query.bin"
        )
        dump(
            runtime.workspace.latent,
            count: config.cacheRecordWidth,
            to: "/tmp/one_layer_latent.bin"
        )
        dump(
            runtime.workspace.attentionOut,
            count: config.numAttentionHeads * config.vHeadDim,
            to: "/tmp/one_layer_attention_out.bin"
        )
        dump(
            runtime.workspace.hcAttentionPre,
            count: config.hcMult,
            to: "/tmp/one_layer_attn_pre.bin"
        )
        dump(
            runtime.workspace.hcAttentionPost,
            count: config.hcMult,
            to: "/tmp/one_layer_attn_post.bin"
        )
        dump(
            runtime.workspace.hcAttentionComb,
            count: config.hcMult * config.hcMult,
            to: "/tmp/one_layer_attn_comb.bin"
        )
        let mixesPointer = mixes.contents().assumingMemoryBound(to: Float.self)
        let mixValues = (0..<((2 + config.hcMult) * config.hcMult)).map { mixesPointer[$0] }
        print("🔎 [dsv41 real] one-layer attn mixes: " + mixValues.map {
            String(format: "%.4f", $0)
        }.joined(separator: " "))
        print("🔎 [dsv41 real] one-layer dumps written to /tmp/one_layer_*.bin")
        withExtendedLifetime(engine) {}
    }

    /// How fast the arena can fill, layer by layer.
    ///
    /// This is the stage a turn spends most of its wall time in: every position reads
    /// six experts per layer from the shards. Timing it on its own says whether a
    /// slow turn is the disk, the pread path or the arena, without a profiler and
    /// without a GPU dispatch in the way.
    func testTheArenaFillThroughput() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 8,
            allowEngramSkip: false
        )
        let streamer = try XCTUnwrap(runtime.expertStreamer)
        let layers = [0, 1, 2, 20, 39]
        for layer in layers {
            let experts = Array(0..<config.numActivatedExperts)
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try streamer.ensureResident(layer: layer, experts: experts)
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            print(
                "🔎 [dsv41 real] layer \(layer) arena fill: "
                    + String(format: "%.3f s for %d experts", seconds, experts.count)
            )
        }
        print("🔎 [dsv41 real] \(streamer.describe())")
        withExtendedLifetime(engine) {}
    }

    /// The conditioning row a DSpark draft would start from, on the real checkpoint.
    ///
    /// The drafter reads `main_hidden` — the target layers' block inputs, means over
    /// the hyper-connection copies, concatenated in ascending layer order — through
    /// `main_proj`. This prints what the device actually captured for a few positions
    /// so the row can be checked against the geometry the checkpoint publishes
    /// (`mtp.0.main_proj` is [5120, 15360], i.e. three 5,120-wide layers end to end)
    /// before anything is built on top of it. A capture that silently wrote zeros, or
    /// one layer's row three times, would otherwise only show up as an acceptance
    /// rate of zero much later.
    func testWhatTheDraftersConditioningRowHoldsOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        XCTAssertEqual(config.dsparkTargetLayerIDs, [37, 38, 39])
        XCTAssertEqual(config.dsparkBlockSize, 5)
        let tokens = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16, 128804, 128822
        ]
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: tokens.count,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            executor.reset()
            executor.capturesDraftInput = true
            let targets = runtime.config.dsparkTargetLayerIDs.sorted()
            let hidden = runtime.config.hiddenSize
            var lines: [String] = []
            for (position, token) in tokens.enumerated() {
                let started = DispatchTime.now().uptimeNanoseconds
                let logits = try executor.forwardToken(token, position: position)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                guard let row = executor.capturedDraftInput(atPosition: position) else {
                    return "position \(position): the capture is empty"
                }
                guard row.count == targets.count * hidden else {
                    return "position \(position): captured \(row.count) values, "
                        + "expected \(targets.count * hidden)"
                }
                guard row.allSatisfy({ $0.isFinite }) else {
                    return "position \(position): the row holds a non-finite value"
                }
                var perLayer: [String] = []
                for (index, layer) in targets.enumerated() {
                    let slice = row[(index * hidden)..<((index + 1) * hidden)]
                    let norm = (slice.reduce(Float(0)) { $0 + $1 * $1 }).squareRoot()
                    let peak = slice.map(abs).max() ?? 0
                    perLayer.append(
                        String(format: "layer %d |x| %.2f max %.3f", layer, norm, peak)
                    )
                }
                let peakLogit = logits.map(abs).max() ?? 0
                let top = logits.indices.max { logits[$0] < logits[$1] } ?? -1
                lines.append(
                    String(format: "position %d: %.1f ms, top %d at %.1f, ", position, elapsed, top, peakLogit)
                        + perLayer.joined(separator: ", ")
                )
            }
            return lines.joined(separator: " | ")
        }.value
        print("🔎 [dsv41 real] drafter conditioning row: \(report)")
        withExtendedLifetime(engine) {}
    }

    /// The seeded draft rings on the real checkpoint.
    ///
    /// Everything the seed touches is published at a shape the synthetic fixture cannot
    /// imitate: `mtp.N.attn.wkv` is FP8 block-scaled (512 x 5 120 with an E8M0 scale
    /// companion, so it takes the block-32 GEMV rather than the bf16 one), `main_proj`
    /// reads three 5 120-wide layers end to end, and the latent rope payload is a
    /// trailing slice of a 512-wide record. This walks a short prefix and prints what
    /// each stage's ring holds, which is the only way to see a mis-bound scale or a rope
    /// slice applied to the wrong offsets before a draft round reads them.
    func testWhatTheDraftRingsHoldOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        XCTAssertEqual(config.dsparkTargetLayerIDs, [37, 38, 39])
        XCTAssertEqual(config.numNextNPredictLayers, 3)
        XCTAssertEqual(config.windowSize, 128, "the published window is the ring's size")
        XCTAssertEqual(config.dsparkBlockSize, 5, "five drafts a round")
        let tokens = [0, 128803, 21750, 2038, 15174, 943, 3939, 270]
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 32,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            executor.reset()
            executor.capturesDraftInput = true
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            // A live turn seeds one row per decoded token, so this is the drafter's own
            // per-turn overhead; a cold seed is the same dispatches over more rows.
            let beforeSeed = DispatchTime.now().uptimeNanoseconds
            let seeded = try executor.seedDraftRings(through: tokens.count)
            let seedMilliseconds =
                Double(DispatchTime.now().uptimeNanoseconds - beforeSeed) / 1_000_000
            guard seeded == tokens.count else {
                return "the seed wrote \(seeded) rows for \(tokens.count) positions"
            }
            // Then one warm row, which is what a live turn actually pays: a decode step
            // adds a single position and the seed projects it alone.
            _ = try executor.forwardToken(270, position: tokens.count)
            let warmStart = DispatchTime.now().uptimeNanoseconds
            let warmRows = try executor.seedDraftRings(through: tokens.count + 1)
            let warmMilliseconds =
                Double(DispatchTime.now().uptimeNanoseconds - warmStart) / 1_000_000
            guard warmRows == 1 else { return "a warm seed wrote \(warmRows) rows" }
            let stages = runtime.config.numNextNPredictLayers
            var lines: [String] = []
            for stage in 0..<stages {
                let rows = executor.draftRing(index: stage)
                guard rows.count == runtime.config.windowSize else {
                    return "stage \(stage): the ring holds \(rows.count) slots"
                }
                // Only the first `tokens.count` slots are seeded; the rest hold
                // whatever the allocation gave them.
                for position in [0, tokens.count - 1] {
                    let row = rows[position % rows.count]
                    let finite = row.allSatisfy { $0.isFinite }
                    let norm = (row.reduce(Float(0)) { $0 + $1 * $1 }).squareRoot()
                    let peak = row.map(abs).max() ?? 0
                    lines.append(
                        String(
                            format: "stage %d slot %d (%d values, %@): |x| %.2f max %.3f",
                            stage, position % rows.count, row.count,
                            finite ? "finite" : "NOT FINITE", norm, peak
                        )
                    )
                }
            }
            // One draft attention over the ring, which is the only place the FP32
            // attention kernel, the 128-row ring and the 64-head geometry meet.
            let attentionStart = DispatchTime.now().uptimeNanoseconds
            let stack = try executor.draftAttentionStack(
                token: 270, startPosition: tokens.count
            )
            let attentionMilliseconds =
                Double(DispatchTime.now().uptimeNanoseconds - attentionStart) / 1_000_000
            guard !stack.isEmpty else { return "the draft attention returned no positions" }
            var attentionLines: [String] = []
            for (position, copies) in stack.enumerated() {
                let collapsed = copies.first ?? []
                let finite = collapsed.allSatisfy { $0.isFinite }
                let norm = (collapsed.reduce(Float(0)) { $0 + $1 * $1 }).squareRoot()
                let peak = collapsed.map(abs).max() ?? 0
                attentionLines.append(
                    String(
                        format: "p%d %@ |x| %.2f max %.3f",
                        position, finite ? "" : "NOT FINITE", norm, peak
                    )
                )
            }
            lines.append(
                String(
                    format: "attention %d positions in %.2f ms: ",
                    stack.count, attentionMilliseconds
                ) + attentionLines.joined(separator: ", ")
            )
            return String(format: "%d rows in %.2f ms, then 1 warm row in %.2f ms; ",
                          seeded, seedMilliseconds, warmMilliseconds)
                + lines.joined(separator: " | ")
        }.value
        print("🔎 [dsv41 real] draft ring seed: \(report)")
        withExtendedLifetime(engine) {}
    }

    /// What verifying a drafted run in ONE batched pass would save over walking it.
    ///
    /// Speculative decoding is the only structure that lets a decode step batch, and
    /// the walk's per-layer routing round trip plus its arena fill are exactly what a
    /// block amortizes: eight drafted positions cost one host round trip per layer
    /// instead of eight, and one expert union instead of eight expert sets. This
    /// prices both arms on the published checkpoint.
    ///
    /// Each arm runs on its own pair of token runs, and the runs are different from
    /// the one before: the arena is shared by the whole runtime, so re-timing the same
    /// tokens reads nothing at all and prices only kernel work. Every run is eight
    /// prompt positions followed by eight drafted ones, the prompt untimed (it leaves
    /// the arena holding what a real turn would have behind it) and the drafted run
    /// timed. Each arm reports the faster of its two runs, because the first pass
    /// over either path is slower for being first.
    func testWhatABatchedVerifyWouldSaveOverAWalk() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let width = 8
        let tokens = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16, 128804, 128822,
            19, 16, 17289, 9372, 344, 10526, 44691, 305, 14, 1060, 15390, 294, 1737, 14, 1843,
            5258, 20450, 377, 3155, 14, 30663, 32453, 19504, 46885, 965, 35868, 20012, 305,
            32604, 965, 23418, 27124, 396, 7457, 1066, 304, 270, 8343, 603
        ]
        let runs = (0..<4).map { run -> ([Int], [Int]) in
            let base = run * width
            return (
                Array(tokens[base..<(base + width)]),
                Array(tokens[(base + width)..<(base + width * 2)])
            )
        }
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: width * 2 + 1,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()

            func walk(_ run: ([Int], [Int])) throws -> (milliseconds: Double, bytes: Int) {
                executor.reset()
                for (position, token) in run.0.enumerated() {
                    _ = try executor.forwardToken(token, position: position)
                }
                let before = runtime.expertStreamer?.bytesRead ?? 0
                let started = DispatchTime.now().uptimeNanoseconds
                for (offset, token) in run.1.enumerated() {
                    _ = try executor.forwardToken(token, position: width + offset)
                }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                return (elapsed, (runtime.expertStreamer?.bytesRead ?? 0) - before)
            }

            func block(_ run: ([Int], [Int])) throws -> (milliseconds: Double, bytes: Int) {
                executor.reset()
                _ = try executor.prefillChunk(run.0)
                let before = runtime.expertStreamer?.bytesRead ?? 0
                let started = DispatchTime.now().uptimeNanoseconds
                _ = try executor.prefillChunk(run.1)
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                return (elapsed, (runtime.expertStreamer?.bytesRead ?? 0) - before)
            }

            let walked = [try walk(runs[0]), try walk(runs[1]), try walk(runs[2])]
            let blocked = [try block(runs[1]), try block(runs[2]), try block(runs[3])]
            let walkMilliseconds = walked.map(\.milliseconds).min() ?? 0
            let blockMilliseconds = blocked.map(\.milliseconds).min() ?? 0
            let walkMegabytes = walked.map { Double($0.bytes) / 1_048_576 }.reduce(0, +)
                / Double(max(1, walked.count))
            let blockMegabytes = blocked.map { Double($0.bytes) / 1_048_576 }.reduce(0, +)
                / Double(max(1, blocked.count))
            return String(
                format: "batched verify vs walk, %d drafted positions: "
                    + "walk %.1f ms (%.1f ms/position, %.0f MB read), "
                    + "block %.1f ms (%.1f ms/position, %.0f MB read) -> %.2fx",
                width,
                walkMilliseconds,
                walkMilliseconds / Double(width),
                walkMegabytes,
                blockMilliseconds,
                blockMilliseconds / Double(width),
                blockMegabytes,
                walkMilliseconds / max(blockMilliseconds, 0.001)
            )
        }.value
        print("🔎 [dsv41 real] \(report)")
        withExtendedLifetime(engine) {}
    }

    /// What one decoded position costs, and which bucket holds it.
    ///
    /// A turn's decode loop is the whole user-visible speed of this model, and it is
    /// one position at a time through the walk: two command buffers per layer, a host
    /// block on the first of each, and an arena fill in between. The block path
    /// reports its buckets, the walk did not, so this turns the walk's timing on and
    /// prints the split instead of guessing from a wall clock. The first positions
    /// are spent filling the arena and are printed but not averaged.
    func testWhatADecodePositionCosts() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        // The prompt a real turn renders followed by the tokens that turn drew, so the
        // routing and the cache depth are the ones a decode loop actually sees. A
        // short prefix routes to a narrow expert set and the arena then serves most of
        // it from the page cache, which is not what a long answer costs.
        let tokens = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16, 128804, 128822,
            19, 16, 17289, 9372, 344, 10526, 44691, 305, 14, 1060, 15390, 294, 1737, 14, 1843,
            5258, 20450, 377, 3155, 14, 30663, 32453, 19504, 46885, 965, 35868, 20012, 305,
            32604, 965, 23418, 27124, 396, 7457, 1066, 304, 270, 8343, 603
        ]
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: tokens.count,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let carryOn = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            executor.reset()
            executor.recordsWalkTiming = true
            var measured: [Double] = []
            var host: [Double] = []
            for (position, token) in tokens.enumerated() {
                executor.resetWalkTiming()
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let timingBefore = runtime.expertStreamer?.timing
                    ?? (reading: 0, allocating: 0, slotsAllocated: 0)
                let started = DispatchTime.now().uptimeNanoseconds
                _ = try executor.forwardToken(token, position: position)
                let nanoseconds = DispatchTime.now().uptimeNanoseconds - started
                let read = (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
                let timingAfter = runtime.expertStreamer?.timing
                    ?? (reading: 0, allocating: 0, slotsAllocated: 0)
                if position % 4 == 0 || position == tokens.count - 1 {
                    print(
                        "🔎 [dsv41 real] position \(position): "
                            + String(format: "%.1f ms, read %.1f MB, pread %.1f ms, "
                                     + "slot alloc %.1f ms (%d new)",
                                     Double(nanoseconds) / 1_000_000,
                                     Double(read) / 1_048_576,
                                     (timingAfter.reading - timingBefore.reading) * 1000,
                                     (timingAfter.allocating - timingBefore.allocating) * 1000,
                                     timingAfter.slotsAllocated - timingBefore.slotsAllocated)
                    )
                    print("🔎 [dsv41 real]   \(executor.walkTiming.describe())")
                }
                if position > 0 {
                    measured.append(Double(nanoseconds) / 1_000_000)
                    host.append(Double(executor.walkTiming.hostNanoseconds) / 1_000_000)
                }
            }
            func mean(_ values: [Double]) -> Double {
                values.reduce(0, +) / Double(max(1, values.count))
            }
            let placement = runtime.expertStreamer?.placementTiming
                ?? (seconds: 0, placements: 0)
            let placementLine = String(
                format: "arena placement %.2f ms over %d pair(s) (%.1f us each); "
                    + "arena %d/%d resident, %.1f%% hit over %d acquisition(s)",
                placement.seconds * 1000,
                placement.placements,
                placement.placements > 0
                    ? placement.seconds / Double(placement.placements) * 1_000_000 : 0,
                runtime.arena.residentCount,
                runtime.arena.capacity,
                runtime.arena.hitRate * 100,
                runtime.arena.acquisitions
            )
            return String(
                format: "decode position over %d position(s): %.1f ms wall, %.1f ms host, "
                    + "so %.1f ms is device and the head; ",
                measured.count,
                mean(measured),
                mean(host),
                mean(measured) - mean(host)
            ) + placementLine
        }.value
        print("🔎 [dsv41 real] \(carryOn)")
        withExtendedLifetime(engine) {}
    }

    func testAPublishedPromptCompletesATurn() async throws {        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        guard let tokenizer = try? DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        ) else {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }

        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(isUser: true, text: userText)],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)
        print(
            "🔎 [dsv41 real] prompt dialect \(dialect.source.rawValue), "
                + "\(promptIds.count) tokens: \(promptIds)"
        )
        XCTAssertFalse(promptIds.isEmpty, "the rendered prompt tokenized to nothing")

        let contextLength = promptIds.count + firstTurnTokens
        // The zero-copy buffers point into this engine's mappings, so the engine has
        // to outlive every read they serve. The app keeps it for the load's lifetime;
        // a local here would unmap the weights as soon as the helper returned.
        // Ablation switch for the two Engram layers: a run with them silenced says
        // whether the memory path is what corrupted a prompt, and it is the only way
        // to ask that question without changing the model on disk.
        let skipEngram = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_ENGRAM_OFF"] == "1"
            || FileManager.default.fileExists(atPath: Self.engramOffMarkerPath)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: contextLength,
            allowEngramSkip: skipEngram
        )
        print("🔎 [dsv41 real] \(runtime.describe())")
        XCTAssertTrue(
            runtime.isRunnable,
            "the published checkpoint is not runnable:\n\(runtime.describe())"
        )
        let forward = runtime.forward()
        XCTAssertTrue(forward.isRunnable, forward.describe())

        let eos = config.eosTokenIDs.map { UInt32(clamping: $0) }
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: firstTurnTokens,
            eosTokenIds: eos
        )

        let draws = DrawLog()
        let first = try await Task.detached { [runtime, promptIds] in
            try DeepSeekV41TurnRunner.run(
                forward: runtime.forward(),
                promptTokenIds: promptIds,
                settings: settings,
                sample: { logits, generated in
                    let token = Self.argmax(logits)
                    draws.record(
                        logits: logits,
                        drawn: token,
                        generated: generated.count,
                        bytesRead: runtime.expertStreamer?.bytesRead ?? 0,
                        hits: runtime.arena.hits,
                        acquisitions: runtime.arena.acquisitions
                    )
                    return token
                },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
            )
        }.value
        print(
            "🔎 [dsv41 real] turn 1: \(first.finishReason) after \(first.tokensGenerated) token(s), "
                + String(format: "prefill %.2f s, decode %.2f s", first.prefillSeconds, first.decodeSeconds)
        )
        print("🔎 [dsv41 real] text: \(first.text.debugDescription)")
        print("🔎 [dsv41 real] ids: \(first.generatedTokenIds)")
        XCTAssertGreaterThan(
            first.tokensGenerated, 0,
            "the published checkpoint produced no token at all (text \(first.text.debugDescription))"
        )

        print(
            "🔎 [dsv41 real] experts after turn: \(runtime.arena.residentCount)/\(runtime.arena.capacity) resident, "
                + String(format: "%.1f%% hit rate", runtime.arena.hitRate * 100)
        )
        print("🔎 [dsv41 real] read: \(draws.summary())")
        print("🔎 [dsv41 real] \(runtime.expertStreamer?.describe() ?? "no streamer")")
        withExtendedLifetime(engine) {}
    }
}
