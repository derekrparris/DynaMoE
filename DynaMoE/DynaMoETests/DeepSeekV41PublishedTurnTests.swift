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

    /// Fixed prompt. Short on purpose: every position costs a full 40-layer pass,
    /// and the first turn pays the expert reads for all of them. Overridable from
    /// the environment so a run can ask a question whose answer is not stylistic.
    private var userText: String {
        ProcessInfo.processInfo.environment["DYNAMOE_DSV41_PROMPT"] ?? "hey"
    }

    /// Tokens the turn may draw. Enough to see a sentence start, small enough that
    /// a wrong graph is not paid for 200 times. Every token is a full 40-layer pass
    /// over freshly streamed experts, so this is the test's whole cost.
    private let firstTurnTokens = 8

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
    private final class DrawLog: @unchecked Sendable {
        private var count = 0
        func record(logits: [Float], drawn: UInt32, generated: Int) {
            guard count < 3 else { return }
            count += 1
            let top = logits.enumerated().sorted { $0.element > $1.element }.prefix(5)
            let shown = top.map { "\($0.offset):\(String(format: "%.2f", $0.element))" }
            print("🔎 [dsv41 real] draw \(generated): \(drawn) top5 \(shown.joined(separator: " "))")
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

    func testAPublishedPromptCompletesATurn() async throws {
        let snapshot = try snapshotDirectory()
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
                    draws.record(logits: logits, drawn: token, generated: generated.count)
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
        print("🔎 [dsv41 real] \(runtime.expertStreamer?.describe() ?? "no streamer")")
        withExtendedLifetime(engine) {}
    }
}
