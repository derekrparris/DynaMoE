//
//  DeepSeekV41PublishedTurnTests.swift
//  DynaMoETests
//
//  The one test that runs the published checkpoint end to end: load the 48 shards,
//  prefill rendered prompts, and draw tokens from the logits the device produced.
//  It runs two shapes — the thinking-off prompt, and the app's default (thinking on
//  with the high level, which puts the checkpoint's own `Reasoning Effort:` line in
//  the System block and opens the reasoning channel).
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
//  routed expert it needs from NVMe, and the second prompt differs at its first
//  tokens so it pays another prefill rather than exercising `resume`. `xcodebuild`
//  does not forward the invoking shell's environment into an app-hosted test, so the
//  switch is a marker file as well as an environment variable. Run it with
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

    /// Fits `seconds = fixed + perToken * tokens + squared * tokens^2` through three
    /// measured points by Cramer's rule, so a tower cost decomposes into the term
    /// that grows with the token count and the one that grows with its square.
    ///
    /// Returns nil when the three points are collinear in that basis, which means
    /// the measurement cannot tell the two growth terms apart and printing a split
    /// would be inventing one.
    private nonisolated static func costModel(
        tokens: [Double],
        seconds: [Double]
    ) -> (fixed: Double, perToken: Double, squared: Double)? {
        guard tokens.count == 3, seconds.count == 3 else { return nil }
        // Rows are the three sample points; columns are 1, t and t^2.
        func determinant(_ a: [Double], _ b: [Double], _ c: [Double]) -> Double {
            a[0] * (b[1] * c[2] - b[2] * c[1])
                - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1])
        }
        let ones: [Double] = [1, 1, 1]
        let squares = tokens.map { $0 * $0 }
        let base = determinant(ones, tokens, squares)
        guard base.isFinite, abs(base) > 1e-9 else { return nil }
        let fixed = determinant(seconds, tokens, squares) / base
        let perToken = determinant(ones, seconds, squares) / base
        let squared = determinant(ones, tokens, seconds) / base
        guard fixed.isFinite, perToken.isFinite, squared.isFinite else { return nil }
        return (fixed, perToken, squared)
    }

    /// A reproducible token stream for the probes, cycled from ids the checkpoint's
    /// own tokenizer produces for ordinary text so routing is not degenerate.
    private static func probeTokens(count: Int) -> [Int] {
        let seed = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16, 128804, 128822,
            19, 16, 17289, 9372, 344, 10526, 44691, 305, 14, 1060, 15390, 294, 1737, 14, 1843,
            5258, 20450, 377, 3155, 14, 30663, 32453, 19504, 46885, 965, 35868, 20012, 305,
            32604, 965, 23418, 27124, 396, 7457, 1066, 304, 270, 8343, 603
        ]
        guard count > 0 else { return [] }
        return (0..<count).map { seed[$0 % seed.count] }
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

    /// Does the sparse-attention rule bite at the published geometry?
    ///
    /// `#28` replaced the port's invented scorer with the checkpoint reference's
    /// two-level rule, and every vector behind that change was generated at a
    /// six-wide config: `index_topk` 3 over 8 produced rows. Those vectors say the
    /// RULE is transcribed correctly; they say nothing about whether it still
    /// changes the answer once the row count passes the budget at 512, which is the
    /// only regime the checkpoint runs in. Below `index_topk` every reachable row is
    /// selected and the scored cut is invisible, so a port that quietly kept reading
    /// the newest rows — the bug `#28` fixed — would pass the entire golden suite.
    ///
    /// So this prefills past the budget on the real weights and asks the model what
    /// it chose. Three things are checked, none of which a synthetic fixture can
    /// decide: the selection at a budget-exceeding position is NOT the newest
    /// `index_topk` reachable rows; it really is the top-k of the scores the device
    /// produced, recomputed independently on the host; and the candidate-block cut
    /// above it is inert at ordinary context, because 2,048 blocks of 8 rows is
    /// 16,384 rows and the checkpoint never reaches them here.
    func testWhatTheIndexerSelectsPastItsBudgetOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let plan = try config.makeLayerPlan()

        // Every compressed layer whose ratio is 1 produces one row per position, so
        // it is the only half that can pass `index_topk` at a context a probe can
        // afford: a ratio-2 layer arrives at the same row count twice as late.
        let ratioOneSources = plan.filter {
            $0.ownsCompressedIndex && $0.compressRatio == 1 && $0.compressedKVSource >= 0
        }
        try XCTSkipIf(
            ratioOneSources.isEmpty,
            "no ratio-1 index source in the layer plan, so nothing passes the budget here"
        )
        try XCTSkipIf(
            config.indexTopK <= 0,
            "no index_topk budget, so the scored cut never runs"
        )

        // Enough positions that the last few sit well past the budget: at 576 the
        // final position ranks 576 reachable rows against a cut of 512.
        let positionCount = min(config.indexTopK + 64, 576)
        let tokens = Self.probeTokens(count: positionCount)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positionCount,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())

        let sources = ratioOneSources.map(\.index)
        let probe = try await Task.detached { [runtime, tokens] () -> (
            [DeepSeekV41IndexSelectionRecord], Double
        ) in
            let executor = runtime.forward()
            executor.reset()
            executor.recordsIndexSelections = true
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try executor.prefillChunk(tokens)
            let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            return (executor.indexSelectionRecords, elapsed)
        }.value
        let records = probe.0
        let seconds = probe.1
        print(
            "🔎 [dsv41 real] prefilled \(tokens.count) positions in "
                + String(format: "%.1f s", seconds)
                + " over \(sources.count) ratio-1 index source(s) "
                + "\(sources.map(String.init).joined(separator: ", ")), "
                + "\(records.count) selection(s) recorded"
        )
        XCTAssertFalse(
            records.isEmpty,
            "the indexer read back no selection at the published geometry"
        )

        // The newest rows the old rule would have taken, as a set: the last `topK`
        // rows under `visibleRows`. Below the budget this IS the whole reachable
        // range, which is why the two rules cannot be told apart there.
        func newestWindow(_ record: DeepSeekV41IndexSelectionRecord) -> Set<Int> {
            let reachable = 0..<record.visibleRows
            return Set(reachable.suffix(record.topK))
        }

        var beyondBudget = 0
        var differsFromNewest = 0
        var oldestAdmitted = Int.max
        var newestSpans: [String] = []
        for record in records where record.producedRows > record.topK {
            beyondBudget += 1
            let selected = Set(record.selected)
            let newest = newestWindow(record)
            if selected != newest {
                differsFromNewest += 1
                let outside = selected.subtracting(newest).sorted()
                if let oldest = selected.min() {
                    oldestAdmitted = min(oldestAdmitted, oldest)
                }
                // The rows a scored cut keeps that a recency cut would have dropped,
                // and how far behind the frontier the deepest of them sits.
                let frontier = record.visibleRows - 1
                if newestSpans.count < 6 {
                    newestSpans.append(
                        "layer \(record.layer) position \(record.position): "
                            + "\(record.producedRows)/\(record.visibleRows) rows, "
                            + "topK \(record.topK), \(outside.count) row(s) outside the newest "
                            + "window, the oldest of them "
                            + "\(frontier - (outside.first ?? frontier)) behind the frontier"
                            + (outside.isEmpty ? "" : " (rows \(outside.prefix(4)))")
                    )
                }
            }
        }
        for line in newestSpans { print("🔎 [dsv41 real] \(line)") }
        print(
            "🔎 [dsv41 real] past the budget at \(beyondBudget) selection(s), "
                + "\(differsFromNewest) of them are not the newest \(config.indexTopK) rows; "
                + (oldestAdmitted == .max
                    ? "no selection reached behind the newest window"
                    : "the deepest admitted row is \(oldestAdmitted)")
        )

        // Which rows won, and why. A selection that differs from the newest window is
        // only meaningful if the scores rank it there: rows admitted from behind the
        // window because of a tie at the cut would mean the rule is decided by the
        // tie-break rather than by the model, and rows admitted because the score
        // decays with row index would mean the cut is a recency window with extra
        // steps. Both are visible here and neither is visible in a golden.
        let deepestPosition = tokens.count - 1
        for record in records where record.position == deepestPosition {
            guard record.producedRows > record.topK else { continue }
            let ranked = record.scores.enumerated()
                .sorted { $0.element == $1.element ? $0.offset < $1.offset : $0.element > $1.element }
                .map(\.offset)
            let cutScore = record.scores[ranked[record.topK - 1]]
            let ties = record.scores.filter { $0 == cutScore }.count
            let newest = record.visibleRows - 1
            let newestRank = ranked.firstIndex(of: newest).map { $0 + 1 } ?? -1
            let oldestRank = ranked.firstIndex(of: 0).map { $0 + 1 } ?? -1
            let finite = record.scores.filter { $0.isFinite }.sorted()
            let median = finite.isEmpty ? 0 : finite[finite.count / 2]
            let positive = record.scores.filter { $0 > 0 }.count
            let zero = record.scores.filter { $0 == 0 }.count
            let distinct = Set(record.scores).count
            print(
                "🔎 [dsv41 real] layer \(record.layer) position \(record.position): "
                    + "cut score \(String(format: "%.4f", cutScore)) held by \(ties) row(s), "
                    + "median \(String(format: "%.4f", median)), "
                    + "range \(String(format: "%.3f", finite.first ?? 0)).."
                    + "\(String(format: "%.3f", finite.last ?? 0)), "
                    + "\(positive) positive / \(zero) exactly zero / "
                    + "\(record.producedRows - positive - zero) negative, "
                    + "\(distinct) distinct score(s); "
                    + "row 0 ranks \(oldestRank), the newest row \(newest) ranks \(newestRank)"
            )
            // WHICH rows scored exactly zero. The rule's rectifier gives a zero to a
            // row no head votes for, which at 32 heads is a coincidence; a contiguous
            // run of them is a region with nothing written in it, and that is a
            // different finding entirely.
            let zeroRows = record.scores.indices.filter { record.scores[$0] == 0 }
            let contiguous = zeroRows.count > 1
                && zeroRows.last! - zeroRows.first! == zeroRows.count - 1
            print(
                "🔎 [dsv41 real]   zero-score rows: \(zeroRows.count), "
                    + (zeroRows.isEmpty ? "none"
                        : "\(zeroRows.first!)..\(zeroRows.last!), "
                            + (contiguous ? "contiguous" : "scattered"))
            )
            // A row scores exactly zero only when every head votes against it, which
            // with 32 heads is a coincidence rather than a pattern. A large block of
            // them is the signature of a key region with nothing written in it, which
            // is exactly what the index-key cache did while it was priced at the FP16
            // stride and addressed at the FP32 one.
            if config.indexNHeads >= 8 {
                XCTAssertLessThanOrEqual(
                    zeroRows.count, record.producedRows / 8,
                    "layer \(record.layer) position \(record.position) scored "
                        + "\(zeroRows.count) of \(record.producedRows) rows exactly zero with "
                        + "\(config.indexNHeads) heads, which does not happen to keys that were "
                        + "written"
                )
            }
        }

        // Does the zero tail grow with the row count, or is it a fixed region? The
        // same layer is compared at the first position past the budget and at the
        // last: a fixed count means rows that were never written, a growing one means
        // the rule is rejecting them.
        let auditLayer = ratioOneSources[0].index
        for record in records
        where record.layer == auditLayer
            && (record.position == config.indexTopK || record.position == deepestPosition) {
            let zeros = record.scores.filter { $0 == 0 }.count
            print(
                "🔎 [dsv41 real] layer \(record.layer) position \(record.position): "
                    + "\(zeros) zero-score row(s) of \(record.producedRows)"
            )
        }

        XCTAssertGreaterThan(
            beyondBudget, 0,
            "no position reached past index_topk, so the scored cut never ran"
        )
        XCTAssertGreaterThan(
            differsFromNewest, 0,
            "every budget-exceeding selection was exactly the newest \(config.indexTopK) "
                + "rows: the scored cut is not doing anything at the published geometry, "
                + "which is the bug #28 fixed"
        )

        // The selection must be the top-k OF THE SCORES, not merely different from a
        // recency window: a row list that arrived from anywhere else would still
        // differ. Recomputing the cut on the host from the read-back scores is the
        // only way to say the numbers the device produced are the ones the gather
        // then bound.
        for record in records {
            XCTAssertEqual(
                record.selected, record.selected.sorted(),
                "layer \(record.layer) position \(record.position) selected rows out of order"
            )
            XCTAssertEqual(
                Set(record.selected).count, record.selected.count,
                "layer \(record.layer) position \(record.position) selected a row twice"
            )
            XCTAssertEqual(
                record.selected.count, min(record.topK, record.visibleRows),
                "layer \(record.layer) position \(record.position) kept "
                    + "\(record.selected.count) of \(record.visibleRows) reachable rows "
                    + "against a cut of \(record.topK)"
            )
            XCTAssertEqual(
                record.scores.count, record.producedRows,
                "layer \(record.layer) position \(record.position) read back "
                    + "\(record.scores.count) scores for \(record.producedRows) rows"
            )
            // The reach filter runs AFTER the cut, so the host recomputation has to
            // do the same to agree: rank every produced row, keep the top `topK`,
            // then drop the rows past the frontier.
            let ranked = record.scores.enumerated()
                .sorted { $0.element == $1.element ? $0.offset < $1.offset : $0.element > $1.element }
                .prefix(record.topK)
                .map(\.offset)
                .filter { $0 < record.visibleRows }
                .sorted()
            XCTAssertEqual(
                record.selected, ranked,
                "layer \(record.layer) position \(record.position) did not read the rows its "
                    + "own scores ranked first"
            )
            // Level one above the top-k, at the rows this checkpoint actually has:
            // 2,048 blocks of 8 rows is 16,384, so the candidate cut is inert here
            // and says so rather than being assumed inert.
            if record.layer == config.candidateSourceLayerID {
                let blocks = DeepSeekV41Indexer.candidateBlocks(
                    scores: record.scores,
                    visibleRows: record.visibleRows,
                    blockSize: max(1, config.candidateBlockSize),
                    topKBlocks: max(0, config.candidateTopKBlocks)
                )
                XCTAssertTrue(
                    blocks.allSatisfy { $0 },
                    "the candidate cut dropped a block at position \(record.position), "
                        + "which the published block budget was supposed to leave inert"
                )
            }
        }

        // Two layers with different queries over the same cache must not choose the
        // same rows. Identical lists would mean the query never entered the scores,
        // which no row-count assertion above can detect.
        let lastPosition = tokens.count - 1
        let samePosition = records.filter { $0.position == lastPosition }
        if samePosition.count > 1 {
            let distinct = Set(samePosition.map { $0.selected })
            print(
                "🔎 [dsv41 real] at position \(lastPosition): \(distinct.count) distinct "
                    + "selection(s) across \(samePosition.count) source layer(s)"
            )
            XCTAssertGreaterThan(
                distinct.count, 1,
                "every index source chose the same rows at position \(lastPosition), "
                    + "so the selection does not depend on the query"
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// Dumps what one index-source layer's indexer was given, for the reference to re-run.
    ///
    /// `DeepSeekV41IndexerGoldenTests` pins `Indexer.forward` at a six-wide config,
    /// which is enough to pin the RULE and nothing about the regime the checkpoint
    /// runs in: 32 heads of 128, a 512-row budget, real activations. The reference
    /// needs three vectors per position that no config can reconstruct -- the pooled
    /// compressed latent `wk` reads, the normed layer input `weights_proj` reads, and
    /// the rank-normed query `wq_b` reads -- so this writes them out and
    /// `Tools/dsv41_goldens.py indexer_real` runs the checkpoint's own `Indexer` on
    /// them. The comparison is offline on purpose: a live run's activations are its
    /// own, and only the SAME activations can be scored both ways.
    ///
    /// Writes `/tmp/dsv41_indexer_inputs.json`; `DSV41_INDEXER_INPUT_DUMP` overrides
    /// the path. The 576-position prefill is the cost, which is why this stays in the
    /// opt-in file.
    func testWhatTheIndexerWasGivenOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let plan = try config.makeLayerPlan()
        guard let source = plan.first(where: {
            $0.ownsCompressedIndex && $0.ownsCompressedKV && $0.compressRatio == 1
        }) else {
            throw XCTSkip("no ratio-1 index source owns both its keys and its latent")
        }
        let positionCount = max(1, min(config.indexTopK + 64, 576))
        let probePosition = positionCount - 1
        let tokens = Self.probeTokens(count: positionCount)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positionCount,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())

        let latentWidth = config.cacheRecordWidth
        guard let latents = runtime.device.makeBuffer(
            length: positionCount * latentWidth * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            return XCTFail("the latent dump buffer could not be allocated")
        }
        let dump = try await Task.detached { [runtime, tokens, latents] () -> (
            DeepSeekV41IndexerInputDump?, [Float]
        ) in
            let executor = runtime.forward()
            executor.reset()
            executor.indexerInputRequest = DeepSeekV41IndexerInputRequest(
                layer: source.index, position: tokens.count - 1, latents: latents
            )
            _ = try executor.prefillChunk(tokens)
            let pointer = latents.contents().assumingMemoryBound(to: Float.self)
            let rows = Array(
                UnsafeBufferPointer(
                    start: pointer, count: tokens.count * latentWidth
                )
            )
            return (executor.indexerInputDump, rows)
        }.value
        let record = try XCTUnwrap(dump.0, "the indexer input dump was never taken")
        let latentRows = dump.1
        XCTAssertEqual(record.layer, source.index)
        XCTAssertEqual(record.position, probePosition)
        XCTAssertEqual(latentRows.count, positionCount * latentWidth)

        func column(_ values: [Float]) -> String {
            values.map { String(format: "%.9g", $0) }.joined(separator: ",")
        }
        let body = """
        {
          "layer": \(record.layer),
          "position": \(record.position),
          "positions": \(positionCount),
          "latent_width": \(latentWidth),
          "hidden": \(record.input.count),
          "query_width": \(record.queryInput.count),
          "latents": [\(column(latentRows))],
          "input": [\(column(record.input))],
          "query_input": [\(column(record.queryInput))],
          "scores": [\(column(record.scores))],
          "selected": [\(record.selected.map(String.init).joined(separator: ","))]
        }
        """
        let path = ProcessInfo.processInfo.environment["DSV41_INDEXER_INPUT_DUMP"]
            ?? "/tmp/dsv41_indexer_inputs.json"
        try body.write(toFile: path, atomically: true, encoding: .utf8)
        print(
            "🔎 [dsv41 real] indexer inputs: layer \(record.layer) position \(record.position), "
                + "\(positionCount) latent(s) of \(latentWidth), \(record.scores.count) score(s), "
                + "\(record.selected.count) selected -> \(path)"
        )
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

    /// DSpark's accept rate on the published checkpoint — the number that decides
    /// whether the drafter is worth verifying at all.
    ///
    /// `#22` priced a batched verify at 70.9 ms/position against a walk's 153.3, so a
    /// round of `dspark_block_size` (5) drafts pays for itself once ~47% of them are
    /// accepted, i.e. about 2.3 of 5. This drafts, walks the target greedily over the
    /// same positions, and counts the longest agreeing prefix — the rule
    /// `DeepSeekV41DSpark.acceptedCount` implements and the only one a linear chain
    /// needs.
    ///
    /// Two continuations rather than one, and a per-STEP rate alongside the total: a
    /// total is a product of a decay curve, and the step at which the curve dies is what
    /// says whether the block is the right width.
    func testWhatDSparkAcceptsOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let steps = config.dsparkBlockSize
        XCTAssertEqual(steps, 5)
        let prompts = [
            [0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16],
            [0, 128803, 1446, 369, 310, 13866, 353, 279, 4685, 327, 16665, 13, 128804]
        ]
        let rounds = 10
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 24 + rounds * (steps + 1) + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await dsparkAcceptance(
            runtime: runtime, prompts: prompts, steps: steps, rounds: rounds
        )
        print("🔎 [dsv41 real] dspark acceptance: \(report)")
        withExtendedLifetime(engine) {}
    }

    /// What a NARROWER draft block accepts — the shape the acceptance curve supports.
    ///
    /// `#25` measured 31% against a ~47% bar at the checkpoint's declared width of
    /// five, and the per-step curve said why: step 0 agreed 80% of the time and
    /// everything past the second step sat at the floor, so the loss was the chain
    /// conditioning each draft on the ones before it. A shorter chain compounds fewer
    /// of those errors.
    ///
    /// It is NOT a prefix of the wide graph, though: the block's own rows are in every
    /// query's key list (the block is not causal inside itself), so a two-wide block is
    /// a different attention than the first two positions of a five-wide one. This
    /// measures that graph, not the truncated logits of the old one.
    ///
    /// Each width gets its own runtime, built and released before the next is built,
    /// because the width sizes the key block, the head's logits buffer and every block
    /// workspace at construction: overriding the config is exactly the graph a
    /// checkpoint declaring that width builds. Width five is in the sweep as the
    /// control and has to reproduce `#25`.
    func testWhatANarrowerDraftBlockAcceptsOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let base = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        XCTAssertEqual(base.dsparkBlockSize, 5)
        let prompts = [
            [0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16],
            [0, 128803, 1446, 369, 310, 13866, 353, 279, 4685, 327, 16665, 13, 128804]
        ]
        let rounds = 10
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        for width in [2, 3, 5] {
            let report = try await dsparkAcceptanceAtWidth(
                width,
                base: base,
                snapshot: snapshot,
                engine: engine,
                prompts: prompts,
                rounds: rounds
            )
            print("🔎 [dsv41 real] dspark acceptance at width \(width): \(report)")
        }
        withExtendedLifetime(engine) {}
    }

    /// One width of the sweep: build a runtime for that block size, run the protocol,
    /// and return while the runtime is still the only one alive.
    ///
    /// The width is a construction parameter, not a knob on a running executor, so a
    /// sweep that shared one runtime would measure the wide graph three times.
    private func dsparkAcceptanceAtWidth(
        _ width: Int,
        base: DeepSeekV41Config,
        snapshot: URL,
        engine: DynaMoeEngine,
        prompts: [[Int]],
        rounds: Int
    ) async throws -> String {
        var config = base
        config.dsparkBlockSize = width
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 24 + rounds * (width + 1) + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        return try await dsparkAcceptance(
            runtime: runtime, prompts: prompts, steps: width, rounds: rounds
        )
    }

    /// The acceptance protocol, shared by every width so the sweep and `#25`'s number
    /// cannot drift: the protocol IS the measurement, and a second copy with one
    /// changed line would report something that cannot be compared.
    ///
    /// It drafts `steps` tokens from the target's last token, walks the TARGET
    /// greedily over the same positions (each position consuming the token the target
    /// itself produced one before), and counts the longest agreeing prefix. The
    /// target's walk advances the cache by `steps` per round whatever the drafts did,
    /// so every round starts from the same state the previous round's target left.
    private func dsparkAcceptance(
        runtime: DeepSeekV41Runtime,
        prompts: [[Int]],
        steps: Int,
        rounds: Int
    ) async throws -> String {
        try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            var summaries: [String] = []
            for (which, prompt) in prompts.enumerated() {
                executor.reset()
                executor.capturesDraftInput = true
                var last = 0
                for (position, token) in prompt.enumerated() {
                    _ = try executor.forwardToken(token, position: position)
                    last = token
                }
                _ = try executor.seedDraftRings(through: prompt.count)

                var acceptedTotal = 0
                var draftedTotal = 0
                var agreedPerStep = [Int](repeating: 0, count: steps)
                var gatedAccepted = 0
                var gatedDrafted = 0
                var draftMilliseconds: [Double] = []
                for _ in 0..<rounds {
                    let startPosition = executor.storedPositions - 1
                    let draftStarted = DispatchTime.now().uptimeNanoseconds
                    let logits = try executor.draftHeadLogits(
                        token: last, startPosition: startPosition
                    )
                    guard logits.count == steps else {
                        return "prompt \(which): the head produced \(logits.count) rows"
                    }
                    let chain = DeepSeekV41DSpark.runChain(
                        steps: steps, seedToken: last, temperature: 0
                    ) { index, previous in
                        guard let step = try? executor.draftStep(
                            index: index, previousToken: previous
                        ), logits.indices.contains(index) else { return nil }
                        return (logits[index], step.bias, step.confidence)
                    }
                    draftMilliseconds.append(
                        Double(DispatchTime.now().uptimeNanoseconds - draftStarted) / 1_000_000
                    )

                    // The target's own greedy choices over the positions the drafts
                    // claim. Each position consumes the token the TARGET produced at the
                    // one before it, which is what a decoding turn does.
                    var choices: [Int] = []
                    var feed = last
                    var position = executor.storedPositions
                    while choices.count < steps {
                        let tokenLogits = try executor.forwardToken(feed, position: position)
                        var best = 0
                        for index in 1..<tokenLogits.count
                        where tokenLogits[index] > tokenLogits[best] {
                            best = index
                        }
                        choices.append(best)
                        feed = best
                        position += 1
                    }
                    let accepted = DeepSeekV41DSpark.acceptedCount(
                        draft: chain.tokens, targetChoices: choices
                    )
                    acceptedTotal += accepted
                    draftedTotal += chain.tokens.count
                    // A step only counts once every earlier step agreed: past a
                    // disagreement the chain is conditioned on a token the target would
                    // not have produced, so its "agreement" would mean nothing.
                    for step in 0..<steps where accepted > step { agreedPerStep[step] += 1 }
                    // What the confidence gate would have submitted anyway.
                    if DeepSeekV41DSpark.shouldSubmit(chain) {
                        gatedAccepted += accepted
                        gatedDrafted += chain.tokens.count
                    }
                    last = feed
                    _ = try executor.seedDraftRings(through: executor.storedPositions)
                }
                let rate = draftedTotal > 0
                    ? Double(acceptedTotal) / Double(draftedTotal) * 100 : 0
                let gatedRate = gatedDrafted > 0
                    ? Double(gatedAccepted) / Double(gatedDrafted) * 100 : 0
                let perStep = agreedPerStep.enumerated().map { index, count in
                    String(format: "s%d %.0f%%", index, Double(count) / Double(rounds) * 100)
                }
                summaries.append(
                    String(
                        format: "p%d %d/%d (%.0f%%), gated %.0f%% of %d, %.0f ms/round, %@",
                        which, acceptedTotal, draftedTotal, rate, gatedRate,
                        gatedDrafted, draftMilliseconds.reduce(0, +) / Double(rounds),
                        perStep.joined(separator: " ")
                    )
                )
            }
            return summaries.joined(separator: " | ")
        }.value
    }

    /// What a verify costs at the widths a drafter might actually submit.
    ///
    /// `#22` priced a batched verify at 70.9 ms/position against a walk's 153.3, but it
    /// measured a block of EIGHT — the width a tree verify wants. `#25` found the
    /// checkpoint's drafter accepts its first draft 80% of the time and its second
    /// 30-60%, so the width a chain would actually submit is two or three, and the
    /// batching that made eight cheap has only two positions to amortize a layer's round
    /// trip and expert union over. This prices the same call at every width, so a narrow
    /// chain's economics rest on a measurement rather than on a projection.
    ///
    /// The widths are INTERLEAVED, one sample each per round, and the first width
    /// measured is not trusted: a first pass over the procedure reads several times slow
    /// (an earlier version measured width 2 at 154.5 ms/position and width 3 at 56.5 in
    /// the same run, which is ordering rather than batching).
    func testWhatAVerifyCostsAtTheWidthsAChainWouldSubmit() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let widths = [1, 2, 3, 4, 5, 8]
        let tokens = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16,
            128804, 128822, 1446, 369, 310, 13866, 353, 279, 4685, 327, 16665, 13
        ]
        let samples = 5
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            // One long continuation, every width drawing from it, so the ring and the
            // arena are in the same state for every arm.
            contextLength: tokens.count + samples * (widths.reduce(0, +) + widths.count),
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            executor.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            var best = [Int: Double]()
            var warm = false
            for round in 0..<(samples + 1) {
                for width in widths {
                    let start = executor.storedPositions
                    let block = (0..<width).map { tokens[($0 + start) % tokens.count] }
                    let began = DispatchTime.now().uptimeNanoseconds
                    _ = try executor.prefillChunk(block)
                    let elapsed =
                        Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000
                    // The first round warms every width's pages and kernels; taking the
                    // best of the rest is what makes the arms comparable.
                    if warm && (best[width].map { elapsed < $0 } ?? true) {
                        best[width] = elapsed
                    }
                }
                warm = true
                _ = round
            }
            let lines = widths.map { width in
                String(format: "w%d %.1f ms/pos", width, (best[width] ?? 0) / Double(width))
            }
            return lines.joined(separator: ", ")
        }.value
        print("🔎 [dsv41 real] verify width sweep: \(report)")
        withExtendedLifetime(engine) {}
    }

    /// What a narrow verify costs against a walk measured in the SAME runtime, round
    /// by round.
    ///
    /// `#25`'s sweep reported width 2 at 105.8 ms/position in one run and 62.9 in
    /// another — a spread larger than the effect it was chasing — because the arms
    /// shared one runtime, ran in a fixed order, and each one advanced a cache the
    /// next then read. `#26` makes width 2 the deciding number (a two-wide chain
    /// accepts 50-55% against a five-wide chain's 31%, so this cost is what decides
    /// whether verifying is a win), so it gets its own measurement with those
    /// confounds removed:
    ///
    /// - Every sample RESETS the executor and re-prefills the same prefix, so every
    ///   arm starts from the same position, the same ring and the same cache contents.
    /// - The arm order rotates each round, so no width always inherits the previous
    ///   arm's leftovers.
    /// - Every round also times a ONE-position walk, and each width is reported as a
    ///   ratio to that walk in the same round. The absolute level has moved 40%
    ///   between runs; the ratio is the part that can be compared.
    ///
    /// The median is reported rather than the best, because the best of a handful of
    /// samples on a warm arena is the number least likely to survive contact with a
    /// real turn.
    func testWhatANarrowVerifyCostsAgainstAWalkInTheSameRuntime() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let prefix = [0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16]
        let arm = [128804, 128822, 1446, 369, 310, 13866, 353]
        let widths = [2, 3]
        let samples = 6
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: prefix.count + arm.count + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()
            // arm 0 is the walk; the rest are verify widths.
            let arms = 1 + widths.count
            var walkMs: [Double] = []
            var verifyMs = [Int: [Double]]()
            var walkMB: [Double] = []
            var verifyMB = [Int: [Double]]()
            for round in 0..<(samples + 1) {
                for offset in 0..<arms {
                    // Rotate so width 2 is not always the first or always the last.
                    let slot = (offset + round) % arms
                    executor.reset()
                    for (position, token) in prefix.enumerated() {
                        _ = try executor.forwardToken(token, position: position)
                    }
                    let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                    let began = DispatchTime.now().uptimeNanoseconds
                    var positions = 1
                    if slot == 0 {
                        _ = try executor.forwardToken(arm[0], position: prefix.count)
                    } else {
                        let width = widths[slot - 1]
                        positions = width
                        _ = try executor.prefillChunk(Array(arm.prefix(width)))
                    }
                    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000
                    let megabytes = Double(
                        (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
                    ) / 1_048_576 / Double(positions)
                    // The first round is the warm-up: pages, kernels and the arena.
                    guard round > 0 else { continue }
                    if slot == 0 {
                        walkMs.append(elapsed / Double(positions))
                        walkMB.append(megabytes)
                    } else {
                        verifyMs[widths[slot - 1], default: []]
                            .append(elapsed / Double(positions))
                        verifyMB[widths[slot - 1], default: []].append(megabytes)
                    }
                }
            }
            func median(_ values: [Double]) -> Double {
                guard !values.isEmpty else { return 0 }
                let sorted = values.sorted()
                return sorted[sorted.count / 2]
            }
            let walk = median(walkMs)
            let lines = widths.map { width -> String in
                let values = verifyMs[width] ?? []
                let scale = walk > 0 ? median(values) / walk : 0
                let bytes = median(verifyMB[width] ?? [])
                return String(
                    format: "w%d %.1f ms/pos = %.2fx a walk (%.1f ms), %.0f MB/pos",
                    width, median(values), scale, walk, bytes
                )
            }
            let samplesLine = String(
                format: "walk %.1f ms/pos over %d samples, %.0f MB/pos",
                walk, walkMs.count, median(walkMB)
            )
            return lines.joined(separator: ", ") + "; " + samplesLine
        }.value
        print("🔎 [dsv41 real] narrow verify vs walk: \(report)")
        withExtendedLifetime(engine) {}
    }

    /// The same narrow verify, priced where it will actually run: over tokens whose
    /// experts the arena does NOT already hold.
    ///
    /// The zero-miss A/B above answers a different question. Its arm tokens are the
    /// same every round, so after the first round the arena serves them and both arms
    /// report 0 MB read; that measures kernel work only and it is the trap `#22`
    /// recorded. A live turn never re-sees a position, so every position it verifies
    /// pays the expert stream, and batching's whole prize is one expert UNION per
    /// block instead of one set per position — which is where `#22`'s 2.16x at width
    /// eight came from and which a width-two block has far less room to amortize.
    ///
    /// Every sample therefore uses a DIFFERENT two- (or three-) token run, taken from
    /// a fixed list rather than from a continuation this test just walked (walking one
    /// would load exactly the experts the arms are supposed to miss), and every sample
    /// re-walks the same prompt and seed so both arms start from identical cache and
    /// arena state. The seed's experts stay hot across samples, exactly as a live
    /// session's context does; the arm's do not, because the arm is where the new
    /// positions are.
    func testWhatANarrowVerifyCostsWhenItsExpertsAreCold() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let prompt = [0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16]
        let seedCount = 2
        let armTokens = [
            128804, 128822, 19, 16, 17289, 9372, 344, 10526, 44691, 305, 14, 1060,
            15390, 294, 1737, 14, 1843, 5258, 20450, 377, 3155, 14, 30663, 32453,
            19504, 46885, 965, 35868, 20012, 305, 32604, 965, 23418, 27124, 396, 7457,
            1066, 304, 270, 8343, 603, 1943, 1943, 1239, 128822, 128804, 344, 4211,
            3961, 573, 3413, 270, 15667, 198, 12694, 25
        ]
        let widths = [2, 3]
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: prompt.count + seedCount + armTokens.count + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let report = try await Task.detached { [runtime] () -> String in
            let executor = runtime.forward()

            // Every sample starts the same way: reset, walk the prompt and the seed.
            func seeded() throws {
                executor.reset()
                for (index, token) in prompt.enumerated() {
                    _ = try executor.forwardToken(token, position: index)
                }
                for offset in 0..<seedCount {
                    _ = try executor.forwardToken(
                        armTokens[offset], position: prompt.count + offset
                    )
                }
            }

            let armOrder = [0] + Array(1...widths.count)
            var walkMs: [Double] = []
            var walkMB: [Double] = []
            var verifyMs = [Int: [Double]]()
            var verifyMB = [Int: [Double]]()
            var cursor = seedCount
            var sample = 0
            while cursor + 3 <= armTokens.count {
                let slot = (sample + 1) % armOrder.count
                let width = slot == 0 ? 1 : widths[slot - 1]
                guard cursor + width <= armTokens.count else { break }
                let run = Array(armTokens[cursor..<(cursor + width)])
                cursor += width
                sample += 1
                // The first sample is warm-up for whichever arm it happens to be.
                let measuring = sample > 1
                try seeded()
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let began = DispatchTime.now().uptimeNanoseconds
                if slot == 0 {
                    for offset in 0..<width {
                        _ = try executor.forwardToken(
                            run[offset], position: prompt.count + seedCount + offset
                        )
                    }
                } else {
                    _ = try executor.prefillChunk(run)
                }
                let elapsed = Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000
                let megabytes = Double(
                    (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
                ) / 1_048_576 / Double(width)
                guard measuring else { continue }
                if slot == 0 {
                    walkMs.append(elapsed / Double(width))
                    walkMB.append(megabytes)
                } else {
                    verifyMs[widths[slot - 1], default: []].append(elapsed / Double(width))
                    verifyMB[widths[slot - 1], default: []].append(megabytes)
                }
            }

            func median(_ values: [Double]) -> Double {
                guard !values.isEmpty else { return 0 }
                let sorted = values.sorted()
                return sorted[sorted.count / 2]
            }
            let walk = median(walkMs)
            let lines = widths.map { width -> String in
                let values = verifyMs[width] ?? []
                let scale = walk > 0 ? median(values) / walk : 0
                return String(
                    format: "w%d %.1f ms/pos = %.2fx a walk (%.1f ms), %.0f MB/pos vs %.0f",
                    width, median(values), scale, walk,
                    median(verifyMB[width] ?? []), median(walkMB)
                )
            }
            let samplesLine = String(
                format: "%d walk samples, %.0f MB/pos", walkMs.count, median(walkMB)
            )
            return lines.joined(separator: ", ") + "; " + samplesLine
        }.value
        print("🔎 [dsv41 real] cold narrow verify vs walk: \(report)")
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

    /// Every position of a batched prefill is the logits the walk produces there.
    ///
    /// This is what `batchedDevicePrefill` was waiting for: the block graph has been
    /// compared with the host on a synthetic checkpoint, but never on the published
    /// one, where the tensor shapes, the 128-row window ring, the ratio-2 compressor
    /// and the expert arena are the real ones. A block of eight is the widest pass a
    /// verify could submit, and the capture asks for every slot, so this compares all
    /// of them rather than only the last — which is where a head that wrote every row
    /// to slot 0, or a block that mixed its positions, would show.
    ///
    /// Two blocks, not one, so the second continues a cache the first filled: a
    /// verify appends to a live turn, and the ring slot and the compressor's group
    /// slot are derived from the absolute position.
    ///
    /// The bar is relative float rounding, not bit equality. A block re-associates
    /// every reduction (the head's rows are one grid, the mixture is one union per
    /// block, the dense stages dispatch over a grid height), so on the published
    /// weights the block and the walk land within ~2e-4 of each other at a logit
    /// scale of ~20; the plain path, without the capture's batched head, is within
    /// ~1.3e-4. A wrong row list or a head that wrote every slot to slot 0 is orders
    /// of magnitude larger, and 5e-4 is still four times below the graph's own
    /// decode tolerance (`DeepSeekV41DeviceGraphParityTests.decodeTolerance`).
    func testABatchedPrefillMatchesTheWalkOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let width = DeepSeekV41DevicePlan.chunkSlots(config: config, contextLength: 32)
        XCTAssertGreaterThan(width, 1, "the published window has to allow a block")
        let positions = width * 2
        let tokens = [
            0, 128803, 21750, 2038, 15174, 943, 3939, 270, 15571, 344, 83779, 16, 128804, 128822,
            19, 16, 17289, 9372, 344, 10526
        ]
        XCTAssertGreaterThanOrEqual(tokens.count, positions)
        let runTokens = Array(tokens.prefix(positions))

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positions + 4,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let result = try await Task.detached { [runtime] () -> (
            worst: Float, position: Int, scale: Float, compared: Int,
            plainWorst: Float, plainPosition: Int, plainScale: Float
        ) in
            let executor = runtime.forward()
            let capture = Set(0..<width)

            executor.reset()
            var walked: [[Float]] = []
            for (position, token) in runTokens.enumerated() {
                walked.append(try executor.forwardToken(token, position: position))
            }

            executor.reset()
            let first = try executor.prefillChunk(
                Array(runTokens.prefix(width)), capturingLogitsAt: capture
            )
            let second = try executor.prefillChunk(
                Array(runTokens.dropFirst(width)), capturingLogitsAt: capture
            )
            var blocked: [[Float]] = []
            for position in 0..<width {
                blocked.append(
                    try XCTUnwrap(first.captured[position], "first block slot \(position)")
                )
            }
            for position in 0..<width {
                blocked.append(
                    try XCTUnwrap(second.captured[position], "second block slot \(position)")
                )
            }
            XCTAssertEqual(blocked.count, walked.count)

            var worst: Float = 0
            var worstPosition = -1
            var worstScale: Float = 1
            for position in walked.indices {
                let expected = walked[position]
                let actual = blocked[position]
                guard actual.count == expected.count else {
                    return (.infinity, position, 1, walked.count, .infinity, position, 1)
                }
                let scale = max(expected.map(abs).max() ?? 1, 1e-6)
                for index in expected.indices {
                    let delta = abs(actual[index] - expected[index]) / scale
                    if delta > worst {
                        worst = delta
                        worstPosition = position
                        worstScale = scale
                    }
                }
            }

            // Control: the same block layers WITHOUT the batched head, so a delta
            // that survives here is the block graph rather than the head's rows.
            var plainWorst: Float = 0
            var plainPosition = -1
            var plainScale: Float = 1
            for position in runTokens.indices {
                executor.reset()
                let plain = try executor.prefillChunk(
                    Array(runTokens.prefix(position + 1))
                ).last
                let expected = walked[position]
                let scale = max(expected.map(abs).max() ?? 1, 1e-6)
                for index in expected.indices {
                    let delta = abs(plain[index] - expected[index]) / scale
                    if delta > plainWorst {
                        plainWorst = delta
                        plainPosition = position
                        plainScale = scale
                    }
                }
            }
            return (
                worst, worstPosition, worstScale, walked.count,
                plainWorst, plainPosition, plainScale
            )
        }.value
        print(
            String(
                format: "🔎 [dsv41 real] batched prefill vs walk: %d positions in two "
                    + "blocks of %d, worst relative delta %.3e at position %d "
                    + "(scale %.3f); plain blocks (no capture) worst %.3e at position "
                    + "%d (scale %.3f)",
                result.compared, width, result.worst, result.position, result.scale,
                result.plainWorst, result.plainPosition, result.plainScale
            )
        )
        XCTAssertLessThan(
            result.worst, 5e-4,
            "position \(result.position) of the batched prefill is off the walk by "
                + "\(result.worst) relative to a \(result.scale) scale"
        )
        XCTAssertLessThan(
            result.plainWorst, 5e-4,
            "the capture-free block path is off the walk by \(result.plainWorst) at "
                + "position \(result.plainPosition), so the delta is not the head"
        )
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

        // The app's DEFAULT shape is thinking on with the high level, and that is
        // the shape the availability flag now advertises: the checkpoint's own
        // `Reasoning Effort:` line opens the System block and the reasoning
        // channel. It differs from the first prompt at the very first tokens, so
        // the second turn is a fresh prefill rather than a reused one.
        let thinkingPrompt = dialect.render(
            turns: [.init(isUser: true, text: userText)],
            cleanSystem: "",
            thinkingEnabled: true,
            effort: .high
        )
        XCTAssertTrue(
            thinkingPrompt.contains(DeepSeekV41PromptDialect.effortInstruction(budget: 100)),
            "the default thinking-on prompt must carry the effort line"
        )
        let thinkingIds = try tokenizer.encode(text: thinkingPrompt)
        XCTAssertFalse(thinkingIds.isEmpty, "the thinking-on prompt tokenized to nothing")
        print("🔎 [dsv41 real] thinking prompt: \(thinkingIds.count) tokens")

        let contextLength = max(promptIds.count, thinkingIds.count) + firstTurnTokens
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

        // Turn 2: the app's default shape (thinking on, high effort). The effort
        // line is the control the availability flag switches on, so the real
        // weights have to see it and still answer.
        let thinkingTurn = try await Task.detached { [runtime, thinkingIds] in
            try DeepSeekV41TurnRunner.run(
                forward: runtime.forward(),
                promptTokenIds: thinkingIds,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
            )
        }.value
        print(
            "🔎 [dsv41 real] thinking turn: \(thinkingTurn.finishReason) after "
                + "\(thinkingTurn.tokensGenerated) token(s), "
                + String(
                    format: "prefill %.2f s, decode %.2f s",
                    thinkingTurn.prefillSeconds, thinkingTurn.decodeSeconds
                )
        )
        print("🔎 [dsv41 real] thinking text: \(thinkingTurn.text.debugDescription)")
        print("🔎 [dsv41 real] thinking ids: \(thinkingTurn.generatedTokenIds)")
        XCTAssertGreaterThan(
            thinkingTurn.tokensGenerated, 0,
            "the effort-line prompt produced no token (text \(thinkingTurn.text.debugDescription))"
        )

        print(
            "🔎 [dsv41 real] experts after turn: \(runtime.arena.residentCount)/\(runtime.arena.capacity) resident, "
                + String(format: "%.1f%% hit rate", runtime.arena.hitRate * 100)
        )
        print("🔎 [dsv41 real] read: \(draws.summary())")
        print("🔎 [dsv41 real] \(runtime.expertStreamer?.describe() ?? "no streamer")")
        withExtendedLifetime(engine) {}
    }

    /// What dialect the checkpoint actually emits when it is asked to call a tool.
    ///
    /// Nothing in this repository defines the tool dialect from the checkpoint's own
    /// files: it ships no chat template at all (`tokenizer_config.json` has no
    /// `chat_template`, so `DeepSeekV41PromptDialect.resolve` falls back) and its
    /// `encoding/` package has no tool path. The tokenizer DOES carry the special ids
    /// the family uses (`tool▁calls▁begin`, `tool▁call▁begin`, `tool▁sep`,
    /// `tool▁call▁end`, `tool▁calls▁end`, and the `output`/`outputs` pair), which
    /// fixes the SPELLING but not the grammar around it: whether the body carries the
    /// literal word "function", and whether the arguments are bare JSON or fenced.
    ///
    /// So this asks the model. It renders the published V3.1 tool block from the
    /// marker strings (never retyped: a rendered view strips the fullwidth bars), runs
    /// a short turn, and prints the raw text with every non-ASCII codepoint escaped,
    /// plus each generated token's piece. A dialect the model does not emit is a
    /// dialect a parser must not look for.
    func testWhatToolDialectTheCheckpointEmits() async throws {
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
        let m = dialect.markers
        let schematic = m.toolCallsBegin + m.toolCallBegin + "tool_call_name"
            + m.toolCallSeparator + "tool_call_arguments" + m.toolCallEnd + m.toolCallsEnd
        // The markers have to be SINGLE ids, or the model is being asked to copy a
        // spelling it never sees as a token and its answer says nothing about the
        // dialect it was trained on.
        for (label, marker) in [
            ("callsBegin", m.toolCallsBegin), ("callBegin", m.toolCallBegin),
            ("sep", m.toolCallSeparator), ("callEnd", m.toolCallEnd),
            ("callsEnd", m.toolCallsEnd), ("outputBegin", m.toolOutputBegin),
            ("outputEnd", m.toolOutputEnd), ("outputsBegin", m.toolOutputsBegin),
            ("outputsEnd", m.toolOutputsEnd), ("assistant", m.assistant)
        ] {
            let ids = (try? tokenizer.encode(text: marker)) ?? []
            print("🔎 [dsv41 real] marker \(label): \(ids) \(marker.debugDescription)")
        }
        let schema = #"{"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"]}"#
        func toolBlock(withFormatInstruction: Bool) -> String {
            var text = """
            You are a helpful assistant.

            ## Tools
            You have access to the following tools:

            ### get_current_time
            Description: Returns the current local time for a city.

            Parameters: \(schema)
            """
            guard withFormatInstruction else { return text }
            text += """


            IMPORTANT: ALWAYS adhere to this exact format for tool use:
            \(schematic)

            Where:
            - `tool_call_name` must be an exact match to one of the available tools
            - `tool_call_arguments` must be valid JSON that strictly follows the tool's Parameters Schema
            - For multiple tool calls, chain them directly without separators or spaces
            """
            return text
        }
        // Four prompts, because one answer cannot separate "the model knows this
        // dialect" from "the model copied the example it was handed". The fourth is
        // the discriminator: the SAME catalogue with the SCHEMATIC swapped for the
        // instruction block this app actually ships, which holds the preamble
        // constant and changes only what the call grammar is taught with.
        let instructionsOnly = toolBlock(withFormatInstruction: false) + "\n\n"
            + DeepSeekV41ToolDialect.instructions()
        let variants: [(label: String, system: String, thinking: Bool)] = [
            ("schematic+chat", toolBlock(withFormatInstruction: true), false),
            ("no-schematic", toolBlock(withFormatInstruction: false), false),
            ("schematic+think", toolBlock(withFormatInstruction: true), true),
            ("instructions-only", instructionsOnly, false)
        ]
        let ask = "What time is it in Paris right now? Use the tool."
        let prompts = variants.map { variant in
            dialect.render(
                turns: [.init(isUser: true, text: ask)],
                cleanSystem: variant.system,
                thinkingEnabled: variant.thinking,
                effort: nil
            )
        }
        let encoded = try prompts.map { try tokenizer.encode(text: $0) }
        for (index, ids) in encoded.enumerated() {
            print("🔎 [dsv41 real] tool probe '\(variants[index].label)': \(ids.count) tokens")
            XCTAssertFalse(ids.isEmpty)
        }

        let drawCount = 64
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: (encoded.map(\.count).max() ?? 0) + drawCount + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let eos = config.eosTokenIDs.map { UInt32(clamping: $0) }
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: drawCount,
            eosTokenIds: eos,
            stopSequences: []
        )
        for (index, promptIds) in encoded.enumerated() {
            let result = try await Task.detached { [runtime, promptIds] in
                let forward = runtime.forward()
                forward.reset()
                return try DeepSeekV41TurnRunner.run(
                    forward: forward,
                    promptTokenIds: promptIds,
                    settings: settings,
                    sample: { logits, _ in Self.argmax(logits) },
                    decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
                )
            }.value
            let label = variants[index].label
            print(
                "🔎 [dsv41 real] tool probe '\(label)': \(result.finishReason) after "
                    + "\(result.tokensGenerated) token(s)"
            )
            print("🔎 [dsv41 real] tool probe '\(label)' text: \(result.text.debugDescription)")
            let pieces = result.generatedTokenIds.map { id in
                ((try? tokenizer.decode(ids: [id])) ?? "?").debugDescription
            }
            print("🔎 [dsv41 real] tool probe '\(label)' pieces: \(pieces.joined(separator: " "))")
            XCTAssertGreaterThan(result.tokensGenerated, 0)
        }
        withExtendedLifetime(engine) {}
    }

    /// Which result shape the checkpoint can READ, once it has made a call.
    ///
    /// The call probe settled the outbound half: this checkpoint emits a plain-text
    /// `<tool_call_name>name</tool_call_name>` plus a `<tool_call_arguments>` JSON
    /// body, even when it is shown the vocabulary's own `tool▁call` special tokens
    /// (all nine of which do tokenize as single ids). The inbound half is not
    /// defined anywhere in the checkpoint, so the loop's choice of result container
    /// is a guess unless it is measured: get it wrong and the model answers from the
    /// pre-call context as if the tool had never run.
    ///
    /// Each variant is the same conversation with a DIFFERENT result container, and
    /// the result carries a value the model could not know (`14:05`). A reply that
    /// states it understood; a reply that ignores it did not.
    func testWhatToolResultShapeTheCheckpointReads() async throws {
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
        let m = dialect.markers
        let system = """
        You are a helpful assistant.

        ## Tools
        You have access to the following tools:

        ### get_current_time
        Description: Returns the current local time for a city.

        Parameters: {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        """
        let ask = "What time is it in Paris right now? Use the tool."
        let call = "I'll check the current time in Paris for you.\n\n"
            + "<tool_call_name>get_current_time</tool_call_name>\n"
            + "<tool_call_arguments>{\"city\": \"Paris\"}</tool_call_arguments>"
        let payload = "{\"city\": \"Paris\", \"local_time\": \"14:05\"}"
        let variants: [(label: String, result: String)] = [
            (
                "output-special",
                m.toolOutputBegin + payload + m.toolOutputEnd
            ),
            (
                "outputs-special",
                m.toolOutputsBegin + m.toolOutputBegin + payload + m.toolOutputEnd
                    + m.toolOutputsEnd
            ),
            ("plain-tool-output", "<tool_call_output>" + payload + "</tool_call_output>"),
            ("tool-response", "<tool_response>" + payload + "</tool_response>"),
            ("bare-json", payload)
        ]
        // The continuation the app would build: the completed call turn, then the
        // result, then a fresh assistant turn with the reasoning channel closed.
        func continuation(_ result: String) -> String {
            m.beginOfSentence
                + m.system
                + dialect.systemBody(cleanSystem: system, thinkingEnabled: false, effort: nil)
                + m.user + ask
                + m.assistant + m.reasoningClose + call + m.endOfSentence
                + result
                + m.assistant + m.reasoningClose
        }
        let encoded = try variants.map { try tokenizer.encode(text: continuation($0.result)) }
        for (index, ids) in encoded.enumerated() {
            print("🔎 [dsv41 real] result probe '\(variants[index].label)': \(ids.count) tokens")
        }

        let drawCount = 48
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: (encoded.map(\.count).max() ?? 0) + drawCount + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let eos = config.eosTokenIDs.map { UInt32(clamping: $0) }
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: drawCount,
            eosTokenIds: eos,
            stopSequences: []
        )
        for (index, promptIds) in encoded.enumerated() {
            let result = try await Task.detached { [runtime, promptIds] in
                let forward = runtime.forward()
                forward.reset()
                return try DeepSeekV41TurnRunner.run(
                    forward: forward,
                    promptTokenIds: promptIds,
                    settings: settings,
                    sample: { logits, _ in Self.argmax(logits) },
                    decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
                )
            }.value
            let label = variants[index].label
            print(
                "🔎 [dsv41 real] result probe '\(label)': \(result.finishReason) after "
                    + "\(result.tokensGenerated) token(s), read=\(result.text.contains("14"))"
            )
            print("🔎 [dsv41 real] result probe '\(label)' text: \(result.text.debugDescription)")
            XCTAssertGreaterThan(result.tokensGenerated, 0)
        }
        withExtendedLifetime(engine) {}
    }

    /// The whole agent path on real weights: a system prompt in this checkpoint's
    /// tool dialect, a question that needs a tool, the freeze hook the engine
    /// installs, and the call it produces.
    ///
    /// The probes above established the shape; this is the assertion that the shape
    /// arrives in an actual turn, that generation stops ON the call rather than
    /// narrating after it, and that `DeepSeekV41ToolDialect` turns what arrived into
    /// an executable call. A dialect that parses a hand-written fixture but not a
    /// generation is a parser for the fixture.
    func testAnAgentTurnFreezesOnAToolCall() async throws {
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
        let system = """
        You are an expert AI software engineering assistant with direct access to local macOS tools.

        # Tools

        ### shell_run
        Description: Runs a shell command and returns its output.
        Parameters: {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

        \(DeepSeekV41ToolDialect.instructions())
        """
        let prompt = dialect.render(
            turns: [.init(isUser: true, text: "List the files in /tmp using the shell_run tool.")],
            cleanSystem: system,
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)

        let drawCount = 96
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: promptIds.count + drawCount + 8,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let eos = config.eosTokenIDs.map { UInt32(clamping: $0) }
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: drawCount,
            eosTokenIds: eos,
            stopSequences: []
        )
        let markers = dialect.markers
        let output = try await Task.detached { [runtime, promptIds] in
            let forward = runtime.forward()
            forward.reset()
            return try DeepSeekV41TurnRunner.run(
                forward: forward,
                promptTokenIds: promptIds,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                shouldFreeze: { text in
                    DeepSeekV41ToolDialect.shouldFreeze(text, markers: markers)
                },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
            )
        }.value
        print(
            "🔎 [dsv41 real] agent turn: \(output.finishReason.rawValue) after "
                + "\(output.tokensGenerated) token(s)"
        )
        print("🔎 [dsv41 real] agent turn text: \(output.text.debugDescription)")
        XCTAssertEqual(
            output.finishReason, .toolFreeze,
            "the turn did not stop on a call: \(output.text.debugDescription)"
        )
        let canonical = DeepSeekV41ToolDialect.canonicalized(output.text, markers: markers)
        let parsed = StreamingToolParser.shared.parseStreamingToolCalls(
            from: canonical, format: .qwenXML
        )
        print(
            "🔎 [dsv41 real] agent turn calls: "
                + parsed.calls.map { "\($0.name)\($0.arguments)" }.joined(separator: ", ")
        )
        XCTAssertEqual(parsed.calls.count, 1)
        XCTAssertEqual(parsed.calls.first?.name, "shell_run")
        // The command itself is the model's own choice (`ls -la /tmp` when it ran),
        // so the assertion is that a real command arrived, not which one.
        let command = parsed.calls.first?.arguments["command"] as? String
        XCTAssertEqual(command?.hasPrefix("ls"), true, "no shell command: \(command ?? "nil")")
        // The result turn the loop appends is the family's one-token output pair
        // around the payload, which is what keeps the continuation token-cheap.
        let resultIds = try tokenizer.encode(
            text: DeepSeekV41ToolDialect.resultTurn(
                payload: "total 0\nfile_a.txt", markers: markers
            )
        )
        XCTAssertEqual(resultIds.first, 128812)
        XCTAssertEqual(resultIds.last, 128813)
        XCTAssertGreaterThan(resultIds.count, 2)
        withExtendedLifetime(engine) {}
    }

    /// What the checkpoint does with the continuation the app actually builds after a
    /// tool result.
    ///
    /// `#27` settled the container (the model read all five shapes the probe tried the
    /// same way) and `#39` closed the SHAPE: the payload rides the call turn's own
    /// end-of-sentence marker inside the `tool▁output` pair, and the next assistant
    /// turn is opened after it. Both facts are pinned as strings and as a token
    /// prefix, but neither has generated a token, so the loop's answer after a tool
    /// result rests on a shape nobody has watched run.
    ///
    /// This walks the loop: the checkpoint is asked for a shell command, the model's
    /// OWN frozen call becomes the turn, the response is built by
    /// `AgentHarness.formatDeepSeekV41ToolResponseTurn` (the same call `ContentView`
    /// makes) and the result is the evidence. A turn that reads the payload names the
    /// files it was handed; a continuation that resumes the cache proves the spliced
    /// prefix is the one the previous turn left resident.
    func testATurnAnswersAfterAToolResultOnThePublishedCheckpoint() async throws {
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
        let markers = dialect.markers
        let system = """
        You are an expert AI software engineering assistant with direct access to local macOS tools.

        # Tools

        ### shell_run
        Description: Runs a shell command and returns its output.
        Parameters: {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}

        \(DeepSeekV41ToolDialect.instructions())
        """
        let ask = "List the files in /tmp using the shell_run tool."
        let prompt = dialect.render(
            turns: [.init(isUser: true, text: ask)],
            cleanSystem: system,
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)

        let drawCount = 64
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: promptIds.count + 320,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let eos = config.eosTokenIDs.map { UInt32(clamping: $0) }
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: drawCount,
            eosTokenIds: eos,
            stopSequences: []
        )

        // The call must be the model's own: a hand-written one would measure a
        // continuation whose call half the checkpoint never produced.
        let call = try await Task.detached { [runtime, promptIds] in
            let forward = runtime.forward()
            forward.reset()
            return try DeepSeekV41TurnRunner.run(
                forward: forward,
                promptTokenIds: promptIds,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                shouldFreeze: { DeepSeekV41ToolDialect.shouldFreeze($0, markers: markers) },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
            )
        }.value
        XCTAssertEqual(
            call.finishReason, .toolFreeze,
            "the turn did not stop on a call: \(call.text.debugDescription)"
        )

        // The response the loop would hand back for that call. `renderToolResultForModel`
        // turns it into the plain text the model reads; the two file names are the
        // sentinel the answer is judged against, because the question never mentioned
        // them.
        let responseJSON = #"{"tool":"shell_run","status":"success","result":{"command":"ls -la /tmp","exit_code":0,"output":"-rw-r--r-- 1 derek wheel 42 Oct 31 09:14 file_a.txt\n-rw-r--r-- 1 derek wheel 57 Oct 31 09:14 file_b.txt"}}"#
        let suffix = await MainActor.run {
            AgentHarness.shared.formatDeepSeekV41ToolResponseTurn(
                responses: [responseJSON],
                thinkingEnabled: false,
                promptDialect: dialect
            )
        }
        XCTAssertTrue(
            suffix.hasPrefix(markers.toolOutputBegin),
            "the result does not open the checkpoint's output pair: \(suffix.debugDescription)"
        )
        XCTAssertTrue(
            suffix.hasSuffix(dialect.assistantPrefix(thinkingEnabled: false)),
            "the continuation did not re-open the assistant turn: \(suffix.debugDescription)"
        )
        XCTAssertFalse(
            suffix.contains("<|im_start|>"),
            "a chatml role marker reached the continuation: \(suffix.debugDescription)"
        )

        let callTurnText = call.text.contains(markers.endOfSentence)
            ? call.text
            : call.text + markers.endOfSentence
        let continuation = prompt + callTurnText + suffix
        let continuationIds = try tokenizer.encode(text: continuation)
        let endTagIds = try tokenizer.encode(text: markers.endOfSentence)
        let suffixIds = try tokenizer.encode(text: suffix)
        let splicedIds = promptIds + call.generatedTokenIds + endTagIds + suffixIds
        let spliceMatches = splicedIds == continuationIds
        print(
            "🔎 [dsv41 real] tool continuation: \(promptIds.count) prompt + "
                + "\(call.generatedTokenIds.count) generated + \(endTagIds.count) end + "
                + "\(suffixIds.count) suffix = \(splicedIds.count) spliced, "
                + "\(continuationIds.count) re-encoded, splice==reencode \(spliceMatches)"
        )
        if !spliceMatches {
            let divergence = zip(splicedIds, continuationIds)
                .enumerated()
                .first { $0.element.0 != $0.element.1 }?
                .offset ?? min(splicedIds.count, continuationIds.count)
            print(
                "🔎 [dsv41 real] tool continuation diverges at \(divergence): "
                    + "spliced \(splicedIds.count), re-encoded \(continuationIds.count)"
            )
        }
        // The app pins the continuation as prompt ids + the real generation ids +
        // the closure, rather than re-encoding the whole transcript. If the two
        // disagree, the model reads a different prompt than the one that was
        // measured here and the resident prefix is no longer the turn's own.
        XCTAssertTrue(
            spliceMatches,
            "the spliced continuation is not the re-encoded one: \(splicedIds.count) "
                + "spliced vs \(continuationIds.count) re-encoded"
        )

        // What the app would propose: every token that went through the cache, which
        // is the prompt plus the generation (the turn's end tag is appended after the
        // freeze and never forwarded). The executor still compares them.
        let forwardedTokens = promptIds.count + call.generatedTokenIds.count - 1
        let prefix = runtime.forward().reusablePrefix(
            proposedPrefix: forwardedTokens,
            promptTokenIds: continuationIds.map { Int($0) }
        )
        print("🔎 [dsv41 real] tool continuation resumes \(prefix) of \(continuationIds.count)")
        // The freeze token is drawn but never forwarded, so the cache holds the
        // prompt plus everything but the last generated id. A larger or smaller
        // count means the pin and the turn it extends no longer describe the same
        // sequence, which is the one thing a post-tool turn cannot afford.
        XCTAssertEqual(
            prefix, forwardedTokens,
            "the continuation resumed \(prefix) of \(continuationIds.count), not the "
                + "call turn's \(forwardedTokens) forwarded tokens"
        )

        let answer = try await Task.detached { [runtime, continuationIds, prefix] in
            try DeepSeekV41TurnRunner.run(
                forward: runtime.forward(),
                promptTokenIds: continuationIds,
                prefixTokens: prefix,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" }
            )
        }.value
        print(
            "🔎 [dsv41 real] tool answer: \(answer.finishReason.rawValue) after "
                + "\(answer.tokensGenerated) token(s), reused \(answer.prefixTokensReused), "
                + String(format: "prefill %.2f s, decode %.2f s",
                         answer.prefillSeconds, answer.decodeSeconds)
        )
        print("🔎 [dsv41 real] tool answer text: \(answer.text.debugDescription)")
        print("🔎 [dsv41 real] tool answer ids: \(answer.generatedTokenIds)")
        XCTAssertGreaterThan(
            answer.tokensGenerated, 0,
            "the continuation produced no token (text \(answer.text.debugDescription))"
        )
        // The question never named a file, so an answer that names one came from the
        // result and not from the context above it.
        XCTAssertTrue(
            answer.text.contains("file_a"),
            "the answer did not read the tool result: \(answer.text.debugDescription)"
        )
        withExtendedLifetime(engine) {}
    }

    /// Whether two prefills of one token stream agree on what the indexer scored.
    ///
    /// The same executor, reset, the same tokens, twice: 576 positions, and zero of the
    /// 576 scores move -- bit for bit, not to a tolerance. That is the invariant every
    /// other numeric claim on this checkpoint rests on, and the count is the full 576 on
    /// purpose: this case moved ALL 576 scores (worst |delta| 0.4195) until "DSV4.1 #31"
    /// fixed the Engram gate's threadgroup reduction race, and running only the length
    /// that passed is how that defect stayed open. See `Benchmarks/PERF_FINDINGS.md`,
    /// "DSV4.1 #31", for the mechanism and `testEveryStageOfTwoPrefillsOfOneStreamAgrees`
    /// for the probe that localises a regression to a layer, a stage and a position.
    func testTwoPrefillsOfOneStreamAgreeOnTheIndexScores() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let positionCount = 576
        let tokens = Self.probeTokens(count: positionCount)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positionCount,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())

        let passes = try await Task.detached { [runtime, tokens] () -> [[Float]] in
            let executor = runtime.forward()
            executor.recordsIndexSelections = true
            var out: [[Float]] = []
            for _ in 0..<2 {
                executor.reset()
                _ = try executor.prefillChunk(tokens)
                let deepest = executor.indexSelectionRecords
                    .filter { $0.layer == 20 }
                    .max { $0.position < $1.position }
                out.append(deepest?.scores ?? [])
            }
            return out
        }.value

        let first = passes[0]
        let second = passes[1]
        print("🔎 [dsv41 real] determinism: \(first.count) vs \(second.count) score(s)")
        XCTAssertEqual(first.count, second.count)
        XCTAssertFalse(first.isEmpty, "no layer-20 selection was recorded")
        guard first.count == second.count, !first.isEmpty else { return }
        var differing = 0
        var worst: Float = 0
        var worstAt = 0
        for (index, pair) in zip(first, second).enumerated() {
            let delta = abs(pair.0 - pair.1)
            if delta > 0 { differing += 1 }
            if delta > worst { worst = delta; worstAt = index }
        }
        let scale = max(Float(1e-6), first.map(abs).max() ?? 1)
        print(
            "🔎 [dsv41 real] determinism: \(differing)/\(first.count) score(s) moved, "
                + "worst |delta| \(worst) at row \(worstAt) against a max |score| \(scale)"
        )
        XCTAssertEqual(differing, 0, "two identical prefills disagree on the index scores")
    }

    /// Whether every stage of two prefills of one stream agrees, and where not.
    ///
    /// `testTwoPrefillsOfOneStreamAgreeOnTheIndexScores` compares the scores a pass
    /// produced; this one compares the workspace at every stage that made them, so a
    /// regression names the layer, the stage and the position instead of only the
    /// value. It is the probe that localised the 576-position nondeterminism
    /// (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #31"): two runs disagreed only at an
    /// Engram stage, with the staged rows, the dequantized rows and the projection
    /// all identical and the gate's own value moving, which is what named a
    /// threadgroup reduction race rather than a bad read.
    ///
    /// It costs one wait per dense stage (the `recordsStageTiming` layout, which is
    /// the only layout where a stage is complete before the next one is encoded), so
    /// it runs on a prefix of the stream rather than the whole prompt. Opt-in with the
    /// other published-checkpoint cases.
    func testEveryStageOfTwoPrefillsOfOneStreamAgrees() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let positionCount = 192
        let block = max(
            1, DeepSeekV41DevicePlan.chunkSlots(config: config, contextLength: positionCount)
        )
        let tokens = Self.probeTokens(count: positionCount)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positionCount,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let engramTablesByLayer = Dictionary(
            uniqueKeysWithValues: config.engramTables.enumerated().map { ($1.layer, $0) }
        )
        let indexLayers: Set<Int> = [2, 8, 14, 20, 24, 28, 32, 36]

        let report = try await Task.detached {
            [runtime, tokens, block, engramTablesByLayer, indexLayers] () -> [String] in
            let executor = runtime.forward()
            executor.recordsStageTiming = true
            var lines: [String] = []
            var baselines: [String: [String: [UInt64]]] = [:]
            var pass = 0
            // A buffer only some layers write still holds the PREVIOUS pass's bytes
            // until its layer runs, so comparing it earlier reports the leftover rather
            // than a divergence.
            func liveDifferences(
                _ before: [String: [UInt64]], _ current: [String: [UInt64]], layer: Int
            ) -> [(name: String, slots: [Int], count: Int)] {
                var out: [(name: String, slots: [Int], count: Int)] = []
                for name in current.keys.sorted() {
                    if (name == "engramKV" || name == "engramGates"),
                       engramTablesByLayer[layer] == nil {
                        continue
                    }
                    if (name == "indexQuery" || name == "indexWeights"),
                       !indexLayers.contains(layer) {
                        continue
                    }
                    // Each table's staging belongs to the ONE layer that owns it.
                    if name.hasPrefix("engramRaw[") || name.hasPrefix("engramValues[") {
                        guard let open = name.firstIndex(of: "["),
                              let close = name.firstIndex(of: "]"),
                              let table = Int(name[name.index(after: open)..<close]),
                              engramTablesByLayer[layer] == table else { continue }
                    }
                    guard let now = current[name], let was = before[name] else { continue }
                    let moved = now.indices.filter { now[$0] != was[$0] }
                    guard !moved.isEmpty else { continue }
                    out.append((name: name, slots: moved, count: now.count))
                }
                return out
            }

            func snapshot(
                _ names: Set<String>?, withEngram: Bool
            ) -> [String: [UInt64]] {
                var digests = executor.slotDigests(names: names)
                if withEngram {
                    for (name, value) in executor.engramDigests() { digests[name] = value }
                }
                return digests
            }

            func compare(
                key: String, current: [String: [UInt64]], layer: Int, header: String
            ) {
                guard let before = baselines[key] else { return }
                let differences = liveDifferences(before, current, layer: layer)
                guard !differences.isEmpty else { return }
                lines.append(header)
                for difference in differences {
                    lines.append(
                        "  \(difference.name): "
                            + "\(difference.slots.map(String.init).joined(separator: ",")) "
                            + "of \(difference.count) slot(s) differ"
                    )
                }
            }

            executor.onBlockEmbedded = { startPosition, _ in
                let key = "block \(startPosition / block) embedding"
                let current = snapshot(["hidden", "streams", "tokenIDs", "hcMix"], withEngram: false)
                if pass == 0 {
                    baselines[key] = current
                } else {
                    compare(
                        key: key, current: current, layer: 0,
                        header: "block \(startPosition / block) disagrees AFTER the embedding"
                    )
                }
            }
            executor.onBlockStageFinished = { layer, startPosition, stage in
                let key = "block \(startPosition / block) layer \(layer) \(stage)"
                var names: Set<String> = ["streams"]
                if stage == "engram" { names.formUnion(["engramKV", "engramGates"]) }
                let current = snapshot(names, withEngram: stage == "engram")
                if pass == 0 {
                    baselines[key] = current
                } else {
                    compare(
                        key: key, current: current, layer: layer,
                        header: "block \(startPosition / block) layer \(layer) "
                            + "disagrees at stage \(stage)"
                    )
                }
            }
            executor.onBlockLayerFinished = { layer, startPosition, _ in
                let key = "block \(startPosition / block) layer \(layer) end"
                let current = snapshot(nil, withEngram: true)
                if pass == 0 {
                    baselines[key] = current
                } else {
                    compare(
                        key: key, current: current, layer: layer,
                        header: "block \(startPosition / block) layer \(layer) "
                            + "disagrees at the layer's end"
                    )
                }
            }

            for passIndex in 0..<2 {
                pass = passIndex
                executor.reset()
                for start in stride(from: 0, to: tokens.count, by: block) {
                    let end = min(start + block, tokens.count)
                    _ = try executor.prefillChunk(Array(tokens[start..<end]))
                }
            }
            return lines
        }.value

        print("🔎 [dsv41 real] per-stage diff: \(report.count) line(s)")
        for line in report { print("🔎 [dsv41 real] \(line)") }
        XCTAssertTrue(report.isEmpty, "two identical prefills disagree")
    }

    // MARK: - The composer's image path

    /// The RGB path the composer uses, priced and bound against each other.
    ///
    /// `testAnImageEntersAPublishedTurn` hands the tower a patch grid it built
    /// itself, so it never runs `DeepSeekV41VisionPreprocess.patches` or the
    /// fit-from-pixels path an attachment takes. This one starts from pixels, and
    /// pins the invariant the engine provisions its cache on: the prompt length the
    /// resize plan prices is the length binding actually produces. A disagreement
    /// there is a cache sized for one prompt and a turn run on another.
    func testAnRGBImagePricesAndBindsOnThePublishedWeights() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 256,
            allowEngramSkip: false
        )
        // The engine's own tower, so this drives the path an attachment takes: the
        // device pass, its one-time weight upload, and the priced length.
        let tower = try XCTUnwrap(
            runtime.visionForward(),
            "the engine could not build the tower from the published shards"
        )
        XCTAssertNotNil(
            tower.device,
            "the published tower should run as one device pass"
        )

        // A 56x56 image, which `fit` scales up to the checkpoint's pixel floor, so
        // this is the smallest admissible picture rather than a toy grid.
        let side = 56
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            for x in 0..<side {
                let offset = (y * side + x) * 3
                rgb[offset] = UInt8(x * 255 / (side - 1))
                rgb[offset + 1] = UInt8(y * 255 / (side - 1))
                rgb[offset + 2] = 64
            }
        }
        let image = DeepSeekV41ImageInput(
            rgb: rgb, width: side, height: side, sha256: "pixels-test"
        )
        let marker = UInt32(spec.imageTokenID)
        let tokenIds: [UInt32] = [1, 2, marker, 3]

        let price = try DeepSeekV41TurnImages.price(
            tokenIds: tokenIds, images: [image], spec: spec
        )
        let estimate = try XCTUnwrap(price.estimates.first)
        XCTAssertEqual(estimate.patchTokens, 1_521, "the pixel floor is the smallest grid")
        XCTAssertEqual(estimate.spanPositions, 184)

        let start = CFAbsoluteTimeGetCurrent()
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: tokenIds,
            images: [image],
            spec: spec,
            rowWidth: config.hiddenSize,
            encode: { input in
                // The tower returns rows; the span's length comes from the same
                // resize plan the price used.
                let fitted = DeepSeekV41VisionLayout.fit(
                    width: input.width, height: input.height, spec: spec
                )
                let rows = try tower.imageSpan(
                    rgb: input.rgb, width: input.width, height: input.height
                )
                return DeepSeekV41ImagePrompt.ImageSpan(
                    rows: rows, columns: fitted.columns, gridRows: fitted.rows
                )
            }
        )
        let seconds = CFAbsoluteTimeGetCurrent() - start
        XCTAssertEqual(
            bound.tokenIds.count, price.promptTokens,
            "the cache is provisioned for the priced length, so the bind must produce it"
        )
        XCTAssertEqual(bound.rows.count, estimate.spanPositions)
        XCTAssertEqual(bound.rows.keys.min(), 1 + 1)
        for row in bound.rows.values {
            XCTAssertEqual(row.count, config.hiddenSize)
            XCTAssertTrue(row.allSatisfy { $0.isFinite }, "the tower produced a non-finite row")
        }
        print(
            "🔎 [dsv41 real] rgb bind: \(side)x\(side) pixels -> \(estimate.patchTokens) patch "
                + "tokens, \(estimate.spanPositions) span positions, priced "
                + String(format: "%.2f s", estimate.seconds) + " against "
                + String(format: "%.2f s", seconds) + " measured (a Debug bundle pays the "
                + "one-time BF16->FP32 upload at -Onone; the app's -O build pays ~0.35 s)"
        )
        withExtendedLifetime(engine) {}
    }


    /// The app's OWN formatter, not the dialect on its own. A chat turn is
    /// rendered by `ChatPromptFormatter.formatPrompt`, which picks the DeepSeek
    /// branch from the model config and then calls `dialect.render`; `#35` proved
    /// the placeholder survives `dialect.render` and binds a span, but not that it
    /// survives the app's path. A marker that tokenized into pieces HERE would
    /// leave literal text in the prompt and bind no rows, which looks exactly like
    /// a model that ignores pictures.
    ///
    /// It needs the checkpoint's tokenizer and config, not its weights, so it runs
    /// in milliseconds.
    func testTheComposedPromptCarriesOneImageTokenPerAttachment() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(
            config.vision, "the published config.json must publish a vision_config"
        )
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        // Decoded the way `LocalModelManager` decodes it, so the formatter takes
        // the branch it takes in the app.
        let modelConfig = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        )
        XCTAssertTrue(
            modelConfig.isDeepSeekV41Model,
            "the formatter only takes the DeepSeek branch when the model config says so"
        )
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)

        func composed(_ text: String) -> String {
            ChatPromptFormatter.formatPrompt(
                prompt: text,
                cleanSystem: "",
                thinkingEnabled: false,
                modelSupportsThinking: true,
                modelConfig: modelConfig,
                thinkingEffort: nil,
                promptDialect: dialect
            )
        }

        let question = "what is in this picture?"
        let plain = composed(question)
        let one = composed(markers.imagePlaceholder + "\n" + question)
        let two = composed(
            markers.imagePlaceholder + "\n" + markers.imagePlaceholder + "\n" + question
        )
        // Controls that put ONE ORDINARY TOKEN where each marker goes, with the
        // whitespace unchanged, so the only difference between the two prompts is
        // which token sits in the marker's place. Counting against the bare
        // question instead gets this wrong: a newline is its own token, and two
        // adjacent newlines are one.
        let oneControl = composed("x\n" + question)
        let twoControl = composed("x\nx\n" + question)
        XCTAssertEqual(
            one.components(separatedBy: markers.imagePlaceholder).count - 1, 1,
            "the formatter must not escape or drop the placeholder"
        )
        XCTAssertEqual(
            two.components(separatedBy: markers.imagePlaceholder).count - 1, 2,
            "two attachments must stay two placeholders"
        )

        let markerOnly = try tokenizer.encode(text: markers.imagePlaceholder)
        XCTAssertEqual(
            markerOnly.map(Int.init), [spec.imageTokenID],
            "the placeholder is one added token, not a run of text tokens"
        )

        let plainIds = try tokenizer.encode(text: plain)
        let oneIds = try tokenizer.encode(text: one)
        let twoIds = try tokenizer.encode(text: two)
        let oneControlIds = try tokenizer.encode(text: oneControl)
        let twoControlIds = try tokenizer.encode(text: twoControl)
        XCTAssertEqual(
            plainIds.filter { Int($0) == spec.imageTokenID }.count, 0,
            "a prompt with no picture must carry no image token"
        )
        XCTAssertEqual(
            oneIds.filter { Int($0) == spec.imageTokenID }.count, 1,
            "one attachment must be one image token"
        )
        XCTAssertEqual(
            twoIds.filter { Int($0) == spec.imageTokenID }.count, 2,
            "two attachments must be two image tokens"
        )
        XCTAssertEqual(
            oneIds.count, oneControlIds.count,
            "the marker must cost exactly what one ordinary token costs"
        )
        XCTAssertEqual(twoIds.count, twoControlIds.count)

        // Tokenizing is only half the contract: `expand` is what binds rows to
        // positions, and it must find one placeholder per image and leave nothing
        // unfilled.
        let cell = [Float](repeating: 0.5, count: config.hiddenSize)
        let span = DeepSeekV41ImagePrompt.ImageSpan(
            rows: Array(
                repeating: cell,
                count: DeepSeekV41VisionLayout.imageTokenCount(columns: 1, rows: 1)
            ),
            columns: 1,
            gridRows: 1
        )
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: twoIds,
            spans: [span, span],
            imageTokenID: spec.imageTokenID,
            rowWidth: config.hiddenSize
        )
        XCTAssertEqual(plan.report.unusedImageCount, 0)
        XCTAssertEqual(plan.report.unfilledPlaceholderCount, 0)
        XCTAssertEqual(plan.rows.count, 2 * span.tokenCount)
        XCTAssertEqual(plan.promptTokens, twoIds.count - 2 + 2 * span.tokenCount)
        print(
            "🔎 [dsv41 real] composer prompt: \(plainIds.count) tokens plain, "
                + "\(oneIds.count) with one image, \(twoIds.count) with two; expand "
                + "binds \(plan.rows.count) row(s), \(plan.report.unfilledPlaceholderCount) "
                + "placeholder(s) unfilled, \(plan.report.unusedImageCount) image(s) unused"
        )
    }

    /// The chat UI's history, at the checkpoint's own tokenizer.
    ///
    /// `DeepSeekV41ChatPromptFormatterTests` pins the rendered SHAPE. This pins the
    /// property the engine's prefix pin actually needs: turn 2's token ids must
    /// extend turn 1's prompt ids plus the turn's generation ids. A string prefix is
    /// necessary but not sufficient — a BPE merge across the boundary would break the
    /// pin while every string assertion still passed — and the boundary here is a
    /// special token, which is why it holds.
    func testAChatTurnsTokensExtendThePreviousPromptPlusGeneration() async throws {
        let snapshot = try snapshotDirectory()
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let modelConfig = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        )
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let system = "You are a careful assistant."
        for thinking in [false, true] {
            let turn1 = ChatPromptFormatter.formatConversation(
                turns: [.init(role: .user, text: "hello there")],
                cleanSystem: system,
                thinkingEnabled: thinking,
                modelSupportsThinking: true,
                modelConfig: modelConfig,
                thinkingEffort: nil,
                promptDialect: dialect
            )
            let generation = thinking
                ? "weighing it" + TurnTextSplitter.canonicalReasoningCloseTag + "an answer"
                : "an answer"
            let turn2 = ChatPromptFormatter.formatConversation(
                turns: [
                    .init(role: .user, text: "hello there"),
                    .init(role: .assistant, text: generation),
                    .init(role: .user, text: "and again"),
                ],
                cleanSystem: system,
                thinkingEnabled: thinking,
                modelSupportsThinking: true,
                modelConfig: modelConfig,
                thinkingEffort: nil,
                promptDialect: dialect
            )
            let promptIds = try tokenizer.encode(text: turn1)
            let generationIds = try tokenizer.encode(text: generation)
            let nextIds = try tokenizer.encode(text: turn2)
            XCTAssertEqual(
                Array(nextIds.prefix(promptIds.count)), promptIds,
                "thinking=\(thinking): turn 2 must re-encode turn 1's prompt to the same ids"
            )
            XCTAssertEqual(
                Array(nextIds.dropFirst(promptIds.count).prefix(generationIds.count)),
                generationIds,
                "thinking=\(thinking): turn 2 must carry the turn's generation ids unchanged"
            )
            XCTAssertGreaterThan(nextIds.count, promptIds.count + generationIds.count)
            print(
                "🔎 [dsv41 real] chat turn: \(promptIds.count) prompt ids + "
                    + "\(generationIds.count) generated, turn 2 is \(nextIds.count)"
            )
        }
    }

    // MARK: - An image in a real turn

    /// The tower at the checkpoint's own weights, and its rows inside a real turn.
    ///
    /// This is the first time the tower runs on the published weights at all: every
    /// other vision test is a synthetic fixture with an independent numpy mirror. What
    /// it proves is the wiring, not the speed: a 2x2 patch grid instead of a
    /// photograph, because `fit` scales any admissible image up to the checkpoint's
    /// `min_pixels` (544x544, 1,521 patch tokens) and the tower is scalar host code.
    /// The 4x4 grid timed beside it makes the per-token cost a slope rather than one
    /// point, which is what says whether this path can ever serve.
    func testAnImageEntersAPublishedTurn() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(
            config.vision, "the published config.json must publish a vision_config"
        )
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }

        // A prompt with one placeholder where a picture belongs. The placeholder is
        // the checkpoint's own string and its id is the config's `image_token_id`, so
        // it must survive tokenization as exactly ONE token: the expansion replaces
        // that one id with the span, and a marker that tokenized into pieces would
        // expand one piece and leave the rest as literal text.
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?")],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)
        XCTAssertFalse(promptIds.isEmpty)
        let placeholderCount = promptIds.filter { Int($0) == spec.imageTokenID }.count
        XCTAssertEqual(
            placeholderCount, 1,
            "the image placeholder must tokenize to exactly one image token; it produced "
                + "\(placeholderCount) in \(promptIds.count) tokens"
        )
        print("🔎 [dsv41 real] image prompt: \(promptIds.count) tokens, image id \(spec.imageTokenID)")

        // A 2x2 patch grid is one aligner cell, so the span is start + row + newline +
        // end. A 4x4 grid is four, timed beside it for the per-token slope.
        let patchVolume = 3 * spec.patchSize * spec.patchSize
        func patches(_ side: Int) -> [[Float]] {
            (0..<(side * side)).map { index in
                (0..<patchVolume).map { dimension in
                    Float((index * 3 + dimension) % 13) / 13 - 0.5
                }
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let contextLength = promptIds.count + spec.maxImageTokens + 8
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: contextLength,
            allowEngramSkip: false
        )
        let source = try XCTUnwrap(
            DeepSeekV41VisionWeights(snapshotDir: snapshot, catalog: runtime.catalog),
            "the tower's tensors could not be resolved from the published shards"
        )
        XCTAssertEqual(source.problems, [])
        XCTAssertEqual(
            source.textHiddenSize, config.hiddenSize,
            "the aligner must project to the text width or a row cannot enter the residual"
        )
        print("🔎 [dsv41 real] vision weights: \(source.tensorCount) tensors")
        let tower = DeepSeekV41VisionForward(
            spec: spec, weights: source, textHiddenSize: config.hiddenSize
        )

        let start = CFAbsoluteTimeGetCurrent()
        let spanRows = try tower.imageSpan(patches: patches(2), rows: 2, columns: 2)
        let oneCellSeconds = CFAbsoluteTimeGetCurrent() - start
        XCTAssertEqual(
            spanRows.count, DeepSeekV41VisionLayout.imageTokenCount(columns: 1, rows: 1)
        )
        for row in spanRows {
            XCTAssertEqual(row.count, config.hiddenSize)
            XCTAssertTrue(row.allSatisfy(\.isFinite), "the tower produced a non-finite row")
        }
        let oneCellSpan = DeepSeekV41ImagePrompt.ImageSpan(
            rows: spanRows, columns: 1, gridRows: 1
        )

        // The per-token slope. Four times the patch tokens at one aligner row per
        // cell: 16 patch tokens -> a 2x2 grid -> 8 span positions.
        let startWide = CFAbsoluteTimeGetCurrent()
        let wideRows = try tower.imageSpan(patches: patches(4), rows: 4, columns: 4)
        let fourCellsSeconds = CFAbsoluteTimeGetCurrent() - startWide
        XCTAssertEqual(wideRows.count, DeepSeekV41VisionLayout.imageTokenCount(columns: 2, rows: 2))

        // The smallest image the checkpoint ACCEPTS: `min_pixels` 295,936 rounds up to
        // 546x546, which at a patch size of 14 is a 39x39 grid, 1521 patch tokens. This
        // is the number that decides whether vision can be served at all, so it is
        // timed rather than extrapolated: the two small grids above measure the slope,
        // this measures the picture.
        let floorStart = CFAbsoluteTimeGetCurrent()
        let floorRows = try tower.imageSpan(patches: patches(39), rows: 39, columns: 39)
        let floorSeconds = CFAbsoluteTimeGetCurrent() - floorStart
        XCTAssertEqual(
            floorRows.count, DeepSeekV41VisionLayout.imageTokenCount(columns: 13, rows: 13)
        )
        for row in floorRows {
            XCTAssertEqual(row.count, config.hiddenSize)
            XCTAssertTrue(row.allSatisfy(\.isFinite), "the tower produced a non-finite row")
        }

        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: promptIds,
            spans: [oneCellSpan],
            imageTokenID: spec.imageTokenID,
            rowWidth: config.hiddenSize
        )
        XCTAssertEqual(plan.report.unusedImageCount, 0)
        XCTAssertEqual(plan.report.unfilledPlaceholderCount, 0)
        XCTAssertEqual(plan.rows.count, oneCellSpan.tokenCount)
        XCTAssertEqual(plan.spans.count, 1)
        print(
            "🔎 [dsv41 real] tower: 4 patch tokens in "
                + String(format: "%.2f s", oneCellSeconds) + ", 16 in "
                + String(format: "%.2f s", fourCellsSeconds) + ", 1521 in "
                + String(format: "%.2f s", floorSeconds) + "; span "
                + "\(oneCellSpan.tokenCount) rows for the 2x2 grid, prompt "
                + "\(promptIds.count) -> \(plan.promptTokens) tokens"
        )
        print(
            "🔎 [dsv41 real] tower floor: "
                + String(format: "%.1f ms per marginal patch token",
                         1_000 * (fourCellsSeconds - oneCellSeconds) / 12)
                + ", smallest image "
                + String(format: "%.2f s", floorSeconds)
                + " (184 span positions)"
        )

        // The LARGEST image the checkpoint accepts, which is the other end of the
        // range and the one nothing has measured. The grid comes from `fit` rather
        // than being chosen here, so this cannot drift from what a real oversized
        // image would become: a 20,000 pixel square collapses onto the token
        // budget, and the patch grid it returns is the biggest admissible one.
        // Opt-in because the cost is quadratic in the token count, so at `-Onone`
        // this would measure the interpreter for minutes rather than the tower.
        if FileManager.default.fileExists(atPath: "/tmp/dynamoe_dsv41_vision_max") {
            let capped = DeepSeekV41VisionLayout.fit(width: 20_000, height: 20_000, spec: spec)
            XCTAssertGreaterThan(capped.tokens, 0)
            XCTAssertLessThanOrEqual(
                capped.tokens, spec.maxImageTokens,
                "fit must return a span inside the checkpoint's own token budget"
            )
            let capSide = capped.width / max(1, spec.patchSize)
            XCTAssertEqual(
                capSide, capped.height / max(1, spec.patchSize),
                "a square image must stay square through the resize plan"
            )
            // The fixed cost, measured instead of inferred. Every encode preads the
            // tower's 970 MB of weights and widens them again, so a warm call is
            // that cost plus its arithmetic; the first call of a process also fills
            // the page cache. Without this the slope between the first and second
            // small grid is negative whenever the fixed cost is small: the cold read
            // lands entirely in the first sample.
            let warmStart = CFAbsoluteTimeGetCurrent()
            _ = try tower.imageSpan(patches: patches(2), rows: 2, columns: 2)
            let warmSeconds = CFAbsoluteTimeGetCurrent() - warmStart
            print(
                "🔎 [dsv41 real] tower warm: a repeated 4-token call in "
                    + String(format: "%.2f s", warmSeconds)
                    + " (the first call pays the cold read, so this is the fixed cost)"
            )

            // Three token counts fit a cost model with a fixed, a linear and a
            // quadratic term, which is the shape of the tower: everything but
            // attention and softmax is linear in the token count, and those two are
            // quadratic. Two points could only draw a line through the origin and
            // would attribute every quadratic second to the linear term.
            let midSide = 63
            let midStart = CFAbsoluteTimeGetCurrent()
            let midRows = try tower.imageSpan(
                patches: patches(midSide), rows: midSide, columns: midSide
            )
            let midSeconds = CFAbsoluteTimeGetCurrent() - midStart
            XCTAssertEqual(
                midRows.count,
                DeepSeekV41VisionLayout.imageTokenCount(
                    columns: (midSide + spec.downsampleRatio - 1) / spec.downsampleRatio,
                    rows: (midSide + spec.downsampleRatio - 1) / spec.downsampleRatio
                )
            )

            let capPatches = capSide * capSide
            let capStart = CFAbsoluteTimeGetCurrent()
            let capRows = try tower.imageSpan(
                patches: patches(capSide), rows: capSide, columns: capSide
            )
            let capSeconds = CFAbsoluteTimeGetCurrent() - capStart
            XCTAssertEqual(
                capRows.count, capped.tokens,
                "the span must be the length the resize plan priced"
            )
            for row in capRows {
                XCTAssertEqual(row.count, config.hiddenSize)
                XCTAssertTrue(row.allSatisfy(\.isFinite), "the tower produced a non-finite row")
            }
            print(
                "🔎 [dsv41 real] tower cap: \(capPatches) patch tokens (\(capSide)x\(capSide)) "
                    + "-> \(capped.columns)x\(capped.rows) aligner grid, \(capRows.count) span "
                    + "positions in " + String(format: "%.2f s", capSeconds)
                    + String(format: " (%.1fx the smallest image, %.2f ms per patch token)",
                             capSeconds / max(floorSeconds, 0.001),
                             1_000 * capSeconds / Double(max(capPatches, 1)))
            )
            // cost(t) = fixed + perToken * t + perTokenSquared * t^2, solved exactly
            // through the three measured points rather than least squares, so the
            // numbers reproduce from the log.
            let fitTokens = [4.0, Double(midSide * midSide), Double(capPatches)]
            let fitSeconds = [warmSeconds, midSeconds, capSeconds]
            if let model = Self.costModel(tokens: fitTokens, seconds: fitSeconds) {
                let quadraticShare = model.squared * fitTokens[2] * fitTokens[2] / fitSeconds[2]
                print(
                    "🔎 [dsv41 real] tower cost model: "
                        + String(format: "%.2f s fixed + %.3f ms per token + %.3e s per token^2",
                                 model.fixed, 1_000 * model.perToken, model.squared)
                        + String(format: "; at %d tokens the quadratic term is %.0f%% of the %.2f s",
                                 capPatches, 100 * quadraticShare, capSeconds)
                        + String(format: " (mid grid %dx%d = %d tokens in %.2f s)",
                                 midSide, midSide, midSide * midSide, midSeconds)
                )
            }
        }

        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: 2,
            eosTokenIds: []
        )
        let draw = try await Task.detached { [runtime, plan] in
            try DeepSeekV41TurnRunner.run(
                forward: runtime.forward(),
                promptTokenIds: plan.tokenIds,
                // A prefix the pin might propose is refused by the loop when a prompt
                // carries image rows, so proposing one here is the assertion.
                prefixTokens: max(0, promptIds.count - 1),
                imageEmbeddings: plan.rows,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                decode: { ids in (try? tokenizer.decode(ids: ids)) ?? "" }
            )
        }.value

        print(
            "🔎 [dsv41 real] image turn: \(draw.finishReason) after \(draw.tokensGenerated) "
                + "token(s), prompt \(draw.promptTokens), reused \(draw.prefixTokensReused), "
                + "prefill " + String(format: "%.2f s", draw.prefillSeconds)
        )
        print("🔎 [dsv41 real] image turn text: \(draw.text)")
        XCTAssertEqual(draw.prefixTokensReused, 0, "an image prompt must not resume a prefix")
        XCTAssertGreaterThan(draw.tokensGenerated, 0)
        withExtendedLifetime(engine) {}
    }

    /// A batched image prefill is the walk's logits, on the checkpoint's own weights.
    ///
    /// The synthetic fixture publishes no Engram layers, so it cannot see the one rule
    /// this path could get wrong on its own: a supplied residual row is an IMAGE
    /// position and takes no part in an n-gram. This checkpoint has two memory layers
    /// (1 and 14), and the span starts at position 2 of a twelve-token prompt, so a
    /// block that inherited the first slot's flag — or left every position live —
    /// hashes an image row into the residual and moves the logits.
    ///
    /// The rows are the tower's own output for the smallest admissible picture, so this
    /// also covers the width and the upload. Two blocks of positions, because the first
    /// window the span crosses is where the ring wraps.
    func testABatchedImagePrefillMatchesTheWalkOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(
                isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?"
            )],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: promptIds.count + spec.maxImageTokens + 8,
            allowEngramSkip: false
        )
        let tower = try XCTUnwrap(runtime.visionForward())
        let catalogue = runtime.catalog
        let source = try XCTUnwrap(
            DeepSeekV41VisionWeights(snapshotDir: snapshot, catalog: catalogue)
        )
        let host = DeepSeekV41VisionForward(
            spec: spec, weights: source, textHiddenSize: config.hiddenSize
        )
        // The smallest admissible picture: 1,521 patch tokens, a 13x13 aligner grid and
        // 184 span positions. Encoded by the tower the turn would use.
        let fitted = DeepSeekV41VisionLayout.fit(width: 56, height: 56, spec: spec)
        let patchVolume = 3 * spec.patchSize * spec.patchSize
        let cells = (0..<(fitted.height / spec.patchSize * (fitted.width / spec.patchSize)))
            .map { index in
                (0..<patchVolume).map { dimension in
                    Float((index * 5 + dimension * 3) % 19) / 19 - 0.5
                }
            }
        let spanRows = try tower.imageSpan(
            patches: cells,
            rows: fitted.height / spec.patchSize,
            columns: fitted.width / spec.patchSize
        )
        XCTAssertEqual(spanRows.count, fitted.columns * (fitted.rows + 1) + 2)
        let hostRows = try host.imageSpan(
            patches: cells,
            rows: fitted.height / spec.patchSize,
            columns: fitted.width / spec.patchSize
        )
        XCTAssertEqual(hostRows.count, spanRows.count)
        var towerDelta: Float = 0
        for (row, expected) in zip(spanRows, hostRows) {
            let scale = max(expected.map(abs).max() ?? 1, 1e-6)
            for index in expected.indices {
                towerDelta = max(towerDelta, abs(row[index] - expected[index]) / scale)
            }
        }
        XCTAssertLessThan(
            towerDelta, 5e-4,
            "the device tower's rows are not the host's, so nothing below is a fair test"
        )
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: promptIds,
            spans: [.init(rows: spanRows, columns: fitted.columns, gridRows: fitted.rows)],
            imageTokenID: spec.imageTokenID,
            rowWidth: config.hiddenSize
        )
        XCTAssertEqual(plan.report.unusedImageCount, 0)
        XCTAssertEqual(plan.report.unfilledPlaceholderCount, 0)

        // Two blocks, so the span crosses a window and the second block continues a
        // cache the first filled.
        let width = DeepSeekV41DevicePlan.chunkSlots(
            config: config, contextLength: runtime.contextLength
        )
        XCTAssertGreaterThan(width, 1)
        let count = min(plan.tokenIds.count, width * 2)
        let tokens = Array(plan.tokenIds.prefix(count)).map { Int($0) }
        let rows = plan.rows.filter { $0.key < count }
        XCTAssertEqual(rows.count, count - 2, "every position after the two text tokens is a row")
        print("🔎 [dsv41 real] image parity: \(count) positions, \(rows.count) image rows")

        let executor = runtime.forward()
        func arm(_ label: String, _ body: () throws -> Void) rethrows {
            executor.reset()
            runtime.arena.reset()
            let before = runtime.expertStreamer?.bytesRead ?? 0
            let blocksBefore = executor.blockMixtureSubmissions
            let fallbacksBefore = executor.perPositionMixtureSubmissions
            let started = DispatchTime.now().uptimeNanoseconds
            try body()
            let ms = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            let bytes = (runtime.expertStreamer?.bytesRead ?? 0) - before
            print(
                "🔎 [dsv41 real] arm \(label): blocks "
                    + "\(executor.blockMixtureSubmissions - blocksBefore) per-position "
                    + "\(executor.perPositionMixtureSubmissions - fallbacksBefore), "
                    + String(format: "%.1f ms, %.0f MB, %.0f%% hit, arena %d",
                             ms,
                             Double(bytes) / 1_048_576,
                             100 * runtime.arena.hitRate,
                             runtime.arena.capacity)
            )
        }
        var chunk = DeepSeekV41ChunkLogits(last: [])
        try arm("chunk+rows") { chunk = try executor.prefillChunk(tokens, embeddings: rows) }
        try arm("chunk text") { _ = try executor.prefillChunk(tokens) }
        var walked: [Float] = []
        try arm("walk") {
            for (position, token) in tokens.enumerated() {
                if let row = rows[position] {
                    walked = try executor.forwardEmbedding(row, token: token, position: position)
                } else {
                    walked = try executor.forwardToken(token, position: position)
                }
            }
        }
        XCTAssertEqual(chunk.last.count, walked.count)
        let scale = max(walked.map(abs).max() ?? 1, 1e-6)
        var delta: Float = 0
        for index in walked.indices {
            delta = max(delta, abs(chunk.last[index] - walked[index]) / scale)
        }
        print(
            "🔎 [dsv41 real] image parity: chunk against walk "
                + String(format: "%.2e", delta) + " relative at a logit scale of "
                + String(format: "%.1f", scale)
        )
        XCTAssertLessThan(
            delta, 5e-4,
            "the batched image prefill is not the walk: an n-gram read a row it should "
                + "not see, or a row landed at the wrong position"
        )
        withExtendedLifetime(engine) {}
    }

    /// Where batching an image prefill stops paying, as the prompt grows.
    ///
    /// At sixteen positions the block form is 1.06x the walk: the arena holds every
    /// expert the span touches, so the only difference is the round trips. The largest
    /// picture is 1,005 positions and touches ~13,000 `(layer, expert)` pairs against
    /// an arena of 3,760, so every pair is read whichever layout runs and the two arms
    /// converge on the disk. This sweeps the length to find where the crossover is, and
    /// prints each arm's expert bytes and which mixture layout ran, because a batching
    /// claim that silently fell back to one position at a time is not a batching claim.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_vision_max_turn`): the longest arm walks 512
    /// positions of the published checkpoint.
    func testWhatBatchingAnImagePrefillSavesAtLength() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_vision_max_turn"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in length sweep; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 1_056,
            allowEngramSkip: false
        )
        let tower = try XCTUnwrap(runtime.visionForward())

        // The largest admissible span, encoded once; prefixes of it are the arms.
        let fitted = DeepSeekV41VisionLayout.fit(width: 20_000, height: 20_000, spec: spec)
        let patchVolume = 3 * spec.patchSize * spec.patchSize
        let side = fitted.height / spec.patchSize
        let cells = (0..<(side * side)).map { index in
            (0..<patchVolume).map { dimension in
                Float((index * 3 + dimension * 7) % 17) / 17 - 0.5
            }
        }
        let spanRows = try tower.imageSpan(patches: cells, rows: side, columns: side)
        let promptIds: [UInt32] = [1, 2, UInt32(spec.imageTokenID), 3]
        let plan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: promptIds,
            spans: [.init(rows: spanRows, columns: fitted.columns, gridRows: fitted.rows)],
            imageTokenID: spec.imageTokenID,
            rowWidth: config.hiddenSize
        )
        print(
            "🔎 [dsv41 real] sweep: \(spanRows.count) span rows for the largest image, "
                + "prompt \(plan.tokenIds.count) positions"
        )

        let executor = runtime.forward()
        let width = DeepSeekV41DevicePlan.chunkSlots(
            config: config, contextLength: runtime.contextLength
        )
        // A discarded warm-up: the first arm of a process pays the arena's slot
        // allocations and the page cache, which is 1.5 s on a sixteen-position arm and
        // would otherwise be read as the block form being slower.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(plan.tokenIds.prefix(8)).map { Int($0) })
        for length in [16, 128, 512, 1_005] {
            let count = min(length, plan.tokenIds.count)
            let tokens = Array(plan.tokenIds.prefix(count)).map { Int($0) }
            let rows = plan.rows.filter { $0.key < count }

            func arm(
                _ body: () throws -> Void
            ) rethrows -> (
                milliseconds: Double, megabytes: Double, hit: Double,
                blocks: Int, fallbacks: Int
            ) {
                executor.reset()
                runtime.arena.reset()
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let blocksBefore = executor.blockMixtureSubmissions
                let fallbacksBefore = executor.perPositionMixtureSubmissions
                let started = DispatchTime.now().uptimeNanoseconds
                try body()
                let milliseconds =
                    Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                return (
                    milliseconds,
                    Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                    runtime.arena.hitRate,
                    executor.blockMixtureSubmissions - blocksBefore,
                    executor.perPositionMixtureSubmissions - fallbacksBefore
                )
            }

            let chunked = try arm {
                _ = try executor.prefillChunk(tokens, embeddings: rows)
            }
            let walked = try arm {
                for (position, token) in tokens.enumerated() {
                    if let row = rows[position] {
                        _ = try executor.forwardEmbedding(row, token: token, position: position)
                    } else {
                        _ = try executor.forwardToken(token, position: position)
                    }
                }
            }
            let positions = Double(count)
            print(
                "🔎 [dsv41 real] sweep \(count) positions (block \(width)): "
                    + "chunk " + String(format: "%.1f", chunked.milliseconds) + " ms ("
                    + String(format: "%.1f", chunked.milliseconds / positions)
                    + " ms/pos, " + String(format: "%.0f", chunked.megabytes) + " MB, "
                    + String(format: "%.0f%%", 100 * chunked.hit) + " hit, "
                    + "\(chunked.blocks) block/\(chunked.fallbacks) per-position), "
                    + "walk " + String(format: "%.1f", walked.milliseconds) + " ms ("
                    + String(format: "%.1f", walked.milliseconds / positions)
                    + " ms/pos, " + String(format: "%.0f", walked.megabytes) + " MB, "
                    + String(format: "%.0f%%", 100 * walked.hit) + " hit) -> "
                    + String(format: "%.2fx", walked.milliseconds / max(chunked.milliseconds, 1))
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// Where a batched prefill's time goes at a published span.
    ///
    /// `#46` answered WHO wins (the block form up to a 1.5x working set, the walk
    /// beyond it) and named the mechanism (a union fill evicts 20-48 pairs at once
    /// where a walk's 6-pair fill keeps what it just used). What it could not say is
    /// where the block form's OWN time goes when it loses: `DeepSeekV41BlockTiming`
    /// had three host buckets that folded the routing readback, the union fill and the
    /// mixture encode into one, no block wall time, and none of the drain/late-start
    /// attribution the walk got in `#50`, so ~215 ms/position of a 353 ms/position
    /// block arm was unattributed by construction.
    ///
    /// This runs both layouts over the same stream in one runtime, each from a reset
    /// executor and a reset arena, printing each arm's own bucket table, its wall time
    /// and its expert bytes. Two rounds with the arm order ROTATED, because a warm page
    /// cache is worth 2x on the first arm (`#51`).
    ///
    /// A text stream rather than a picture: the lengths are what matter here and the
    /// tower's 6.9 s encode would be paid twice for nothing. The working-set ratio the
    /// arm reaches is printed beside the numbers so the regime can be read off rather
    /// than assumed.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_block_span` for 128 positions,
    /// `/tmp/dynamoe_dsv41_block_span_long` for the 1,005-position arm, which is ~20
    /// minutes at `-O`).
    func testWhereABatchedPrefillsTimeGoesOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_block_span"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in block attribution; touch \(markerPath) to run"
        )
        let longMarkerPath = "/tmp/dynamoe_dsv41_block_span_long"
        let lengths = FileManager.default.fileExists(atPath: longMarkerPath)
            ? [128, 1_005] : [128]

        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")

        /// The prompt of `length` tokens: the paragraph repeated, rotated once per
        /// repetition so the sequence is not periodic. A plain repeat cycles the same
        /// routing and leaves the arena warm, which is a different regime from any real
        /// prompt; the arm's own misses per position are printed so the regime it did
        /// reach is read off rather than assumed.
        func prompt(_ length: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * 13) % stream.count]
            }
        }

        let longest = lengths.max() ?? 128
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: longest + 1,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let width = DeepSeekV41DevicePlan.chunkSlots(
            config: config, contextLength: runtime.contextLength
        )
        let slots = runtime.arena.capacity
        print(
            "🔎 [dsv41 real] block attribution: block width \(width), arena \(slots) slot(s), "
                + "stream repeats every \(stream.count) token(s)"
        )

        // One discarded warm-up so the first arm does not pay the slot allocations.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        for length in lengths {
            let tokens = prompt(length)
            let positions = Double(max(1, length))

            func blockArm() throws -> (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41BlockTiming) {
                executor.reset()
                runtime.arena.reset()
                executor.resetBlockTiming()
                executor.recordsBlockTiming = true
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
                let started = DispatchTime.now().uptimeNanoseconds
                _ = try executor.prefillChunk(tokens)
                let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                executor.recordsBlockTiming = false
                return (
                    wall,
                    Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                    (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                    executor.blockTiming
                )
            }

            func walkArm() throws -> (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41WalkTiming) {
                executor.reset()
                runtime.arena.reset()
                executor.resetWalkTiming()
                executor.recordsWalkTiming = true
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
                let started = DispatchTime.now().uptimeNanoseconds
                for (position, token) in tokens.enumerated() {
                    _ = try executor.forwardToken(token, position: position)
                }
                let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
                executor.recordsWalkTiming = false
                return (
                    wall,
                    Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                    (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                    executor.walkTiming
                )
            }

            var bestBlock: (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41BlockTiming)?
            var bestWalk: (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41WalkTiming)?
            for round in 0..<2 {
                // Rotated: whoever runs first reads a colder cache, which is worth ~2x
                // on the fill and would otherwise decide the comparison.
                if round % 2 == 0 {
                    let arm = try blockArm()
                    let other = try walkArm()
                    if bestBlock == nil || arm.wall < bestBlock!.wall { bestBlock = arm }
                    if bestWalk == nil || other.wall < bestWalk!.wall { bestWalk = other }
                } else {
                    let other = try walkArm()
                    let arm = try blockArm()
                    if bestBlock == nil || arm.wall < bestBlock!.wall { bestBlock = arm }
                    if bestWalk == nil || other.wall < bestWalk!.wall { bestWalk = other }
                }
            }
            let block = try XCTUnwrap(bestBlock)
            let walk = try XCTUnwrap(bestWalk)
            XCTAssertGreaterThan(block.timing.layers, 0, "the block switch recorded nothing")
            XCTAssertGreaterThan(walk.timing.layers, 0, "the walk switch recorded nothing")

            let per = block.timing.millisecondsPerPosition()
            print(
                String(
                    format: "🔎 [dsv41 real] %d positions (arena %d slot(s)): block %.1f "
                        + "ms/pos (%.1f s), %.0f MB and %d misses (%.1f per position); "
                        + "walk %.1f ms/pos (%.1f s), %.0f MB and %d misses (%.1f per "
                        + "position) -> %.2fx",
                    length,
                    slots,
                    block.wall / positions,
                    block.wall / 1_000,
                    block.bytes,
                    block.misses,
                    Double(block.misses) / positions,
                    walk.wall / positions,
                    walk.wall / 1_000,
                    walk.bytes,
                    walk.misses,
                    Double(walk.misses) / positions,
                    walk.wall / max(block.wall, 1)
                )
            )
            print("🔎 [dsv41 real] " + block.timing.describe())
            print(
                "🔎 [dsv41 real] block buckets per position: wall "
                    + String(format: "%.1f", per.wall)
                    + ", embedding " + String(format: "%.1f", per.embedding)
                    + ", dense encode " + String(format: "%.1f", per.denseEncode)
                    + ", front wait " + String(format: "%.1f", per.frontWait)
                    + ", routing " + String(format: "%.1f", per.routing)
                    + ", expert fill " + String(format: "%.1f", per.expertFill)
                    + ", mixture encode " + String(format: "%.1f", per.mixtureEncode)
                    + ", device " + String(format: "%.1f", per.denseDevice + per.mixtureDevice)
                    + " ms (drained front "
                    + String(format: "%.1f",
                        Double(block.timing.frontDrainNanoseconds)
                            / Double(max(1, block.timing.positionCount)) / 1_000_000)
                    + ", mixture "
                    + String(format: "%.1f",
                        Double(block.timing.mixtureDrainNanoseconds)
                            / Double(max(1, block.timing.positionCount)) / 1_000_000)
                    + ", late start front "
                    + String(format: "%.1f",
                        Double(block.timing.frontLateStartNanoseconds)
                            / Double(max(1, block.timing.positionCount)) / 1_000_000)
                    + ", mixture "
                    + String(format: "%.1f",
                        Double(block.timing.mixtureLateStartNanoseconds)
                            / Double(max(1, block.timing.positionCount)) / 1_000_000)
                    + ")"
            )
            print("🔎 [dsv41 real] " + walk.timing.describe())
        }

        // The dense-half ranking at the widths the checkpoint ships, which `#46`'s
        // tables never had: every stage table in the journal was taken on a synthetic
        // fixture, so a real weight's per-stage cost has never been attributed.
        executor.reset()
        runtime.arena.reset()
        executor.resetStageTiming()
        executor.recordsStageTiming = true
        executor.recordsBlockTiming = true
        _ = try executor.prefillChunk(Array(stream.prefix(8)))
        executor.recordsStageTiming = false
        executor.recordsBlockTiming = false
        let stages = executor.stageTiming
        XCTAssertFalse(stages.isEmpty, "no stage reported a device window")
        let total = Double(stages.values.reduce(0, +))
        print(
            "🔎 [dsv41 real] one real block of 8 positions, dense stages "
                + "(ms for the block, share of the dense half):"
        )
        for (stage, nanoseconds) in stages.sorted(by: { $0.value > $1.value }) {
            print(
                "🔎 [dsv41 real]   "
                    + String(
                        format: "%-42@ %8.3f ms  %5.1f%%",
                        stage as NSString,
                        Double(nanoseconds) / 1_000_000,
                        Double(nanoseconds) / max(total, 1) * 100
                    )
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// The same attribution on the prompt that actually loses: the largest picture.
    ///
    /// `#46` measured the batched form at 0.74x the walk on the composer's own 1,005
    /// position picture prompt, against 1.26x on a bare text span of the same length,
    /// and attributed the loss to bulk eviction. The bytes do not support that on their
    /// own — the two layouts read the same 218 GB on that prompt — so this runs the two
    /// arms with the instrument above attached to each and prints the split, plus the
    /// per-position cost of both layouts on a prompt whose positions are mostly IMAGE
    /// rows. Both arms are 2x slower per position on this prompt than on text of the
    /// same length (`#46`: 353 and 266 ms against 122 and 150), which is a fact about
    /// the graph rather than the layout and is what this case exists to attribute.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_block_span_picture`): the tower encode plus two
    /// rotated rounds of a thousand-position prefill is ~25 minutes at `-O`.
    func testWhereAPicturesPrefillGoesOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_block_span_picture"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in picture attribution; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let engine = try DynaMoeEngine(filePath: snapshot.path)

        // The composer's own path, because the regime is the picture's CONTENT and not
        // its length. `#46`'s bare-span arm (a synthetic patch grid, 5,533 distinct
        // pairs) measured 1.26x, and a synthetic grid reproduces that arm rather than
        // this one: the same grid through this instrument reads 5,552 pairs and 1.67x.
        // The composer's prompt touches 12,228, i.e. 3.25x the arena, which is the case
        // that loses. So the pixels are a real gradient and the rows are the tower's.
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let rendered = dialect.render(
            turns: [.init(
                isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?"
            )],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: rendered)
        let side = 4_096
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            let rowBase = y * side * 3
            for x in 0..<side {
                let offset = rowBase + x * 3
                rgb[offset] = UInt8(x & 0xFF)
                rgb[offset + 1] = UInt8(y & 0xFF)
                rgb[offset + 2] = 96
            }
        }
        let image = DeepSeekV41ImageInput(
            rgb: rgb, width: side, height: side, sha256: "largest-image"
        )
        let price = try DeepSeekV41TurnImages.price(
            tokenIds: promptIds, images: [image], spec: spec
        )
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: price.promptTokens + 1,
            allowEngramSkip: false
        )
        let tower = try XCTUnwrap(runtime.visionForward())
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: promptIds,
            images: [image],
            spec: spec,
            rowWidth: config.hiddenSize
        ) { input in
            let fitted = DeepSeekV41VisionLayout.fit(
                width: input.width, height: input.height, spec: spec
            )
            let rows = try tower.imageSpan(
                rgb: input.rgb, width: input.width, height: input.height
            )
            return DeepSeekV41ImagePrompt.ImageSpan(
                rows: rows, columns: fitted.columns, gridRows: fitted.rows
            )
        }
        let tokens = bound.tokenIds.map { Int($0) }
        let positions = Double(max(1, tokens.count))
        let slots = runtime.arena.capacity
        XCTAssertEqual(tokens.count, price.promptTokens, "the price must equal the bind")
        print(
            "🔎 [dsv41 real] picture attribution: \(bound.rows.count) image rows, prompt "
                + "\(tokens.count) positions, arena \(slots) slot(s)"
        )

        let executor = runtime.forward()
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(tokens.prefix(8)))

        func blockArm() throws -> (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41BlockTiming) {
            executor.reset()
            runtime.arena.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try executor.prefillChunk(tokens, embeddings: bound.rows)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsBlockTiming = false
            return (
                wall,
                Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                executor.blockTiming
            )
        }

        func walkArm() throws -> (wall: Double, bytes: Double, misses: Int, timing: DeepSeekV41WalkTiming) {
            executor.reset()
            runtime.arena.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            for (position, token) in tokens.enumerated() {
                if let row = bound.rows[position] {
                    _ = try executor.forwardEmbedding(row, token: token, position: position)
                } else {
                    _ = try executor.forwardToken(token, position: position)
                }
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsWalkTiming = false
            return (
                wall,
                Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                executor.walkTiming
            )
        }

        // ONE round, unlike the text case: each arm here reads ~218 GB, which is larger
        // than this machine's RAM, so the rotation exists to protect an arm from a page
        // cache the other arm warmed and there is no such cache to protect against.
        let block = try blockArm()
        let walk = try walkArm()
        XCTAssertGreaterThan(block.timing.layers, 0, "the block switch recorded nothing")
        XCTAssertGreaterThan(walk.timing.layers, 0, "the walk switch recorded nothing")
        print(
            String(
                format: "🔎 [dsv41 real] picture prefill: block %.1f ms/pos (%.1f s), "
                    + "%.0f MB, %d misses; walk %.1f ms/pos (%.1f s), %.0f MB, %d misses "
                    + "-> %.2fx",
                block.wall / positions,
                block.wall / 1_000,
                block.bytes,
                block.misses,
                walk.wall / positions,
                walk.wall / 1_000,
                walk.bytes,
                walk.misses,
                walk.wall / max(block.wall, 1)
            )
        )
        print("🔎 [dsv41 real] " + block.timing.describe())
        print("🔎 [dsv41 real] " + walk.timing.describe())
        withExtendedLifetime(engine) {}
    }

    /// Whether batching actually loses on the largest picture, decided inside ONE
    /// process.
    ///
    /// `#46` measured this prompt at 0.74x batched and `#52` re-ran it at 1.57x, on the
    /// same positions, the same misses and the same bytes. Both times the arm that ran
    /// FIRST won, and the page cache cannot explain that: each arm reads ~213 GB
    /// against 128 GB of RAM, so there is no cache for a rotation to protect, and the
    /// two layouts read the same bytes on that prompt. A sign that flips between
    /// processes is not a property of the layout, and the shipped turn's decision to
    /// WALK an image prompt rests on it.
    ///
    /// So this alternates the two layouts inside one process and prints EVERY round
    /// rather than a best-of, which makes three things readable: the per-round ratio
    /// (is the sign constant?), each arm's drift across rounds (does the machine slow
    /// as it streams?), and the identity pairs (block of round 0 against block of round
    /// 2, walk of round 1 against walk of round 3), which are the harness's own noise
    /// floor under the rotation `#51` requires. Thermal state is printed beside every
    /// round, because a throttle is the obvious way for whichever arm runs first to
    /// keep winning.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_prefill_sign`; `/tmp/dynamoe_dsv41_prefill_sign_quick`
    /// halves it to two rounds, which is the cheap harness check rather than the
    /// measurement).
    func testWhetherTheBatchedPrefillDecisionHoldsInsideOneProcess() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_prefill_sign"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in prefill sign revalidation; touch \(markerPath) to run"
        )
        let rounds = FileManager.default.fileExists(
            atPath: "/tmp/dynamoe_dsv41_prefill_sign_quick"
        ) ? 2 : 4

        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let engine = try DynaMoeEngine(filePath: snapshot.path)

        // The composer's own path, on the composer's own largest picture: `#52` showed
        // that a synthetic patch grid at this same resolution is a different routing
        // regime (5,552 distinct pairs against this prompt's 12,176), so the pixels
        // have to be the ones an attachment would carry.
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let rendered = dialect.render(
            turns: [.init(
                isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?"
            )],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: rendered)
        let side = 4_096
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            let rowBase = y * side * 3
            for x in 0..<side {
                let offset = rowBase + x * 3
                rgb[offset] = UInt8(x & 0xFF)
                rgb[offset + 1] = UInt8(y & 0xFF)
                rgb[offset + 2] = 96
            }
        }
        let image = DeepSeekV41ImageInput(
            rgb: rgb, width: side, height: side, sha256: "largest-image"
        )
        let price = try DeepSeekV41TurnImages.price(
            tokenIds: promptIds, images: [image], spec: spec
        )
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: price.promptTokens + 1,
            allowEngramSkip: false
        )
        let tower = try XCTUnwrap(runtime.visionForward())
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: promptIds,
            images: [image],
            spec: spec,
            rowWidth: config.hiddenSize
        ) { input in
            let fitted = DeepSeekV41VisionLayout.fit(
                width: input.width, height: input.height, spec: spec
            )
            let rows = try tower.imageSpan(
                rgb: input.rgb, width: input.width, height: input.height
            )
            return DeepSeekV41ImagePrompt.ImageSpan(
                rows: rows, columns: fitted.columns, gridRows: fitted.rows
            )
        }
        let tokens = bound.tokenIds.map { Int($0) }
        XCTAssertEqual(tokens.count, price.promptTokens, "the price must equal the bind")
        try measurePrefillSign(
            label: "largest picture, \(bound.rows.count) image rows",
            runtime: runtime,
            tokens: tokens,
            rows: bound.rows,
            rounds: rounds
        )
        withExtendedLifetime(engine) {}
    }

    /// The two layouts over one prompt, inside ONE process, printing every round.
    ///
    /// Shared by the picture case and the text case because the finding is about the
    /// PROCESS rather than about the prompt: a process's first pass is not comparable
    /// with the passes after it, and both `#46` and `#52` measured their winning arm as
    /// the first pass of a fresh process. What a layout comparison needs is the steady
    /// state, so this alternates the order every round, prints every round rather than
    /// a best-of, and prints the identity pairs (block of round 1 against round 3, walk
    /// of round 2 against round 4) as the harness's own noise floor.
    ///
    /// Returns the per-round `walk / block` ratios, so a caller can see whether the
    /// sign held.
    @discardableResult
    private func measurePrefillSign(
        label: String,
        runtime: DeepSeekV41Runtime,
        tokens: [Int],
        rows: [Int: [Float]],
        rounds: Int
    ) throws -> [Double] {
        let positions = Double(max(1, tokens.count))
        print(
            "🔎 [dsv41 real] prefill sign, \(label): prompt \(tokens.count) positions, "
                + "\(rows.count) supplied row(s), arena \(runtime.arena.capacity) slot(s), "
                + "\(rounds) round(s)"
        )

        let executor = runtime.forward()
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(tokens.prefix(8)))

        /// One arm's whole-prompt pass, with the layout's own clock and counters. The
        /// two layouts are priced from the instrument that belongs to each, because a
        /// walk has no block buckets and a block has no per-position timeline.
        struct Arm {
            var wall: Double
            var bytes: Double
            var misses: Int
            var device: Double
            var fill: Double
            var layers: Int
            /// Slot `MTLBuffer`s this pass created, and the wall time that cost. One
            /// buffer per slot is allocated on first use and never freed, so a fresh
            /// runtime allocates 3,760 of them (67 GB) during its FIRST prefill and none
            /// afterwards. That is the state change the buckets turn on.
            var slots: Int
            var allocating: Double
            /// The layout's own whole-vs-parts line, so a round that comes out slow
            /// says WHICH bucket it went to rather than only that it was slow.
            var detail: String
        }

        func blockArm() throws -> Arm {
            executor.reset()
            runtime.arena.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let slotsBefore = runtime.expertStreamer?.timing.slotsAllocated ?? 0
            let allocatingBefore = runtime.expertStreamer?.timing.allocating ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try executor.prefillChunk(tokens, embeddings: rows)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsBlockTiming = false
            let timing = executor.blockTiming
            let per = timing.millisecondsPerPosition()
            return Arm(
                wall: wall,
                bytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                device: per.denseDevice + per.mixtureDevice,
                fill: per.expertFill,
                layers: timing.layers,
                slots: (runtime.expertStreamer?.timing.slotsAllocated ?? 0) - slotsBefore,
                allocating: (runtime.expertStreamer?.timing.allocating ?? 0) - allocatingBefore,
                detail: timing.describe()
            )
        }

        func walkArm() throws -> Arm {
            executor.reset()
            runtime.arena.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let slotsBefore = runtime.expertStreamer?.timing.slotsAllocated ?? 0
            let allocatingBefore = runtime.expertStreamer?.timing.allocating ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            for (position, token) in tokens.enumerated() {
                if let row = rows[position] {
                    _ = try executor.forwardEmbedding(row, token: token, position: position)
                } else {
                    _ = try executor.forwardToken(token, position: position)
                }
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsWalkTiming = false
            let timing = executor.walkTiming
            let per = timing.millisecondsPerPosition()
            return Arm(
                wall: wall,
                bytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                device: per.frontDevice + per.mixtureDevice,
                fill: per.expertFill,
                layers: timing.layers,
                slots: (runtime.expertStreamer?.timing.slotsAllocated ?? 0) - slotsBefore,
                allocating: (runtime.expertStreamer?.timing.allocating ?? 0) - allocatingBefore,
                detail: timing.describe()
            )
        }

        var blockRuns: [Arm] = []
        var walkRuns: [Arm] = []
        var ratios: [Double] = []
        let processStart = Date()
        for round in 0..<rounds {
            // Rotated by round, so whichever arm runs first alternates: that ordering
            // is the variable the two processes disagreed on.
            let blockFirst = round % 2 == 0
            let block: Arm
            let walk: Arm
            if blockFirst {
                block = try blockArm()
                walk = try walkArm()
            } else {
                walk = try walkArm()
                block = try blockArm()
            }
            blockRuns.append(block)
            walkRuns.append(walk)
            let ratio = walk.wall / max(block.wall, 1)
            ratios.append(ratio)
            print(
                String(
                    format: "🔎 [dsv41 real] round %d/%d (+%.0f s, %@ first, thermal %@): "
                        + "block %.1f ms/pos (%.1f s), %.0f MB, %d misses, device %.1f, "
                        + "fill %.1f; walk %.1f ms/pos (%.1f s), %.0f MB, %d misses, "
                        + "device %.1f, fill %.1f -> %.2fx",
                    round + 1,
                    rounds,
                    Date().timeIntervalSince(processStart),
                    blockFirst ? "block" : "walk",
                    String(describing: ProcessInfo.processInfo.thermalState),
                    block.wall / positions,
                    block.wall / 1_000,
                    block.bytes,
                    block.misses,
                    block.device,
                    block.fill,
                    walk.wall / positions,
                    walk.wall / 1_000,
                    walk.bytes,
                    walk.misses,
                    walk.device,
                    walk.fill,
                    ratio
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real]   slot buffers created: block %d "
                        + "(%.1f s), walk %d (%.1f s)",
                    block.slots,
                    block.allocating,
                    walk.slots,
                    walk.allocating
                )
            )
            print("🔎 [dsv41 real]   block: " + block.detail)
            print("🔎 [dsv41 real]   walk: " + walk.detail)
        }

        XCTAssertTrue(
            blockRuns.allSatisfy { $0.layers > 0 }, "the block switch recorded nothing"
        )
        XCTAssertTrue(
            walkRuns.allSatisfy { $0.layers > 0 }, "the walk switch recorded nothing"
        )

        /// Each layout's own spread across rounds, in ms per position, so the per-round
        /// ratios can be read against the noise they were taken in.
        func spread(_ runs: [Arm]) -> String {
            let perPosition = runs.map { $0.wall / positions }
            let low = perPosition.min() ?? 0
            let high = perPosition.max() ?? 0
            return String(
                format: "%.1f..%.1f ms/pos (%.2fx spread)",
                low, high, high / max(low, 1)
            )
        }

        let blockAlways = ratios.allSatisfy { $0 > 1 }
        let walkAlways = ratios.allSatisfy { $0 < 1 }
        let verdict = blockAlways
            ? "block faster in EVERY round"
            : (walkAlways ? "walk faster in EVERY round" : "the sign is NOT stable")
        print(
            "🔎 [dsv41 real] sign across \(rounds) round(s), ratios "
                + ratios.map { String(format: "%.2f", $0) }.joined(separator: ", ")
                + ": \(verdict)"
        )
        print(
            "🔎 [dsv41 real] spread: block " + spread(blockRuns)
                + ", walk " + spread(walkRuns)
        )
        if rounds >= 4 {
            // Identity pairs: same layout, same position in the round order, different
            // round. Their ratio is what this harness can call noise.
            let blockPair = blockRuns[0].wall / max(blockRuns[2].wall, 1)
            let walkPair = walkRuns[1].wall / max(walkRuns[3].wall, 1)
            print(
                String(
                    format: "🔎 [dsv41 real] identity pairs: block round 1 / round 3 "
                        + "%.2fx, walk round 2 / round 4 %.2fx",
                    blockPair,
                    walkPair
                )
            )
        }
        // Both layouts must have read the same prompt, or a ratio between them says
        // nothing about the layout. `#52` measured the two byte counts 0.3% apart.
        let blockBytes = blockRuns.map(\.bytes)
        let walkBytes = walkRuns.map(\.bytes)
        let byteGap = abs((blockBytes.min() ?? 0) - (walkBytes.min() ?? 0))
        let byteShare = byteGap / max(walkBytes.min() ?? 1, 1)
        print(
            String(
                format: "🔎 [dsv41 real] bytes per arm: block %.0f-%.0f MB, walk "
                    + "%.0f-%.0f MB, layouts %.2f%% apart",
                blockBytes.min() ?? 0,
                blockBytes.max() ?? 0,
                walkBytes.min() ?? 0,
                walkBytes.max() ?? 0,
                byteShare * 100
            )
        )
        // The decision statistic: every round AFTER the process's first pass. That
        // first pass is the one `#46` and `#52` each measured their winner on, and it
        // is fast in every bucket (host encode, disk and device alike), so it cannot be
        // part of a layout comparison.
        if ratios.count > 1 {
            let steady = Array(ratios.dropFirst())
            let mean = steady.reduce(0, +) / Double(steady.count)
            let verdict = steady.allSatisfy { $0 > 1 }
                ? "block faster in every one"
                : (steady.allSatisfy { $0 < 1 } ? "walk faster in every one" : "mixed")
            print(
                String(
                    format: "🔎 [dsv41 real] steady state (rounds 2-%d): %.2fx on "
                        + "average, %@",
                    rounds,
                    mean,
                    verdict
                )
            )
        }
        return ratios
    }

    /// The same A/B on a TEXT prompt, because the batched form is the SHIPPED text
    /// prefill and `#52` measured 1.46x at 128 positions on a round whose block arm was
    /// the first pass of a fresh process.
    ///
    /// The stream is the paragraph `#52` used, rotated 13 tokens per repetition so the
    /// sequence is not periodic: a plain repeat cycles the same routing and leaves the
    /// arena warm, which is a different regime from any real prompt.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_prefill_sign_text`; `_quick` halves the rounds).
    func testWhetherBatchingATextPrefillPaysInsideOneProcess() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_prefill_sign_text"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in text prefill sign revalidation; touch \(markerPath) to run"
        )
        let rounds = FileManager.default.fileExists(
            atPath: "/tmp/dynamoe_dsv41_prefill_sign_text_quick"
        ) ? 2 : 4

        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * 13) % stream.count]
            }
        }

        // 128 positions: the length `#52` reported 1.46x at, and the length where a
        // round trip costs the walk the most per position.
        // 128 positions is the length `#52` reported 1.46x at, and it is where a round
        // trip costs the walk the most per position. `_long` adds 1,005, where `#52`
        // reported 1.24x: the length where a batching win would matter most, and the one
        // number a shipped text prefill would notice if it came out below 1.
        let lengths = FileManager.default.fileExists(
            atPath: "/tmp/dynamoe_dsv41_prefill_sign_text_long"
        ) ? [128, 1_005] : [128]

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: (lengths.max() ?? 128) + 1,
            allowEngramSkip: false
        )
        // One runtime for BOTH lengths on purpose: the second call's round 1 is then not
        // the runtime's first pass, which is the state every pass after it lives in.
        for length in lengths {
            try measurePrefillSign(
                label: "text, \(length) positions",
                runtime: runtime,
                tokens: prompt(length),
                rows: [:],
                rounds: rounds
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// What the largest picture the composer accepts costs a person, end to end.
    ///
    /// The tower's own cost is measured and published (`#43`: 6.74 s for the largest
    /// admissible span), but a turn is not the tower. What someone waits for is the
    /// encode PLUS the prefill of the ~1,000-position span PLUS the first generated
    /// token. The prefill is where the shape of this path showed: until `#46` the turn
    /// handed a whole prompt to `prefillChunk` only when every position was a token, so
    /// an image prompt was WALKED, one full layer pass and one expert set per position.
    /// The walk is timed here as the baseline against the shipped turn, in one runtime
    /// and from a reset arena each, with each arm reporting its own bytes.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_vision_max_turn`): a thousand walked positions stream
    /// hundreds of gigabytes of routed experts, which is minutes at `-O` and far worse
    /// at `-Onone`. This is the measurement, not a regression test.
    func testWhatTheLargestImageCostsEndToEnd() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_vision_max_turn"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in largest-image turn measurement; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(
                isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?"
            )],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)
        XCTAssertEqual(
            promptIds.filter { Int($0) == spec.imageTokenID }.count, 1,
            "the prompt must carry exactly one image placeholder"
        )

        // The largest picture the composer lets through: a side above 4,096 pixels is
        // refused before the tower sees it, and every side from here up collapses onto
        // the checkpoint's own token budget, so this is the worst case an attachment
        // can reach rather than a grid chosen to look bad.
        let side = 4_096
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            let rowBase = y * side * 3
            for x in 0..<side {
                let offset = rowBase + x * 3
                rgb[offset] = UInt8(x & 0xFF)
                rgb[offset + 1] = UInt8(y & 0xFF)
                rgb[offset + 2] = 96
            }
        }
        let image = DeepSeekV41ImageInput(
            rgb: rgb, width: side, height: side, sha256: "largest-image"
        )

        // Price first, exactly as `launchTurn` does: this is the number the composer
        // chip shows, and it is the length the cache is provisioned for.
        let priceStart = CFAbsoluteTimeGetCurrent()
        let price = try DeepSeekV41TurnImages.price(
            tokenIds: promptIds, images: [image], spec: spec
        )
        let priceSeconds = CFAbsoluteTimeGetCurrent() - priceStart
        let estimate = try XCTUnwrap(price.estimates.first)
        XCTAssertEqual(estimate.patchTokens, 8_649, "the token budget caps the patch grid")
        XCTAssertEqual(estimate.spanPositions, 994, "the span is what the prompt pays for")
        print(
            "🔎 [dsv41 real] largest image: \(side)x\(side) pixels -> \(estimate.patchTokens) "
                + "patch tokens, \(estimate.columns)x\(estimate.rows) aligner grid, "
                + "\(estimate.spanPositions) span positions, prompt \(promptIds.count) -> "
                + "\(price.promptTokens) tokens, priced "
                + String(format: "%.2f s", estimate.seconds) + " in "
                + String(format: "%.4f s", priceSeconds)
        )

        let maxTokens = 2
        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: maxTokens,
            eosTokenIds: []
        )
        let contextLength = max(
            1, min(config.maxPositionEmbeddings, maxTokens + price.promptTokens)
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: contextLength,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let tower = try XCTUnwrap(
            runtime.visionForward(), "the engine could not build the tower from the shards"
        )
        XCTAssertNotNil(tower.device, "the published tower should run as one device pass")

        // The encode and the bind, timed as one stage. A fresh process pays the
        // tower's one-time weight upload inside this arm, which is what the first
        // picture of a session costs; a repeat separates the upload from the encode.
        func bindOnce() throws -> (tokenIds: [UInt32], rows: [Int: [Float]], seconds: Double) {
            let started = CFAbsoluteTimeGetCurrent()
            let bound = try DeepSeekV41TurnImages.bind(
                tokenIds: promptIds,
                images: [image],
                spec: spec,
                rowWidth: config.hiddenSize
            ) { input in
                let fitted = DeepSeekV41VisionLayout.fit(
                    width: input.width, height: input.height, spec: spec
                )
                let rows = try tower.imageSpan(
                    rgb: input.rgb, width: input.width, height: input.height
                )
                return DeepSeekV41ImagePrompt.ImageSpan(
                    rows: rows, columns: fitted.columns, gridRows: fitted.rows
                )
            }
            return (bound.tokenIds, bound.rows, CFAbsoluteTimeGetCurrent() - started)
        }
        let first = try bindOnce()
        XCTAssertEqual(
            first.tokenIds.count, price.promptTokens, "the price must equal the bind"
        )
        XCTAssertEqual(first.rows.count, estimate.spanPositions)
        XCTAssertEqual(first.rows.keys.min(), 1 + 1, "the span starts after the first text token")
        for position in [first.rows.keys.min(), first.rows.keys.max()].compactMap({ $0 }) {
            let row = try XCTUnwrap(first.rows[position])
            XCTAssertEqual(row.count, config.hiddenSize)
            XCTAssertTrue(row.allSatisfy(\.isFinite), "the tower produced a non-finite row")
        }
        let repeated = try bindOnce()
        print(
            "🔎 [dsv41 real] largest image encode: "
                + String(format: "%.2f s", first.seconds) + " cold (weight upload included), "
                + String(format: "%.2f s", repeated.seconds) + " repeated, against a "
                + String(format: "%.2f s", estimate.seconds) + " estimate"
        )

        // Arm 1, the shipped turn: an image prompt is WALKED, one full layer pass and one
        // expert set per position. This is the user-visible latency.
        runtime.forward().reset()
        runtime.arena.reset()
        let walkBefore = runtime.expertStreamer?.bytesRead ?? 0
        let turnStart = CFAbsoluteTimeGetCurrent()
        let draw = try await Task.detached { [runtime, first, settings] in
            try DeepSeekV41TurnRunner.run(
                forward: runtime.forward(),
                promptTokenIds: first.tokenIds,
                imageEmbeddings: first.rows,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                decode: { ids in (try? tokenizer.decode(ids: ids)) ?? "" }
            )
        }.value
        let turnSeconds = CFAbsoluteTimeGetCurrent() - turnStart
        let walkBytes = (runtime.expertStreamer?.bytesRead ?? 0) - walkBefore
        let walkHits = runtime.arena.hitRate
        XCTAssertEqual(draw.prefixTokensReused, 0, "an image prompt must not resume a prefix")
        XCTAssertGreaterThan(draw.tokensGenerated, 0)
        XCTAssertEqual(draw.promptTokens, first.tokenIds.count)

        // Arm 2, the batched prefill the turn does NOT use, on the same prompt and rows
        // from a reset arena: the counterfactual that decides the shipped path.
        runtime.forward().reset()
        runtime.arena.reset()
        let chunkBefore = runtime.expertStreamer?.bytesRead ?? 0
        let blocksBefore = runtime.forward().blockMixtureSubmissions
        let chunkStart = CFAbsoluteTimeGetCurrent()
        _ = try await Task.detached { [runtime, first] () -> Bool in
            _ = try runtime.forward().prefillChunk(
                first.tokenIds.map { Int($0) }, embeddings: first.rows
            )
            return true
        }.value
        let chunkSeconds = CFAbsoluteTimeGetCurrent() - chunkStart
        let chunkBytes = (runtime.expertStreamer?.bytesRead ?? 0) - chunkBefore
        let chunkHits = runtime.arena.hitRate
        let chunkBlocks = runtime.forward().blockMixtureSubmissions - blocksBefore
        let chunkFallbacks = runtime.forward().perPositionMixtureSubmissions

        let positions = Double(first.tokenIds.count)
        let walkMilliseconds = 1_000 * draw.prefillSeconds / positions
        let chunkMilliseconds = 1_000 * chunkSeconds / positions
        print(
            "🔎 [dsv41 real] largest image prefill, \(first.tokenIds.count) positions: "
                + "walk (shipped) " + String(format: "%.2f s", draw.prefillSeconds)
                + String(format: " (%.1f ms/position, %.0f MB read, %.0f%% hit), ",
                         walkMilliseconds,
                         Double(walkBytes) / 1_048_576,
                         100 * walkHits)
                + "one chunk " + String(format: "%.2f s", chunkSeconds)
                + String(format: " (%.1f ms/position, %.0f MB read, %.0f%% hit, ",
                         chunkMilliseconds,
                         Double(chunkBytes) / 1_048_576,
                         100 * chunkHits)
                + "\(chunkBlocks) block/\(chunkFallbacks) per-position) -> "
                + String(format: "%.2fx", draw.prefillSeconds / max(chunkSeconds, 0.001))
        )
        XCTAssertGreaterThan(
            chunkSeconds, draw.prefillSeconds,
            "the batched image prefill measured FASTER for the largest span, so the "
                + "shipped path's comment (and this test) says it is why the turn still "
                + "walks. Re-measure before changing either."
        )
        let firstToken = draw.timeToFirstTokenSeconds ?? 0
        print(
            "🔎 [dsv41 real] largest image turn: "
                + "encode " + String(format: "%.2f s", first.seconds)
                + " + prefill " + String(format: "%.2f s", draw.prefillSeconds)
                + " + first token " + String(format: "%.2f s", firstToken)
                + " = " + String(format: "%.2f s", first.seconds + draw.prefillSeconds + firstToken)
                + " to the first visible token (batched prefill would be "
                + String(
                    format: "%.2f s",
                    first.seconds + chunkSeconds + firstToken
                )
                + "), "
                + "\(draw.finishReason) after \(draw.tokensGenerated) token(s), "
                + "text \(draw.text.debugDescription)"
        )
        print(
            "🔎 [dsv41 real] largest image batching verdict: batching the span costs "
                + String(format: "%.2f s", chunkSeconds - draw.prefillSeconds)
                + " of the prefill, i.e. the walk is "
                + String(format: "%.2fx", chunkSeconds / max(draw.prefillSeconds, 0.001))
                + " the block form at this working set"
        )
        print("🔎 [dsv41 real] \(runtime.describe())")
        withExtendedLifetime(engine) {}
    }

    /// Where a walked image prefill spends its time, per layer.
    ///
    /// `#46` measured the largest picture at 255.98-297.98 s of prefill and named the
    /// arena's fill policy as the next lever, but that was an inference from a byte
    /// count rather than an observation: 218,886 MB at the disk's 13-14 GB/s ceiling is
    /// seventeen seconds, so the other ~250 s of a walked position is something else. A
    /// walk commits two buffers per layer and blocks on the host twice, which is why a
    /// position can cost far more than the sum of its kernels, and `recordsWalkTiming`
    /// is the instrument that separates the five candidates: host encode, host wait,
    /// the expert fill, the mixture encode, and each half's own device window.
    ///
    /// Three arms, one runtime, each from a reset arena and a cleared timing
    /// accumulator, each reporting its own bytes and resize counters:
    ///
    /// - the shipped path, the composer's own prompt and the tower's own rows;
    /// - the same arm again, so a first-arm effect cannot pass for a finding;
    /// - a same-length walk of TEXT tokens, which is what an image span is paying for
    ///   beyond its own length. An image row costs the tower's encode; once the row is
    ///   in the residual, a position is a position.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_walk_cost`): a thousand walked positions is four
    /// minutes at `-O`, and this is a measurement rather than a regression test.
    func testWhereAWalkedImagePrefillSpendsItsTime() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_walk_cost"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in walked-prefill measurement; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision)
        let tokenizer: DynaMoeTokenizer
        do {
            tokenizer = try DynaMoeTokenizer(
                tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
            )
        } catch {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let markers = DeepSeekV41VisionMarkers.familyDefault
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(
                isUser: true, text: "\(markers.imagePlaceholder)\nwhat is in this picture?"
            )],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt)

        // The same picture the end-to-end case uses, because the answer has to be about
        // the prompt a person actually waits for.
        let side = 4_096
        var rgb = [UInt8](repeating: 0, count: side * side * 3)
        for y in 0..<side {
            let rowBase = y * side * 3
            for x in 0..<side {
                let offset = rowBase + x * 3
                rgb[offset] = UInt8(x & 0xFF)
                rgb[offset + 1] = UInt8(y & 0xFF)
                rgb[offset + 2] = 96
            }
        }
        let image = DeepSeekV41ImageInput(
            rgb: rgb, width: side, height: side, sha256: "walk-cost"
        )
        let price = try DeepSeekV41TurnImages.price(
            tokenIds: promptIds, images: [image], spec: spec
        )
        let contextLength = max(
            1, min(config.maxPositionEmbeddings, price.promptTokens + 2)
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: contextLength,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let tower = try XCTUnwrap(runtime.visionForward())
        let bound = try DeepSeekV41TurnImages.bind(
            tokenIds: promptIds,
            images: [image],
            spec: spec,
            rowWidth: config.hiddenSize
        ) { input in
            let fitted = DeepSeekV41VisionLayout.fit(
                width: input.width, height: input.height, spec: spec
            )
            let rows = try tower.imageSpan(
                rgb: input.rgb, width: input.width, height: input.height
            )
            return DeepSeekV41ImagePrompt.ImageSpan(
                rows: rows, columns: fitted.columns, gridRows: fitted.rows
            )
        }
        XCTAssertEqual(bound.tokenIds.count, price.promptTokens, "the price must equal the bind")
        XCTAssertEqual(bound.rows.count, try XCTUnwrap(price.estimates.first).spanPositions)
        let tokens = bound.tokenIds.map { Int($0) }
        let rows = bound.rows

        // The fixture `#46`'s length sweep used, at the same span length, because that
        // sweep's walk measured 98% hit and 99,062 MB where the composer's walk measured
        // 95% and 218,886 MB at the same position count. If this arm reproduces the
        // sweep's number on the same runtime, the difference between the two tables is
        // the fixture's routing rather than the length, which is the whole question.
        let syntheticFit = DeepSeekV41VisionLayout.fit(
            width: 20_000, height: 20_000, spec: spec
        )
        let patchVolume = 3 * spec.patchSize * spec.patchSize
        let patchSide = syntheticFit.height / spec.patchSize
        let syntheticCells = (0..<(patchSide * patchSide)).map { index in
            (0..<patchVolume).map { dimension in
                Float((index * 3 + dimension * 7) % 17) / 17 - 0.5
            }
        }
        let syntheticSpanRows = try tower.imageSpan(
            patches: syntheticCells, rows: patchSide, columns: patchSide
        )
        let syntheticPlan = try DeepSeekV41ImagePrompt.expand(
            tokenIds: [1, 2, UInt32(spec.imageTokenID), 3],
            spans: [.init(
                rows: syntheticSpanRows,
                columns: syntheticFit.columns,
                gridRows: syntheticFit.rows
            )],
            imageTokenID: spec.imageTokenID,
            rowWidth: config.hiddenSize
        )

        // A same-length prose walk. Long enough not to be periodic, so its routing is a
        // fair comparison rather than one sentence's experts repeated.
        var prose = ""
        for index in 0..<60 {
            prose += "Line \(index) says the quick brown fox jumps over \(index + 3) lazy dogs. "
        }
        let fillerIds = try tokenizer.encode(text: prose).map { Int($0) }
        XCTAssertGreaterThan(fillerIds.count, 200, "the prose walk needs distinct tokens")

        let executor = runtime.forward()
        executor.recordsWalkTiming = true
        let capacity = runtime.arena.capacity
        print(
            "🔎 [dsv41 real] walk cost: \(tokens.count) positions, \(rows.count) image rows, "
                + "\(capacity) arena slots"
        )

        struct Arm {
            let label: String
            let seconds: Double
            let milliseconds: Double
            let megabytes: Double
            let hit: Double
            let misses: Int
            let evictions: Int
            let resident: Int
            let acquisitions: Int
            let readSeconds: Double
            let allocateSeconds: Double
            let placeSeconds: Double
            let placements: Int
            let timing: DeepSeekV41WalkTiming
        }

        func walk(_ label: String, _ tokenIds: [Int], _ embeddingRows: [Int: [Float]]) throws -> Arm {
            executor.reset()
            runtime.arena.reset()
            executor.resetWalkTiming()
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let loadedBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let readBefore = runtime.expertStreamer?.timing.reading ?? 0
            let allocateBefore = runtime.expertStreamer?.timing.allocating ?? 0
            let placeBefore = runtime.expertStreamer?.placementTiming.seconds ?? 0
            let placementsBefore = runtime.expertStreamer?.placementTiming.placements ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            for (position, token) in tokenIds.enumerated() {
                if let row = embeddingRows[position] {
                    _ = try executor.forwardEmbedding(row, token: token, position: position)
                } else {
                    _ = try executor.forwardToken(token, position: position)
                }
            }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            let positions = Double(tokenIds.count)
            return Arm(
                label: label,
                seconds: seconds,
                milliseconds: 1_000 * seconds / positions,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                hit: runtime.arena.hitRate,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - loadedBefore,
                evictions: runtime.arena.evictions,
                resident: runtime.arena.residentCount,
                acquisitions: runtime.arena.acquisitions,
                readSeconds: (runtime.expertStreamer?.timing.reading ?? 0) - readBefore,
                allocateSeconds: (runtime.expertStreamer?.timing.allocating ?? 0) - allocateBefore,
                placeSeconds: (runtime.expertStreamer?.placementTiming.seconds ?? 0) - placeBefore,
                placements: (runtime.expertStreamer?.placementTiming.placements ?? 0) - placementsBefore,
                timing: executor.walkTiming
            )
        }

        func report(_ arm: Arm) {
            let per = arm.timing.millisecondsPerPosition()
            print(
                "🔎 [dsv41 real] walk arm \(arm.label): "
                    + String(format: "%.2f s (%.1f ms/position)", arm.seconds, arm.milliseconds)
                    + String(format: ", %.0f MB read, %.1f%% hit, %d misses, %d evictions, ", arm.megabytes, 100 * arm.hit, arm.misses, arm.evictions)
                    + "\(arm.resident)/\(capacity) resident, \(arm.acquisitions) acquisitions"
            )
            print(
                "🔎 [dsv41 real]   pread " + String(format: "%.1f s at %.0f MB/s", arm.readSeconds, arm.megabytes / max(arm.readSeconds, 0.001))
                    + String(format: ", allocate %.1f s, place %.1f s over %d pairs", arm.allocateSeconds, arm.placeSeconds, arm.placements)
            )
            print(
                "🔎 [dsv41 real]   per position: front encode " + String(format: "%.1f", per.frontEncode)
                    + String(format: " + front wait %.1f + indexer wait %.1f + expert fill %.1f"
                             + " + mixture encode %.1f = host %.1f ms",
                             per.frontWait,
                             arm.timing.millisecondsPerPositionIndexerWait(),
                             per.expertFill,
                             per.mixtureEncode,
                             per.frontEncode + per.frontWait + per.expertFill + per.mixtureEncode
                                 + arm.timing.millisecondsPerPositionIndexerWait())
                    + String(format: "; device front %.1f + mixture %.1f ms", per.frontDevice, per.mixtureDevice)
            )
            print("🔎 [dsv41 real]   \(arm.timing.describe())")
        }

        let first = try walk("shipped walk (image rows)", tokens, rows)
        report(first)
        let second = try walk("shipped walk, repeated", tokens, rows)
        report(second)
        let textOnly = (0..<tokens.count).map { Int(fillerIds[$0 % fillerIds.count]) }
        let proseWalk = try walk("same length, text tokens", textOnly, [:])
        report(proseWalk)
        let syntheticWalk = try walk(
            "sweep fixture (synthetic patch rows)",
            syntheticPlan.tokenIds.map { Int($0) },
            syntheticPlan.rows
        )
        report(syntheticWalk)

        // The comparison that makes this a measurement rather than a table: a position
        // that carries an image row and a position that carries a token run the same
        // graph, so the two arms should cost the same per position once the tower is out
        // of the picture. The routing breadth is content-dependent and printed above,
        // which is why this ratio is reported with both arms' miss counts beside it.
        print(
            "🔎 [dsv41 real] walk verdict: an image span costs "
                + String(format: "%.1f ms/position", first.milliseconds)
                + " against a same-length text walk's "
                + String(format: "%.1f ms/position", proseWalk.milliseconds)
                + String(format: " (%.2fx); the repeat measured %.1f ms/position, so the arm is %.0f%% reproducible",
                         first.milliseconds / max(proseWalk.milliseconds, 0.001),
                         second.milliseconds,
                         100 * second.milliseconds / max(first.milliseconds, 0.001))
        )
        print(
            "🔎 [dsv41 real] sweep reconciliation: \(syntheticPlan.tokenIds.count) synthetic-row "
                + "positions cost " + String(format: "%.1f ms/position", syntheticWalk.milliseconds)
                + String(format: " and %.0f MB (%.1f%% hit, %d misses), against the composer's ",
                         syntheticWalk.megabytes, 100 * syntheticWalk.hit, syntheticWalk.misses)
                + String(format: "%.1f ms and %.0f MB", first.milliseconds, first.megabytes)
                + String(format: " -> about %d distinct pairs against %d slots",
                         syntheticWalk.misses + capacity, first.misses + capacity)
        )
        print("🔎 [dsv41 real] \(runtime.describe())")
        withExtendedLifetime(engine) {}
    }

    /// Where a walked position's idle time actually sits.
    ///
    /// The bucket table above says a walked position blocks in `waitUntilCompleted` for
    /// 146 ms and that the mixture half's device window had not opened for 112 of those
    /// milliseconds. That is a fact about a queue rather than a cause: the GPU can be
    /// idle because the host had not committed the work (the expert fill), because the
    /// driver had not picked a committed buffer up, or because the half before it was
    /// still running. `recordsWalkTimeline` keeps each buffer's commit time, the moment
    /// the host began waiting, and its device window, which is what separates them.
    ///
    /// Text tokens on purpose: the walk's round-trip structure is the same whatever a
    /// position carries (the length sweep above shows a text walk of the same length
    /// paying 87 ms of the same idle), and this case is about the structure, so it must
    /// not carry a tower's cost as well.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_walk_cost`).
    func testWhereAWalkedPositionsIdleTimeSits() throws {
        let markerPath = "/tmp/dynamoe_dsv41_walk_cost"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in walk timeline; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 512,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        // Two positions are recorded, and they are not the first two: a walk's first
        // positions pay pipeline warm-up and an empty arena, so a timeline taken there
        // describes a cold start rather than the steady state a thousand-position span
        // spends its time in.
        let positions = 402
        let recordedFrom = positions - executor.walkTimelineLimit
        let tokens = (0..<positions).map { 1_000 + ($0 * 37) }

        executor.reset()
        runtime.arena.reset()
        executor.recordsWalkTiming = true
        executor.resetWalkTiming()

        var wall: [Int: Double] = [:]
        for (position, token) in tokens.enumerated() {
            if position == recordedFrom {
                executor.recordsWalkTimeline = true
                executor.resetWalkTiming()
            }
            let started = DispatchTime.now().uptimeNanoseconds
            _ = try executor.forwardToken(token, position: position)
            if position >= recordedFrom {
                wall[position] = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            }
        }

        let perPosition = executor.walkTimeline.count / max(1, executor.walkTimelinePositions)
        print(
            "🔎 [dsv41 real] walk timeline: \(executor.walkTimelinePositions) position(s) x "
                + "\(perPosition) buffer(s) at positions \(recordedFrom)..."
        )
        for index in 0..<executor.walkTimelinePositions {
            let slice = Array(executor.walkTimeline[
                (index * perPosition)..<min((index + 1) * perPosition, executor.walkTimeline.count)
            ])
            guard let base = slice.first?.commitNanoseconds else { continue }
            let milliseconds = { (value: UInt64) in Double(value) / 1_000_000 }
            var device = 0.0
            var blocked = 0.0
            var pickup = 0.0
            var idle = 0.0
            var previousEnd: UInt64 = base
            for entry in slice {
                blocked += entry.waitMilliseconds
                guard entry.windowEndNanoseconds > entry.windowStartNanoseconds else { continue }
                device += entry.windowMilliseconds
                pickup += entry.pickupMilliseconds
                if entry.windowStartNanoseconds > previousEnd {
                    idle += milliseconds(entry.windowStartNanoseconds - previousEnd)
                }
                previousEnd = max(previousEnd, entry.windowEndNanoseconds)
            }
            let position = recordedFrom + index
            print(
                "🔎 [dsv41 real]   position \(position): wall "
                    + String(format: "%.1f ms", 1_000 * (wall[position] ?? 0))
                    + String(format: " = device %.1f ms (%.0f%%) in %d buffer(s)", device, 100 * device / max(1, 1_000 * (wall[position] ?? 1)), slice.count)
                    + String(format: ", host blocked %.1f ms (%.0f%%), GPU idle between buffers %.1f ms",
                             blocked,
                             100 * blocked / max(1, 1_000 * (wall[position] ?? 1)),
                             idle)
                    + String(format: ", commit-to-start %.1f ms (%.2f ms/buffer)", pickup, pickup / Double(max(1, slice.count)))
            )

            // Idle is the GPU with no window open, so a buffer's wait only counts when
            // nothing else was queued: from the later of its own commit and the previous
            // window's end. Waiting behind work already in the queue is a queue wait,
            // not idle, and a per-buffer "commit to start" figure cannot tell them
            // apart -- summing it as idle is what makes 88 submissions look like 2.9 ms
            // of start latency each.
            var boundaries: [String: (idle: Double, queue: Double, window: Double, count: Int)] = [:]
            var previous: DeepSeekV41WalkTimelineEntry?
            for entry in slice
            where entry.windowEndNanoseconds > entry.windowStartNanoseconds {
                defer { previous = entry }
                let name = walkBoundaryClass(entry.stage)
                var running = boundaries[name] ?? (0, 0, 0, 0)
                let ready = max(
                    entry.commitNanoseconds, previous?.windowEndNanoseconds ?? 0
                )
                if entry.windowStartNanoseconds > ready {
                    running.idle += milliseconds(entry.windowStartNanoseconds - ready)
                }
                if entry.windowStartNanoseconds > entry.commitNanoseconds {
                    running.queue += milliseconds(
                        entry.windowStartNanoseconds - entry.commitNanoseconds
                    )
                }
                running.window += entry.windowMilliseconds
                running.count += 1
                boundaries[name] = running
            }
            let idleTotal = boundaries.values.reduce(0) { $0 + $1.idle }
            let queueTotal = boundaries.values.reduce(0) { $0 + $1.queue }
            print(
                "🔎 [dsv41 real]     idle with nothing queued "
                    + String(format: "%.1f ms of %.1f ms wall (%.0f%%), queue wait %.1f ms",
                             idleTotal, 1_000 * (wall[position] ?? 0),
                             100 * idleTotal / max(1, 1_000 * (wall[position] ?? 1)), queueTotal)
            )
            for (name, running) in boundaries.sorted(by: { $0.key < $1.key }) {
                print(
                    "🔎 [dsv41 real]     boundary into \(name) (\(running.count)): idle "
                        + String(format: "%.1f ms (%.2f/buffer), queue wait %.1f ms "
                            + "(%.2f/buffer), window %.1f ms (%.2f/buffer)",
                            running.idle, running.idle / Double(max(1, running.count)),
                            running.queue, running.queue / Double(max(1, running.count)),
                            running.window, running.window / Double(max(1, running.count)))
                )
            }
            if index == 0 {
                for entry in slice.prefix(10) {
                    print(
                        "🔎 [dsv41 real]     \(entry.stage): committed "
                            + String(format: "%.2f ms", milliseconds(entry.commitNanoseconds - base))
                            + ", started at " + String(format: "%.2f ms", milliseconds(entry.windowStartNanoseconds - base))
                            + " for " + String(format: "%.2f ms", entry.windowMilliseconds)
                            + ", host waited " + String(format: "%.3f ms", entry.waitMilliseconds)
                    )
                }
                // Twenty consecutive boundaries, so the shape is visible rather than
                // summarised: only the first few of a position start from a cold queue.
                previous = nil
                for entry in slice.prefix(20)
                where entry.windowEndNanoseconds > entry.windowStartNanoseconds {
                    defer { previous = entry }
                    let ready = max(
                        entry.commitNanoseconds, previous?.windowEndNanoseconds ?? 0
                    )
                    let idle = entry.windowStartNanoseconds > ready
                        ? milliseconds(entry.windowStartNanoseconds - ready) : 0
                    let hostGap = previous.map {
                        entry.commitNanoseconds > $0.windowEndNanoseconds
                            ? milliseconds(entry.commitNanoseconds - $0.windowEndNanoseconds) : 0
                    } ?? 0
                    print(
                        "🔎 [dsv41 real]       \(previous?.stage ?? "start") -> "
                            + "\(entry.stage): host gap "
                            + String(format: "%.3f ms, idle %.3f ms, window %.3f ms",
                                     hostGap, idle, entry.windowMilliseconds)
                    )
                }
            }
        }
        // The same probe that reads a tenth of a millisecond in a fresh process, run
        // here instead: if a submission starts late only after 402 walked positions, the
        // cause is the machine state a long turn leaves behind, not the submission.
        if let probe = try? DeepSeekV41CommitStartupProbe.make() {
            for _ in 0..<5 { _ = probe.serialSample() }
            var pickups: [Double] = []
            for _ in 0..<21 {
                if let sample = probe.serialSample(dispatches: 20) {
                    pickups.append(sample.pickupMilliseconds)
                }
            }
            pickups.sort()
            print(
                "🔎 [dsv41 real] after the walk, a 20-launch buffer still starts in "
                    + String(format: "%.3f ms (median %.3f, worst %.3f) over %d attempt(s)",
                             pickups.first ?? 0,
                             pickups.isEmpty ? 0 : pickups[pickups.count / 2],
                             pickups.last ?? 0, pickups.count)
            )
        }
        print("🔎 [dsv41 real] \(runtime.describe())")
        withExtendedLifetime(engine) {}
    }

    /// Which half of a layer a committed buffer carries, read from the executor's own
    /// stage label. A boundary is attributed to the buffer that follows it, because the
    /// gap belongs to whatever was supposed to start next.
    private func walkBoundaryClass(_ stage: String) -> String {
        if stage.hasSuffix("attention") { return "front half" }
        if stage.hasSuffix("experts") { return "mixture half" }
        if stage.hasSuffix("indexer") { return "indexer split" }
        return "other (\(stage))"
    }

    // MARK: - Layer hand-off

    /// What one layer's dense-to-expert hand-off costs, and what the layer had to read
    /// to get there.
    ///
    /// `#48` put the walk's only genuine idle at this boundary -- 2.1-2.3 ms per layer
    /// with the queue empty -- and could not reproduce it with a probe beside the walk
    /// in any shape: launch count, grid size, bound bytes, resource count, barriers,
    /// zero-copy memory, a read of what the host just wrote, the walk's own
    /// alternation, a saturated co-tenant, or the same probe inside this process after
    /// 402 positions. A probe cannot reproduce the one thing that might matter: the
    /// LAYER. This case is 40 boundaries with each layer's own fill, routing and encode
    /// attached, over three arms that hold the submission pattern fixed and vary only
    /// what the layer did:
    ///
    ///   1. distinct tokens, the steady state a long walk spends its time in;
    ///   2. an eight-token cycle, which leaves the arena holding every pair the walk
    ///      asks for, so the fill reads nothing and the gap can be attributed to it or
    ///      cleared of it;
    ///   3. the same distinct tokens with a one-dispatch buffer committed immediately
    ///      before each expert half, which separates "this process will not start work"
    ///      from "this buffer will not start".
    ///
    /// Opt-in because it walks the real checkpoint three times.
    func testWhatEachLayersExpertHalfHandoffCosts() throws {
        let markerPath = "/tmp/dynamoe_dsv41_layer_gaps"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in layer hand-off attribution; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 512,
            allowEngramSkip: false
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()

        let positions = 202
        let distinct = (0..<positions).map { 1_000 + ($0 * 37) }
        let cycled = (0..<positions).map { 1_000 + (($0 % 8) * 37) }

        // Discarded: it is what leaves the page cache holding this arm's 555 MB, so the
        // paired arms below are not the first-read arm paying for a cold disk.
        _ = try layerGapArm(executor: executor, tokens: distinct, probe: 0, touch: false)

        // Alternating A/B on the same tokens and the same warm page cache. The read-back
        // is the only difference between the pair, so an ordering or caching effect shows
        // up as the two "off" arms disagreeing with each other.
        var paired: [(label: String, touch: Bool, samples: [DeepSeekV41LayerGapSample])] = []
        for round in ["A", "B"] {
            for touch in [false, true] {
                let label = (touch ? "read-back " : "plain      ") + round
                let arm = try layerGapArm(
                    executor: executor, tokens: distinct, probe: 0, touch: touch
                )
                paired.append((label, touch, arm.samples))
                reportLayerGaps(
                    label + (touch ? " (fill read back on the committing thread)" : " (no read-back)"),
                    samples: arm.samples
                )
                reportParts(label, arm: arm)
            }
        }
        reportPairedArms(paired)

        // The cache is reset by the arm so the walk starts from position 0 again; the
        // ARENA deliberately is not, because this arm's point is a fill that reads
        // nothing. It is the floor: the same 40 boundaries with no I/O at all.
        let warm = try layerGapArm(executor: executor, tokens: cycled, probe: 0, touch: false)
        reportLayerGaps("warm (8-token cycle, zero-byte fill)", samples: warm.samples)
        reportParts("warm (8-token cycle, zero-byte fill)", arm: warm)

        // A third fill level, between the zero-byte arm and the full one, because two
        // points can only say "the gap needs I/O"; three can say whether it scales with
        // it, which is what makes the arena's hit rate worth what it is worth.
        let partial = (0..<positions).map { 1_000 + (($0 % 96) * 37) }
        let midway = try layerGapArm(
            executor: executor, tokens: partial, probe: 0, touch: false
        )
        reportLayerGaps("partial (96-token cycle)", samples: midway.samples)
        reportParts("partial (96-token cycle)", arm: midway)

        let probed = try layerGapArm(executor: executor, tokens: distinct, probe: 1, touch: false)
        reportLayerGaps(
            "probed (one trivial buffer before the expert half)", samples: probed.samples
        )
        reportProbeArm(samples: probed.samples)
        reportParts("probed (one trivial buffer)", arm: probed)

        // `#50`: several trivial buffers committed back to back just before the expert
        // half. `#49`'s single probe said the expert half waits while a trivial buffer
        // does not; it could not say whether the delay is the expert BUFFER or the act
        // of committing anything else into an idle queue. The second and third probes
        // are that question: if they pick up in a tenth of a millisecond too, the
        // stacked commits are cheap and the expert half's delay is its own.
        let stacked = try layerGapArm(executor: executor, tokens: distinct, probe: 3, touch: false)
        reportLayerGaps(
            "stacked (three trivial buffers before the expert half)", samples: stacked.samples
        )
        reportProbeArm(samples: stacked.samples)
        reportParts("stacked (three trivial buffers)", arm: stacked)

        // One buffer holding eight trivial dispatches, more than a mixture has stages. If
        // this one waits like the expert half, the stall follows the work a buffer
        // carries; if it starts at once, the stall is the expert buffer's own resources.
        let loaded = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false, dispatches: 8
        )
        reportLayerGaps(
            "loaded (one eight-dispatch trivial buffer before the expert half)",
            samples: loaded.samples
        )
        reportProbeArm(samples: loaded.samples)
        reportParts("loaded (one eight-dispatch trivial buffer)", arm: loaded)

        // The probe that binds what the expert half binds and does nothing with it. If
        // its pickup jumps to the expert half's, the delay follows the RESOURCES a buffer
        // references rather than the work it carries.
        let arenaBound = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false, binding: 1
        )
        reportLayerGaps(
            "arena-bound (probe reads one resident expert's slot)", samples: arenaBound.samples
        )
        reportProbeArm(samples: arenaBound.samples)
        reportParts("arena-bound (probe reads one arena slot)", arm: arenaBound)

        // The control for the same probe against a large allocation the CPU never wrote:
        // the gathered key rows, which only the GPU has touched.
        let workspaceBound = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false, binding: 2
        )
        reportLayerGaps(
            "workspace-bound (probe reads the gathered rows)", samples: workspaceBound.samples
        )
        reportProbeArm(samples: workspaceBound.samples)
        reportParts("workspace-bound (probe reads the gathered rows)", arm: workspaceBound)

        // `#50`'s last standing difference: the expert half references far more distinct
        // allocations than any probe above. If that were the cost, thirty of them at the
        // ~3 us each a two-buffer probe's 0.1 ms implies would be a millisecond.
        let wide = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false, binding: 3
        )
        reportLayerGaps(
            "wide (probe binds thirty shard allocations)", samples: wide.samples
        )
        reportProbeArm(samples: wide.samples)
        reportParts("wide (probe binds thirty shard allocations)", arm: wide)

        // The surviving difference, and the one the arena slots really do have: the CPU
        // wrote them, a few hundred microseconds before the expert half reads them. A
        // probe that reads a buffer the host has just written tests that directly. The
        // small arm bounds how much of any effect is the write itself.
        let dirtySmall = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false,
            binding: 4, dirtyBytes: 4096
        )
        reportLayerGaps(
            "dirty 4 KB (probe reads a buffer the host just wrote)",
            samples: dirtySmall.samples
        )
        reportProbeArm(samples: dirtySmall.samples)
        reportParts("dirty 4 KB (probe reads a host-written buffer)", arm: dirtySmall)

        let dirtyLarge = try layerGapArm(
            executor: executor, tokens: distinct, probe: 1, touch: false,
            binding: 4, dirtyBytes: 16 << 20
        )
        reportLayerGaps(
            "dirty 16 MB (probe reads a buffer the host just wrote)",
            samples: dirtyLarge.samples
        )
        reportProbeArm(samples: dirtyLarge.samples)
        reportParts("dirty 16 MB (probe reads a host-written buffer)", arm: dirtyLarge)

        reportFillCurve([
            ("zero-byte fill", warm.samples),
            ("partial fill", midway.samples),
            ("full fill (plain A)", paired[0].samples),
            ("full fill (plain B)", paired[2].samples)
        ])

        print("🔎 [dsv41 real] \(runtime.describe())")
        withExtendedLifetime(engine) {}
    }

    /// The A/B itself: two rounds, alternating, so a caching or ordering effect appears
    /// as the two read-back-off arms disagreeing rather than as a difference between the
    /// conditions.
    private func reportPairedArms(
        _ arms: [(label: String, touch: Bool, samples: [DeepSeekV41LayerGapSample])]
    ) {
        func stats(_ samples: [DeepSeekV41LayerGapSample]) -> (mean: Double, median: Double) {
            let idle = samples.map(\.idleMilliseconds).sorted()
            guard !idle.isEmpty else { return (0, 0) }
            return (idle.reduce(0, +) / Double(idle.count), idle[idle.count / 2])
        }
        var off: [Double] = []
        var on: [Double] = []
        for arm in arms {
            let reading = stats(arm.samples)
            print(
                "🔎 [dsv41 real] pair \(arm.label): idle mean "
                    + String(format: "%.3f ms, median %.3f ms over %d layer(s)",
                             reading.mean, reading.median, arm.samples.count)
            )
            if arm.touch { on.append(reading.mean) } else { off.append(reading.mean) }
        }
        guard !off.isEmpty, !on.isEmpty else { return }
        let offMean = off.reduce(0, +) / Double(off.count)
        let onMean = on.reduce(0, +) / Double(on.count)
        print(
            "🔎 [dsv41 real] pair summary: no read-back "
                + String(format: "%.3f ms against %.3f ms with it (%.2fx)",
                         offMean, onMean, offMean / max(onMean, 0.0001))
                + String(format: "; the two off arms differ by %.3f ms",
                         off.count > 1 ? (off.max()! - off.min()!) : 0)
        )
    }

    /// Three fill levels against the same 40 boundaries: the gap is a property of the
    /// walk's I/O, so the question is whether it tracks how much was read.
    private func reportFillCurve(
        _ arms: [(String, [DeepSeekV41LayerGapSample])]
    ) {
        for (label, samples) in arms {
            guard !samples.isEmpty else { continue }
            let fill = samples.map { Double($0.fillBytes) / 1_048_576 }.reduce(0, +)
                / Double(samples.count)
            let idle = samples.map(\.idleMilliseconds)
            let mean = idle.reduce(0, +) / Double(idle.count)
            let median = idle.sorted()[idle.count / 2]
            print(
                "🔎 [dsv41 real] curve " + label + ": fill "
                    + String(format: "%.1f MB/layer -> idle mean %.3f ms, median %.3f ms",
                             fill, mean, median)
            )
        }
    }

    /// Walks `tokens` with the hand-off recorder off, then records the LAST position
    /// only, so the samples are one steady-state position's layers rather than every
    /// position's behind-the-warm-up average.
    ///
    /// `probe` is how many trivial buffers are committed before each expert half, `0`
    /// for none. The position record comes from the same walk, which is what lets the
    /// layer records be checked against the position they were taken from.
    private func layerGapArm(
        executor: DeepSeekV41MetalForward,
        tokens: [Int],
        probe: Int,
        touch: Bool,
        dispatches: Int = 1,
        binding: Int = 0,
        dirtyBytes: Int = 0
    ) throws -> LayerGapArm {
        executor.reset()
        executor.recordsLayerGaps = false
        executor.recordsMixtureStartProbe = false
        executor.touchesFilledRanges = touch
        for (position, token) in tokens.enumerated() where position < tokens.count - 1 {
            _ = try executor.forwardToken(token, position: position)
        }
        executor.resetWalkTiming()
        executor.recordsLayerGaps = true
        executor.recordsMixtureStartProbe = probe > 0
        executor.mixtureStartProbeDepth = max(1, probe)
        executor.mixtureStartProbeDispatches = max(1, dispatches)
        executor.mixtureStartProbeBinding = binding
        executor.mixtureStartProbeDirtyBytes = dirtyBytes
        _ = try executor.forwardToken(tokens[tokens.count - 1], position: tokens.count - 1)
        executor.recordsLayerGaps = false
        executor.recordsMixtureStartProbe = false
        executor.mixtureStartProbeDepth = 1
        executor.mixtureStartProbeDispatches = 1
        executor.mixtureStartProbeBinding = 0
        executor.mixtureStartProbeDirtyBytes = 0
        executor.touchesFilledRanges = false
        let samples = executor.layerGaps
        let position = executor.layerGapPositions.last
        // The instrument's own contract, so a broken recorder cannot be read as a result:
        // one record per layer, every expert half reporting a device window, and four
        // pieces that tile the position rather than overlapping it. A sum that EXCEEDS the
        // position would mean two of the pieces are the same interval.
        XCTAssertEqual(samples.count, executor.plan.count, "one record per layer")
        XCTAssertTrue(samples.allSatisfy(\.hasWindow), "every expert half must report a window")
        if let position, !samples.isEmpty {
            let summed = samples.map {
                $0.frontWindowMilliseconds + $0.hostGapMilliseconds
                    + $0.idleMilliseconds + $0.mixtureWindowMilliseconds
            }.reduce(0, +)
            XCTAssertLessThanOrEqual(
                summed, position.wallMilliseconds * 1.02,
                "the layer records must tile the position, not overlap it"
            )
            XCTAssertGreaterThanOrEqual(
                summed, position.wallMilliseconds * 0.8,
                "the layer records must account for most of the position"
            )
        } else {
            XCTFail("the recorded position must have a wall record")
        }
        return LayerGapArm(samples: samples, position: position)
    }

    /// One arm's reading: the layer records, and the position record beside them.
    private struct LayerGapArm {
        let samples: [DeepSeekV41LayerGapSample]
        let position: DeepSeekV41GapPositionSample?

        func partsMilliseconds() -> (front: Double, handoff: Double, mixture: Double) {
            (
                samples.map(\.frontWindowMilliseconds).reduce(0, +),
                samples.map { $0.hostGapMilliseconds + $0.idleMilliseconds }.reduce(0, +),
                samples.map(\.mixtureWindowMilliseconds).reduce(0, +)
            )
        }
    }

    /// The control `#49` did not have: do the layer records add up to the position?
    ///
    /// A layer's four pieces tile the position with no overlap, so their sum has to land
    /// just under the position's wall time, with the remainder being the host's work
    /// between one layer's expert half retiring and the next layer's dense half being
    /// encoded. A sum that EXCEEDS the wall time means two pieces are the same interval,
    /// and every conclusion drawn from the split is wrong.
    private func reportParts(_ label: String, arm: LayerGapArm) {
        guard let position = arm.position, !arm.samples.isEmpty else {
            print("🔎 [dsv41 real] parts (\(label)): no position record")
            return
        }
        let parts = arm.partsMilliseconds()
        let summed = parts.front + parts.handoff + parts.mixture
        print(
            "🔎 [dsv41 real] parts (\(label)): position "
                + String(format: "%.1f ms = dense halves %.1f + hand-off %.1f + expert halves %.1f",
                         position.wallMilliseconds, parts.front, parts.handoff, parts.mixture)
                + String(format: " -> %.1f ms accounted (%.1f%%) over %d layer(s), %.1f ms unaccounted",
                         summed,
                         100 * summed / max(position.wallMilliseconds, 0.0001),
                         arm.samples.count,
                         position.wallMilliseconds - summed)
        )
    }

    /// Prints one arm's boundaries, then the correlations the slice is asking about:
    /// does the gap track what the layer read, does it survive a layer that read
    /// nothing, and is it different on a layer that also runs the compressed half.
    private func reportLayerGaps(
        _ label: String,
        samples: [DeepSeekV41LayerGapSample]
    ) {
        guard !samples.isEmpty else {
            print("🔎 [dsv41 real] layer gaps (\(label)): no samples")
            return
        }
        print("🔎 [dsv41 real] layer gaps (\(label)): \(samples.count) layer(s)")
        for sample in samples.sorted(by: { $0.layer < $1.layer }) {
            print(
                "🔎 [dsv41 real]   layer " + String(format: "%2d", sample.layer)
                    + (sample.compressed ? " c" : "  ")
                    + String(format: ": idle %.3f ms, host gap %.3f ms (routing %.3f + encode %.3f)",
                             sample.idleMilliseconds,
                             sample.hostGapMilliseconds,
                             Double(sample.routingNanoseconds) / 1_000_000,
                             Double(sample.mixtureEncodeNanoseconds) / 1_000_000)
                    + String(format: ", fill %.1f MB / %d miss (%d new slot) in %.3f + %.3f ms",
                             Double(sample.fillBytes) / 1_048_576,
                             sample.fillMisses,
                             sample.fillSlotsAllocated,
                             1_000 * sample.fillReadSeconds,
                             1_000 * sample.fillPlacementSeconds)
                    + String(format: ", %d expert(s), dense %.3f ms, window %.3f ms",
                             sample.expertCount,
                             sample.frontWindowMilliseconds,
                             sample.mixtureWindowMilliseconds)
            )
        }

        let idle = samples.map(\.idleMilliseconds).sorted()
        let total = idle.reduce(0, +)
        let median = idle[idle.count / 2]
        let windowed = samples.filter(\.hasWindow).count
        print(
            "🔎 [dsv41 real]   idle total " + String(format: "%.1f ms", total)
                + String(format: " (mean %.3f, median %.3f, min %.3f, max %.3f per layer), %d/%d windowed",
                         total / Double(samples.count), median, idle.first ?? 0,
                         idle.last ?? 0, windowed, samples.count)
        )
        // The host's own gap is GPU idle too: the dense half is drained before the
        // fill starts, so the routing readback, the pread and the encode all run with
        // nothing queued. Counting only the post-commit part understates the GPU's
        // idle budget by however long the fill took.
        let hostGaps = samples.map(\.hostGapMilliseconds)
        let device = samples.map(\.mixtureWindowMilliseconds).reduce(0, +)
        print(
            "🔎 [dsv41 real]   host gap total " + String(format: "%.1f ms", hostGaps.reduce(0, +))
                + String(format: " (mean %.3f ms/layer), expert halves ran %.1f ms",
                         hostGaps.reduce(0, +) / Double(samples.count), device)
                + String(format: " -> GPU idle including the host's gap %.1f ms of %.1f ms unaccounted",
                         hostGaps.reduce(0, +) + total,
                         hostGaps.reduce(0, +) + total + device)
        )

        // The attribution the arm exists for: a layer whose experts were all resident
        // read nothing, so a gap that survives there cannot be the fill -- neither the
        // pread nor the dirty lines its worker threads leave for the GPU to pick up.
        let quiet = samples.filter { $0.fillBytes == 0 }
        let busy = samples.filter { $0.fillBytes > 0 }
        let meanQuiet = quiet.isEmpty
            ? 0 : quiet.map(\.idleMilliseconds).reduce(0, +) / Double(quiet.count)
        let meanBusy = busy.isEmpty
            ? 0 : busy.map(\.idleMilliseconds).reduce(0, +) / Double(busy.count)
        print(
            "🔎 [dsv41 real]   idle with a zero-byte fill: "
                + String(format: "%.3f ms over %d layer(s); with a non-zero fill %.3f ms over %d",
                         meanQuiet, quiet.count, meanBusy, busy.count)
        )
        // `#50`'s split of the same two groups: on a layer that read nothing the host's
        // own gap is what the pread would have been, and what is left is the device
        // failing to start the buffer it was given.
        let quietGap = quiet.isEmpty
            ? 0 : quiet.map(\.hostGapMilliseconds).reduce(0, +) / Double(quiet.count)
        let busyGap = busy.isEmpty
            ? 0 : busy.map(\.hostGapMilliseconds).reduce(0, +) / Double(busy.count)
        print(
            "🔎 [dsv41 real]   host gap with a zero-byte fill: "
                + String(format: "%.3f ms over %d layer(s); with a non-zero fill %.3f ms over %d",
                         quietGap, quiet.count, busyGap, busy.count)
        )
        let compressed = samples.filter(\.compressed)
        let windowOnly = samples.filter { !$0.compressed }
        if !compressed.isEmpty, !windowOnly.isEmpty {
            print(
                "🔎 [dsv41 real]   idle on the compressed half: "
                    + String(format: "%.3f ms over %d layer(s); window-only %.3f ms over %d",
                             compressed.map(\.idleMilliseconds).reduce(0, +)
                                / Double(compressed.count),
                             compressed.count,
                             windowOnly.map(\.idleMilliseconds).reduce(0, +)
                                / Double(windowOnly.count),
                             windowOnly.count)
            )
        }
        print(
            "🔎 [dsv41 real]   the recorded layer read "
                + String(format: "%.1f MB in %d miss(es)",
                         samples.map { Double($0.fillBytes) / 1_048_576 }.reduce(0, +),
                         samples.map(\.fillMisses).reduce(0, +))
        )
    }

    /// The probe arm's own reading, slot by slot: trivial buffers committed back to back
    /// just before the expert half. If the first one waits too, the delay belongs to the
    /// process; if every one starts at once and the expert half behind them does not,
    /// the delay is the expert buffer's, and not the act of committing into an idle
    /// queue (`#50`).
    private func reportProbeArm(samples: [DeepSeekV41LayerGapSample]) {
        let depth = samples.map(\.probes.count).max() ?? 0
        guard depth > 0 else {
            print("🔎 [dsv41 real]   probe: no probe window reported")
            return
        }
        for slot in 0..<depth {
            let probed = samples.compactMap { sample -> DeepSeekV41GapProbe? in
                guard sample.probes.count > slot else { return nil }
                let probe = sample.probes[slot]
                return probe.hasWindow ? probe : nil
            }
            guard !probed.isEmpty else {
                print("🔎 [dsv41 real]   probe \(slot): no window reported")
                continue
            }
            let windows = probed.map(\.windowMilliseconds).sorted()
            let pickups = probed.map(\.pickupMilliseconds).sorted()
            print(
                "🔎 [dsv41 real]   probe \(slot): \(probed.count)/\(samples.count) windowed, window "
                    + String(format: "median %.4f ms, max %.4f; pickup median %.4f ms, max %.4f",
                             windows[windows.count / 2], windows.last ?? 0,
                             pickups[pickups.count / 2], pickups.last ?? 0)
            )
        }
        // The discriminating pair: the probes and the expert half were committed into
        // the same queue within a few hundred microseconds of each other, with nothing
        // in flight. A probe that starts in a tenth of a millisecond while the expert
        // half waits milliseconds means the delay is the expert buffer's own content.
        let mixes = samples.map(\.pickupMilliseconds).sorted()
        print(
            "🔎 [dsv41 real]   probe against the expert half's pickup median "
                + String(format: "%.4f ms (max %.4f)", mixes[mixes.count / 2], mixes.last ?? 0)
        )
        // The first probe is the one committed immediately after the dense half drained,
        // so a probe that is still running when the expert half starts means the queue
        // never went empty; a probe that finished first and left the queue idle again is
        // the case where the gap is the buffer's.
        let first = samples.compactMap { $0.probes.first }
        let overlaps = first.filter { $0.windowEndNanoseconds > $0.windowStartNanoseconds }
            .filter { probe in
                samples.contains { sample in
                    probe.windowEndNanoseconds > sample.mixtureWindowStartNanoseconds
                        && probe.windowStartNanoseconds <= sample.mixtureWindowStartNanoseconds
                }
            }.count
        print(
            "🔎 [dsv41 real]   the first probe ran into the expert half in \(overlaps) of "
                + "\(samples.count) layer(s)"
        )
    }
}
