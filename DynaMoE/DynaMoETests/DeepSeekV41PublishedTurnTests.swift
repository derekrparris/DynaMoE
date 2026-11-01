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
}
