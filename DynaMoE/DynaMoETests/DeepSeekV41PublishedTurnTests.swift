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

    /// The host's own memory accounting, read from the kernel rather than guessed at.
    ///
    /// `#53` found every throughput bucket of the batched prefill 1.3-1.6x slower once a
    /// process had warmed, while the walked arm stayed flat, and left the cause open. The
    /// leading candidate is unified-memory pressure: the arena holds 67 GB of
    /// `storageModeShared` slot buffers that are not wired, so the kernel is free to
    /// compress or evict them, which would slow the pread that writes them and the GPU
    /// that reads them at the same time. That is the only mechanism found so far that
    /// moves the disk, the host encoder and the device windows together, and the kernel
    /// publishes the evidence directly: `phys_footprint`, the GRAPHICS ledger and how
    /// much of it is compressed, the process's remaining limit, decompressions and
    /// swapins. A compressor that fills, a graphics footprint that shrinks, a shrinking
    /// limit or swap activity would show the mechanism; a flat reading rules it out.
    ///
    /// Page counts are not used here: every field below is already in bytes.
    private func hostMemoryCounters() -> (
        footprintMB: Double,
        graphicsMB: Double,
        graphicsCompressedMB: Double,
        compressedMB: Double,
        limitRemainingMB: Double,
        purgeableMB: Double,
        decompressions: Int,
        swapins: Int,
        pressureLevel: Int,
        systemCompressorGB: Double,
        systemFreeGB: Double,
        swapUsedGB: Double
    ) {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
        )
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), rebound, &count)
            }
        }
        var pressure: Int32 = 0
        var pressureSize = MemoryLayout<Int32>.size
        let pressureStatus = sysctlbyname(
            "kern.memorystatus_vm_pressure_level", &pressure, &pressureSize, nil, 0
        )
        // The MACHINE's own view, not this process's: the decay `#53` found tracks how
        // squeezed unified memory is at the time, and a task-level reading cannot tell a
        // process that is big from a machine that has no room left.
        var statistics = vm_statistics64_data_t()
        var statisticsCount = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size
        )
        let statisticsStatus = withUnsafeMutablePointer(to: &statistics) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(statisticsCount)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &statisticsCount)
            }
        }
        let systemPageSize = Double(vm_kernel_page_size) / 1_073_741_824
        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        let swapStatus = sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0)
        let ok = status == KERN_SUCCESS
        let megabytes = { (bytes: Int64) -> Double in Double(bytes) / 1_048_576 }
        return (
            ok ? Double(info.phys_footprint) / 1_048_576 : -1,
            ok ? megabytes(info.ledger_tag_graphics_footprint) : -1,
            ok ? megabytes(info.ledger_tag_graphics_footprint_compressed) : -1,
            ok ? megabytes(Int64(bitPattern: info.compressed)) : -1,
            ok ? megabytes(Int64(bitPattern: info.limit_bytes_remaining)) : -1,
            ok ? megabytes(info.ledger_purgeable_nonvolatile) : -1,
            ok ? Int(info.decompressions) : -1,
            ok ? Int(info.ledger_swapins) : -1,
            pressureStatus == 0 ? Int(pressure) : -1,
            statisticsStatus == KERN_SUCCESS
                ? Double(statistics.compressor_page_count) * systemPageSize : -1,
            statisticsStatus == KERN_SUCCESS
                ? Double(statistics.free_count) * systemPageSize : -1,
            swapStatus == 0 ? Double(swap.xsu_used) / 1_073_741_824 : -1
        )
    }

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
    ///
    /// `wireArena` pins the arena's residency regime. Every measurement below was
    /// taken with the arena unwired (`#20`-`#56`), so the harness defaults to that
    /// and only the `#57` policy case asks for the shipping default; a run that
    /// silently wired 90 GB would move every earlier slice's numbers.
    private func makeRuntime(
        snapshot: URL,
        engine: DynaMoeEngine,
        config: DeepSeekV41Config,
        contextLength: Int,
        allowEngramSkip: Bool,
        wireArena: Bool? = false,
        arenaFraction: Double? = nil
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
            allowEngramSkip: allowEngramSkip,
            wireArena: wireArena,
            arenaFraction: arenaFraction
        ) else {
            throw XCTSkip(
                "the runtime did not build: "
                    + (DeepSeekV41Runtime.lastInitializationError ?? "no reason recorded")
            )
        }
        // `#58` made the arena's frequency rule the shipping default. Every table in
        // `#20`-`#57` was taken under LRU, so the harness pins that order unless the
        // case is measuring the rule itself; a run that silently changed replacement
        // would move all of them.
        runtime.arena.victimPolicy = .leastRecentlyUsed
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
                    ?? (reading: 0, allocating: 0, slotsAllocated: 0, wiring: 0)
                let started = DispatchTime.now().uptimeNanoseconds
                _ = try executor.forwardToken(token, position: position)
                let nanoseconds = DispatchTime.now().uptimeNanoseconds - started
                let read = (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
                let timingAfter = runtime.expertStreamer?.timing
                    ?? (reading: 0, allocating: 0, slotsAllocated: 0, wiring: 0)
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

    /// The API path's prompt, from a real request body to the checkpoint's ids.
    ///
    /// Slice #77 lets the local server accept image content parts, and the traffic it
    /// sees is a different shape from the composer's: the picture arrives base64 in an
    /// `image_url` part, the request may carry a history, and the prompt is rendered by
    /// `formatConversation` rather than `formatPrompt`. Two things have to hold, and
    /// neither is visible from the pure-logic tests: the marker must survive BOTH
    /// renderers (a marker that tokenized into pieces here would leave literal text in
    /// the prompt and bind no rows), and the number of image tokens must equal the
    /// number of attachments, because `TurnImages.price` refuses the turn when the
    /// placeholder count and the image count disagree.
    ///
    /// Like its neighbour this needs the checkpoint's tokenizer and config, not its
    /// weights, so it runs in milliseconds.
    func testAnAPIRequestWithAnImageCarriesOneImageToken() async throws {
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
        let modelConfig = try JSONDecoder().decode(
            ModelConfig.self,
            from: Data(contentsOf: snapshot.appendingPathComponent("config.json"))
        )
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let markers = DeepSeekV41VisionMarkers.familyDefault

        // A real 2x2 PNG, base64-encoded: the smallest thing ImageIO will decode, and
        // the shape an SDK actually sends.
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(
            data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        context.setFillColor(red: 0.8, green: 0.2, blue: 0.1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        let image = try XCTUnwrap(context.makeImage())
        let base64 = try XCTUnwrap(
            NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
        ).base64EncodedString()

        let body = """
        {"model":"any","messages":[
            {"role":"user","content":"hello"},
            {"role":"assistant","content":"hi"},
            {"role":"user","content":[
                {"type":"text","text":"what is in this picture?"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,\(base64)"}}
            ]}
        ]}
        """
        let request = try JSONDecoder().decode(OpenAIChatRequest.self, from: Data(body.utf8))
        let split = try OpenAIAdapter.splitMessages(request.messages)
        XCTAssertEqual(split.orderedImages.count, 1, "the request carries one picture")
        let resolved = try ServerController.attachImageMarkers(
            turns: split.turns, images: split.images, marker: markers.imagePlaceholder
        )
        XCTAssertEqual(resolved.images.count, 1)
        XCTAssertTrue(
            resolved.turns[2].text.hasPrefix(markers.imagePlaceholder),
            "the marker must head the turn that carried the picture"
        )
        XCTAssertFalse(
            resolved.turns[0].text.contains(markers.imagePlaceholder),
            "no other turn may carry a marker"
        )

        let rendered = ChatPromptFormatter.formatConversation(
            turns: resolved.turns,
            cleanSystem: "",
            thinkingEnabled: false,
            modelSupportsThinking: true,
            modelConfig: modelConfig,
            thinkingEffort: nil,
            promptDialect: dialect
        )
        let ids = try tokenizer.encode(text: rendered)
        let imageTokens = ids.filter { Int($0) == spec.imageTokenID }.count
        XCTAssertEqual(imageTokens, 1, "one attachment must render as exactly one image token")

        // And the engine's own gate agrees: one placeholder for one image, so the
        // price/render it computes accepts the turn.
        let priced = try DeepSeekV41TurnImages.price(
            tokenIds: ids,
            images: resolved.images.map {
                DeepSeekV41ImageInput(rgb: $0.rgb, width: $0.width, height: $0.height, sha256: $0.sha256)
            },
            spec: spec
        )
        XCTAssertGreaterThan(priced.promptTokens, ids.count)
        print(
            "🔎 [dsv41 real] API image request: \(ids.count) prompt tokens with "
                + "\(imageTokens) image token(s); priced \(priced.promptTokens) after expansion "
                + "(+\(priced.extraPositions)), \(priced.estimates.count) estimate(s), "
                + String(format: "%.2f s", priced.seconds) + " of tower"
        )
    }

    /// The same request, over the real server, on the real weights: does an image
    /// turn actually answer?
    ///
    /// Slice #77 let the local server accept image content parts and pinned the
    /// assembly: the marker heads its turn, one attachment renders one image token,
    /// and the prompt budget is the EXPANDED length. What it could not pin is the
    /// wiring. An app-hosted test has no model loaded, so #77's request stopped at
    /// the planner, and every case in this file drives `DeepSeekV41TurnRunner`
    /// directly rather than through the server. Everything downstream of the plan
    /// is therefore the one part of the image path no test has run: `images`
    /// reaching `CompletionEngine.launchTurn`, which prices the spans, provisions
    /// the cache for the expanded prompt, runs the tower, binds the rows and
    /// asserts the bound length equals the priced one. A mistake anywhere along it
    /// is a refusal or a 500, never a wrong token, so the whole path is either
    /// exercised here or not at all.
    ///
    /// That the picture really entered the prompt is checked as a DELTA. The server
    /// renders its own prompt (system date included), so a test that re-derived it
    /// would be pinning a second copy of the template; instead the same text is
    /// sent twice, plain and with one attachment, and the counted prompt tokens
    /// must differ by exactly the marker's own cost plus the span the resize plan
    /// predicts. Both numbers come from outside the server: the marker and the
    /// newline after it are two tokens at the checkpoint's tokenizer (`#35`
    /// measured 10 -> 12 on the composer's prompt), and the span is
    /// `DeepSeekV41VisionPricing.estimate` (the same function the engine prices
    /// with) on the pixels `DeepSeekV41ImagePixels.decode` produces.
    func testAnAPIRequestWithAnImageAnswersOverTheLocalServer() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let spec = try XCTUnwrap(config.vision, "the checkpoint must publish a vision config")
        guard let tokenizer = try? DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        ) else {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
        let modelConfig = try XCTUnwrap(
            ModelConfig.load(from: snapshot),
            "the checkpoint's config.json did not decode as a ModelConfig"
        )
        // The server's own gate, asserted first: when it is false the request below
        // is refused before a pixel is read, which is a different failure from a
        // broken image path and must not be read as one.
        XCTAssertTrue(
            ModelConfig.supportsVision(modelPath: snapshot.path),
            "the loaded checkpoint must advertise the tower the request will use"
        )

        // The load the app performs, in the app's order: index the checkpoint, split
        // any shard past the device's buffer limit, map every segment zero-copy, then
        // publish it to the shared engine the server's turns run on. The engine owns
        // the mappings every buffer points into, so it has to outlive the whole case.
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
                      bytesNoCopy: pointer,
                      length: length,
                      options: .storageModeShared,
                      deallocator: nil
                  )
            else { continue }
            buffers[shard.index] = buffer
        }
        print(
            "🔎 [dsv41 real] live server: \(indexed.tensors.count) tensors in "
                + "\(indexed.shards.count) shard(s), mapped \(buffers.count)/"
                + "\(summary.shards.count) segment(s), \(indexed.sizeGb) GB"
        )
        CompletionEngine.shared.configure(
            summary: summary,
            modelConfig: modelConfig,
            tokenizer: tokenizer,
            shardBuffers: buffers,
            activeLoadedModelPath: snapshot.path
        )
        defer {
            CompletionEngine.shared.configure(
                summary: nil,
                modelConfig: nil,
                tokenizer: nil,
                shardBuffers: [:],
                activeLoadedModelPath: nil
            )
        }

        // The smallest thing ImageIO will decode, which the checkpoint's own
        // `min_pixels` floor turns into its smallest admissible encode. That floor,
        // not this 2x2, is what the turn pays for: 1,521 patch tokens on a 13x13
        // aligner grid, i.e. 184 prompt positions.
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        let context = try XCTUnwrap(CGContext(
            data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
            space: space, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ))
        context.setFillColor(red: 0.9, green: 0.15, blue: 0.1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        let cgImage = try XCTUnwrap(context.makeImage())
        let png = try XCTUnwrap(
            NSBitmapImageRep(cgImage: cgImage).representation(using: .png, properties: [:])
        )
        let pixels = try DeepSeekV41ImagePixels.decode(png)
        let estimate = try XCTUnwrap(
            DeepSeekV41VisionPricing.estimate(
                width: pixels.width, height: pixels.height, spec: spec
            ),
            "the resize plan must price an image the decoder accepted"
        )
        XCTAssertGreaterThan(
            estimate.spanPositions, 1,
            "the span has to exceed the placeholder it replaces or there is nothing to check"
        )

        let port = 8791
        let question = "what is in this picture?"
        try await withLocalServer(on: port) { store in
            let plain = try await postChat(port: port, text: question, imageBase64: nil)
            let pictured = try await postChat(
                port: port, text: question, imageBase64: png.base64EncodedString()
            )
            print(
                "🔎 [dsv41 real] live image turn: plain \(plain.promptTokens) prompt tokens / "
                    + "\(plain.completionTokens) generated, pictured \(pictured.promptTokens) / "
                    + "\(pictured.completionTokens), span \(estimate.spanPositions) position(s) "
                    + "over \(estimate.patchTokens) patch token(s), "
                    + String(format: "%.2f s", estimate.seconds) + " of tower"
            )
            print(
                "🔎 [dsv41 real] live image answer: \(pictured.content.debugDescription) "
                    + "(finish \(pictured.finishReason ?? "nil"))"
            )
            // The picture must cost the prompt the marker plus its span, and nothing
            // else: a prompt that never saw the marker would count the same as the
            // plain one, and a span that bound nothing would cost one token.
            XCTAssertEqual(
                pictured.promptTokens - plain.promptTokens,
                2 + estimate.spanPositions - 1,
                "the attachment must add its marker plus its span; plain "
                    + "\(plain.promptTokens), pictured \(pictured.promptTokens), span "
                    + "\(estimate.spanPositions)"
            )
            XCTAssertGreaterThan(
                pictured.completionTokens, 0, "the image turn generated no token at all"
            )
            XCTAssertFalse(
                pictured.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "the image turn returned no text (finish \(pictured.finishReason ?? "nil"))"
            )

            // The request-log row is the other half of the contract: the length the
            // client was billed is the length the server recorded, and a picture is
            // NAMED in the captured body rather than carried in it.
            let rows = await store.fetch(limit: 20)
            let logged = rows.filter { $0.path == "/v1/chat/completions" }
            XCTAssertEqual(logged.count, 2, "both turns must be logged")
            XCTAssertTrue(
                logged.contains {
                    $0.inputTokens == pictured.promptTokens
                        && ($0.requestBody ?? "").contains("\"image_parts\":1")
                },
                "the image turn's row must carry the expanded prompt length and name its "
                    + "picture: \(logged.map { "\($0.inputTokens ?? -1) \($0.requestBody ?? "")" })"
            )
        }
        withExtendedLifetime(engine) {}
    }

    /// Runs `body` with the local server listening on `port`, then puts everything
    /// back.
    ///
    /// A private port rather than the app's 8787: a developer's own DynaMoE may be
    /// running, and a bind failure would otherwise read as a connection error. Every
    /// default the server consults is saved and restored, and the request log is
    /// pointed at an in-memory store so an app-hosted run never appends to the
    /// developer's real log.
    private func withLocalServer(
        on port: Int,
        _ body: (ServerRequestLogStore) async throws -> Void
    ) async throws {
        let defaults = UserDefaults.standard
        let savedEnabled = defaults.object(forKey: ServerController.enabledKey)
        let savedPort = defaults.object(forKey: ServerController.portKey)
        let savedKey = defaults.object(forKey: ServerController.apiKeyKey)
        let savedCapture = defaults.object(forKey: ServerRequestLogRecorder.captureBodiesKey)
        let store = ServerRequestLogStore.inMemory()

        func restore() {
            if let savedEnabled {
                defaults.set(savedEnabled, forKey: ServerController.enabledKey)
            } else {
                defaults.removeObject(forKey: ServerController.enabledKey)
            }
            if let savedPort {
                defaults.set(savedPort, forKey: ServerController.portKey)
            } else {
                defaults.removeObject(forKey: ServerController.portKey)
            }
            if let savedKey {
                defaults.set(savedKey, forKey: ServerController.apiKeyKey)
            } else {
                defaults.removeObject(forKey: ServerController.apiKeyKey)
            }
            if let savedCapture {
                defaults.set(savedCapture, forKey: ServerRequestLogRecorder.captureBodiesKey)
            } else {
                defaults.removeObject(forKey: ServerRequestLogRecorder.captureBodiesKey)
            }
            ServerRequestLogRecorder.shared.useOverrideStore(nil)
        }

        defaults.set(true, forKey: ServerController.enabledKey)
        defaults.set(port, forKey: ServerController.portKey)
        // The case sends unauthenticated requests, so a key saved in the app's
        // settings would turn both of them into a 401.
        defaults.removeObject(forKey: ServerController.apiKeyKey)
        // Metadata names the parts; only a captured body shows the `image_parts`
        // count the case asserts on.
        defaults.set(true, forKey: ServerRequestLogRecorder.captureBodiesKey)
        ServerRequestLogRecorder.shared.useOverrideStore(store)

        // A server another case left running is on ITS port, and `start()` is a
        // no-op while one is up.
        if ServerController.shared.isRunning {
            await ServerController.shared.stop()
        }
        ServerController.shared.start()
        do {
            let deadline = Date().addingTimeInterval(20)
            var listening = false
            while Date() < deadline {
                if let (_, response) = try? await URLSession.shared.data(
                    from: URL(string: "http://127.0.0.1:\(port)/health")!
                ), (response as? HTTPURLResponse)?.statusCode == 200 {
                    listening = true
                    break
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            try XCTSkipUnless(
                listening, "the local server never answered /health on port \(port)"
            )
            try await body(store)
        } catch {
            await ServerController.shared.stop()
            restore()
            throw error
        }
        await ServerController.shared.stop()
        restore()
    }

    /// What one non-streaming chat completion reported back.
    private struct ChatTurn {
        var content: String
        var promptTokens: Int
        var completionTokens: Int
        var finishReason: String?
    }

    /// Posts one OpenAI chat completion to the local server, text or text+image.
    private func postChat(
        port: Int, text: String, imageBase64: String?
    ) async throws -> ChatTurn {
        var parts = ["{\"type\":\"text\",\"text\":\"\(text)\"}"]
        if let imageBase64, !imageBase64.isEmpty {
            parts.append(
                "{\"type\":\"image_url\",\"image_url\":"
                    + "{\"url\":\"data:image/png;base64,\(imageBase64)\"}}"
            )
        }
        // Long enough to leave the app's default thinking-on prompt's reasoning
        // channel and write a sentence about the picture. Sampling is the server's
        // default (temperature 0, top-k 50), so the text is near-greedy and a run
        // that changes it says more about the graph than about the sampler.
        let body = """
        {"model":"any","max_tokens":24,"messages":[
            {"role":"user","content":[\(parts.joined(separator: ","))]}
        ]}
        """
        var request = URLRequest(
            url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        )
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data(body.utf8)
        // A turn is a full prefill plus its decode, and the expanded prompt is the
        // smallest span the checkpoint accepts, still ~200 positions.
        request.timeoutInterval = 900
        let (data, response) = try await URLSession.shared.data(for: request)
        let http = try XCTUnwrap(response as? HTTPURLResponse)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any],
            "the server's reply was not a JSON object: "
                + String(decoding: data, as: UTF8.self)
        )
        guard http.statusCode == 200 else {
            XCTFail("\(http.statusCode) from the local server: \(json)")
            return ChatTurn(content: "", promptTokens: -1, completionTokens: -1, finishReason: nil)
        }
        let choice = try XCTUnwrap((json["choices"] as? [[String: Any]])?.first)
        let message = try XCTUnwrap(choice["message"] as? [String: Any])
        let usage = try XCTUnwrap(json["usage"] as? [String: Any])
        return ChatTurn(
            content: message["content"] as? String ?? "",
            promptTokens: usage["prompt_tokens"] as? Int ?? -1,
            completionTokens: usage["completion_tokens"] as? Int ?? -1,
            finishReason: choice["finish_reason"] as? String
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
                    + ", indexer wait " + String(format: "%.1f", per.indexerWait)
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

    /// What turns a batched prefill's state, from the kernel's own numbers.
    ///
    /// `#53` established the effect and left the cause open: every throughput bucket of
    /// the batched arm is 1.3-1.6x slower once a process has warmed (text at 128
    /// positions went dense encode 31.5 -> 42.0 ms/position, expert fill 66.2 -> 102.9 at
    /// 10.4 -> 6.7 GB/s, device 54.0 -> 69.0), on byte and miss counts identical to the
    /// digit, while the walked arm holds device 62.3-63.9 flat. The app lives in the slow
    /// state for every prefill after its first, so a cause here is worth 30-60% on every
    /// batched prefill.
    ///
    /// Two candidate mechanisms are decidable from `task_vm_info` alone. Unified-memory
    /// pressure would show in the GRAPHICS ledger being compressed, in a shrinking limit,
    /// in decompressions or in swapins; a Metal heap that degrades under repeated 67 GB
    /// allocation would show as a steady `currentAllocatedSize` against a falling fill
    /// rate with every host counter flat. The passes also settle the SHAPE of the
    /// transition: eight in a row say whether it is a step at the arena's first
    /// commitment or a slide, and a short prompt at the end says whether it is about the
    /// arena's residency or something process-wide.
    ///
    /// Note what the per-block split costs: `onBlockLayerFinished` ends with a
    /// `settlePending()` of its own, so installing it serialises the block path a layer at
    /// a time. The wall time does not move (232.4 against 234.1 ms/position on the same
    /// arm, measured both ways), but the recorded front wait collapses from 117.9 to 27.1
    /// ms/position because there is never queued work to wait on, and the buckets stop
    /// tiling the wall. Its ABSOLUTE numbers are therefore not comparable with a run
    /// without the hook; its ratios across passes are, because every pass carries it.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_state_probe`), about seven minutes at `-O`.
    @MainActor
    func testWhatTurnsABatchedPrefillsStateInsideOneProcess() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_state_probe"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in prefill state probe; touch \(markerPath) to run"
        )

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
        // Rotated once per repetition so the sequence is not periodic: a plain repeat
        // cycles the same routing and leaves the arena warm (`#52`). Two rotations give
        // two prompts of the same shape that route differently, which is what separates
        // "this pass allocated the arena" from "this pass is not the first one".
        func prompt(_ length: Int, rotation: Int = 13) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())

        /// One phase: build nothing, just measure the passes this runtime is given, with
        /// the layout's buckets, the streamer's counters and the kernel's memory
        /// accounting on one line per pass.
        func probe(
            label: String,
            runtime: DeepSeekV41Runtime,
            rounds: Int,
            shortPrompt: Bool
        ) throws {
            let executor = runtime.forward()
            let before = hostMemoryCounters()
            print(
                String(
                    format: "🔎 [dsv41 real] state probe, %@: arena %d slot(s) of "
                        + "17.93 MB (%.0f GB), metal %.1f GB allocated against a %.1f GB "
                        + "working set; before any pass footprint %.1f GB, graphics %.1f "
                        + "GB (compressed %.2f), compressed %.1f GB, pressure %d",
                    label,
                    runtime.arena.capacity,
                    Double(runtime.arena.capacity) * 17.93 / 1_024,
                    Double(device.currentAllocatedSize) / 1_073_741_824,
                    Double(device.recommendedMaxWorkingSetSize) / 1_073_741_824,
                    before.footprintMB / 1_024,
                    before.graphicsMB / 1_024,
                    before.graphicsCompressedMB / 1_024,
                    before.compressedMB / 1_024,
                    before.pressureLevel
                )
            )

            func pass(_ tokens: [Int], label: String, index: Int) throws {
                executor.reset()
                runtime.arena.reset()
                executor.resetBlockTiming()
                executor.recordsBlockTiming = true
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
                let slotsBefore = runtime.expertStreamer?.timing.slotsAllocated ?? 0
                // The hook fires once per (block, layer), so the first forty deltas are
                // the FIRST block's layers: the only ones that can allocate slots. A first
                // pass whose advantage sits there is about allocation; one whose advantage
                // is spread over all 126 blocks is about the process's state.
                var firstBlockMilliseconds = 0.0
                var laterBlocksMilliseconds = 0.0
                var layerCallbacks = 0
                var lastCallback: UInt64 = 0
                executor.onBlockLayerFinished = { _, _, _ in
                    let now = DispatchTime.now().uptimeNanoseconds
                    defer {
                        lastCallback = now
                        layerCallbacks += 1
                    }
                    guard lastCallback != 0 else { return }
                    let milliseconds = Double(now - lastCallback) / 1_000_000
                    if layerCallbacks < 40 {
                        firstBlockMilliseconds += milliseconds
                    } else {
                        laterBlocksMilliseconds += milliseconds
                    }
                }
                let start = DispatchTime.now().uptimeNanoseconds
                _ = try executor.prefillChunk(tokens)
                executor.onBlockLayerFinished = nil
                let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                executor.recordsBlockTiming = false
                let timing = executor.blockTiming
                let per = timing.millisecondsPerPosition()
                let positions = Double(max(1, tokens.count))
                let fillSeconds = Double(timing.expertFillNanoseconds) / 1_000_000_000
                let rate = fillSeconds > 0
                    ? Double(timing.fillBytes) / 1_073_741_824 / fillSeconds : 0
                let memory = hostMemoryCounters()
                XCTAssertGreaterThan(timing.layers, 0, "the block switch recorded nothing")
                print(
                    String(
                        format: "🔎 [dsv41 real]   %@ pass %d: %.1f ms/pos, encode "
                            + "%.1f, front wait %.1f, fill %.1f at %.1f GB/s (%.0f MB, %d "
                            + "misses), device %.1f, slots +%d | footprint %.1f GB, "
                            + "graphics %.1f GB (compressed %.2f), compressed %.1f GB, "
                            + "decompressions %d, swapins %d, pressure %d, metal %.1f GB, "
                            + "machine compressor %.1f GB / free %.1f GB / swap %.1f GB | "
                            + "block 0 %.0f ms, later blocks %.0f ms",
                        label,
                        index,
                        wall / positions,
                        per.denseEncode,
                        per.frontWait,
                        per.expertFill,
                        rate,
                        Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                        (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                        per.denseDevice + per.mixtureDevice,
                        (runtime.expertStreamer?.timing.slotsAllocated ?? 0) - slotsBefore,
                        memory.footprintMB / 1_024,
                        memory.graphicsMB / 1_024,
                        memory.graphicsCompressedMB / 1_024,
                        memory.compressedMB / 1_024,
                        memory.decompressions,
                        memory.swapins,
                        memory.pressureLevel,
                        Double(device.currentAllocatedSize) / 1_073_741_824,
                        memory.systemCompressorGB,
                        memory.systemFreeGB,
                        memory.swapUsedGB,
                        firstBlockMilliseconds,
                        laterBlocksMilliseconds
                    )
                )
            }

            // The arm `#53` measured at 205.0 ms/position on its first pass and 296-305
            // afterwards, and the cheapest one that shows the effect. The prompts
            // alternate between two rotations so every pass carries a fresh expert set:
            // the first pass is the only one that ALLOCATES the arena's 3,760 slot
            // buffers, so if the second pass on a different prompt is as fast as the
            // first, the allocation is what the effect is about; if it is slow, the state
            // is the process's and not the buffers'.
            for index in 1...rounds {
                let rotation = index % 2 == 1 ? 13 : 7
                try pass(
                    prompt(128, rotation: rotation),
                    label: label + (rotation == 13 ? " P" : " Q"),
                    index: (index + 1) / 2
                )
            }
            if shortPrompt {
                // A prompt that touches ~2,660 `(layer, expert)` pairs instead of ~5,020.
                // If the state turns on the arena's own residency this is the control that
                // says so; if it is process-wide it stays slow.
                for index in 1...2 {
                    try pass(prompt(32), label: label + ", 32 positions", index: index)
                }
            }
        }

        // Phase A: the shipped arena, a quarter of RAM at the shipped default.
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: 129,
                allowEngramSkip: false
            )
            try probe(label: "default arena", runtime: runtime, rounds: 4, shortPrompt: false)
        }

        // A quarter-size arena was tried here as the control and WITHDRAWN, because it
        // cannot separate the hypothesis: with 1,202 slots the same 128-position prompt
        // misses 10,342 times instead of 5,021, i.e. 16.2 misses per fill against 7.8, and
        // `#51` measured that a fill's rate is a function of exactly that (one miss reads
        // at 3.8-4.7 GB/s, six at 13-43). The arm duly read 6.0-6.5 GB/s at a higher
        // compressed footprint and a pressure level of 2, so its rate says nothing about
        // compression. Separating the two needs the same misses per fill at two different
        // compression states, which no in-process knob reaches.
        withExtendedLifetime(engine) {}
    }

    // MARK: - The fill's own decay

    /// What makes a process's Nth read slower than its first.
    ///
    /// `#54` ruled out every mechanism it could name for the warmed-prefill penalty and
    /// left one fact standing: the expert fill re-reads the same 90,025 MB at 7.5 GB/s on
    /// a runtime's first pass and 5.5 on its fourth, with misses per fill unchanged. No
    /// counter the runtime owns moves with that (they are flat or falling), so the
    /// question has to be asked from OUTSIDE the fill: did the process's read slow down,
    /// or the fill's?
    ///
    /// The probe below is the control the fill never had. It reads a fixed 1.1 GB of
    /// expert ranges six ways, before the first pass and after every one:
    ///
    ///   * `host`: the checkpoint's own ranges into one host buffer. Between phases a
    ///     pass churns 90 GB of other experts, more than this machine can cache, so each
    ///     phase's reading is cold again rather than free. The probe after the rest is
    ///     the exception on purpose: it is the one that reads warm, at 67-88 GB/s, and it
    ///     is what says the file cache itself still works.
    ///   * `host repeat`: the same ranges immediately again, i.e. the WARM rate. If the
    ///     warm rate holds while the cold rate falls, the cache still works and it is the
    ///     disk's cold read that changed.
    ///   * `metal`: the same (already warm) ranges into arena-shaped
    ///     `storageModeShared` buffers, the destination the fill actually writes. A warm
    ///     arm that slows here is the destination, not the disk.
    ///   * `scratch`: a file this test writes itself, the same byte count, that nothing
    ///     else reads. A decay here is the machine's, not this checkpoint's page cache.
    ///   * `fresh host` / `fresh metal`: the same WARM bytes into destinations this
    ///     process has not written, once as plain anonymous memory and once as
    ///     arena-shaped slots. These are the arms that separate "the machine serves
    ///     fewer bytes per second" from "making a destination page available costs
    ///     more", because the source is already in the file cache and only the
    ///     destination differs. They are measured last in a phase, being the only arms
    ///     that allocate.
    ///
    /// Two more phases answer the rest of it. A twenty-second rest separates "time since
    /// the process started" from "bytes this process has read", and a SECOND runtime
    /// built in the same aged process separates the runtime's own 67 GB of slot buffers
    /// from the process itself. The second runtime is behind its own marker because two
    /// arenas at once is 134 GB on a 128 GB machine.
    ///
    /// What it found (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #55"): nothing outside the
    /// fill decays (the disk serves 13.6-15.2 GB/s at every phase, warm reads 83-96), the
    /// fill decays WITHIN its first pass from 15.3 GB/s in its first byte quartile to 4.5
    /// in its third, and the only arm that lands on the fill's own rate is a FRESH METAL
    /// destination: 0.8-3.4 GB/s for the same warm bytes into newly made slots, against
    /// 90-97 into resident ones and 21-28 into fresh host memory, and 48.2 once a rest
    /// gave the machine 3.4 GB of free memory back.
    ///
    /// Opt-in: it reads tens of gigabytes and, with the second marker, builds two
    /// runtimes.
    ///
    ///     touch /tmp/dynamoe_dsv41_fill_decay         # four passes and six probes
    ///     touch /tmp/dynamoe_dsv41_fill_decay_fresh   # plus a second runtime
    @MainActor
    func testWhatMakesAProcessesReadsSlowerThanItsFirst() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_fill_decay"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in fill decay probe; touch \(markerPath) to run"
        )
        let freshMarker = "/tmp/dynamoe_dsv41_fill_decay_fresh"

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
        // The same stream and the same 128 positions `#53` and `#54` measured, rotated
        // once per repetition so a pass does not cycle the routing and leave the arena
        // warm (`#52`).
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let positions = 128

        /// One prefill of the shipped block path, with the streamer's per-fill samples
        /// on, printed the way `#54` printed its passes so the two tables read together.
        func pass(_ runtime: DeepSeekV41Runtime, _ tokens: [Int], label: String) {
            guard let streamer = runtime.expertStreamer else {
                print("🔎 [dsv41 real]   \(label): no expert streamer, nothing to measure")
                return
            }
            let executor = runtime.forward()
            executor.reset()
            runtime.arena.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            streamer.recordsFillSamples = true
            streamer.clearFillSamples()
            let slotsBefore = streamer.timing.slotsAllocated
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                _ = try executor.prefillChunk(tokens)
            } catch {
                print("🔎 [dsv41 real]   \(label) refused: \(error)")
                executor.recordsBlockTiming = false
                streamer.recordsFillSamples = false
                return
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsBlockTiming = false
            streamer.recordsFillSamples = false
            let timing = executor.blockTiming
            let samples = streamer.fillSamples
            let per = timing.millisecondsPerPosition()
            let positionsInPass = Double(max(1, tokens.count))
            let fillSeconds = Double(timing.expertFillNanoseconds) / 1_000_000_000
            let rate = fillSeconds > 0 ? Double(timing.fillBytes) / 1_000_000_000 / fillSeconds : 0
            XCTAssertGreaterThan(timing.layers, 0, "the block switch recorded nothing")
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ pass: %.1f ms/pos, encode %.1f, front "
                        + "wait %.1f, fill %.1f at %.1f GB/s (%.0f MB, %d misses), device "
                        + "%.1f, slots +%d",
                    label,
                    wall / positionsInPass,
                    per.denseEncode,
                    per.frontWait,
                    per.expertFill,
                    rate,
                    Double(timing.fillBytes) / 1_048_576,
                    timing.fillMisses,
                    per.denseDevice + per.mixtureDevice,
                    streamer.timing.slotsAllocated - slotsBefore
                )
            )
            guard !samples.isEmpty else { return }
            // Where in the pass the slow fills are. A rate that falls as the pass
            // accumulates bytes is a within-pass effect (something about having read);
            // a pass whose quartiles are flat and whose average still fell against the
            // previous pass is a step between passes (something about being older).
            var quartileBytes = [0, 0, 0, 0]
            var quartileSeconds = [0.0, 0.0, 0.0, 0.0]
            let totalBytes = samples.reduce(0) { $0 + $1.bytes }
            var running = 0
            for sample in samples {
                let bucket = totalBytes > 0
                    ? min(3, running * 4 / totalBytes) : 0
                quartileBytes[bucket] += sample.bytes
                quartileSeconds[bucket] += sample.seconds
                running += sample.bytes
            }
            let quartiles = (0..<4).map { index in
                quartileSeconds[index] > 0
                    ? Double(quartileBytes[index]) / 1_000_000_000 / quartileSeconds[index]
                    : 0
            }
            let rates = samples.filter { $0.bytes > 0 }.map(\.gigabytesPerSecond).sorted()
            let minimum = rates.first ?? 0
            let median = rates.isEmpty ? 0 : rates[rates.count / 2]
            let maximum = rates.last ?? 0
            let memory = hostMemoryCounters()
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ fills: %d fills, per-fill %.1f/%.1f/%.1f "
                        + "GB/s (min/median/max), by cumulative bytes Q1..Q4 %.1f %.1f "
                        + "%.1f %.1f; footprint %.1f GB, graphics %.1f GB (compressed "
                        + "%.2f), compressed %.1f GB, decompressions %d, pressure %d, "
                        + "metal %.1f GB",
                    label,
                    samples.count,
                    minimum,
                    median,
                    maximum,
                    quartiles[0],
                    quartiles[1],
                    quartiles[2],
                    quartiles[3],
                    memory.footprintMB / 1_024,
                    memory.graphicsMB / 1_024,
                    memory.graphicsCompressedMB / 1_024,
                    memory.compressedMB / 1_024,
                    memory.decompressions,
                    memory.pressureLevel,
                    Double(device.currentAllocatedSize) / 1_073_741_824
                )
            )
        }

        // ── Phase A: one runtime, four passes, a rest, and a probe at every step ──
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false
            )
            let streamer = try XCTUnwrap(
                runtime.expertStreamer,
                "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
            )
            let geometry = streamer.geometry
            let slot = geometry.slotBytes
            let layerPlan = runtime.plan.map(\.index)

            // Sixty-four fixed pairs, advancing one layer at a time so the probe spans
            // the whole checkpoint the way a pass does rather than sitting in one shard.
            let probePairs: [(layer: Int, expert: Int)] = (0..<64).map { index in
                (
                    layer: layerPlan[index % layerPlan.count],
                    expert: (index * 5 + 1) % config.numRoutedExperts
                )
            }
            let probeBytes = probePairs.count * slot

            /// The probe's ranges, laid out exactly as a fill lays out a slot's: six
            /// contiguous ranges per pair, grouped per shard, destination by index.
            func batches(
                destination: (Int) -> UnsafeMutableRawPointer
            ) -> [Dsv41ProbeBatches.Batch] {
                var byShard: [UInt32: [DeepSeekV41ExpertShardReader.Range]] = [:]
                for (index, pair) in probePairs.enumerated() {
                    guard let gate = runtime.weights.tensor(
                        .expertW1, layer: pair.layer, expert: pair.expert
                    ),
                        let upT = runtime.weights.tensor(
                            .expertW3, layer: pair.layer, expert: pair.expert
                        ),
                        let downT = runtime.weights.tensor(
                            .expertW2, layer: pair.layer, expert: pair.expert
                        ) else { continue }
                    let gateScale = runtime.weights.scaleTensor(
                        .expertW1, layer: pair.layer, expert: pair.expert
                    )
                    let upScale = runtime.weights.scaleTensor(
                        .expertW3, layer: pair.layer, expert: pair.expert
                    )
                    let downScale = runtime.weights.scaleTensor(
                        .expertW2, layer: pair.layer, expert: pair.expert
                    )
                    let base = destination(index)
                    func range(
                        _ tensor: DeepSeekV41DeviceTensor?,
                        offset: Int,
                        length: Int
                    ) -> DeepSeekV41ExpertShardReader.Range? {
                        guard length > 0, let tensor else { return nil }
                        return DeepSeekV41ExpertShardReader.Range(
                            fileOffset: tensor.byteOffset,
                            destination: base.advanced(by: offset),
                            length: length
                        )
                    }
                    let pieces = [
                        range(
                            gate, offset: geometry.gateWeightOffset,
                            length: geometry.gateWeightBytes
                        ),
                        range(
                            gateScale, offset: geometry.gateScaleOffset,
                            length: geometry.gateScaleBytes
                        ),
                        range(
                            upT, offset: geometry.upWeightOffset,
                            length: geometry.upWeightBytes
                        ),
                        range(
                            upScale, offset: geometry.upScaleOffset,
                            length: geometry.upScaleBytes
                        ),
                        range(
                            downT, offset: geometry.downWeightOffset,
                            length: geometry.downWeightBytes
                        ),
                        range(
                            downScale, offset: geometry.downScaleOffset,
                            length: geometry.downScaleBytes
                        )
                    ].compactMap { $0 }
                    byShard[gate.shardIndex, default: []].append(contentsOf: pieces)
                }
                return byShard.map { Dsv41ProbeBatches.Batch(shard: $0.key, ranges: $0.value) }
            }

            // One host destination and one set of arena-shaped Metal buffers, allocated
            // once and held for the whole phase: a phase then measures the read and not
            // the destination's first touch. Reusing the Metal buffers is also the fill's
            // own later-pass situation, where every slot is already allocated.
            let hostBase = UnsafeMutableRawPointer.allocate(
                byteCount: probeBytes, alignment: 4096
            )
            defer { hostBase.deallocate() }
            var metalBuffers: [MTLBuffer] = []
            for _ in probePairs.indices {
                guard let buffer = device.makeBuffer(
                    length: slot, options: .storageModeShared
                ) else { break }
                metalBuffers.append(buffer)
            }
            let metalIsComplete = metalBuffers.count == probePairs.count
            if !metalIsComplete {
                print("🔎 [dsv41 real] only \(metalBuffers.count) probe slots were allocated")
            }

            // The independent control: a file this test writes, read through the same
            // reader class the fill uses, so the only thing it shares with the checkpoint
            // is the machine.
            let scratchPath = "/tmp/dsv41_fill_decay_scratch.bin"
            let chunk = 8 << 20
            let block = Data(repeating: 0xA5, count: chunk)
            var scratchWritten = FileManager.default.createFile(atPath: scratchPath, contents: nil)
            if scratchWritten, let handle = FileHandle(forWritingAtPath: scratchPath) {
                var written = 0
                while written < probeBytes {
                    let piece = min(chunk, probeBytes - written)
                    handle.write(block.prefix(piece))
                    written += piece
                }
                scratchWritten = (try? handle.synchronize()) != nil
                try? handle.close()
            }
            defer { try? FileManager.default.removeItem(atPath: scratchPath) }
            var scratchReader: DeepSeekV41ExpertShardReader?
            if scratchWritten {
                scratchReader = try? DeepSeekV41ExpertShardReader(
                    url: URL(fileURLWithPath: scratchPath)
                )
            }
            XCTAssertNotNil(scratchReader, "the scratch file could not be written or opened")

            let scratchBatches: [Dsv41ProbeBatches.Batch] = (0..<probePairs.count).map { index in
                Dsv41ProbeBatches.Batch(
                    shard: 0,
                    ranges: [DeepSeekV41ExpertShardReader.Range(
                        fileOffset: index * slot,
                        destination: hostBase.advanced(by: index * slot),
                        length: slot
                    )]
                )
            }

            let probeReaders = try Dsv41ProbeBatches.OpenShards(
                snapshot: snapshot, indices: batches(destination: { _ in hostBase }).map(\.shard)
            )

            func measure(
                _ label: String,
                _ list: [Dsv41ProbeBatches.Batch],
                _ readers: [UInt32: DeepSeekV41ExpertShardReader]
            ) -> Double {
                guard !list.isEmpty else { return 0 }
                let run = Dsv41ProbeBatches(
                    readers: readers, batches: list, chunkBytes: 0, sequential: false
                )
                let result = run.run()
                if let failure = run.failure {
                    print("🔎 [dsv41 real]   probe \(label) refused: \(failure)")
                    return 0
                }
                return Double(result.bytes) / 1_000_000_000 / max(result.seconds, 1e-9)
            }

            /// The same warm bytes into a destination this process has never written:
            /// 1.1 GB of fresh pages, timed from the first byte of the read. It is the
            /// one arm that separates "the machine serves fewer bytes per second" from
            /// "writing the bytes costs more", because the data is already in the file
            /// cache and only the destination differs.
            func measureFreshHost(_ label: String) -> Double {
                let fresh = UnsafeMutableRawPointer.allocate(
                    byteCount: probeBytes, alignment: 4096
                )
                defer { fresh.deallocate() }
                return measure(label, batches { fresh.advanced(by: $0 * slot) },
                               probeReaders.readers)
            }

            /// The same again into freshly made arena-shaped slots, which is the
            /// destination the fill writes. The two fresh arms are the pair that says
            /// whether the destination TYPE matters as well as its residency: they read
            /// identical bytes, at the same instant, differing only in what they write.
            func measureFreshMetal(_ label: String) -> Double {
                var fresh: [MTLBuffer] = []
                for _ in probePairs.indices {
                    guard let buffer = device.makeBuffer(
                        length: slot, options: .storageModeShared
                    ) else { break }
                    fresh.append(buffer)
                }
                guard fresh.count == probePairs.count else { return 0 }
                return measure(label, batches { fresh[$0].contents() }, probeReaders.readers)
            }

            func probe(_ label: String) {
                let host = batches { hostBase.advanced(by: $0 * slot) }
                var metal: [Dsv41ProbeBatches.Batch] = []
                if metalIsComplete {
                    metal = batches { metalBuffers[$0].contents() }
                }
                let scratchRate = scratchReader.map { opened in
                    measure("scratch", scratchBatches, [0: opened])
                } ?? 0
                let cold = measure("host", host, probeReaders.readers)
                let warm = measure("host repeat", host, probeReaders.readers)
                let metalRate = measure("metal", metal, probeReaders.readers)
                // Last, because these allocate: the arms above are meant to be read as
                // a quiet machine, and a 1.1 GB allocation at zero free memory is not.
                let freshHost = measureFreshHost("fresh host")
                let freshMetal = measureFreshMetal("fresh metal")
                let memory = hostMemoryCounters()
                print(
                    String(
                        format: "🔎 [dsv41 real]   probe %@: host cold %.1f GB/s, host "
                            + "repeat %.1f (warm), metal %.1f (warm), scratch %.1f (cold), "
                            + "fresh host %.1f (warm data), fresh metal %.1f, "
                            + "%.0f MB an arm | footprint %.1f GB, graphics %.1f GB "
                            + "(compressed %.2f), compressed %.1f GB, decompressions %d, "
                            + "swapins %d, pressure %d, metal %.1f GB, machine "
                            + "compressor %.1f GB / free %.1f GB / swap %.1f GB",
                        label,
                        cold,
                        warm,
                        metalRate,
                        scratchRate,
                        freshHost,
                        freshMetal,
                        Double(probeBytes) / 1_048_576,
                        memory.footprintMB / 1_024,
                        memory.graphicsMB / 1_024,
                        memory.graphicsCompressedMB / 1_024,
                        memory.compressedMB / 1_024,
                        memory.decompressions,
                        memory.swapins,
                        memory.pressureLevel,
                        Double(device.currentAllocatedSize) / 1_073_741_824,
                        memory.systemCompressorGB,
                        memory.systemFreeGB,
                        memory.swapUsedGB
                    )
                )
            }

            probe("before any pass")
            for index in 1...3 {
                let rotation = index % 2 == 1 ? 13 : 7
                pass(runtime, prompt(positions, rotation: rotation), label: "P\(index)")
                probe("after P\(index)")
            }
            // A rest issues no I/O and lets the compressor settle, so it separates "this
            // process has been alive for a while" from "this process has read a lot".
            print("🔎 [dsv41 real]   resting 20 s with no reads")
            try await Task.sleep(for: .seconds(20))
            probe("after the rest")
            pass(runtime, prompt(positions, rotation: 7), label: "P4 after the rest")
            probe("after P4")
            print(
                String(
                    format: "🔎 [dsv41 real] metal %d -> %.1f GB allocated once the phase ends",
                    metalBuffers.count,
                    Double(device.currentAllocatedSize) / 1_073_741_824
                )
            )
            probeReaders.close()
        }

        // ── Phase B: a SECOND runtime in the same aged process ──────────────────
        guard FileManager.default.fileExists(atPath: freshMarker) else {
            print(
                "🔎 [dsv41 real] fresh-runtime phase skipped (touch \(freshMarker) to run it)"
            )
            withExtendedLifetime(engine) {}
            return
        }
        let beforeSecond = hostMemoryCounters()
        print(
            String(
                format: "🔎 [dsv41 real] second runtime: metal %.1f GB allocated, pressure "
                    + "%d, swap %.1f GB before it is built",
                Double(device.currentAllocatedSize) / 1_073_741_824,
                beforeSecond.pressureLevel,
                beforeSecond.swapUsedGB
            )
        )
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false
            )
            pass(runtime, prompt(positions, rotation: 13), label: "fresh runtime, aged process")
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: - Slot residency

    /// Whether an arena's slots can stay out of the compressor, and what that buys.
    ///
    /// `#55` ended the search for the baked-in prefill penalty with a destination
    /// difference: the same warm bytes into freshly made `storageModeShared` slots run at
    /// 0.8-3.4 GB/s on a machine with no free memory, against 90-97 into slots the
    /// process already owns and 21-28 into fresh host memory, while the disk, the page
    /// cache and the process's own read path stay flat. Its counters name the compressor
    /// (free memory 64.8 -> 0.1 GB, machine compressor 0.1 -> 28-44 GB, decompressions
    /// 3.5M -> 27.3M), but a counter that moves with the fill is a correlation: the fill
    /// writes 90 GB a pass and the machine compresses tens of gigabytes a pass, and
    /// nothing has yet shown the second is caused by the first.
    ///
    /// A wired page is the one destination the compressor cannot take, so the probe
    /// below writes a fixed 1.1 GB of expert ranges that are already in the file cache
    /// (the source is held constant on purpose) into destinations that differ in exactly
    /// one property:
    ///
    ///   * `resident metal` — slots written at the previous probe, cold to this one.
    ///   * `fresh metal` — the `#55` arm, reproduced.
    ///   * `touched metal` — the same new slots, zeroed before the clock starts. Slow
    ///     `fresh` beside fast `touched` is page materialization, i.e. the allocator and
    ///     not the compressor.
    ///   * `wired host` / `wired metal` — `mmap` + `mlock`, once as plain memory and once
    ///     through `makeBuffer(bytesNoCopy:)`, which is the shape an arena fix would
    ///     actually take. `mlock` refusing is an answer too.
    ///   * `fresh host` / `touched host` — the same two properties without Metal, which
    ///     separates "Metal's allocation path" from "a destination page on this machine".
    ///   * `resident footprint` — warm bytes rotated through a destination four times
    ///     the probe's size, reporting every window. The fill's footprint is 67 GB; if
    ///     the rate is a function of how much destination a process has already written,
    ///     it shows here and nowhere else.
    ///
    /// Every arm reports the decompressions it caused, because that is the mechanism and
    /// the rate is only its symptom.
    ///
    /// The passes are the second half. After `#55`'s squeeze the arena's own slots are
    /// wired in rungs and a pass is timed at each, with misses per fill printed beside
    /// the rate: `#54` and `#55` both warn that a rate alone says nothing here, because
    /// a fill's rate is a function of its misses and a smaller arena changes it by
    /// itself. A wired arena cannot be compressed, so the machine has to find the room
    /// somewhere else, and where it found it is printed on the same line.
    ///
    /// Opt-in: it reads hundreds of gigabytes and wires tens of them.
    ///
    ///     touch /tmp/dynamoe_dsv41_slot_residency
    @MainActor
    func testWhetherAnArenasSlotsCanStayOutOfTheCompressor() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_slot_residency"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in slot residency probe; touch \(markerPath) to run"
        )

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
        // The same stream and the same 128 positions `#53`-`#55` measured, rotated per
        // pass so a pass does not cycle the routing and leave the arena warm (`#52`).
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        let positions = 128

        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positions + 1,
            allowEngramSkip: false
        )
        let streamer = try XCTUnwrap(
            runtime.expertStreamer,
            "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
        )
        let geometry = streamer.geometry
        let slot = geometry.slotBytes
        // A `bytesNoCopy` Metal buffer over wired memory must start on a page, so the
        // probe's slots are laid out at a padded stride. The ranges requested are
        // unchanged, which is what keeps this comparable to a fill.
        let page = max(4096, Int(vm_kernel_page_size))
        let stride = (slot + page - 1) / page * page
        let layerPlan = runtime.plan.map(\.index)

        // Sixty-four fixed pairs, advancing one layer at a time so the probe spans the
        // whole checkpoint the way a pass does rather than sitting in one shard.
        let probePairs: [(layer: Int, expert: Int)] = (0..<64).map { index in
            (
                layer: layerPlan[index % layerPlan.count],
                expert: (index * 5 + 1) % config.numRoutedExperts
            )
        }
        let probeBytes = probePairs.count * stride

        /// The probe's ranges, laid out exactly as a fill lays out a slot's: six
        /// contiguous ranges per pair, grouped per shard, the destination by index.
        func batches(_ destination: (Int) -> UnsafeMutableRawPointer) -> [Dsv41ProbeBatches.Batch] {
            var byShard: [UInt32: [DeepSeekV41ExpertShardReader.Range]] = [:]
            for (index, pair) in probePairs.enumerated() {
                guard let gate = runtime.weights.tensor(
                    .expertW1, layer: pair.layer, expert: pair.expert
                ),
                    let upT = runtime.weights.tensor(
                        .expertW3, layer: pair.layer, expert: pair.expert
                    ),
                    let downT = runtime.weights.tensor(
                        .expertW2, layer: pair.layer, expert: pair.expert
                    ) else { continue }
                let gateScale = runtime.weights.scaleTensor(
                    .expertW1, layer: pair.layer, expert: pair.expert
                )
                let upScale = runtime.weights.scaleTensor(
                    .expertW3, layer: pair.layer, expert: pair.expert
                )
                let downScale = runtime.weights.scaleTensor(
                    .expertW2, layer: pair.layer, expert: pair.expert
                )
                let base = destination(index)
                func range(
                    _ tensor: DeepSeekV41DeviceTensor?,
                    offset: Int,
                    length: Int
                ) -> DeepSeekV41ExpertShardReader.Range? {
                    guard length > 0, let tensor else { return nil }
                    return DeepSeekV41ExpertShardReader.Range(
                        fileOffset: tensor.byteOffset,
                        destination: base.advanced(by: offset),
                        length: length
                    )
                }
                let pieces = [
                    range(gate, offset: geometry.gateWeightOffset, length: geometry.gateWeightBytes),
                    range(gateScale, offset: geometry.gateScaleOffset, length: geometry.gateScaleBytes),
                    range(upT, offset: geometry.upWeightOffset, length: geometry.upWeightBytes),
                    range(upScale, offset: geometry.upScaleOffset, length: geometry.upScaleBytes),
                    range(downT, offset: geometry.downWeightOffset, length: geometry.downWeightBytes),
                    range(
                        downScale, offset: geometry.downScaleOffset,
                        length: geometry.downScaleBytes
                    )
                ].compactMap { $0 }
                byShard[gate.shardIndex, default: []].append(contentsOf: pieces)
            }
            return byShard.map { Dsv41ProbeBatches.Batch(shard: $0.key, ranges: $0.value) }
        }

        // One descriptor per shard file, opened once for the whole run so no arm pays
        // for `open` in the rate it reports.
        var probeShards: [UInt32] = []
        for pair in probePairs {
            if let gate = runtime.weights.tensor(
                .expertW1, layer: pair.layer, expert: pair.expert
            ) {
                probeShards.append(gate.shardIndex)
            }
        }
        let probeReaders = try Dsv41ProbeBatches.OpenShards(
            snapshot: snapshot, indices: probeShards
        )
        defer { probeReaders.close() }

        /// Times one arm and reports the decompressions it caused. That pairing is the
        /// point: a rate says the write was slow, the decompression count says whether
        /// the kernel had to reconstruct the destination first.
        func measure(
            _ label: String,
            _ list: [Dsv41ProbeBatches.Batch]
        ) -> (rate: Double, decompressions: Int) {
            guard !list.isEmpty else { return (0, 0) }
            let before = hostMemoryCounters().decompressions
            let run = Dsv41ProbeBatches(
                readers: probeReaders.readers, batches: list, chunkBytes: 0, sequential: false
            )
            let result = run.run()
            let after = hostMemoryCounters().decompressions
            if let failure = run.failure {
                print("🔎 [dsv41 real]   arm \(label) refused: \(failure)")
                return (0, 0)
            }
            return (
                Double(result.bytes) / 1_000_000_000 / max(result.seconds, 1e-9),
                after - before
            )
        }

        // Destinations written at every probe and held for the whole run: the fill's
        // later-pass situation, where every slot is already allocated and was written
        // before the pass that is now being measured.
        let residentHost = UnsafeMutableRawPointer.allocate(
            byteCount: probeBytes, alignment: page
        )
        defer { residentHost.deallocate() }
        var residentMetal: [MTLBuffer] = []
        for _ in probePairs.indices {
            guard let buffer = device.makeBuffer(
                length: stride, options: .storageModeShared
            ) else { break }
            residentMetal.append(buffer)
        }
        XCTAssertEqual(
            residentMetal.count, probePairs.count,
            "the resident probe slots could not be allocated"
        )

        // Four times the probe's own size, allocated before the first pass while the
        // machine still has memory, so a rate that falls with how much destination this
        // process has written is visible inside one arm. Four rather than eight because
        // a 67 GB arena already leaves this machine at 0.1 GB free, and a probe that has
        // to evict its own destination to allocate the arm after it would be measuring
        // itself.
        var footprint: [MTLBuffer] = []
        for _ in 0..<(probePairs.count * 4) {
            guard let buffer = device.makeBuffer(
                length: stride, options: .storageModeShared
            ) else { break }
            footprint.append(buffer)
        }
        print(
            String(
                format: "🔎 [dsv41 real] probe slot %.1f MB at a %.1f MB stride, %d pairs "
                    + "= %.2f GB an arm; resident footprint %.2f GB (%d slots)",
                Double(slot) / 1_048_576,
                Double(stride) / 1_048_576,
                probePairs.count,
                Double(probeBytes) / 1_073_741_824,
                Double(footprint.count * stride) / 1_073_741_824,
                footprint.count
            )
        )

        func residentMetalArm() -> (rate: Double, decompressions: Int) {
            guard residentMetal.count == probePairs.count else { return (0, 0) }
            return measure(
                "metal resident", batches { residentMetal[$0].contents() }
            )
        }

        func freshMetalArm(touched: Bool) -> (rate: Double, decompressions: Int) {
            var buffers: [MTLBuffer] = []
            for _ in probePairs.indices {
                guard let buffer = device.makeBuffer(
                    length: stride, options: .storageModeShared
                ) else { break }
                buffers.append(buffer)
            }
            guard buffers.count == probePairs.count else { return (0, 0) }
            if touched {
                // Untimed on purpose: this arm asks what the READ costs once the
                // destination's pages exist, not what making them costs.
                for buffer in buffers { memset(buffer.contents(), 0, stride) }
            }
            return measure(
                touched ? "metal touched" : "metal fresh",
                batches { buffers[$0].contents() }
            )
        }

        func freshHostArm(touched: Bool) -> (rate: Double, decompressions: Int) {
            let base = UnsafeMutableRawPointer.allocate(
                byteCount: probeBytes, alignment: page
            )
            defer { base.deallocate() }
            if touched { memset(base, 0, probeBytes) }
            return measure(
                touched ? "host touched" : "host fresh",
                batches { base.advanced(by: $0 * stride) }
            )
        }

        /// Warm bytes through the whole multi-gigabyte destination, one window at a
        /// time. A rate that holds is a destination that is simply memory; a rate that
        /// falls window by window is the footprint itself, and the fill's is 67 GB.
        func footprintArm() -> [Double] {
            guard footprint.count >= probePairs.count * 2 else { return [] }
            var rates: [Double] = []
            var start = 0
            while start + probePairs.count <= footprint.count {
                let window = Array(footprint[start..<(start + probePairs.count)])
                rates.append(
                    measure("metal footprint", batches { window[$0].contents() }).rate
                )
                start += probePairs.count
            }
            return rates
        }

        func probe(_ label: String) {
            // The source: read cold once so the arms that follow it read the same bytes
            // WARM from the file cache. Without this every destination arm would be
            // measuring the disk.
            let cold = measure(
                "host cold", batches { residentHost.advanced(by: $0 * stride) }
            )
            let warm = measure(
                "host warm", batches { residentHost.advanced(by: $0 * stride) }
            )
            let resident = residentMetalArm()
            let fresh = freshMetalArm(touched: false)
            let touched = freshMetalArm(touched: true)

            // The arm the slice is about. `mlock` is attempted before anything is timed
            // and its result is printed, because a refusal (a machine with no room to
            // keep the pages) is itself the finding.
            var wiredHost = (rate: 0.0, decompressions: 0)
            var wiredMetal = (rate: 0.0, decompressions: 0)
            var wireNote = "not attempted"
            if let region = Dsv41WiredRegion(bytes: probeBytes, wire: true) {
                if region.wireResult == 0 {
                    wireNote = "wired"
                    wiredHost = measure(
                        "host wired",
                        batches { region.base.advanced(by: $0 * stride) }
                    )
                    var buffers: [MTLBuffer] = []
                    for index in probePairs.indices {
                        let start = region.base.advanced(by: index * stride)
                        guard let buffer = device.makeBuffer(
                            bytesNoCopy: start,
                            length: stride,
                            options: .storageModeShared,
                            deallocator: nil
                        ) else { break }
                        buffers.append(buffer)
                    }
                    if buffers.count == probePairs.count {
                        wiredMetal = measure(
                            "metal wired", batches { buffers[$0].contents() }
                        )
                    }
                } else {
                    wireNote = "mlock refused \(region.wireResult)"
                }
            }

            let freshHost = freshHostArm(touched: false)
            let touchedHost = freshHostArm(touched: true)
            let windows = footprintArm()
            let memory = hostMemoryCounters()
            let wiring = streamer.wiring
            print(
                String(
                    format: "🔎 [dsv41 real]   probe %@: cold %.1f | warm %.1f | resident "
                        + "metal %.1f (dec %d) | fresh metal %.1f (dec %d) | touched metal "
                        + "%.1f (dec %d) | %@ host %.1f (dec %d) metal %.1f (dec %d) | "
                        + "fresh host %.1f | touched host %.1f",
                    label,
                    cold.rate,
                    warm.rate,
                    resident.rate,
                    resident.decompressions,
                    fresh.rate,
                    fresh.decompressions,
                    touched.rate,
                    touched.decompressions,
                    wireNote,
                    wiredHost.rate,
                    wiredHost.decompressions,
                    wiredMetal.rate,
                    wiredMetal.decompressions,
                    freshHost.rate,
                    touchedHost.rate
                )
            )
            if !windows.isEmpty {
                let text = windows.map { String(format: "%.1f", $0) }.joined(separator: " ")
                print(
                    "🔎 [dsv41 real]     footprint windows, each 1.1 GB warm: \(text) GB/s"
                )
            }
            print(
                String(
                    format: "🔎 [dsv41 real]     footprint %.1f GB, graphics %.1f GB "
                        + "(compressed %.2f), compressed %.1f GB, decompressions %d, "
                        + "swapins %d, pressure %d, metal %.1f GB, arena wired %.2f GB "
                        + "(%d refusals), machine compressor %.1f GB / free %.1f GB / "
                        + "swap %.1f GB",
                    memory.footprintMB / 1_024,
                    memory.graphicsMB / 1_024,
                    memory.graphicsCompressedMB / 1_024,
                    memory.compressedMB / 1_024,
                    memory.decompressions,
                    memory.swapins,
                    memory.pressureLevel,
                    Double(device.currentAllocatedSize) / 1_073_741_824,
                    Double(wiring.bytes) / 1_073_741_824,
                    wiring.failures,
                    memory.systemCompressorGB,
                    memory.systemFreeGB,
                    memory.swapUsedGB
                )
            )
        }

        /// One prefill of the shipped block path, with the streamer's per-fill samples on
        /// and, when a budget is given, the arena's slots wired first.
        func pass(_ label: String, rotation: Int, wiring budget: Int? = nil) {
            if let budget {
                let started = DispatchTime.now().uptimeNanoseconds
                let state = streamer.wireResidentSlots(budgetBytes: budget)
                let seconds = Double(
                    DispatchTime.now().uptimeNanoseconds - started
                ) / 1_000_000_000
                let request = budget >= Int.max / 4
                    ? "the whole arena"
                    : String(format: "%.1f GB", Double(budget) / 1_073_741_824)
                print(
                    String(
                        format: "🔎 [dsv41 real]   wiring to %@: %.2f GB wired in "
                            + "%.1f s, %d refusal(s)",
                        request,
                        Double(state.bytes) / 1_073_741_824,
                        seconds,
                        state.failures
                    )
                )
            }
            let executor = runtime.forward()
            executor.reset()
            runtime.arena.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            streamer.recordsFillSamples = true
            streamer.clearFillSamples()
            let before = hostMemoryCounters()
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                _ = try executor.prefillChunk(prompt(positions, rotation: rotation))
            } catch {
                print("🔎 [dsv41 real]   \(label) refused: \(error)")
                executor.recordsBlockTiming = false
                streamer.recordsFillSamples = false
                return
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsBlockTiming = false
            streamer.recordsFillSamples = false
            let timing = executor.blockTiming
            let per = timing.millisecondsPerPosition()
            let samples = streamer.fillSamples
            let fillSeconds = Double(timing.expertFillNanoseconds) / 1_000_000_000
            let rate = fillSeconds > 0
                ? Double(timing.fillBytes) / 1_000_000_000 / fillSeconds : 0
            let after = hostMemoryCounters()
            let missesPerFill = samples.isEmpty
                ? 0 : Double(timing.fillMisses) / Double(samples.count)
            XCTAssertGreaterThan(timing.layers, 0, "the block switch recorded nothing")
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ pass: %.1f ms/pos, fill %.1f GB/s (%.0f "
                        + "MB, %d misses, %.1f per fill), device %.1f, wall %.0f s, "
                        + "decompressions +%d, compressor %.1f -> %.1f GB, arena wired "
                        + "%.2f GB",
                    label,
                    wall / Double(positions),
                    rate,
                    Double(timing.fillBytes) / 1_048_576,
                    timing.fillMisses,
                    missesPerFill,
                    per.denseDevice + per.mixtureDevice,
                    wall / 1_000,
                    after.decompressions - before.decompressions,
                    before.systemCompressorGB,
                    after.systemCompressorGB,
                    Double(streamer.wiring.bytes) / 1_073_741_824
                )
            )
        }

        probe("quiet machine")
        pass("P1", rotation: 13)
        probe("after P1")
        pass("P2 wired 8 GB", rotation: 7, wiring: 8 << 30)
        probe("after P2")
        pass("P3 wired 24 GB", rotation: 13, wiring: 24 << 30)
        probe("after P3")
        // The rung that matters for a fix: a slot that cannot be compressed cannot be
        // compressor input at all, so the whole arena is wired and the same pass is
        // timed again. The rungs before it price the transition, because a wired arena
        // has to leave the machine somewhere to find the room the compressor was using.
        pass("P4 fully wired", rotation: 7, wiring: Int.max / 2)
        probe("after P4")
        withExtendedLifetime(engine) {}
    }

    // MARK: - Arena residency policy

    /// What the shipping wiring policy is worth, paying the lock where it is paid.
    ///
    /// `#56` established the mechanism and the magnitude by wiring the arena
    /// RETROACTIVELY: every slot was locked before the pass being timed, so the pass
    /// itself was pure win. A shipping policy cannot do that. Slots are allocated on
    /// first use, so the policy wires each slot inside the fill that creates it and
    /// the first pass pays the `mlock` cost; only the passes after it see the
    /// regime `#56` measured.
    ///
    /// This case builds the runtime through the app's own path: the memory plan
    /// prices the arena, `wireBudgetBytes` grants it or refuses it, and the streamer
    /// wires each slot as it allocates it. It then prints the grant and times three
    /// rotated passes of the same 128-position stream, with misses per fill beside
    /// every rate (`#54`'s trap) and the wiring counters, the lock time and the
    /// decompressions of each pass beside it.
    ///
    /// Pass 1 is expected to be the slow one and pass 2+ to land in `#56`'s
    /// 171-174 ms/position band. If pass 1 is not slower, the lock is cheap enough
    /// that the policy costs a user nothing they would notice; if pass 2 is not in
    /// the band, the shipping shape does not reproduce the measurement and the
    /// retroactive seam was measuring a machine a user never has.
    ///
    /// Opt-in: it reads hundreds of gigabytes and wires tens of them.
    ///
    ///     touch /tmp/dynamoe_dsv41_wired_policy
    @MainActor
    func testWhatTheArenaResidencyPolicyIsWorthIncludingItsLock() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_wired_policy"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in arena residency policy probe; touch \(markerPath) to run"
        )

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
        // The same stream and the same 128 positions `#53`-`#56` measured, rotated
        // per pass so a pass does not cycle the routing and leave the arena warm.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        // `wireArena: nil` is the shipping default: the memory plan decides. Every
        // other measurement in this file pins the unwired regime instead.
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positions + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        let streamer = try XCTUnwrap(
            runtime.expertStreamer,
            "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
        )
        let granted = streamer.wiringState
        print("🔎 [dsv41 real] \(streamer.describe())")
        print(
            String(
                format: "🔎 [dsv41 real] wiring grant: enabled %@, budget %.2f GB, "
                    + "refused %@, %d refusal(s), arena %d slots of %.1f MB",
                granted.isEnabled ? "yes" : "no",
                Double(granted.budgetBytes) / 1_073_741_824,
                granted.isRefused ? "yes" : "no",
                granted.failures,
                streamer.capacity,
                Double(streamer.geometry.slotBytes) / 1_048_576
            )
        )
        try XCTSkipUnless(
            granted.isEnabled,
            "the memory plan did not afford wiring this arena on a machine with "
                + "\(String(format: "%.1f", Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824)) GB"
        )

        /// One prefill of the shipped block path, with the policy already engaged.
        /// Returns the fill rate so the caller can assert on the passes that should
        /// be wired; pass 1 also pays the lock, so it is not held to that band.
        @discardableResult
        func pass(_ label: String, rotation: Int) -> Double {
            let executor = runtime.forward()
            executor.reset()
            runtime.arena.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            streamer.recordsFillSamples = true
            streamer.clearFillSamples()
            let wiringBefore = streamer.wiringState
            let timingBefore = streamer.timing
            let before = hostMemoryCounters()
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                _ = try executor.prefillChunk(prompt(positions, rotation: rotation))
            } catch {
                print("🔎 [dsv41 real]   \(label) refused: \(error)")
                executor.recordsBlockTiming = false
                streamer.recordsFillSamples = false
                return 0
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsBlockTiming = false
            streamer.recordsFillSamples = false
            let timing = executor.blockTiming
            let per = timing.millisecondsPerPosition()
            let samples = streamer.fillSamples
            let fillSeconds = Double(timing.expertFillNanoseconds) / 1_000_000_000
            let rate = fillSeconds > 0
                ? Double(timing.fillBytes) / 1_000_000_000 / fillSeconds : 0
            let missesPerFill = samples.isEmpty
                ? 0 : Double(timing.fillMisses) / Double(samples.count)
            let after = hostMemoryCounters()
            let wiringAfter = streamer.wiringState
            let timingAfter = streamer.timing
            XCTAssertGreaterThan(timing.layers, 0, "the block switch recorded nothing")
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ pass: %.1f ms/pos, fill %.1f GB/s (%.0f "
                        + "MB, %d misses, %.1f per fill), device %.1f, wall %.0f s, "
                        + "decompressions +%d, compressor %.1f -> %.1f GB, wired "
                        + "%.2f -> %.2f GB (+%d slots, %.2f s locking), refusals %d",
                    label,
                    wall / Double(positions),
                    rate,
                    Double(timing.fillBytes) / 1_048_576,
                    timing.fillMisses,
                    missesPerFill,
                    per.denseDevice + per.mixtureDevice,
                    wall / 1_000,
                    after.decompressions - before.decompressions,
                    before.systemCompressorGB,
                    after.systemCompressorGB,
                    Double(wiringBefore.bytes) / 1_073_741_824,
                    Double(wiringAfter.bytes) / 1_073_741_824,
                    wiringAfter.slots - wiringBefore.slots,
                    timingAfter.wiring - timingBefore.wiring,
                    wiringAfter.failures
                )
            )
            // The mechanism, not a mood: a wired fill reads at the disk's own rate
            // (`#56`: 13.0 against the disk's 13.6-15.2) and an unwired one at
            // 4.6-6.7 on this machine. The caller asserts this on the passes after
            // the one that paid the lock.
            return rate
        }

        let quiet = hostMemoryCounters()
        print(
            String(
                format: "🔎 [dsv41 real] before: compressor %.1f GB, free %.1f GB, "
                    + "pressure %d, decompressions %d",
                quiet.systemCompressorGB,
                quiet.systemFreeGB,
                quiet.pressureLevel,
                quiet.decompressions
            )
        )
        pass("P1 policy on", rotation: 13)
        let second = pass("P2 after wiring", rotation: 7)
        let third = pass("P3 after wiring", rotation: 13)
        XCTAssertGreaterThan(second, 8, "P2 filled at \(second) GB/s, the unwired regime")
        XCTAssertGreaterThan(third, 8, "P3 filled at \(third) GB/s, the unwired regime")

        // The A/B the policy is actually asking, on the same runtime and the same
        // stream: withdraw it and let a pass grow the compressor into the pages again.
        // That is the regime `#55`/`#56` measured, and it is what the policy is
        // preventing rather than repairing.
        streamer.disableSlotWiring()
        let unlocked = hostMemoryCounters()
        print(
            String(
                format: "🔎 [dsv41 real] policy withdrawn: wired 0.00 GB, compressor "
                    + "%.1f GB, free %.1f GB",
                unlocked.systemCompressorGB,
                unlocked.systemFreeGB
            )
        )
        let unprotected = pass("P4 policy off", rotation: 7)
        let afterUnprotected = hostMemoryCounters()
        print(
            String(
                format: "🔎 [dsv41 real] after P4: compressor %.1f GB (was %.1f), "
                    + "decompressions +%d, fill %.1f GB/s",
                afterUnprotected.systemCompressorGB,
                unlocked.systemCompressorGB,
                afterUnprotected.decompressions - unlocked.decompressions,
                unprotected
            )
        )

        // Restore it the way `#56` did, retroactively, which is also the expensive way
        // once a pass has compressed the pages: the policy paid 2 s for fresh slots
        // as it allocated them, and this pays for pages the compressor already took.
        let restoredStarted = DispatchTime.now().uptimeNanoseconds
        let restored = streamer.wireResidentSlots(budgetBytes: Int.max / 2)
        let restoredSeconds = Double(
            DispatchTime.now().uptimeNanoseconds - restoredStarted
        ) / 1_000_000_000
        print(
            String(
                format: "🔎 [dsv41 real] retroactive restore: %.2f GB wired in %.1f s, "
                    + "%d refusal(s)",
                Double(restored.bytes) / 1_073_741_824,
                restoredSeconds,
                restored.failures
            )
        )
        let restoredRate = pass("P5 policy restored", rotation: 13)
        XCTAssertGreaterThan(
            restoredRate, 8, "P5 filled at \(restoredRate) GB/s, the unwired regime"
        )

        let wiring = streamer.wiringState
        let allocated = streamer.timing.slotsAllocated
        XCTAssertFalse(wiring.isRefused, "the policy withdrew: \(wiring.failures) refusal(s)")
        XCTAssertGreaterThan(wiring.bytes, 0, "the shipping policy wired nothing")
        XCTAssertLessThanOrEqual(wiring.bytes, wiring.budgetBytes)
        XCTAssertEqual(
            wiring.slots, allocated,
            "every allocated slot must be wired under the shipping policy"
        )
        print(
            String(
                format: "🔎 [dsv41 real] after: %d of %d slots wired, %.2f of %.2f GB, "
                    + "%d refusal(s), %.2f s locking, compressor %.1f GB",
                wiring.slots,
                allocated,
                Double(wiring.bytes) / 1_073_741_824,
                Double(wiring.budgetBytes) / 1_073_741_824,
                wiring.failures,
                streamer.timing.wiring,
                hostMemoryCounters().systemCompressorGB
            )
        )
        withExtendedLifetime(engine) {}
    }

    // MARK: - What the misses are made of

    /// Whether the residual expert traffic can be attacked at all.
    ///
    /// `#57` shipped the arena's residency policy and left the wired fill at 14.4-14.5
    /// GB/s, i.e. on the disk's own ceiling, with **8.1-8.2 misses per fill** unchanged
    /// by anything the policy did. That made the miss rate the next lever, and `#21`
    /// had already said eviction was not where a short run's bytes went -- the two are
    /// compatible only if a pass's misses are its COMPULSORY working set, pairs read
    /// once and never needed again, because no replacement rule can remove a read that
    /// has not happened before.
    ///
    /// That is a question with an exact answer, and it needs no policy: classify each
    /// miss as a pair this window has not loaded yet or a pair it loaded and then
    /// evicted (`DeepSeekV41ExpertArena.recordsMissComposition`). The first kind is the
    /// floor a perfect oracle would still pay; only the second can any policy win back.
    /// `#54`'s quarter-arena control already hints at the answer -- 1,202 slots missed
    /// 10,342 times against 5,021 at 3,760 -- so capacity does something, and the
    /// question is whether it is replacing first touches or re-fetches.
    ///
    /// Six passes over `#52`-`#57`'s 128-position stream, all with the shipping arena
    /// and the shipping residency policy, plus a seventh on a second runtime:
    ///
    ///   * `A1` from `arena.reset()`: a COLD arena, i.e. what `#54`-`#57` measured.
    ///   * `A2` and `A3`, WITHOUT a reset: the steady state a live turn sees, where the
    ///     arena already holds the previous pass's surviving pairs. If the cold pass's
    ///     misses were compulsory, a warm pass pays only what the new text touches; if
    ///     they were re-fetches, the warm pass pays them again.
    ///   * `C1` from `arena.reset()` and `C2`/`C3` warm, with the arena's replacement
    ///     rule switched to `leastFrequentlyUsed`: the same arena, the same slots, the
    ///     same stream. The cold passes are the control -- routing does not consult the
    ///     arena, so they must touch an identical working set -- and the warm ones are
    ///     the A/B, because a warm pass is what a live turn pays.
    ///   * `B1` and `B2` on a runtime built with a 0.72 arena fraction (4,637 slots,
    ///     81.19 GB): the same two passes at a bigger capacity, which `#21` could not
    ///     measure because its working set fit both sizes.
    ///
    /// Every pass prints the decomposition in absolute terms and the bytes each side is
    /// worth (`slotBytes` per miss), because "the miss rate" is only actionable if the
    /// re-fetch column is a meaningful share of 89 GB.
    ///
    /// Then a SECOND runtime at a 0.72 arena fraction, one cold pass of the same stream.
    /// Routing is a deterministic function of the activations, so the compulsory column
    /// must come out identical and only the re-fetch column can move -- which makes the
    /// capacity question arithmetic instead of an inference. `#21` compared 0.6 against
    /// 0.72 and found the bytes identical, but it compared them at a position where the
    /// working set fit BOTH capacities, so that arm could not see a difference by
    /// construction.
    ///
    /// Opt-in: it reads hundreds of gigabytes and builds two runtimes.
    ///
    ///     touch /tmp/dynamoe_dsv41_miss_composition
    @MainActor
    func testWhatTheResidualMissesAreMadeOf() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_miss_composition"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in miss composition probe; touch \(markerPath) to run"
        )

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
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128

        /// What one pass cost and what its misses were made of.
        struct Outcome {
            let rate: Double
            let millisecondsPerPosition: Double
            let misses: Int
            let firstTouch: Int
            let refetch: Int
            let evictions: Int
            let bytes: Int
        }

        /// One prefill of the shipped block path. `resetting` is the only difference
        /// between the cold and warm arms, so every other counter is comparable.
        func pass(
            _ label: String,
            runtime: DeepSeekV41Runtime,
            rotation: Int,
            resetting: Bool
        ) -> Outcome {
            let executor = runtime.forward()
            let arena = runtime.arena
            guard let streamer = runtime.expertStreamer else {
                XCTFail("\(label): the runtime has no expert streamer")
                return Outcome(
                    rate: 0, millisecondsPerPosition: 0, misses: 0,
                    firstTouch: 0, refetch: 0, evictions: 0, bytes: 0
                )
            }
            executor.reset()
            if resetting { arena.reset() }
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            // Set after the reset on purpose: the flag describes the instrument, the
            // counters describe the window (`#58`).
            arena.recordsMissComposition = true
            streamer.recordsFillSamples = true
            streamer.clearFillSamples()
            let acquisitionsBefore = arena.acquisitions
            let hitsBefore = arena.hits
            let evictionsBefore = arena.evictions
            let compositionBefore = arena.missComposition
            let residentBefore = arena.residentCount
            let wiringBefore = streamer.wiringState
            let before = hostMemoryCounters()
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                _ = try executor.prefillChunk(prompt(positions, rotation: rotation))
            } catch {
                XCTFail("\(label) refused: \(error)")
                executor.recordsBlockTiming = false
                streamer.recordsFillSamples = false
                return Outcome(
                    rate: 0, millisecondsPerPosition: 0, misses: 0,
                    firstTouch: 0, refetch: 0, evictions: 0, bytes: 0
                )
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsBlockTiming = false
            let timing = executor.blockTiming
            let fills = streamer.fillSamples.count
            streamer.recordsFillSamples = false
            let per = timing.millisecondsPerPosition()
            let fillSeconds = Double(timing.expertFillNanoseconds) / 1_000_000_000
            let rate = fillSeconds > 0
                ? Double(timing.fillBytes) / 1_000_000_000 / fillSeconds : 0
            let missesPerFill = fills > 0 ? Double(timing.fillMisses) / Double(fills) : 0
            let acquisitions = arena.acquisitions - acquisitionsBefore
            let hits = arena.hits - hitsBefore
            let misses = acquisitions - hits
            let evictions = arena.evictions - evictionsBefore
            let composition = arena.missComposition
            let firstTouch = composition.firstTouch - compositionBefore.firstTouch
            let refetch = composition.refetch - compositionBefore.refetch
            let after = hostMemoryCounters()
            let wiringAfter = streamer.wiringState
            let slotBytes = streamer.geometry.slotBytes
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ pass: %.1f ms/pos, fill %.1f GB/s (%.0f MB, "
                        + "%d misses, %.1f per fill over %d fills), device %.1f, wall %.0f s, "
                        + "decompressions +%d, resident %d -> %d",
                    label,
                    wall / Double(positions),
                    rate,
                    Double(timing.fillBytes) / 1_048_576,
                    misses,
                    missesPerFill,
                    fills,
                    per.denseDevice + per.mixtureDevice,
                    wall / 1_000,
                    after.decompressions - before.decompressions,
                    residentBefore,
                    arena.residentCount
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ composition: %d acquisitions, %d hits "
                        + "(%.1f%%), %d misses = %d first touch + %d re-fetch (%.1f%%), "
                        + "%d eviction(s); %.1f distinct/position, %.1f GB compulsory "
                        + "against %.2f GB of arena, %.1f GB re-read, %.1f GB/position read",
                    label,
                    acquisitions,
                    hits,
                    acquisitions > 0 ? 100 * Double(hits) / Double(acquisitions) : 0,
                    misses,
                    firstTouch,
                    refetch,
                    misses > 0 ? 100 * Double(refetch) / Double(misses) : 0,
                    evictions,
                    Double(firstTouch) / Double(positions),
                    Double(firstTouch * slotBytes) / 1_073_741_824,
                    Double(arena.capacity * slotBytes) / 1_073_741_824,
                    Double(refetch * slotBytes) / 1_073_741_824,
                    Double(misses * slotBytes) / Double(positions) / 1_048_576
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ residency: wired %.2f -> %.2f GB (+%d slots, "
                        + "%d refusal(s), refused %@), %.0f GB free, compressor %.1f GB "
                        + "-> %.1f GB",
                    label,
                    Double(wiringBefore.bytes) / 1_073_741_824,
                    Double(wiringAfter.bytes) / 1_073_741_824,
                    wiringAfter.slots - wiringBefore.slots,
                    wiringAfter.failures,
                    wiringAfter.isRefused ? "yes" : "no",
                    after.systemFreeGB,
                    before.systemCompressorGB,
                    after.systemCompressorGB
                )
            )
            // No rate assertion here on purpose: `#56`/`#57`'s "wired means the disk's
            // own 13.6-15.2 GB/s" holds for the shipped arena, and the wide arm below is
            // wired and reads at 6.3 because the machine has no free memory left for
            // anything the process did NOT wire. The caller asserts where the regime is
            // the shipped one.
            // The instrument's own arithmetic: every miss is classified exactly once,
            // and the streamer's fill count is the arena's miss count.
            XCTAssertEqual(
                firstTouch + refetch, misses,
                "\(label): the composition does not cover the misses"
            )
            XCTAssertEqual(
                timing.fillMisses, misses,
                "\(label): the fill loaded \(timing.fillMisses) experts, the arena missed \(misses)"
            )
            return Outcome(
                rate: rate,
                millisecondsPerPosition: wall / Double(positions),
                misses: misses,
                firstTouch: firstTouch,
                refetch: refetch,
                evictions: evictions,
                bytes: misses * slotBytes
            )
        }

        var cold: [(label: String, outcome: Outcome)] = []
        var warm: [(label: String, outcome: Outcome)] = []
        var narrowArenaGB = 0.0
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            let streamer = try XCTUnwrap(
                runtime.expertStreamer,
                "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
            )
            let wiring = streamer.wiringState
            print("🔎 [dsv41 real] \(streamer.describe())")
            print(
                String(
                    format: "🔎 [dsv41 real] arena %d slots of %.1f MB (%.2f GB), wired %@ "
                        + "(%.2f GB), refused %@",
                    runtime.arena.capacity,
                    Double(streamer.geometry.slotBytes) / 1_048_576,
                    Double(runtime.arena.capacity * streamer.geometry.slotBytes) / 1_073_741_824,
                    wiring.isEnabled ? "yes" : "no",
                    Double(wiring.bytes) / 1_073_741_824,
                    wiring.isRefused ? "yes" : "no"
                )
            )
            narrowArenaGB = Double(runtime.arena.capacity * streamer.geometry.slotBytes)
                / 1_073_741_824
            let quiet = hostMemoryCounters()
            print(
                String(
                    format: "🔎 [dsv41 real] before: compressor %.1f GB, free %.1f GB, "
                        + "pressure %d, decompressions %d",
                    quiet.systemCompressorGB,
                    quiet.systemFreeGB,
                    quiet.pressureLevel,
                    quiet.decompressions
                )
            )
            for arm in [("A1 cold, LRU", 13, true), ("A2 warm, LRU", 7, false), ("A3 warm, LRU", 13, false)] {
                let outcome = pass(arm.0, runtime: runtime, rotation: arm.1, resetting: arm.2)
                // The shipped regime (a wired 65.84 GB arena on this machine) reads at
                // or above the disk's own 13.6-15.2 GB/s in its warm passes (`#56`).
                if !arm.2 {
                    XCTAssertGreaterThan(
                        outcome.rate, 8, "\(arm.0) filled at \(outcome.rate) GB/s, the unwired regime"
                    )
                }
                if arm.2 { cold.append((arm.0, outcome)) } else { warm.append((arm.0, outcome)) }
            }
            let lruCold = cold.last?.outcome

            // The policy A/B: same arena, same slots, same stream. Only the rule a miss
            // replaces by changes, so the compulsory column has to come out identical --
            // routing does not consult the arena -- and only the re-fetch column can move.
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            for arm in [
                ("C1 cold, frequency", 13, true),
                ("C2 warm, frequency", 7, false),
                ("C3 warm, frequency", 13, false)
            ] {
                let outcome = pass(arm.0, runtime: runtime, rotation: arm.1, resetting: arm.2)
                if !arm.2 {
                    XCTAssertGreaterThan(
                        outcome.rate, 8, "\(arm.0) filled at \(outcome.rate) GB/s, the unwired regime"
                    )
                }
                if arm.2 { cold.append((arm.0, outcome)) } else { warm.append((arm.0, outcome)) }
            }
            let frequencyCold = cold.last?.outcome
            if let lruCold, let frequencyCold {
                // Routing does not consult the arena, so the working set is a control: the
                // two cold passes must have touched exactly the same pairs, and only the
                // column a policy can move may differ.
                XCTAssertEqual(
                    frequencyCold.firstTouch, lruCold.firstTouch,
                    "the same stream touched a different working set, so the two policies are "
                        + "not being compared on the same passes"
                )
                print(
                    String(
                        format: "🔎 [dsv41 real] policy control: both cold passes touched %d "
                            + "pairs; LRU missed %d (%d re-fetch), frequency missed %d (%d)",
                        lruCold.firstTouch,
                        lruCold.misses,
                        lruCold.refetch,
                        frequencyCold.misses,
                        frequencyCold.refetch
                    )
                )
            }
            // A warm pass is the regime a live turn is in, and the two policies' warm
            // passes are the comparison. Printed as a table rather than one delta because
            // the arms follow different predecessors, so a reader can match rotations.
            for arm in warm {
                print(
                    String(
                        format: "🔎 [dsv41 real]   %@ pass summary: %d misses (%d re-fetch, "
                            + "%.1f GB), re-read %.1f GB, %.1f ms/pos",
                        arm.label,
                        arm.outcome.misses,
                        arm.outcome.refetch,
                        Double(arm.outcome.bytes) / 1_073_741_824,
                        Double(arm.outcome.refetch * streamer.geometry.slotBytes) / 1_073_741_824,
                        arm.outcome.millisecondsPerPosition
                    )
                )
            }
            print(
                String(
                    format: "🔎 [dsv41 real] after the six passes: %d eviction(s) in the last "
                        + "window, %d distinct pairs touched there of %d slots, compressor %.1f GB",
                    runtime.arena.evictions,
                    runtime.arena.distinctPairs,
                    runtime.arena.capacity,
                    hostMemoryCounters().systemCompressorGB
                )
            )
            runtime.shutdown()
        }

        // The capacity arm. Two runtimes at once is 147 GB on a 128 GB machine, so the
        // first is shut down before the second is built; routing does not depend on the
        // arena, so the compulsory column is the control that says the two passes saw
        // the same stream.
        let wide = 0.72
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil,
                arenaFraction: wide
            )
            let streamer = try XCTUnwrap(
                runtime.expertStreamer,
                "no expert streamer at \(wide): \(runtime.expertStreamerProblem ?? "no reason")"
            )
            let wiring = streamer.wiringState
            print(
                String(
                    format: "🔎 [dsv41 real] wide arena: fraction %.2f, %d slots of %.1f MB "
                        + "(%.2f GB), wired %@ (%.2f GB), refused %@; memory plan says %@",
                    wide,
                    runtime.arena.capacity,
                    Double(streamer.geometry.slotBytes) / 1_048_576,
                    Double(runtime.arena.capacity * streamer.geometry.slotBytes) / 1_073_741_824,
                    wiring.isEnabled ? "yes" : "no",
                    Double(wiring.bytes) / 1_073_741_824,
                    wiring.isRefused ? "yes" : "no",
                    runtime.memoryPlan.describe(expertSlots: runtime.arena.capacity)
                )
            )
            let outcome = pass("B1 cold, wider arena", runtime: runtime, rotation: 13, resetting: true)
            // The capacity question is asked by the WARM pass: a cold one pays its
            // working set's first touches whatever the arena holds, and only a warm one
            // shows how much of the previous pass survived.
            let wideWarm = pass(
                "B2 warm, wider arena", runtime: runtime, rotation: 7, resetting: false
            )
            let wideWiring = streamer.wiringState
            // The policy is all-or-nothing, so the arm is only a capacity measurement if
            // the whole 81.19 GB actually stayed wired. `mlock` refusals would have
            // withdrawn it and the arm would be the unwired regime instead.
            XCTAssertFalse(
                wideWiring.isRefused,
                "the wider arena was refused after \(wideWiring.failures) lock failure(s), so "
                    + "this is not a capacity arm"
            )
            XCTAssertGreaterThan(wideWiring.bytes, 0, "the wider arena wired nothing")
            let slotBytes = streamer.geometry.slotBytes
            let narrowWarm = warm.first?.outcome
            if let narrowWarm {
                print(
                    String(
                        format: "🔎 [dsv41 real] warm capacity A/B (rotation 7, no reset): "
                            + "%.2f GB of arena -> %d misses (%d re-fetch, %.1f GB read, "
                            + "%.1f ms/pos); %.2f GB -> %d misses (%d re-fetch, %.1f GB read, "
                            + "%.1f ms/pos), i.e. %+.1f%% read and %+.1f ms/position",
                        narrowArenaGB,
                        narrowWarm.misses,
                        narrowWarm.refetch,
                        Double(narrowWarm.bytes) / 1_073_741_824,
                        narrowWarm.millisecondsPerPosition,
                        Double(runtime.arena.capacity * slotBytes) / 1_073_741_824,
                        wideWarm.misses,
                        wideWarm.refetch,
                        Double(wideWarm.bytes) / 1_073_741_824,
                        wideWarm.millisecondsPerPosition,
                        100 * Double(wideWarm.bytes - narrowWarm.bytes)
                            / Double(max(1, narrowWarm.bytes)),
                        wideWarm.millisecondsPerPosition - narrowWarm.millisecondsPerPosition
                    )
                )
            }
            let narrow = cold.first?.outcome
            if let narrow {
                // The whole point of the arm: routing is deterministic, so the number of
                // pairs this pass touched cannot depend on how many slots could hold
                // them. Only the column a policy could win may move.
                XCTAssertEqual(
                    outcome.firstTouch, narrow.firstTouch,
                    "the two capacities touched different working sets, so the passes are "
                        + "not comparable: routing must not depend on the arena"
                )
                print(
                    String(
                        format: "🔎 [dsv41 real] capacity A/B: %.2f GB of arena -> %d misses "
                            + "(%.1f GB read, %.1f ms/pos); %.2f GB -> %d misses (%.1f GB "
                            + "read, %.1f ms/pos). Re-fetch %d against %d, i.e. %+.1f%% read "
                            + "and %+.1f ms/position",
                        narrowArenaGB,
                        narrow.misses,
                        Double(narrow.bytes) / 1_073_741_824,
                        narrow.millisecondsPerPosition,
                        Double(runtime.arena.capacity * slotBytes) / 1_073_741_824,
                        outcome.misses,
                        Double(outcome.bytes) / 1_073_741_824,
                        outcome.millisecondsPerPosition,
                        narrow.refetch,
                        outcome.refetch,
                        100 * Double(outcome.bytes - narrow.bytes) / Double(max(1, narrow.bytes)),
                        outcome.millisecondsPerPosition - narrow.millisecondsPerPosition
                    )
                )
            }
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// Whether the replacement rule is worth anything to a DECODE, not only to a prefill.
    ///
    /// `#58` measured the rule on a 128-position prefill and left one caveat standing: a
    /// live turn's decode loop is the WALK -- one position at a time through the same
    /// arena -- and no walk had ever run under frequency. `#21` measured a 53-position
    /// walk whose 3,739 distinct pairs just fit 3,760 slots, which is why it saw no
    /// evictions to tune; `#58` then showed a 128-position working set does not fit. This
    /// asks the walk side of that: at the length where the working set overflows, does
    /// the pair a miss replaces by move the decode?
    ///
    /// One runtime, one token stream, six passes. Every arm replays the SAME stream from
    /// position 0, so every arm issues the identical acquisition sequence -- routing
    /// does not consult the arena -- and only the pair a miss replaces changes. The cold
    /// passes are the control: their first-touch column must match exactly, or the two
    /// policies are not being compared on the same sequence. Warm passes do not reset,
    /// so they measure the steady state a live turn is in, which is the number the rule
    /// has to move.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_walk_policy`, plus the class's own marker), because a
    /// cold walk of 128 positions reads the whole routed working set off disk.
    func testWhatTheReplacementRuleIsWorthToAWalk() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_walk_policy"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in walk policy probe; touch \(markerPath) to run"
        )

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
        // The same 128-position stream `#52`-`#58` measured, so this slice's working set
        // is comparable with theirs position for position.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)

        /// What one walked pass cost and what its misses were made of.
        struct Outcome {
            let millisecondsPerPosition: Double
            let readBytes: Int
            let readRate: Double
            let misses: Int
            let firstTouch: Int
            let refetch: Int
            let distinctPairs: Int
        }

        /// One walk of the stream. `resetting` is the only difference between a cold and
        /// a warm arm, so every counter is comparable.
        func pass(
            _ label: String,
            runtime: DeepSeekV41Runtime,
            resetting: Bool
        ) -> Outcome {
            let executor = runtime.forward()
            let arena = runtime.arena
            guard let streamer = runtime.expertStreamer else {
                XCTFail("\(label): the runtime has no expert streamer")
                return Outcome(
                    millisecondsPerPosition: 0, readBytes: 0, readRate: 0, misses: 0,
                    firstTouch: 0, refetch: 0, distinctPairs: 0
                )
            }
            executor.reset()
            if resetting { arena.reset() }
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            // Set after the reset on purpose: the flag describes the instrument, the
            // counters describe the window (`#58`).
            arena.recordsMissComposition = true
            streamer.recordsFillSamples = true
            streamer.clearFillSamples()
            let acquisitionsBefore = arena.acquisitions
            let hitsBefore = arena.hits
            let evictionsBefore = arena.evictions
            let compositionBefore = arena.missComposition
            let residentBefore = arena.residentCount
            let bytesBefore = streamer.bytesRead
            let readingBefore = streamer.timing.reading
            let before = hostMemoryCounters()
            let start = DispatchTime.now().uptimeNanoseconds
            do {
                for (position, token) in tokens.enumerated() {
                    _ = try executor.forwardToken(token, position: position)
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            let timing = executor.walkTiming
            let per = timing.millisecondsPerPosition()
            let fills = streamer.fillSamples.count
            let sampledMisses = streamer.fillSamples.reduce(0) { $0 + $1.misses }
            streamer.recordsFillSamples = false
            let acquisitions = arena.acquisitions - acquisitionsBefore
            let hits = arena.hits - hitsBefore
            let misses = acquisitions - hits
            let evictions = arena.evictions - evictionsBefore
            let composition = arena.missComposition
            let firstTouch = composition.firstTouch - compositionBefore.firstTouch
            let refetch = composition.refetch - compositionBefore.refetch
            let readBytes = streamer.bytesRead - bytesBefore
            let readSeconds = streamer.timing.reading - readingBefore
            let rate = readSeconds > 0 ? Double(readBytes) / 1_000_000_000 / readSeconds : 0
            let after = hostMemoryCounters()
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ walk: %.1f ms/pos, read %.1f GB at %.1f GB/s, "
                        + "%d miss(es) (%.1f per fill over %d fills), device %.1f ms/pos, "
                        + "resident %d -> %d, compressor %.1f -> %.1f GB",
                    label,
                    wall / Double(positions),
                    Double(readBytes) / 1_073_741_824,
                    rate,
                    misses,
                    fills > 0 ? Double(misses) / Double(fills) : 0,
                    fills,
                    per.frontDevice + per.mixtureDevice,
                    residentBefore,
                    arena.residentCount,
                    before.systemCompressorGB,
                    after.systemCompressorGB
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ composition: %d acquisitions, %d hits "
                        + "(%.1f%%), %d misses = %d first touch + %d re-fetch (%.1f%%), "
                        + "%d eviction(s), %d distinct pair(s) of %d slot(s); %.1f GB read "
                        + "per position",
                    label,
                    acquisitions,
                    hits,
                    acquisitions > 0 ? 100 * Double(hits) / Double(acquisitions) : 0,
                    misses,
                    firstTouch,
                    refetch,
                    misses > 0 ? 100 * Double(refetch) / Double(misses) : 0,
                    evictions,
                    arena.distinctPairs,
                    arena.capacity,
                    Double(readBytes) / Double(positions) / 1_048_576
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ walk buckets: front encode %.1f, front wait "
                        + "%.1f, expert fill %.1f, mixture encode %.1f, mixture device %.1f "
                        + "ms/position",
                    label,
                    per.frontEncode,
                    per.frontWait,
                    per.expertFill,
                    per.mixtureEncode,
                    per.mixtureDevice
                )
            )
            // The instrument's own arithmetic: every miss is classified exactly once,
            // and the fills the streamer sampled are the arena's misses.
            XCTAssertEqual(
                firstTouch + refetch, misses,
                "\(label): the composition does not cover the misses"
            )
            XCTAssertEqual(
                sampledMisses, misses,
                "\(label): the fills loaded \(sampledMisses) experts, the arena missed \(misses)"
            )
            return Outcome(
                millisecondsPerPosition: wall / Double(positions),
                readBytes: readBytes,
                readRate: rate,
                misses: misses,
                firstTouch: firstTouch,
                refetch: refetch,
                distinctPairs: arena.distinctPairs
            )
        }

        var cold: [(label: String, outcome: Outcome)] = []
        var warm: [(label: String, outcome: Outcome)] = []
        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            let streamer = try XCTUnwrap(
                runtime.expertStreamer,
                "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
            )
            // `makeRuntime` pins LRU so every `#20`-`#57` table stays comparable; this
            // case is measuring the rule, so it says which one each arm is under.
            print("🔎 [dsv41 real] \(streamer.describe())")
            print(
                String(
                    format: "🔎 [dsv41 real] walk policy probe: %d positions, arena %d slots "
                        + "of %.1f MB (%.2f GB), policy %@",
                    positions,
                    runtime.arena.capacity,
                    Double(streamer.geometry.slotBytes) / 1_048_576,
                    Double(runtime.arena.capacity * streamer.geometry.slotBytes) / 1_073_741_824,
                    runtime.arena.victimPolicy == .leastFrequentlyUsed ? "frequency" : "LRU"
                )
            )
            for arm in [
                ("A1 cold, LRU", true), ("A2 warm, LRU", false), ("A3 warm, LRU", false)
            ] {
                let outcome = pass(arm.0, runtime: runtime, resetting: arm.1)
                if arm.1 { cold.append((arm.0, outcome)) } else { warm.append((arm.0, outcome)) }
            }
            let lruCold = cold.last?.outcome

            // The policy A/B: same arena, same slots, same token stream. Only the rule a
            // miss replaces by changes, so the compulsory column has to come out
            // identical -- routing does not consult the arena -- and only the re-fetch
            // column can move.
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            for arm in [
                ("C1 cold, frequency", true),
                ("C2 warm, frequency", false),
                ("C3 warm, frequency", false)
            ] {
                let outcome = pass(arm.0, runtime: runtime, resetting: arm.1)
                if arm.1 { cold.append((arm.0, outcome)) } else { warm.append((arm.0, outcome)) }
            }
            let frequencyCold = cold.last?.outcome
            if let lruCold, let frequencyCold {
                // Routing does not consult the arena, so the working set is a control:
                // the two cold walks must have touched exactly the same pairs, and only
                // the column a policy can move may differ.
                XCTAssertEqual(
                    frequencyCold.firstTouch, lruCold.firstTouch,
                    "the same stream touched a different working set, so the two policies "
                        + "are not being compared on the same passes"
                )
                print(
                    String(
                        format: "🔎 [dsv41 real] policy control: both cold walks touched %d "
                            + "pairs (%d of %d slots); LRU missed %d (%d re-fetch), "
                            + "frequency missed %d (%d re-fetch)",
                        lruCold.firstTouch,
                        lruCold.distinctPairs,
                        runtime.arena.capacity,
                        lruCold.misses,
                        lruCold.refetch,
                        frequencyCold.misses,
                        frequencyCold.refetch
                    )
                )
            }
            // The comparison: the warm arms are the steady state a live turn decodes in.
            let lruWarm = warm.prefix(2).map(\.outcome)
            let frequencyWarm = warm.suffix(2).map(\.outcome)
            func mean(_ values: [Double]) -> Double {
                values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
            }
            if lruWarm.count == 2, frequencyWarm.count == 2 {
                let lruWall = mean(lruWarm.map(\.millisecondsPerPosition))
                let frequencyWall = mean(frequencyWarm.map(\.millisecondsPerPosition))
                let lruRead = mean(lruWarm.map { Double($0.readBytes) / 1_073_741_824 })
                let frequencyRead = mean(frequencyWarm.map { Double($0.readBytes) / 1_073_741_824 })
                let lruMisses = mean(lruWarm.map { Double($0.misses) })
                let frequencyMisses = mean(frequencyWarm.map { Double($0.misses) })
                print(
                    String(
                        format: "🔎 [dsv41 real] warm walk A/B: LRU %.1f ms/pos, %.1f GB read, "
                            + "%.0f misses; frequency %.1f ms/pos, %.1f GB read, %.0f misses "
                            + "i.e. %+.1f%% wall, %+.1f%% read, %+.1f%% misses",
                        lruWall,
                        lruRead,
                        lruMisses,
                        frequencyWall,
                        frequencyRead,
                        frequencyMisses,
                        100 * (frequencyWall - lruWall) / max(0.001, lruWall),
                        100 * (frequencyRead - lruRead) / max(0.001, lruRead),
                        100 * (frequencyMisses - lruMisses) / max(1, lruMisses)
                    )
                )
                // A warm walk must not spend its time on the disk: whoever picks the
                // victim, the fill is the smaller bucket and the walk stays far above the
                // unwired regime's 6-7 GB/s. This is a regime check, not a win.
                for arm in warm {
                    XCTAssertGreaterThan(
                        arm.outcome.readRate, 8,
                        "\(arm.label) filled at \(arm.outcome.readRate) GB/s, the "
                            + "unwired regime"
                    )
                }
            }
            print(
                String(
                    format: "🔎 [dsv41 real] after the six walks: %d eviction(s) in the last "
                        + "window, compressor %.1f GB, free %.1f GB",
                    runtime.arena.evictions,
                    hostMemoryCounters().systemCompressorGB,
                    hostMemoryCounters().systemFreeGB
                )
            )
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// Which stage holds a walked position's device time, on the checkpoint itself.
    ///
    /// `#59` closed the arena's replacement question and left a warm decode at 101-122
    /// ms/position with 53.8 ms of device windows -- the largest bucket, and the one no
    /// published table has ever attributed. Every stage table in
    /// `Benchmarks/PERF_FINDINGS.md` comes from a synthetic fixture, and the one table
    /// taken on the real checkpoint is a BLOCK of eight positions: its three biggest
    /// stages (`queryUp`, `outputUp`, `outputDown`, 57% between them) are GEMV
    /// projections whose cost there is the per-position activation traffic a block
    /// re-requests once per output row (`#18` priced it at 1.9 GB of requests for one
    /// block), and that term grows with the block width. A walk asks for ONE position's
    /// worth, so the block's ranking is not evidence about a decode and this asks it
    /// directly.
    ///
    /// Two arms, one runtime, one 128-position stream, the shipping frequency policy:
    ///
    /// 1. baseline -- the single-buffer layout with `recordsWalkTiming`, which is the
    ///    decomposition `#59` quoted;
    /// 2. stages -- `recordsWalkStageTiming`, which commits every stage of every half
    ///    in its own command buffer, in the same order on the same queue, so each
    ///    reports its own device window.
    ///
    /// The stages arm has to return the same logits as the baseline and its windows
    /// have to add up to the baseline's device total, so the table is read against a
    /// number a different layout produced rather than against itself.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_walk_stages`, plus the class's own marker): a cold
    /// walk of the stream reads the whole routed working set off disk.
    func testWhereAWalkedPositionsDeviceTimeGoes() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_walk_stages"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in walk stage probe; touch \(markerPath) to run"
        )

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
        // The same 128-position stream `#52`-`#59` measured, so this slice's buckets are
        // comparable with theirs position for position.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)

        /// One walked pass: its wall, its bucket instrument, and the logits the last
        /// position produced (so the two layouts can be compared for answer).
        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let lastLogits: [Float]
        }

        func pass(
            _ label: String,
            executor: DeepSeekV41MetalForward,
            arena: DeepSeekV41ExpertArena,
            streamer: DeepSeekV41ExpertStreamer?,
            splittingStages: Bool
        ) -> Outcome? {
            executor.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            executor.recordsWalkStageTiming = splittingStages
            let acquisitionsBefore = arena.acquisitions
            let hitsBefore = arena.hits
            let bytesBefore = streamer?.bytesRead ?? 0
            let start = DispatchTime.now().uptimeNanoseconds
            var logits: [Float] = []
            do {
                for (position, token) in tokens.enumerated() {
                    logits = try executor.forwardToken(token, position: position)
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
                return nil
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            executor.recordsWalkStageTiming = false
            let device = executor.walkTiming.frontDeviceNanoseconds
                + executor.walkTiming.mixtureDeviceNanoseconds
            let acquisitions = arena.acquisitions - acquisitionsBefore
            let misses = acquisitions - (arena.hits - hitsBefore)
            let bytes = (streamer?.bytesRead ?? 0) - bytesBefore
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f ms/position, "
                        + "%d miss(es), %.1f GB read",
                    label,
                    wall / Double(positions),
                    Double(device) / Double(positions) / 1_000_000,
                    misses,
                    Double(bytes) / 1_073_741_824
                )
            )
            print("🔎 [dsv41 real]   \(label) buckets: " + executor.walkTiming.describe())
            return Outcome(
                wallMilliseconds: wall, timing: executor.walkTiming, lastLogits: logits
            )
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            // `makeRuntime` pins LRU so every `#20`-`#58` table stays comparable; a
            // decode in the shipping build uses the frequency rule (`#59`), so both arms
            // here do too.
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            // A discarded warm-up: the first pass of a young process pays the arena's
            // slot creation and wiring (`#54`, `#57`) and the page cache is cold, and
            // neither belongs to a stage. Both measured arms then walk the warm arena a
            // live turn decodes in. They are not reset between the arms on purpose:
            // `#59` showed the rule converges over warm passes, and a reset would turn
            // both into cold passes whose fill is the biggest bucket and whose wall
            // says nothing about a decode. The device windows do not depend on the hit
            // rate, so the layout A/B stays valid across the two.
            executor.reset()
            runtime.arena.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            let slots = runtime.arena.capacity
            let slotMB = runtime.expertStreamer.map {
                Double($0.geometry.slotBytes) / 1_048_576
            } ?? 0
            print(
                String(
                    format: "🔎 [dsv41 real] walk stage probe: %d positions, arena %d slots "
                        + "of %.1f MB, policy %@, warm-up done",
                    positions,
                    slots,
                    slotMB,
                    runtime.arena.victimPolicy == .leastFrequentlyUsed ? "frequency" : "LRU"
                )
            )

            let baseline = try XCTUnwrap(
                pass(
                    "baseline, one buffer per half",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    splittingStages: false
                )
            )
            let stages = try XCTUnwrap(
                pass(
                    "stages, one buffer per stage",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    splittingStages: true
                )
            )

            // The layout must not change the answer: every stage is submitted in the
            // same order on the same queue and binds the same buffers.
            let scale = baseline.lastLogits.map(abs).max() ?? 1
            let worst = zip(baseline.lastLogits, stages.lastLogits)
                .map { abs($0 - $1) }
                .max() ?? 0
            XCTAssertLessThan(
                worst / max(scale, 1e-6), 2e-3,
                "the per-stage layout changed the logits"
            )
            print(
                String(
                    format: "🔎 [dsv41 real] layout control: worst logit delta %.2e absolute "
                        + "of a %.2f scale (%.2e relative)",
                    worst,
                    scale,
                    worst / max(scale, 1e-6)
                )
            )

            // The table has to cover every position and add up to the device total the
            // baseline measured in the other layout.
            XCTAssertEqual(stages.timing.positions, positions)
            XCTAssertEqual(executor.walkStagePositions, positions)
            XCTAssertEqual(
                stages.timing.frontUnwindowedBuffers + stages.timing.mixtureUnwindowedBuffers,
                0,
                "a half's buffer reported no device window"
            )
            let stageTotal = executor.walkStageTiming.values.reduce(0, +)
            XCTAssertEqual(
                stageTotal,
                stages.timing.frontDeviceNanoseconds + stages.timing.mixtureDeviceNanoseconds,
                "the per-stage windows do not add up to the half totals"
            )

            let frontNames = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
            let table = executor.walkStageTable()
            // Every stage this checkpoint actually dispatches has to appear: a stage
            // that silently vanished from the split layout would still add up, because
            // both sides count the same buffers. `.compressPool` is the one stage that
            // never dispatches on its own (the pool runs inside `.indexer`).
            var expected = frontNames
                .union(DeepSeekV41DevicePlan.backHalf.map(\.rawValue))
            expected.remove(DeepSeekV41ForwardStages.Stage.compressPool.rawValue)
            let reported = Set(table.map(\.stage))
            XCTAssertTrue(
                expected.subtracting(reported).isEmpty,
                "the walk's stage table is missing "
                    + "\(expected.subtracting(reported).sorted())"
            )

            let denseTotal = table
                .filter { frontNames.contains($0.stage) }
                .reduce(0) { $0 + $1.millisecondsPerPosition }
            let mixtureTotal = table
                .filter { !frontNames.contains($0.stage) }
                .reduce(0) { $0 + $1.millisecondsPerPosition }
            let totalPerPosition = Double(stageTotal)
                / Double(max(1, executor.walkStagePositions)) / 1_000_000
            print(
                String(
                    format: "🔎 [dsv41 real] walked position by stage: %.2f ms/position "
                        + "total, %.2f dense, %.2f mixture, over %d position(s) and %d stage(s)",
                    totalPerPosition,
                    denseTotal,
                    mixtureTotal,
                    executor.walkStagePositions,
                    table.count
                )
            )
            // The same split `#59`'s closing sentence was written from, now with the
            // drains and late starts beside it: a wait that is mostly its own drain is
            // the GPU working, and one that is mostly late start is not.
            func perPosition(_ nanoseconds: UInt64) -> String {
                String(format: "%.1f", Double(nanoseconds) / Double(positions) / 1_000_000)
            }
            print(
                "🔎 [dsv41 real] baseline per-position buckets: front device "
                    + perPosition(baseline.timing.frontDeviceNanoseconds)
                    + ", mixture device "
                    + perPosition(baseline.timing.mixtureDeviceNanoseconds)
                    + ", front wait "
                    + perPosition(baseline.timing.frontWaitNanoseconds)
                    + " (of which drained front "
                    + perPosition(baseline.timing.frontDrainNanoseconds)
                    + ", drained mixture "
                    + perPosition(baseline.timing.mixtureDrainNanoseconds)
                    + ", late start front "
                    + perPosition(baseline.timing.frontLateStartNanoseconds)
                    + ", mixture "
                    + perPosition(baseline.timing.mixtureLateStartNanoseconds)
                    + "), indexer wait "
                    + String(format: "%.1f", baseline.timing.millisecondsPerPositionIndexerWait())
                    + " ms"
            )
            print(
                "🔎 [dsv41 real] walked position by stage "
                    + "(ms/position, share of the position's measured device time):"
            )
            for entry in table {
                print(
                    "🔎 [dsv41 real]   "
                        + String(
                            format: "%-40@ %7.3f ms %5.1f%%  %@",
                            entry.stage as NSString,
                            entry.millisecondsPerPosition,
                            entry.share * 100,
                            frontNames.contains(entry.stage) ? "dense" : "mixture"
                        )
                )
            }
            XCTAssertGreaterThan(
                baseline.timing.positions, 0, "the baseline instrument reported nothing"
            )
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// What the shared expert's coalesced forms are worth on a real decode.
    ///
    /// `#60` attributed a walked position and found the shared expert the single
    /// largest stage: 12.96 ms of 54.3 ms of device time, at 109-113 GB/s while the
    /// routed experts read 352 GB/s on the same kind of bytes. The cause was the
    /// dispatch shape — `dsv41_fp8_block32_gemv` had a coalesced sibling but the FUSED
    /// swiglu and the down accumulate did not, so the projection the model runs on
    /// every layer of every token still gave a thread a whole row.
    ///
    /// Two arms, one runtime, one 128-position stream, the shipping frequency policy,
    /// both under `recordsWalkStageTiming` so the `sharedExpert` window is measured the
    /// same way in both:
    ///
    /// 1. one-thread — `disablesFP8FusedRows`, the forms the checkpoint ran before;
    /// 2. coalesced — the shipped choice (`fp8GemvWantsWideGrid`).
    ///
    /// The two forms sum in a different order, so the answer is compared within a bar
    /// rather than bit for bit; everything else in the graph is unchanged.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_shared_stage`, plus the class's own marker).
    func testWhatTheCoalescedSharedExpertIsWorthToAWalk() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_shared_stage"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in shared-expert probe; touch \(markerPath) to run"
        )

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
        // The same 128-position stream `#52`-`#60` measured, so the buckets line up
        // with theirs position for position.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)

        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let lastLogits: [Float]
            let table: [(stage: String, millisecondsPerPosition: Double, share: Double)]
            let sharedExpertPerPosition: Double
            /// Logits sampled along the walk, so the two forms' drift can be read as a
            /// trend rather than only at the last position: a reduction-order
            /// difference starts at FP32 rounding and is amplified by forty layers, so
            /// where it starts is what says whether it is rounding or a defect.
            let sampledLogits: [Int: [Float]]
        }

        let sampledPositions: Set<Int> = [0, 1, 2, 3, 7, 15, 31, 63, 127]

        func pass(
            _ label: String,
            executor: DeepSeekV41MetalForward,
            arena: DeepSeekV41ExpertArena,
            streamer: DeepSeekV41ExpertStreamer?,
            oneThread: Bool,
            lanes: Int = 0
        ) -> Outcome? {
            executor.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            executor.recordsWalkStageTiming = true
            executor.disablesFP8FusedRows = oneThread
            executor.fp8FusedLaneWidthOverride = lanes
            let acquisitionsBefore = arena.acquisitions
            let hitsBefore = arena.hits
            let bytesBefore = streamer?.bytesRead ?? 0
            let start = DispatchTime.now().uptimeNanoseconds
            var logits: [Float] = []
            var sampled: [Int: [Float]] = [:]
            do {
                for (position, token) in tokens.enumerated() {
                    logits = try executor.forwardToken(token, position: position)
                    if sampledPositions.contains(position) { sampled[position] = logits }
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
                return nil
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            executor.recordsWalkStageTiming = false
            executor.disablesFP8FusedRows = false
            executor.fp8FusedLaneWidthOverride = 0
            let device = executor.walkTiming.frontDeviceNanoseconds
                + executor.walkTiming.mixtureDeviceNanoseconds
            let acquisitions = arena.acquisitions - acquisitionsBefore
            let misses = acquisitions - (arena.hits - hitsBefore)
            let bytes = (streamer?.bytesRead ?? 0) - bytesBefore
            let table = executor.walkStageTable()
            let shared = table.first { $0.stage == DeepSeekV41ForwardStages.Stage.sharedExpert.rawValue }
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f ms/position, "
                        + "sharedExpert %.3f ms/position, %d miss(es), %.1f GB read",
                    label,
                    wall / Double(positions),
                    Double(device) / Double(positions) / 1_000_000,
                    shared?.millisecondsPerPosition ?? -1,
                    misses,
                    Double(bytes) / 1_073_741_824
                )
            )
            print("🔎 [dsv41 real]   \(label) buckets: " + executor.walkTiming.describe())
            return Outcome(
                wallMilliseconds: wall,
                timing: executor.walkTiming,
                lastLogits: logits,
                table: table,
                sharedExpertPerPosition: shared?.millisecondsPerPosition ?? -1,
                sampledLogits: sampled
            )
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            // A discarded warm-up, exactly as `#59`/`#60`: the first pass pays the
            // arena's slot creation and wiring, and the two measured arms are then warm
            // and are not reset between arms.
            executor.reset()
            runtime.arena.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            print(
                String(
                    format: "🔎 [dsv41 real] shared-expert probe: %d positions, arena %d slots, "
                        + "policy frequency, warm-up done",
                    positions,
                    runtime.arena.capacity
                )
            )

            let oneThread = try XCTUnwrap(
                pass(
                    "one-thread FP8 fused forms",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    oneThread: true
                )
            )
            let coalesced = try XCTUnwrap(
                pass(
                    "coalesced FP8 fused forms, shipped lanes",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    oneThread: false
                )
            )
            // The shipped form at the OTHER lane width: a different reduction tree over
            // the same math. It is the control the answer comparison needs — whatever
            // two widths of one kernel drift is the graph's sensitivity to a
            // rounding-level change, and the one-thread form can be read against it.
            let controlLanes = try XCTUnwrap(
                pass(
                    "coalesced FP8 fused forms, 64 lanes (control)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    oneThread: false,
                    lanes: 64
                )
            )

            // The forms reduce in a different order, so the answer is compared the way
            // the other form sweeps compare it — but with a control, because forty
            // layers amplify a rounding-level change. The 32- and 64-lane arms are the
            // SAME kernel with a different reduction tree over the same arithmetic, so
            // whatever they drift is the graph's own sensitivity. If the one-thread arm
            // drifts no further than that, the coalesced form is arithmetic-equivalent;
            // if it drifts much further, it is reading something different.
            func worstRelative(_ lhs: [Float], _ rhs: [Float]) -> (absolute: Float, relative: Double) {
                let scale = max(lhs.map(abs).max() ?? 1, 1)
                let worst = zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
                return (worst, Double(worst / scale))
            }
            let formDrift = worstRelative(oneThread.lastLogits, coalesced.lastLogits)
            let controlDrift = worstRelative(controlLanes.lastLogits, coalesced.lastLogits)
            print(
                String(
                    format: "🔎 [dsv41 real] answer control: one-thread vs 64 lanes %.2e "
                        + "absolute (%.2e relative), shipped vs 64 lanes %.2e absolute "
                        + "(%.2e relative), ratio %.1fx",
                    formDrift.absolute,
                    formDrift.relative,
                    controlDrift.absolute,
                    controlDrift.relative,
                    controlDrift.relative > 0
                        ? formDrift.relative / controlDrift.relative
                        : 0
                )
            )
            let worst = formDrift.absolute
            let scale = oneThread.lastLogits.map(abs).max() ?? 1
            // The answer-level requirement, checked at every sampled position: a
            // rounding-level difference must not change a decision. The drift itself is
            // expected to be larger than the 32-vs-64 control's, because the one-thread
            // form sums 160 blocks in ONE dependent chain where a threadgroup sums a
            // shallow tree — that is the old form's error, not the new one's.
            for position in sampledPositions.sorted() {
                guard let a = oneThread.sampledLogits[position],
                      let b = coalesced.sampledLogits[position] else { continue }
                let argmaxA = a.enumerated().max { $0.element < $1.element }?.offset
                let argmaxB = b.enumerated().max { $0.element < $1.element }?.offset
                XCTAssertEqual(
                    argmaxA, argmaxB,
                    "the greedy token at position \(position) changed with the kernel form"
                )
            }
            // Where the drift sits along the stream: it is not monotone, which is what
            // amplification of a 1e-7 stage difference looks like rather than a bias.
            for position in sampledPositions.sorted() {
                guard let a = oneThread.sampledLogits[position],
                      let b = coalesced.sampledLogits[position] else { continue }
                let localScale = max(1, a.map(abs).max() ?? 1)
                let delta = zip(a, b).map { abs($0 - $1) }.max() ?? 0
                print(
                    String(
                        format: "🔎 [dsv41 real]   drift at position %d: %.2e absolute "
                            + "(%.2e relative of %.2f)",
                        position,
                        delta,
                        delta / localScale,
                        localScale
                    )
                )
            }

            // The kernel's own arithmetic on real activations. The two arms' logits
            // have been through forty layers each, and a rounding-level difference is
            // amplified by that; the shared expert's fused output at layer 0 depends on
            // nothing but the normed input and its own weights, so the two forms are
            // compared THERE, once, on the same position.
            let hidden = config.hiddenSize
            let sharedWidth = executor.shapes.sharedIntermediate
            guard let device = MTLCreateSystemDefaultDevice() else {
                XCTFail("no Metal device for the feed-forward probes")
                return
            }
            guard let normedProbe = device.makeBuffer(
                      length: hidden * MemoryLayout<Float>.stride,
                      options: .storageModeShared
                  ),
                  let sharedProbe = device.makeBuffer(
                      length: sharedWidth * MemoryLayout<Float>.stride,
                      options: .storageModeShared
                  ) else {
                XCTFail("the feed-forward probes did not allocate")
                return
            }
            func capturedSharedOutput(oneThread: Bool) -> (normed: [Float], shared: [Float])? {
                executor.reset()
                executor.disablesFP8FusedRows = oneThread
                executor.parityProbes = DeepSeekV41ParityProbes(
                    ffnNormed: normedProbe,
                    sharedIntermediate: sharedProbe,
                    sharedExpertCaptureLayer: 0
                )
                do {
                    _ = try executor.forwardToken(tokens[0], position: 0)
                } catch {
                    XCTFail("the capture walk refused: \(error)")
                    return nil
                }
                executor.disablesFP8FusedRows = false
                executor.parityProbes = DeepSeekV41ParityProbes()
                return (
                    floats(normedProbe, count: hidden),
                    floats(sharedProbe, count: sharedWidth)
                )
            }
            func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
                let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: pointer, count: count))
            }
            let perThreadCapture = try XCTUnwrap(
                capturedSharedOutput(oneThread: true), "the one-thread capture did not run"
            )
            let coalescedCapture = try XCTUnwrap(
                capturedSharedOutput(oneThread: false), "the coalesced capture did not run"
            )
            do {
                let perThread = perThreadCapture
                let coalescedOutput = coalescedCapture
                func worst(_ a: [Float], _ b: [Float]) -> (absolute: Float, relative: Double) {
                    let scale = max(a.map(abs).max() ?? 1, 1e-9)
                    let delta = zip(a, b).map { abs($0 - $1) }.max() ?? 0
                    return (delta, Double(delta / scale))
                }
                let inputDrift = worst(perThread.normed, coalescedOutput.normed)
                let outputDrift = worst(perThread.shared, coalescedOutput.shared)
                print(
                    String(
                        format: "🔎 [dsv41 real] layer-0 shared expert on real activations: "
                            + "input drift %.2e (%.2e relative, %.2f scale), gate/up drift "
                            + "%.2e (%.2e relative, %.2f scale)",
                        inputDrift.absolute,
                        inputDrift.relative,
                        perThread.normed.map(abs).max() ?? 0,
                        outputDrift.absolute,
                        outputDrift.relative,
                        perThread.shared.map(abs).max() ?? 0
                    )
                )
                XCTAssertEqual(
                    inputDrift.absolute, 0,
                    "the two arms must feed the shared expert the same normed input"
                )
                XCTAssertLessThan(
                    outputDrift.relative, 1e-5,
                    "the coalesced fused kernel computes a different gate/up output"
                )
            }

            XCTAssertEqual(oneThread.timing.positions, positions)
            XCTAssertEqual(coalesced.timing.positions, positions)
            XCTAssertGreaterThan(oneThread.sharedExpertPerPosition, 0)
            XCTAssertGreaterThan(coalesced.sharedExpertPerPosition, 0)

            let speedup = oneThread.sharedExpertPerPosition / coalesced.sharedExpertPerPosition
            let savedPerPosition = oneThread.sharedExpertPerPosition
                - coalesced.sharedExpertPerPosition
            print(
                String(
                    format: "🔎 [dsv41 real] sharedExpert %.3f -> %.3f ms/position "
                        + "(%.2fx, %.3f ms saved per position; %.1f%% of the baseline's "
                        + "device time)",
                    oneThread.sharedExpertPerPosition,
                    coalesced.sharedExpertPerPosition,
                    speedup,
                    savedPerPosition,
                    savedPerPosition
                        / (Double(oneThread.timing.frontDeviceNanoseconds
                            + oneThread.timing.mixtureDeviceNanoseconds)
                            / Double(positions) / 1_000_000) * 100
                )
            )

            for (label, outcome) in [("one-thread", oneThread), ("coalesced", coalesced)] {
                let frontNames = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
                let denseTotal = outcome.table
                    .filter { frontNames.contains($0.stage) }
                    .reduce(0) { $0 + $1.millisecondsPerPosition }
                let mixtureTotal = outcome.table
                    .filter { !frontNames.contains($0.stage) }
                    .reduce(0) { $0 + $1.millisecondsPerPosition }
                print(
                    String(
                        format: "🔎 [dsv41 real] %@ stage table: %.2f dense, %.2f mixture, "
                            + "%d stage(s); top five %@",
                        label,
                        denseTotal,
                        mixtureTotal,
                        outcome.table.count,
                        outcome.table.prefix(5)
                            .map { String(format: "%@ %.2f", $0.stage, $0.millisecondsPerPosition) }
                            .joined(separator: ", ") as NSString
                    )
                )
            }
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// What the mHC coefficient pair's two dispatches are worth on a real decode.
    ///
    /// `#60` attributed a walked position and left two stages far below the machine's
    /// measured GEMV ceiling: `hcAttentionCoefficients` 4.09 ms/position and
    /// `hcFFNCoefficients` 3.37, i.e. 102 us per layer for a 1.97 MB FP32 weight read,
    /// **19 GB/s** where the dense GEMVs run at 332-397. `#61` then closed the stage
    /// ranked above them, so the pair was the largest item left that was not at the
    /// ceiling — and it is two dispatches, not one: the projection `#9` had already
    /// given a threadgroup per (slot, logit), and the split, which is ONE thread per
    /// position running the schedule's twenty Sinkhorn iterations.
    ///
    /// The probe prices each half on its own, because "the stage is slow" is not a
    /// diagnosis:
    ///
    /// 1. **the projection** — `#9` swept the threadgroup width at a BLOCK's
    ///    occupancy (192 threadgroups, where 256 lanes wins). A walk's grid is 24, and
    ///    `testWhatOneHyperConnectionProjectionCostsAtThePublishedWidths` measures that
    ///    grid at 0.022 ms with one dependent accumulator pair and 0.006 with four; so
    ///    the arms here are the one-chain kernel and the four-chain one at 256, 512 and
    ///    the plan's width;
    /// 2. **the split** — the Sinkhorn's 4x4 matrix lives in DEVICE memory, so every
    ///    one of ~160 normalization passes is a chain of dependent loads and stores on
    ///    one thread. The register-resident form is A/B'd against it, and the
    ///    `Sinkhorn 1` / `10 iteration` arms price one iteration of the chain;
    /// 3. **the fixed part** — `projection x2` / `x4` and `split x2` repeat a dispatch
    ///    (idempotent, so the graph's answer cannot move) and the slope IS that
    ///    dispatch's in-situ cost. An isolated kernel table cannot give this: in a walk
    ///    the read competes with everything else a position streams.
    ///
    /// The last arms run the PRODUCTION layout — one command buffer per half, no
    /// per-stage windows — because a per-stage buffer holds one dispatch and the GPU
    /// may be clocked for that dispatch alone; `frontDevice` is then the dense half as
    /// a turn pays it.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_hcwalk_stage`, plus the class's own marker).
    func testWhatTheHyperConnectionProjectionAndSplitAreWorthToAWalk() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_hcwalk_stage"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in mHC probe; touch \(markerPath) to run"
        )

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
        // The 128-position stream `#52`-`#61` measured, so the tables line up.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)
        let sampledPositions: Set<Int> = [0, 1, 2, 3, 7, 15, 31, 63, 127]

        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let lastLogits: [Float]
            let table: [(stage: String, millisecondsPerPosition: Double, share: Double)]
            let attentionCoefficients: Double
            let feedForwardCoefficients: Double
            let sampledLogits: [Int: [Float]]
        }

        func pass(
            _ label: String,
            executor: DeepSeekV41MetalForward,
            arena: DeepSeekV41ExpertArena,
            streamer: DeepSeekV41ExpertStreamer?,
            lanes: Int,
            oneChain: Bool = false,
            splitRegisters: Bool = true,
            projectionRepeats: Int = 1,
            splitRepeats: Int = 1,
            splitLanes: Int = 0,
            sinkhornIterations: Int? = nil,
            staged: Bool = true
        ) -> Outcome? {
            executor.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            executor.recordsWalkStageTiming = staged
            executor.hyperConnectionLaneWidthOverride = lanes
            executor.disablesHyperConnectionX4 = oneChain
            executor.disablesHyperConnectionSplitRegisters = !splitRegisters
            executor.hyperConnectionProjectionRepeats = projectionRepeats
            executor.hyperConnectionSplitRepeats = splitRepeats
            executor.hyperConnectionSplitLanesOverride = splitLanes
            executor.hyperConnectionSinkhornIterationsOverride = sinkhornIterations
            let bytesBefore = streamer?.bytesRead ?? 0
            let start = DispatchTime.now().uptimeNanoseconds
            var logits: [Float] = []
            var sampled: [Int: [Float]] = [:]
            do {
                for (position, token) in tokens.enumerated() {
                    logits = try executor.forwardToken(token, position: position)
                    if sampledPositions.contains(position) { sampled[position] = logits }
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
                return nil
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            executor.recordsWalkStageTiming = false
            executor.hyperConnectionLaneWidthOverride = 0
            executor.disablesHyperConnectionX4 = false
            executor.disablesHyperConnectionSplitRegisters = false
            executor.hyperConnectionProjectionRepeats = 1
            executor.hyperConnectionSplitRepeats = 1
            executor.hyperConnectionSplitLanesOverride = 0
            executor.hyperConnectionSinkhornIterationsOverride = nil
            let table = executor.walkStageTable()
            func window(_ stage: DeepSeekV41ForwardStages.Stage) -> Double {
                table.first { $0.stage == stage.rawValue }?.millisecondsPerPosition ?? -1
            }
            let device = executor.walkTiming.frontDeviceNanoseconds
                + executor.walkTiming.mixtureDeviceNanoseconds
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f ms/position, "
                        + "hcAttn %.3f + hcFFN %.3f = %.3f ms/position, %.1f GB read",
                    label,
                    wall / Double(positions),
                    Double(device) / Double(positions) / 1_000_000,
                    window(.hcAttentionCoefficients),
                    window(.hcFFNCoefficients),
                    window(.hcAttentionCoefficients) + window(.hcFFNCoefficients),
                    Double((streamer?.bytesRead ?? 0) - bytesBefore) / 1_073_741_824
                )
            )
            return Outcome(
                wallMilliseconds: wall,
                timing: executor.walkTiming,
                lastLogits: logits,
                table: table,
                attentionCoefficients: window(.hcAttentionCoefficients),
                feedForwardCoefficients: window(.hcFFNCoefficients),
                sampledLogits: sampled
            )
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            // A discarded warm-up, as `#59`-`#61`: the first pass pays slot creation and
            // wiring, and the measured arms are then warm and are not reset between.
            executor.reset()
            runtime.arena.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            let mixCount = (2 + config.hcMult) * config.hcMult
            print(
                String(
                    format: "🔎 [dsv41 real] mHC probe: %d positions, %d lanes at a walk, "
                        + "%d mix logits, warm-up done",
                    positions,
                    DeepSeekV41DevicePlan.hyperConnectionLaneWidth(slots: 1),
                    mixCount
                )
            )

            let baseline = try XCTUnwrap(
                pass(
                    "one-chain kernel, 256 lanes (what shipped)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 256,
                    oneChain: true,
                    splitRegisters: false
                )
            )
            let projectionOnly = try XCTUnwrap(
                pass(
                    "four-chain projection, device-memory split",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 256,
                    splitRegisters: false
                )
            )
            let splitOnly = try XCTUnwrap(
                pass(
                    "one-chain projection, register split",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 256,
                    oneChain: true
                )
            )
            let control = try XCTUnwrap(
                pass(
                    "four-chain kernel, 512 lanes (control width)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 512
                )
            )
            let shipped = try XCTUnwrap(
                pass(
                    "four-chain kernel, \(DeepSeekV41DevicePlan.hyperConnectionLaneWidth(slots: 1)) lanes (shipped)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0
                )
            )
            // The in-situ slope of each half of the stage. An isolated kernel table
            // cannot say this: in a walk the same dispatch reads weights that nothing
            // else just touched, while the surrounding position streams 35 GB.
            let projectionTwice = try XCTUnwrap(
                pass(
                    "shipped, projection dispatched twice",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    projectionRepeats: 2
                )
            )
            let projectionFourTimes = try XCTUnwrap(
                pass(
                    "shipped, projection dispatched four times",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    projectionRepeats: 4
                )
            )
            let splitTwice = try XCTUnwrap(
                pass(
                    "shipped, split dispatched twice",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    splitRepeats: 2
                )
            )
            // Why a one-thread dispatch costs what it does: a wider threadgroup (the
            // kernel returns at once on the lanes past `slotCount`), and a shorter
            // Sinkhorn chain. Neither arm's answer is compared; they price the body.
            let splitWide = try XCTUnwrap(
                pass(
                    "shipped, split at 32 lanes",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    splitLanes: 32
                )
            )
            let sinkhornOne = try XCTUnwrap(
                pass(
                    "shipped, Sinkhorn 1 iteration",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    sinkhornIterations: 1
                )
            )
            let sinkhornTen = try XCTUnwrap(
                pass(
                    "shipped, Sinkhorn 10 iterations",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    sinkhornIterations: 10
                )
            )

            func worstRelative(
                _ lhs: [Float], _ rhs: [Float]
            ) -> (absolute: Float, relative: Double) {
                let scale = max(lhs.map(abs).max() ?? 1, 1)
                let worst = zip(lhs, rhs).map { abs($0 - $1) }.max() ?? 0
                return (worst, Double(worst / scale))
            }
            for position in sampledPositions.sorted() {
                guard let a = baseline.sampledLogits[position],
                      let b = shipped.sampledLogits[position] else { continue }
                let argmaxA = a.enumerated().max { $0.element < $1.element }?.offset
                let argmaxB = b.enumerated().max { $0.element < $1.element }?.offset
                XCTAssertEqual(
                    argmaxA, argmaxB,
                    "the greedy token at position \(position) changed with the kernel form"
                )
            }
            let drift = worstRelative(baseline.lastLogits, shipped.lastLogits)
            let controlDrift = worstRelative(control.lastLogits, shipped.lastLogits)
            print(
                String(
                    format: "🔎 [dsv41 real] answer control: one-chain vs shipped %.2e absolute "
                        + "(%.2e relative), 512 vs shipped %.2e absolute (%.2e relative), "
                        + "ratio %.1fx",
                    drift.absolute,
                    drift.relative,
                    controlDrift.absolute,
                    controlDrift.relative,
                    controlDrift.relative > 0 ? drift.relative / controlDrift.relative : 0
                )
            )

            // The projection's OWN arithmetic at layer 0, where a lane-width change can
            // be read before forty layers amplify it.
            guard let device = MTLCreateSystemDefaultDevice() else {
                XCTFail("no Metal device for the coefficient probes")
                return
            }
            guard let attentionProbe = device.makeBuffer(
                      length: mixCount * MemoryLayout<Float>.stride,
                      options: .storageModeShared
                  ),
                  let feedForwardProbe = device.makeBuffer(
                      length: mixCount * MemoryLayout<Float>.stride,
                      options: .storageModeShared
                  ) else {
                XCTFail("the coefficient probes did not allocate")
                return
            }
            func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
                let pointer = buffer.contents().assumingMemoryBound(to: Float.self)
                return Array(UnsafeBufferPointer(start: pointer, count: count))
            }
            func capturedCoefficients(
                lanes: Int, oneChain: Bool
            ) -> ([Float], [Float])? {
                executor.reset()
                executor.hyperConnectionLaneWidthOverride = lanes
                executor.disablesHyperConnectionX4 = oneChain
                executor.parityProbes = DeepSeekV41ParityProbes(
                    attentionLogits: attentionProbe,
                    feedForwardLogits: feedForwardProbe,
                    hyperConnectionCaptureLayer: 0
                )
                do {
                    _ = try executor.forwardToken(tokens[0], position: 0)
                } catch {
                    XCTFail("the coefficient capture refused: \(error)")
                    return nil
                }
                executor.hyperConnectionLaneWidthOverride = 0
                executor.disablesHyperConnectionX4 = false
                executor.parityProbes = DeepSeekV41ParityProbes()
                return (
                    floats(attentionProbe, count: mixCount),
                    floats(feedForwardProbe, count: mixCount)
                )
            }
            let baselineCapture = try XCTUnwrap(
                capturedCoefficients(lanes: 256, oneChain: true),
                "the one-chain capture did not run"
            )
            let shippedCapture = try XCTUnwrap(
                capturedCoefficients(lanes: 0, oneChain: false), "the shipped capture did not run"
            )
            do {
                let attention = worstRelative(baselineCapture.0, shippedCapture.0)
                let feedForward = worstRelative(baselineCapture.1, shippedCapture.1)
                print(
                    String(
                        format: "🔎 [dsv41 real] layer-0 coefficient logits: attention "
                            + "%.2e absolute (%.2e relative of %.2f), FFN %.2e (%.2e of %.2f)",
                        attention.absolute,
                        attention.relative,
                        baselineCapture.0.map(abs).max() ?? 0,
                        feedForward.absolute,
                        feedForward.relative,
                        baselineCapture.1.map(abs).max() ?? 0
                    )
                )
                XCTAssertLessThan(
                    attention.relative, 1e-5,
                    "the four-chain form computes a different attention coefficient"
                )
                XCTAssertLessThan(
                    feedForward.relative, 1e-5,
                    "the four-chain form computes a different FFN coefficient"
                )
            }

            XCTAssertEqual(baseline.timing.positions, positions)
            XCTAssertEqual(shipped.timing.positions, positions)
            XCTAssertGreaterThan(baseline.attentionCoefficients, 0)
            XCTAssertGreaterThan(shipped.attentionCoefficients, 0)
            let before = baseline.attentionCoefficients + baseline.feedForwardCoefficients
            let after = shipped.attentionCoefficients + shipped.feedForwardCoefficients
            print(
                String(
                    format: "🔎 [dsv41 real] mHC pair %.3f -> %.3f ms/position "
                        + "(%.2fx, %.3f ms saved; %.1f%% of the baseline's device time); "
                        + "projection alone (four chains, device-memory split) %.3f, "
                        + "register split alone (one chain) %.3f",
                    before,
                    after,
                    before / after,
                    before - after,
                    (before - after)
                        / (Double(baseline.timing.frontDeviceNanoseconds
                            + baseline.timing.mixtureDeviceNanoseconds)
                            / Double(positions) / 1_000_000) * 100,
                    projectionOnly.attentionCoefficients + projectionOnly.feedForwardCoefficients,
                    splitOnly.attentionCoefficients + splitOnly.feedForwardCoefficients
                )
            )
            func coefficients(_ outcome: Outcome) -> Double {
                outcome.attentionCoefficients + outcome.feedForwardCoefficients
            }
            // 80 dispatches per position: the projection and the split, once per
            // sub-layer, twice per layer, forty layers.
            let extraProjection = (coefficients(projectionTwice) - coefficients(shipped))
                / 80 * 1_000
            let extraProjectionFour = (coefficients(projectionFourTimes) - coefficients(shipped))
                / (3 * 80) * 1_000
            let extraSplit = (coefficients(splitTwice) - coefficients(shipped))
                / 80 * 1_000
            let oneIteration = (coefficients(sinkhornOne) - coefficients(shipped))
                / 80 * 1_000
            let tenIterations = (coefficients(sinkhornTen) - coefficients(shipped))
                / 80 * 1_000
            print(
                String(
                    format: "🔎 [dsv41 real] in-situ slope per extra dispatch (us): projection "
                        + "%.2f / %.2f, split %.2f, split at 32 lanes %.2f, Sinkhorn 1 iter "
                        + "%.2f, 10 iters %.2f; one stage window is %.2f us",
                    extraProjection,
                    extraProjectionFour,
                    extraSplit,
                    (coefficients(splitWide) - coefficients(shipped)) / 80 * 1_000,
                    oneIteration,
                    tenIterations,
                    coefficients(shipped) / 80 * 1_000
                )
            )
            // The same two knobs in the PRODUCTION layout: one command buffer per half,
            // no per-stage windows. A per-stage buffer holds one dispatch, so the GPU
            // may be clocked for that dispatch alone; a turn's buffer holds twenty and
            // the dense half's own time is what a decode pays. If the split's cost is
            // still here, it is the chain; if it is not, it is the instrument.
            let halfShipped = try XCTUnwrap(
                pass(
                    "half layout, shipped",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    staged: false
                )
            )
            let halfBaseline = try XCTUnwrap(
                pass(
                    "half layout, one-chain and device-memory split",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 256,
                    oneChain: true,
                    splitRegisters: false,
                    staged: false
                )
            )
            let halfSinkhornOne = try XCTUnwrap(
                pass(
                    "half layout, Sinkhorn 1 iteration",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    sinkhornIterations: 1,
                    staged: false
                )
            )
            let halfSplitTwice = try XCTUnwrap(
                pass(
                    "half layout, split dispatched twice",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    lanes: 0,
                    splitRepeats: 2,
                    staged: false
                )
            )
            func dense(_ outcome: Outcome) -> Double {
                Double(outcome.timing.frontDeviceNanoseconds)
                    / Double(positions) / 1_000_000
            }
            print(
                String(
                    format: "🔎 [dsv41 real] half layout dense half: %.3f ms/position as "
                        + "shipped, %.3f before #62 (%.3f ms won); Sinkhorn 1 %.3f "
                        + "(%.3f ms of the dense half is 19 iterations), split x2 %.3f "
                        + "(one split costs %.2f us in situ)",
                    dense(halfShipped),
                    dense(halfBaseline),
                    dense(halfBaseline) - dense(halfShipped),
                    dense(halfSinkhornOne),
                    dense(halfShipped) - dense(halfSinkhornOne),
                    dense(halfSplitTwice),
                    (dense(halfSplitTwice) - dense(halfShipped)) / 80 * 1_000
                )
            )
            for (label, outcome) in [
                ("one-chain, 256", baseline),
                ("four-chain projection only", projectionOnly),
                ("register split only", splitOnly),
                ("four-chain, 512", control),
                ("shipped", shipped),
                ("shipped, projection x2", projectionTwice),
                ("shipped, projection x4", projectionFourTimes),
                ("shipped, split x2", splitTwice),
                ("shipped, split 32 lanes", splitWide),
                ("shipped, Sinkhorn 1", sinkhornOne),
                ("shipped, Sinkhorn 10", sinkhornTen),
            ] {
                let frontNames = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
                let denseTotal = outcome.table
                    .filter { frontNames.contains($0.stage) }
                    .reduce(0) { $0 + $1.millisecondsPerPosition }
                let mixtureTotal = outcome.table
                    .filter { !frontNames.contains($0.stage) }
                    .reduce(0) { $0 + $1.millisecondsPerPosition }
                print(
                    String(
                        format: "🔎 [dsv41 real] %@ stage table: %.2f dense, %.2f mixture; "
                            + "top five %@",
                        label,
                        denseTotal,
                        mixtureTotal,
                        outcome.table.prefix(5)
                            .map { String(format: "%@ %.2f", $0.stage, $0.millisecondsPerPosition) }
                            .joined(separator: ", ") as NSString
                    )
                )
            }
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// What a router stage is made of on a walk, dispatch by dispatch.
    ///
    /// `#62` left `router` the last large stage of a walked position that is not at a
    /// bandwidth ceiling: 3.6 ms/position, i.e. 90 us per layer for a 3.93 MB BF16
    /// projection plus a top-k scan that is ONE thread over 384 experts. The stage is
    /// two dispatches and nothing had separated them, so this repeats each in place
    /// (both are idempotent — the projection rewrites the same logits and the scan
    /// rewrites the same indices — so the graph's answer cannot move) and takes the
    /// slope of the stage window. The half-layout arms repeat the exercise with one
    /// command buffer per half, so the result is not a per-stage-table artifact.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_router_stage`, plus the class's own marker).
    func testWhatARouterStageIsMadeOfOnAWalk() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_router_stage"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in router probe; touch \(markerPath) to run"
        )

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
        // The 128-position stream `#52`-`#62` measured.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)
        let sampledPositions: Set<Int> = [0, 1, 2, 3, 7, 15, 31, 63, 127]

        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let lastLogits: [Float]
            let table: [(stage: String, millisecondsPerPosition: Double, share: Double)]
            let routerPerPosition: Double
            let sampledLogits: [Int: [Float]]
        }

        func pass(
            _ label: String,
            executor: DeepSeekV41MetalForward,
            streamer: DeepSeekV41ExpertStreamer?,
            projectionRepeats: Int = 1,
            topKRepeats: Int = 1,
            laneTopK: Bool = true,
            staged: Bool = true
        ) -> Outcome? {
            executor.reset()
            executor.resetWalkTiming()
            executor.recordsWalkTiming = true
            executor.recordsWalkStageTiming = staged
            executor.routerProjectionRepeats = projectionRepeats
            executor.routerTopKRepeats = topKRepeats
            executor.disablesRouterTopKLanes = !laneTopK
            let bytesBefore = streamer?.bytesRead ?? 0
            let start = DispatchTime.now().uptimeNanoseconds
            var logits: [Float] = []
            var sampled: [Int: [Float]] = [:]
            do {
                for (position, token) in tokens.enumerated() {
                    logits = try executor.forwardToken(token, position: position)
                    if sampledPositions.contains(position) { sampled[position] = logits }
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
                return nil
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            executor.recordsWalkStageTiming = false
            executor.routerProjectionRepeats = 1
            executor.routerTopKRepeats = 1
            executor.disablesRouterTopKLanes = false
            let table = executor.walkStageTable()
            let router = table.first {
                $0.stage == DeepSeekV41ForwardStages.Stage.router.rawValue
            }?.millisecondsPerPosition ?? -1
            let device = executor.walkTiming.frontDeviceNanoseconds
                + executor.walkTiming.mixtureDeviceNanoseconds
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f ms/position, "
                        + "router %.3f ms/position (%.2f us/layer), %.1f GB read",
                    label,
                    wall / Double(positions),
                    Double(device) / Double(positions) / 1_000_000,
                    router,
                    router / 40 * 1_000,
                    Double((streamer?.bytesRead ?? 0) - bytesBefore) / 1_073_741_824
                )
            )
            return Outcome(
                wallMilliseconds: wall,
                timing: executor.walkTiming,
                lastLogits: logits,
                table: table,
                routerPerPosition: router,
                sampledLogits: sampled
            )
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            executor.reset()
            runtime.arena.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            print(
                "🔎 [dsv41 real] router probe: \(positions) positions, "
                    + "\(config.numRoutedExperts) experts over \(config.hiddenSize) wide, warm-up done"
            )

            let baseline = try XCTUnwrap(
                pass(
                    "one-thread top-k (what shipped)", executor: executor,
                    streamer: runtime.expertStreamer, laneTopK: false
                )
            )
            let shipped = try XCTUnwrap(
                pass("lane top-k (shipped)", executor: executor, streamer: runtime.expertStreamer)
            )
            let projectionTwice = try XCTUnwrap(
                pass(
                    "projection x2", executor: executor, streamer: runtime.expertStreamer,
                    projectionRepeats: 2
                )
            )
            let projectionFour = try XCTUnwrap(
                pass(
                    "projection x4", executor: executor, streamer: runtime.expertStreamer,
                    projectionRepeats: 4
                )
            )
            let topKTwice = try XCTUnwrap(
                pass(
                    "top-k x2", executor: executor, streamer: runtime.expertStreamer,
                    topKRepeats: 2
                )
            )
            let halfShipped = try XCTUnwrap(
                pass(
                    "half layout, shipped", executor: executor, streamer: runtime.expertStreamer,
                    staged: false
                )
            )
            let halfBaseline = try XCTUnwrap(
                pass(
                    "half layout, one-thread top-k", executor: executor,
                    streamer: runtime.expertStreamer, laneTopK: false, staged: false
                )
            )
            let halfProjection = try XCTUnwrap(
                pass(
                    "half layout, projection x2", executor: executor,
                    streamer: runtime.expertStreamer, projectionRepeats: 2, staged: false
                )
            )
            let halfTopK = try XCTUnwrap(
                pass(
                    "half layout, top-k x2", executor: executor, streamer: runtime.expertStreamer,
                    topKRepeats: 2, staged: false
                )
            )

            // The repeats are idempotent and the lane form is exact, so the answer may
            // not move AT ALL: any drift means either a repeat is not repeating what
            // the stage does, or the merge's total order differs from the scan's.
            for (label, outcome) in [
                ("projection x2", projectionTwice), ("top-k x2", topKTwice),
                ("half layout", halfShipped), ("one-thread top-k", baseline),
            ] {
                XCTAssertEqual(
                    outcome.lastLogits, shipped.lastLogits,
                    "\(label) changed the answer"
                )
            }
            for position in sampledPositions.sorted() {
                guard let a = shipped.sampledLogits[position],
                      let b = projectionFour.sampledLogits[position] else { continue }
                XCTAssertEqual(
                    a, b, "the projection repeats changed the answer at position \(position)"
                )
            }

            let projectionSlope = (projectionTwice.routerPerPosition
                - shipped.routerPerPosition) / 40 * 1_000
            let projectionSlopeFour = (projectionFour.routerPerPosition
                - shipped.routerPerPosition) / (3 * 40) * 1_000
            let topKSlope = (topKTwice.routerPerPosition - shipped.routerPerPosition)
                / 40 * 1_000
            print(
                String(
                    format: "🔎 [dsv41 real] router stage %.3f -> %.3f ms/position "
                        + "(%.2fx, %.3f ms saved); per layer %.2f -> %.2f us; "
                        + "in-situ slope per extra dispatch: projection %.2f / %.2f us, "
                        + "top-k (lane form) %.2f us",
                    baseline.routerPerPosition,
                    shipped.routerPerPosition,
                    baseline.routerPerPosition / shipped.routerPerPosition,
                    baseline.routerPerPosition - shipped.routerPerPosition,
                    baseline.routerPerPosition / 40 * 1_000,
                    shipped.routerPerPosition / 40 * 1_000,
                    projectionSlope,
                    projectionSlopeFour,
                    topKSlope
                )
            )
            func dense(_ outcome: Outcome) -> Double {
                Double(outcome.timing.frontDeviceNanoseconds)
                    / Double(positions) / 1_000_000
            }
            print(
                String(
                    format: "🔎 [dsv41 real] half layout dense half: %.3f ms/position as "
                        + "shipped, %.3f before #63 (%.3f ms won); projection x2 %.3f "
                        + "(one costs %.2f us), top-k x2 %.3f (one costs %.2f us)",
                    dense(halfShipped),
                    dense(halfBaseline),
                    dense(halfBaseline) - dense(halfShipped),
                    dense(halfProjection),
                    (dense(halfProjection) - dense(halfShipped)) / 40 * 1_000,
                    dense(halfTopK),
                    (dense(halfTopK) - dense(halfShipped)) / 40 * 1_000
                )
            )
            for (label, outcome) in [
                ("one-thread top-k", baseline), ("shipped", shipped),
                ("projection x2", projectionTwice), ("projection x4", projectionFour),
                ("top-k x2", topKTwice),
            ] {
                let frontNames = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
                let denseTotal = outcome.table
                    .filter { frontNames.contains($0.stage) }
                    .reduce(0) { $0 + $1.millisecondsPerPosition }
                print(
                    String(
                        format: "🔎 [dsv41 real] %@ stage table: %.2f dense; top five %@",
                        label,
                        denseTotal,
                        outcome.table.prefix(5)
                            .map { String(format: "%@ %.2f", $0.stage, $0.millisecondsPerPosition) }
                            .joined(separator: ", ") as NSString
                    )
                )
            }
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// What an event-gated expert half is worth to a walked position: NOTHING.
    ///
    /// `#64` split the indexer's readback into its GPU drain and the host's own work and
    /// found the host's part is nothing, which left the walk's structure as the last
    /// lever: every layer stops the queue for the routing readback, then streams the
    /// experts, and only then hands the GPU the half that reads them. The GPU is idle for
    /// the whole fill, and when the half is committed the queue has been empty long
    /// enough that the buffer's window opens ~0.3 ms later
    /// (`mixtureLateStartNanoseconds`, `#22` and `#50`) — 12 ms of a warm position.
    ///
    /// The one change that could move that under the fill is a DEFERRED BIND: the expert
    /// half binds only what the arena PLACED, so it can be encoded and committed before
    /// the bytes exist, gated on a shared event the read signals. `#65` built it, and
    /// this probe is why it is not shipped. At MATCHED read volume — the second control
    /// and the second gated arm read the same 1,559 misses and the same 27.3 GB, because
    /// they run back to back on one arena — the wall is the same within noise and
    /// `mixtureLateStartNanoseconds` is the same, so the gate neither removes the pickup
    /// nor moves it. A gated buffer does not park on its event; it resumes through the
    /// same driver path a commit uses, which is what
    /// `DeepSeekV41CommandStartCostTests` measures from the other side.
    ///
    /// The arms are a rotation (control, gated, gated, control) because the page cache
    /// and the arena's frequency counts are both still converging over the first pass, so
    /// a gated arm placed second would read faster for reasons the gate did not cause.
    /// Comparing the two SECOND arms is the whole control: arm 1 of either kind pays the
    /// cold fill (35.0 GB against 27.3) and is not comparable with anything.
    ///
    /// The gate changes no arithmetic, so the answer is asserted BIT-identical across
    /// every arm and every sampled position: a scheduling change that moves a logit is a
    /// bug, not a win. That assertion is what caught the first cut of the gate encoding
    /// its wait AFTER the kernels it was supposed to hold.
    func testWhatAGatedMixtureCommitSavesOnAWalk() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_gatewalk"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in gate probe; touch \(markerPath) to run"
        )

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
        // The 128-position stream every assembly from `#52` on measured.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)
        let sampledPositions: Set<Int> = [0, 1, 2, 3, 7, 15, 31, 63, 127]

        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let lastLogits: [Float]
            let sampledLogits: [Int: [Float]]
            let misses: Int
            let bytes: Int
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            // The shipping rule, as every probe from `#59` on has used.
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            func pass(_ label: String, gated: Bool) -> Outcome? {
                executor.reset()
                executor.resetWalkTiming()
                executor.gatesMixtureOnExpertFill = gated
                executor.recordsWalkTiming = true
                executor.recordsWalkStageTiming = false
                let acquisitionsBefore = runtime.arena.acquisitions
                let hitsBefore = runtime.arena.hits
                let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
                let start = DispatchTime.now().uptimeNanoseconds
                var logits: [Float] = []
                var sampled: [Int: [Float]] = [:]
                do {
                    for (position, token) in tokens.enumerated() {
                        logits = try executor.forwardToken(token, position: position)
                        if sampledPositions.contains(position) { sampled[position] = logits }
                    }
                } catch {
                    XCTFail("\(label) refused at a position: \(error)")
                    return nil
                }
                let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
                executor.recordsWalkTiming = false
                let acquisitions = runtime.arena.acquisitions - acquisitionsBefore
                let misses = acquisitions - (runtime.arena.hits - hitsBefore)
                let bytes = (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
                let timing = executor.walkTiming
                let per = timing.millisecondsPerPosition()
                let divisor = Double(max(1, positions)) * 1_000_000
                print(
                    String(
                        format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f, "
                            + "%d miss(es), %.1f GB read; front wait %.1f, fill %.1f, "
                            + "mixture encode %.1f; drained mixture %.1f, late start "
                            + "mixture %.1f, late start front %.1f",
                        label,
                        wall / Double(positions),
                        per.frontDevice + per.mixtureDevice,
                        misses,
                        Double(bytes) / 1_073_741_824,
                        per.frontWait,
                        per.expertFill,
                        per.mixtureEncode,
                        Double(timing.mixtureDrainNanoseconds) / divisor,
                        Double(timing.mixtureLateStartNanoseconds) / divisor,
                        Double(timing.frontLateStartNanoseconds) / divisor
                    )
                )
                return Outcome(
                    wallMilliseconds: wall,
                    timing: timing,
                    lastLogits: logits,
                    sampledLogits: sampled,
                    misses: misses,
                    bytes: bytes
                )
            }

            // A discarded warm-up: slot creation, slot wiring and a cold page cache all
            // belong to the process, not to either arm.
            executor.reset()
            runtime.arena.reset()
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
            print(
                String(
                    format: "🔎 [dsv41 real] gate probe: %d positions, arena %d slots, "
                        + "warm-up done",
                    positions,
                    runtime.arena.capacity
                )
            )

            // The rotation: the two controls bracket the two gated arms, so a warming
            // cache shows up as a control-versus-control difference instead of as a win.
            let controlFirst = try XCTUnwrap(
                pass("control (commit after the fill), arm 1", gated: false)
            )
            let gatedFirst = try XCTUnwrap(
                pass("gated (commit before the fill), arm 1", gated: true)
            )
            let gatedSecond = try XCTUnwrap(
                pass("gated (commit before the fill), arm 2", gated: true)
            )
            let controlSecond = try XCTUnwrap(
                pass("control (commit after the fill), arm 2", gated: false)
            )

            // The gate moves WHEN the half is submitted, not what it computes, so the
            // answer has to be identical to the bit at every sampled position.
            for (label, outcome) in [
                ("gated arm 1", gatedFirst), ("gated arm 2", gatedSecond),
                ("control arm 2", controlSecond),
            ] {
                XCTAssertEqual(
                    outcome.lastLogits, controlFirst.lastLogits,
                    "\(label) moved the last position's logits"
                )
                XCTAssertEqual(
                    Set(outcome.sampledLogits.keys), Set(controlFirst.sampledLogits.keys),
                    "\(label) sampled a different set of positions"
                )
                for (position, logits) in controlFirst.sampledLogits {
                    XCTAssertEqual(
                        outcome.sampledLogits[position], logits,
                        "\(label) moved position \(position)'s logits"
                    )
                }
            }
            print(
                "🔎 [dsv41 real] gate control: every sampled position's logits are "
                    + "bit-identical across all four arms"
            )

            // The instrument has to have covered the whole stream, or the wall it
            // reports is not this stream's.
            for (label, outcome) in [
                ("control arm 1", controlFirst), ("gated arm 1", gatedFirst),
                ("gated arm 2", gatedSecond), ("control arm 2", controlSecond),
            ] {
                XCTAssertEqual(outcome.timing.positions, positions, "\(label) lost positions")
                XCTAssertEqual(
                    outcome.timing.frontUnwindowedBuffers
                        + outcome.timing.mixtureUnwindowedBuffers,
                    0,
                    "\(label) reported a half with no device window"
                )
            }

            func perPosition(_ nanoseconds: UInt64) -> Double {
                Double(nanoseconds) / Double(positions) / 1_000_000
            }
            // The pairing is the SECOND arms: they run back to back, so they share a page
            // cache and an arena state and read the same bytes. The FIRST arms are the
            // cold pair and differ by the whole fill (35.0 GB against 27.3), which is why
            // an average over all four would report a win the gate did not earn.
            let wallCold = controlFirst.wallMilliseconds / Double(positions)
            let wallWarmControl = controlSecond.wallMilliseconds / Double(positions)
            let wallWarmGated = gatedSecond.wallMilliseconds / Double(positions)
            let wallColdGated = gatedFirst.wallMilliseconds / Double(positions)
            let lateWarmControl = perPosition(controlSecond.timing.mixtureLateStartNanoseconds)
            let lateWarmGated = perPosition(gatedSecond.timing.mixtureLateStartNanoseconds)
            let deviceWarmControl = perPosition(
                controlSecond.timing.frontDeviceNanoseconds
                    + controlSecond.timing.mixtureDeviceNanoseconds
            )
            let deviceWarmGated = perPosition(
                gatedSecond.timing.frontDeviceNanoseconds
                    + gatedSecond.timing.mixtureDeviceNanoseconds
            )
            print(
                String(
                    format: "🔎 [dsv41 real] gate result: at MATCHED read (%d against %d "
                        + "miss(es), %.1f against %.1f GB) wall %.1f against %.1f "
                        + "ms/position (%.1f of difference), mixture late start %.1f "
                        + "against %.1f, device %.1f against %.1f; the cold arms read "
                        + "%.1f and %.1f GB and their walls are %.1f and %.1f",
                    controlSecond.misses,
                    gatedSecond.misses,
                    Double(controlSecond.bytes) / 1_073_741_824,
                    Double(gatedSecond.bytes) / 1_073_741_824,
                    wallWarmControl,
                    wallWarmGated,
                    wallWarmGated - wallWarmControl,
                    lateWarmControl,
                    lateWarmGated,
                    deviceWarmControl,
                    deviceWarmGated,
                    Double(controlFirst.bytes) / 1_073_741_824,
                    Double(gatedFirst.bytes) / 1_073_741_824,
                    wallCold,
                    wallColdGated
                )
            )
            // The finding, stated as a bound rather than as a coincidence: if the gate
            // moved the pickup under the fill at all, the warm pair would differ by more
            // than the ~1 ms that separates two arms reading the same bytes.
            XCTAssertLessThan(
                abs(wallWarmGated - wallWarmControl), 2.0,
                "the gate changed the wall at matched read volume"
            )
            XCTAssertEqual(controlSecond.misses, gatedSecond.misses, "the arms did not read alike")
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// How many dispatches a walked position issues, and what one costs in situ.
    ///
    /// `#65` closed the deferred-bind expert half and left the next lever as "fuse
    /// kernels rather than reschedule them", with a ~1,080 dispatches/position estimate
    /// (27 pipeline stages x 40 layers) and a driver-level ~3.5 us per dispatch of
    /// WINDOW growth. Both are inferences, and the second was taken in the least
    /// favourable conditions for it: a buffer of trivial dispatches committed into an
    /// EMPTY queue, where nothing else can hide the cost. In the graph a dispatch shares
    /// its buffer with real work the GPU is already saturated by, so the marginal
    /// dispatch may be nearly free — and if it is, no fusion is worth writing.
    ///
    /// So both are measured here, on the same 128-position stream `#52`-`#65` used:
    ///
    /// 1. **The count** — `countsDispatches` counts what the encoder actually issues, by
    ///    stage. A stage is not a dispatch: `.routedExperts` issues a clear plus two per
    ///    active expert (13 at six), `.sharedExpert` two, `.router` two, the mHC
    ///    coefficient stage two, and `.outputDown` one per block-diagonal group.
    /// 2. **The marginal cost** — `injectedDispatchRepeats` adds K clears of a 4-float
    ///    scratch buffer to every stage's own command buffer, so an injected arm's count
    ///    is the control's plus exactly 27 x layers x K, and the slope of the wall (and
    ///    of the device window) IS what removing a dispatch would save.
    ///
    /// The injected arms have to return bit-identical logits: the extra dispatches write
    /// a buffer nothing reads. A second control runs last to say whether the wall drifted
    /// between the arms, and a stage-split arm carries the same census so each stage's
    /// dispatches can be read beside the window `#60` ranked.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_dispatch_census`, plus the class's own marker).
    func testWhatAWalkedPositionsDispatchesCost() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_dispatch_census"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in dispatch census probe; touch \(markerPath) to run"
        )

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
        // The stream `#52`-`#65` measured, position for position.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        func prompt(_ length: Int, rotation: Int) -> [Int] {
            (0..<length).map { index in
                let repetition = index / stream.count
                return stream[(index + repetition * rotation) % stream.count]
            }
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let positions = 128
        let tokens = prompt(positions, rotation: 7)
        let stageCount = DeepSeekV41DevicePlan.frontHalf.count
            + DeepSeekV41DevicePlan.backHalf.count
        let injectedPerPositionPerRepeat = stageCount * config.numHiddenLayers

        /// One walked pass: what it cost, what it issued, and what it produced.
        struct Outcome {
            let wallMilliseconds: Double
            let timing: DeepSeekV41WalkTiming
            let census: [String: Int]
            let censusPositions: Int
            let stageSeconds: [String: UInt64]
            let misses: Int
            let bytes: Int
            let lastLogits: [Float]
        }

        func pass(
            _ label: String,
            executor: DeepSeekV41MetalForward,
            arena: DeepSeekV41ExpertArena,
            streamer: DeepSeekV41ExpertStreamer?,
            repeats: Int,
            counting: Bool,
            splittingStages: Bool = false
        ) -> Outcome? {
            executor.reset()
            executor.resetWalkTiming()
            executor.resetDispatchCensus()
            executor.recordsWalkTiming = true
            executor.recordsWalkStageTiming = splittingStages
            executor.countsDispatches = counting
            executor.injectedDispatchRepeats = repeats
            let acquisitionsBefore = arena.acquisitions
            let hitsBefore = arena.hits
            let bytesBefore = streamer?.bytesRead ?? 0
            let start = DispatchTime.now().uptimeNanoseconds
            var logits: [Float] = []
            do {
                for (position, token) in tokens.enumerated() {
                    logits = try executor.forwardToken(token, position: position)
                }
            } catch {
                XCTFail("\(label) refused at a position: \(error)")
                return nil
            }
            let wall = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            executor.recordsWalkTiming = false
            executor.recordsWalkStageTiming = false
            executor.countsDispatches = false
            executor.injectedDispatchRepeats = 0
            let acquisitions = arena.acquisitions - acquisitionsBefore
            let misses = acquisitions - (arena.hits - hitsBefore)
            let bytes = (streamer?.bytesRead ?? 0) - bytesBefore
            let issued = executor.dispatchCensus.values.reduce(0, +)
            let device = executor.walkTiming.frontDeviceNanoseconds
                + executor.walkTiming.mixtureDeviceNanoseconds
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, device %.1f, "
                        + "%d miss(es), %.1f GB read, %d dispatches (%@ per position)",
                    label,
                    wall / Double(positions),
                    Double(device) / Double(positions) / 1_000_000,
                    misses,
                    Double(bytes) / 1_073_741_824,
                    issued,
                    counting ? String(format: "%.1f", Double(issued) / Double(positions)) : "n/a"
                )
            )
            return Outcome(
                wallMilliseconds: wall,
                timing: executor.walkTiming,
                census: executor.dispatchCensus,
                censusPositions: executor.dispatchCensusPositions,
                stageSeconds: executor.walkStageTiming,
                misses: misses,
                bytes: bytes,
                lastLogits: logits
            )
        }

        do {
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: positions + 1,
                allowEngramSkip: false,
                wireArena: nil
            )
            runtime.arena.victimPolicy = .leastFrequentlyUsed
            let executor = runtime.forward()

            // Discarded warm-ups, three of them because that is what the arena's read
            // takes to converge: `#65` saw 35.0, 28.8, 27.3 and 27.3 GB over four arms,
            // so an arm that runs before the third is measuring the fill rather than a
            // dispatch. Everything below reads the same bytes, and the assertions at the
            // end are what hold that.
            for warmUp in 1...3 {
                _ = pass(
                    "warm-up \(warmUp)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 0,
                    counting: false
                )
            }
            print(
                String(
                    format: "🔎 [dsv41 real] dispatch census: %d positions, %d stage(s) per "
                        + "layer, %d layer(s), arena %d slots, warm-up done",
                    positions,
                    stageCount,
                    config.numHiddenLayers,
                    runtime.arena.capacity
                )
            )

            // The count first, with its stage windows: the census is the target list, and
            // the injected arms below are read against it.
            let stages = try XCTUnwrap(
                pass(
                    "stages, one buffer per stage",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 0,
                    counting: true,
                    splittingStages: true
                )
            )
            let control = try XCTUnwrap(
                pass(
                    "control",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 0,
                    counting: true
                )
            )
            let injectedOne = try XCTUnwrap(
                pass(
                    "injected 1/stage",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 1,
                    counting: true
                )
            )
            let injectedFour = try XCTUnwrap(
                pass(
                    "injected 4/stage",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 4,
                    counting: true
                )
            )
            let controlAgain = try XCTUnwrap(
                pass(
                    "control (repeat, census off)",
                    executor: executor,
                    arena: runtime.arena,
                    streamer: runtime.expertStreamer,
                    repeats: 0,
                    counting: false
                )
            )

            // An injected dispatch writes a scratch buffer nobody reads, so every arm
            // has to answer with the same logits, to the bit.
            for (label, outcome) in [
                ("injected 1/stage", injectedOne), ("injected 4/stage", injectedFour),
                ("control arm 2", controlAgain),
            ] {
                XCTAssertEqual(
                    outcome.lastLogits, control.lastLogits,
                    "\(label) moved the last position's logits"
                )
            }
            print(
                "🔎 [dsv41 real] dispatch control: every arm's last position is "
                    + "bit-identical"
            )

            // The instrument has to have covered the whole stream, and the injection
            // has to be exactly the count it claims: one clear per stage per layer.
            XCTAssertEqual(control.timing.positions, positions)
            XCTAssertEqual(control.censusPositions, positions)
            XCTAssertEqual(stages.censusPositions, positions)
            let controlTotal = control.census.values.reduce(0, +)
            let oneTotal = injectedOne.census.values.reduce(0, +)
            let fourTotal = injectedFour.census.values.reduce(0, +)
            XCTAssertGreaterThan(controlTotal, 0, "the census counted nothing")
            XCTAssertEqual(
                oneTotal - controlTotal,
                injectedPerPositionPerRepeat * positions,
                "the injected arm did not issue exactly one clear per stage per layer"
            )
            XCTAssertEqual(
                fourTotal - controlTotal,
                4 * injectedPerPositionPerRepeat * positions,
                "the injected arm did not issue exactly four clears per stage per layer"
            )
            // The split layout submits the same dispatches in more buffers.
            XCTAssertEqual(
                stages.census.values.reduce(0, +), controlTotal,
                "the per-stage layout changed the dispatch count"
            )

            let dispatchesPerPosition = Double(controlTotal) / Double(positions)
            print(
                String(
                    format: "🔎 [dsv41 real] dispatch count: %.1f dispatches per walked "
                        + "position over %d layer(s); the handover's 27 stages x 40 layers "
                        + "estimate was %d",
                    dispatchesPerPosition,
                    config.numHiddenLayers,
                    stageCount * config.numHiddenLayers
                )
            )
            // The marginal cost. Two injected arms and a control, so the slope does not
            // rest on one difference, and a second control that says whether the wall
            // drifted while the arms ran. The arms are matched-read (the warm-ups are
            // what make them so), which is the assertion below rather than an assumption.
            let wallControl = control.wallMilliseconds / Double(positions)
            let wallOne = injectedOne.wallMilliseconds / Double(positions)
            let wallFour = injectedFour.wallMilliseconds / Double(positions)
            let wallControlAgain = controlAgain.wallMilliseconds / Double(positions)
            let deviceControl = Double(
                control.timing.frontDeviceNanoseconds + control.timing.mixtureDeviceNanoseconds
            ) / Double(positions) / 1_000_000
            let deviceOne = Double(
                injectedOne.timing.frontDeviceNanoseconds
                    + injectedOne.timing.mixtureDeviceNanoseconds
            ) / Double(positions) / 1_000_000
            let deviceFour = Double(
                injectedFour.timing.frontDeviceNanoseconds
                    + injectedFour.timing.mixtureDeviceNanoseconds
            ) / Double(positions) / 1_000_000
            let injectedOneCount = Double(injectedPerPositionPerRepeat)
            let injectedFourCount = Double(injectedPerPositionPerRepeat) * 4
            // ms per dispatch; multiplied by 1,000 where it is printed as microseconds.
            let perDispatchWallOne = (wallOne - wallControl) / injectedOneCount
            let perDispatchWallFour = (wallFour - wallControl) / injectedFourCount
            let perDispatchDeviceOne = (deviceOne - deviceControl) / injectedOneCount
            let perDispatchDeviceFour = (deviceFour - deviceControl) / injectedFourCount
            print(
                String(
                    format: "🔎 [dsv41 real] injected dispatch: wall %.1f (control) / %.1f "
                        + "(+1) / %.1f (+4) / %.1f (control again) ms/position, device "
                        + "%.1f / %.1f / %.1f; that is %.3f and %.3f us of wall, %.3f and "
                        + "%.3f us of device, per injected dispatch",
                    wallControl,
                    wallOne,
                    wallFour,
                    wallControlAgain,
                    deviceControl,
                    deviceOne,
                    deviceFour,
                    perDispatchWallOne * 1_000,
                    perDispatchWallFour * 1_000,
                    perDispatchDeviceOne * 1_000,
                    perDispatchDeviceFour * 1_000
                )
            )
            XCTAssertEqual(control.misses, injectedOne.misses, "the arms did not read alike")
            XCTAssertEqual(control.misses, injectedFour.misses, "the arms did not read alike")
            XCTAssertEqual(
                control.bytes, injectedFour.bytes,
                "the injected arms read a different number of bytes"
            )
            XCTAssertEqual(
                control.misses, controlAgain.misses,
                "the repeated control read a different number of bytes"
            )
            let perDispatchWall = (perDispatchWallOne + perDispatchWallFour) / 2
            let perDispatchDevice = (perDispatchDeviceOne + perDispatchDeviceFour) / 2

            print(
                "🔎 [dsv41 real] dispatches per position by stage, most first, beside the "
                    + "stage's device window (`#60`'s ranking, same stream) and what its "
                    + "dispatches cost at the measured rate:"
            )
            let frontNames = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
            let stageDivisor = Double(max(1, stages.censusPositions))
            func stageMilliseconds(_ stage: String) -> Double {
                Double(stages.stageSeconds[stage] ?? 0) / stageDivisor / 1_000_000
            }
            // Read from the arm's own copy, not from the executor: `pass` clears the live
            // census at the start of every arm, so the last one to run is empty.
            let ordered = stages.census
                .map { (stage: $0.key, count: Double($0.value) / stageDivisor) }
                .sorted { $0.count > $1.count }
            for entry in ordered {
                let milliseconds = stageMilliseconds(entry.stage)
                print(
                    "🔎 [dsv41 real]   "
                        + String(
                            format: "%-40@ %7.1f dispatches %5.1f%%  %6.3f ms  %5.2f ms "
                                + "of dispatch  %@",
                            entry.stage as NSString,
                            entry.count,
                            entry.count / max(dispatchesPerPosition, 1) * 100,
                            milliseconds,
                            entry.count * perDispatchDevice,
                            frontNames.contains(entry.stage) ? "dense" : "mixture"
                        )
                )
            }
            // Every stage that issues anything has to appear, and the ones that never
            // issue a dispatch on their own must not.
            var expected = frontNames.union(DeepSeekV41DevicePlan.backHalf.map(\.rawValue))
            expected.remove(DeepSeekV41ForwardStages.Stage.compressPool.rawValue)
            let counted = Set(stages.census.keys)
            XCTAssertTrue(
                expected.subtracting(counted).isEmpty,
                "the census is missing \(expected.subtracting(counted).sorted())"
            )

            // The two numbers the next slice has to be planned against: what the whole
            // dispatch floor is, and the most a fusion of every stage into one dispatch
            // could return (which is the relinquished bound, not a proposal).
            let dispatchFloor = dispatchesPerPosition * perDispatchDevice
            let fusionCeiling = (dispatchesPerPosition - Double(stageCount * config.numHiddenLayers))
                * perDispatchDevice
            print(
                String(
                    format: "🔎 [dsv41 real] the dispatch floor: %.1f dispatches x %.3f us "
                        + "= %.2f ms of a %.1f ms device position and %.2f ms of a %.1f ms "
                        + "wall position, where the driver sweep's 3.5 us would have said "
                        + "%.2f ms; fusing every stage into one dispatch, the most that "
                        + "removing dispatches ALONE could return, is %.2f ms",
                    dispatchesPerPosition,
                    perDispatchDevice * 1_000,
                    dispatchFloor,
                    deviceControl,
                    dispatchesPerPosition * perDispatchWall,
                    wallControl,
                    dispatchesPerPosition * 0.0035,
                    fusionCeiling
                )
            )
            runtime.shutdown()
        }
        withExtendedLifetime(engine) {}
    }

    /// A page-aligned anonymous mapping this test can wire.
    ///
    /// `mlock` is the only in-process way to tell the kernel a page may be neither
    /// evicted nor compressed, which is what an arena slot would have to become for the
    /// fill's destination to stop being compressor input (`#56`). It is used both as a
    /// host destination and, through `makeBuffer(bytesNoCopy:)`, as the backing of
    /// arena-shaped Metal slots, so the two arms differ in Metal and nothing else.
    private final class Dsv41WiredRegion {
        let base: UnsafeMutableRawPointer
        let bytes: Int
        let wireResult: Int32

        init?(bytes requested: Int, wire: Bool) {
            let page = max(4096, Int(vm_kernel_page_size))
            let length = (requested + page - 1) / page * page
            let mapped = mmap(
                nil, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0
            )
            guard let mapped, mapped != MAP_FAILED else { return nil }
            base = mapped
            bytes = length
            wireResult = wire ? mlock(mapped, length) : 0
        }

        deinit {
            if wireResult == 0 { munlock(base, bytes) }
            munmap(base, bytes)
        }
    }

    /// Issues one probe's reads the way a fill issues its own: one batch per shard, the
    /// batches concurrently, each shard's ranges fanned out by the reader.
    ///
    /// The same shape as `DeepSeekV41ShardBatchContext`, and for the same reason: a
    /// probe that read its shards one after another would be latency bound at 3.8-4.7
    /// GB/s (`#51`) and would not be the rate the fill is paying.
    private final class Dsv41ProbeBatches: @unchecked Sendable {
        struct Batch: @unchecked Sendable {
            let shard: UInt32
            let ranges: [DeepSeekV41ExpertShardReader.Range]
        }

        /// One descriptor per shard file, opened once and held for the whole phase so a
        /// probe does not pay for `open` in the arm it is measuring.
        final class OpenShards: @unchecked Sendable {
            let readers: [UInt32: DeepSeekV41ExpertShardReader]

            init(snapshot: URL, indices: [UInt32]) throws {
                let files = ((try? FileManager.default.contentsOfDirectory(
                    atPath: snapshot.path
                )) ?? []).filter { $0.hasSuffix(".safetensors") }.sorted()
                var opened: [UInt32: DeepSeekV41ExpertShardReader] = [:]
                for shard in Set(indices) {
                    guard Int(shard) < files.count else { continue }
                    opened[shard] = try DeepSeekV41ExpertShardReader(
                        url: snapshot.appendingPathComponent(files[Int(shard)])
                    )
                }
                readers = opened
            }

            func close() {
                for reader in readers.values { reader.closeNow() }
            }
        }

        private let readers: [UInt32: DeepSeekV41ExpertShardReader]
        private let batches: [Batch]
        private let chunkBytes: Int
        private let sequential: Bool
        private let lock = NSLock()
        private var recordedFailure: String?

        init(
            readers: [UInt32: DeepSeekV41ExpertShardReader],
            batches: [Batch],
            chunkBytes: Int,
            sequential: Bool
        ) {
            self.readers = readers
            self.batches = batches
            self.chunkBytes = chunkBytes
            self.sequential = sequential
        }

        var bytes: Int {
            batches.reduce(0) { total, batch in
                total + batch.ranges.reduce(0) { $0 + $1.length }
            }
        }

        var failure: String? {
            lock.lock()
            defer { lock.unlock() }
            return recordedFailure
        }

        func run() -> (seconds: Double, bytes: Int) {
            let started = DispatchTime.now().uptimeNanoseconds
            let one: @Sendable (Int) -> Void = { self.read($0) }
            if batches.count >= 2, !sequential {
                DispatchQueue.concurrentPerform(iterations: batches.count, execute: one)
            } else {
                for index in batches.indices { one(index) }
            }
            return (
                Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000,
                bytes
            )
        }

        private func read(_ index: Int) {
            let batch = batches[index]
            guard let reader = readers[batch.shard] else {
                note("no descriptor for shard \(batch.shard)")
                return
            }
            do {
                try reader.read(batch.ranges, chunkBytes: chunkBytes, sequential: sequential)
            } catch {
                note("\(error)")
            }
        }

        private func note(_ text: String) {
            lock.lock()
            if recordedFailure == nil { recordedFailure = text }
            lock.unlock()
        }
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

    // MARK: - The live turn's expert fill

    /// What a live turn's expert fill costs, measured in the regime a live turn is
    /// actually in, and what hiding it could be worth.
    ///
    /// `#66` closed the dispatch floor and left a walked position as 36.7 ms of device
    /// work, a mixture late start of 11.6, an indexer drain of 9.4, an encode of 4.6 and
    /// a fill of **4.2 ms** -- measured on a stream the probe had already walked three
    /// times, so its expert bytes looked cheap. `#22` and `#49` both measured a live
    /// turn's fill an order of magnitude larger (67 ms, and 556 MB a position with
    /// 86-93 ms of hand-off idle behind it). Three probes, three fills, and every later
    /// slice quotes whichever one it happened to run.
    ///
    /// The difference is the regime, not the model. A probe that re-walks one stream is
    /// warm in two independent ways and a live turn's text is warm in neither:
    ///
    ///   1. **The arena** -- `#59` showed a cold pass is 95.3% compulsory first touches
    ///      while a warm pass re-fetches 83-100% of its misses, so the two are the same
    ///      bytes with different causes and only the second is a replacement question.
    ///   2. **The page cache** -- the first read of an expert's bytes comes off the disk
    ///      at 3.7-4.0 GB/s and every read after it, while it lasts, out of the cache.
    ///
    /// So this case walks four arms on ONE runtime and reports each arm's whole budget,
    /// per position, from the layer records `#49` built. Those carry the fill's own
    /// bytes, misses, `pread` seconds and placement seconds beside the four intervals
    /// that tile a position, which is what makes the hand-off decomposable:
    ///
    ///   `hand-off = routing readback + pread + placement + mixture encode + idle`
    ///
    ///   * `cold` -- a stream this process has never walked, starting from
    ///     `arena.reset()`. A live turn's first positions in both senses.
    ///   * `warm` -- the same stream walked again. `#59`'s steady state, and what every
    ///     table from `#60` on has actually been measuring.
    ///   * `cycled` -- an eight-token cycle, discarded: it loads its own small working
    ///     set so the arm after it can measure a fill that reads nothing.
    ///   * `floor` -- the same cycle again, every pair resident. `#49`'s zero-byte-fill
    ///     arm, and the control the other arms are priced against.
    ///
    /// The deliverable is the number the handover asked for and no table in this journal
    /// carries: **what hiding the whole expert fill behind the GPU would be worth**, read
    /// as `wall(cold) - wall(floor)` and `wall(warm) - wall(floor)` -- the read, the
    /// hand-off idle it induces, and nothing else, at one submission shape and one
    /// machine. It is a ceiling: it assumes the read becomes free rather than merely
    /// overlapped.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_live_fill`, plus the class's own marker).
    func testWhatALiveTurnsExpertFillCosts() throws {
        let markerPath = "/tmp/dynamoe_dsv41_live_fill"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in live-fill probe; touch \(markerPath) to run"
        )
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
        // The 128-position stream every assembly from `#52` on measured, so this arm's
        // numbers are line-for-line comparable with that table's.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8, "the stream must span more than one block")
        let positions = 128
        let tokens = (0..<positions).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 7) % stream.count]
        }
        // Eight tokens cycling: the arena can hold every pair this walk asks for, so once
        // it has been walked the fill has nothing to read.
        let cycled = (0..<positions).map { index -> Int in
            stream[(index % 8) % stream.count]
        }
        // The process warm-up: a third stream, so slot creation, wiring, the pipeline
        // caches and the shard mappings are paid before any arm is timed, and none of the
        // measured arms' bytes are in the page cache because of it.
        let seed = (0..<16).map { index in
            stream[(index * 5 + 3) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: positions + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        // The shipping rule, and the miss classifier `#59` added: a miss that is a first
        // touch is a floor no replacement rule can remove, and only the rest is
        // reclaimable.
        runtime.arena.victimPolicy = .leastFrequentlyUsed
        runtime.arena.recordsMissComposition = true
        let executor = runtime.forward()

        _ = try liveFillArm(
            runtime: runtime,
            executor: executor,
            tokens: seed,
            resetArena: false,
            label: "warm-up (discarded)"
        )
        print(
            String(
                format: "🔎 [dsv41 real] live fill: %d positions, arena %d slots, "
                    + "%@ rule, warm-up done",
                positions,
                runtime.arena.capacity,
                runtime.arena.victimPolicy == .leastFrequentlyUsed ? "frequency" : "LRU"
            )
        )

        let cold = try liveFillArm(
            runtime: runtime,
            executor: executor,
            tokens: tokens,
            resetArena: true,
            label: "cold (live turn)"
        )
        let warm = try liveFillArm(
            runtime: runtime,
            executor: executor,
            tokens: tokens,
            resetArena: false,
            label: "warm (steady state)"
        )
        _ = try liveFillArm(
            runtime: runtime,
            executor: executor,
            tokens: cycled,
            resetArena: false,
            label: "cycled (discarded: loads its own set)"
        )
        let floor = try liveFillArm(
            runtime: runtime,
            executor: executor,
            tokens: cycled,
            resetArena: false,
            label: "floor (no fill)"
        )

        for arm in [cold, warm, floor] { reportLiveFill(arm) }

        // The arms are the same weights and the same stream, so the two stream-A arms
        // have to return the same logits to the bit; only the arena and the page cache
        // differ. That is the control that says this instrument measures the fill rather
        // than having changed the graph.
        XCTAssertEqual(
            cold.lastLogits, warm.lastLogits,
            "the second walk of the same stream moved the last position's logits"
        )
        print(
            "🔎 [dsv41 real] live fill control: the cold and warm arms' logits are "
                + "bit-identical at the last position"
        )

        // A cold pass is compulsory (`#59`: 95.3% first touches), so it is what a live
        // turn pays and no rule can take it away. Asserted rather than assumed, because a
        // probe whose warm-up leaked its pairs into the arena would read like the warm
        // arm and quietly retire the conclusion.
        let coldShare = Double(cold.firstTouch) / Double(max(1, cold.misses))
        XCTAssertGreaterThan(
            coldShare, 0.9,
            "a cold arm must be compulsory first touches, not re-fetches"
        )
        // The floor has to actually be a floor, or the prize is priced against a phantom.
        // It cannot be a ZERO-byte one under the shipping rule: `leastFrequentlyUsed`'s
        // aging halves every count once a capacity-worth of misses has gone by, so even an
        // eight-token cycle's own small set is retired and re-read. What makes it a floor
        // anyway is that those bytes are in the page cache -- it reads them at tens of
        // GB/s -- so its fill is a couple of milliseconds and the prize is priced against
        // a position that has one.
        XCTAssertLessThan(
            floor.preadPerPosition, 0.25 * cold.preadPerPosition,
            "the floor arm must spend far less on the fill than the cold one"
        )
        XCTAssertLessThan(
            Double(floor.misses), 0.25 * Double(cold.misses),
            "the floor arm must miss far less often than the cold one"
        )
        XCTAssertGreaterThan(
            floor.gigabytesPerSecond, cold.gigabytesPerSecond,
            "the floor arm's bytes must be served faster than the cold arm's"
        )
        // Everything except the fill is held fixed across the arms, so the difference in
        // the dense halves and the expert halves is measurement noise rather than a
        // second cause. Stated as a bound: 2 ms of a ~18 ms half.
        XCTAssertLessThan(
            abs(cold.densePerPosition - floor.densePerPosition), 2.0,
            "the dense halves moved between arms, so the arms are not comparable"
        )
        XCTAssertLessThan(
            abs(cold.mixturePerPosition - floor.mixturePerPosition), 2.0,
            "the expert halves moved between arms, so the arms are not comparable"
        )

        let coldPrize = cold.wallPerPosition - floor.wallPerPosition
        let warmPrize = warm.wallPerPosition - floor.wallPerPosition
        print(
            String(
                format: "🔎 [dsv41 real] live fill prize: a cold position reads %.0f MB "
                    + "at %.1f GB/s (%.1f ms of pread) and spends %.1f ms in the "
                    + "hand-off, so hiding the whole fill is %.1f ms of a %.1f ms "
                    + "position (%.0f%%); the warm arena's %.0f MB is %.1f ms of %.1f "
                    + "(%.0f%%), against a floor of %.1f ms",
                cold.megabytesPerPosition,
                cold.gigabytesPerSecond,
                cold.preadPerPosition,
                cold.handoffPerPosition,
                coldPrize,
                cold.wallPerPosition,
                100 * coldPrize / max(cold.wallPerPosition, 0.0001),
                warm.megabytesPerPosition,
                warmPrize,
                warm.wallPerPosition,
                100 * warmPrize / max(warm.wallPerPosition, 0.0001),
                floor.wallPerPosition
            )
        )
        print(String(format: "🔎 [dsv41 real] %@", runtime.describe()))
        runtime.shutdown()
        withExtendedLifetime(engine) {}
    }

    /// One arm's per-position record: the four intervals that tile the position, the
    /// hand-off broken into the pieces that sum to it, and what the layer read.
    private struct LiveFillPosition {
        let position: Int
        let wallMilliseconds: Double
        let denseMilliseconds: Double
        let handoffMilliseconds: Double
        let mixtureMilliseconds: Double
        let routingMilliseconds: Double
        let idleMilliseconds: Double
        let encodeMilliseconds: Double
        let misses: Int
        let bytes: Int
        let preadMilliseconds: Double
        let placementMilliseconds: Double
        let prepareMilliseconds: Double
        let allocateMilliseconds: Double
        let wireMilliseconds: Double

        var tiledMilliseconds: Double {
            denseMilliseconds + handoffMilliseconds + mixtureMilliseconds
        }

        /// What the fill costs the position: the bytes, plus how long the GPU sat idle
        /// waiting for them. The readback and the encode are the graph's own work and are
        /// not the fill's.
        var fillMilliseconds: Double {
            prepareMilliseconds + preadMilliseconds + placementMilliseconds
                + idleMilliseconds
        }

        /// The per-miss half of the preparation: catalog lookups, read ranges, bindings.
        var resolveMilliseconds: Double {
            max(0, prepareMilliseconds - allocateMilliseconds - wireMilliseconds)
        }
    }

    private struct LiveFillArm {
        let label: String
        let records: [LiveFillPosition]
        let misses: Int
        let firstTouch: Int
        let refetch: Int
        let bytes: Int
        let evictions: Int
        let resident: Int
        let capacity: Int
        let lastLogits: [Float]

        private func mean(_ value: (LiveFillPosition) -> Double) -> Double {
            guard !records.isEmpty else { return 0 }
            return records.map(value).reduce(0, +) / Double(records.count)
        }

        var wallPerPosition: Double { mean { $0.wallMilliseconds } }
        var densePerPosition: Double { mean { $0.denseMilliseconds } }
        var handoffPerPosition: Double { mean { $0.handoffMilliseconds } }
        var mixturePerPosition: Double { mean { $0.mixtureMilliseconds } }
        var routingPerPosition: Double { mean { $0.routingMilliseconds } }
        var idlePerPosition: Double { mean { $0.idleMilliseconds } }
        var encodePerPosition: Double { mean { $0.encodeMilliseconds } }
        var fillPerPosition: Double { mean { $0.fillMilliseconds } }
        var missesPerPosition: Double { mean { Double($0.misses) } }
        var megabytesPerPosition: Double { mean { Double($0.bytes) / 1_048_576 } }
        var preadPerPosition: Double { mean { $0.preadMilliseconds } }
        var placementPerPosition: Double { mean { $0.placementMilliseconds } }
        var preparePerPosition: Double { mean { $0.prepareMilliseconds } }
        var allocatePerPosition: Double { mean { $0.allocateMilliseconds } }
        var wirePerPosition: Double { mean { $0.wireMilliseconds } }
        var resolvePerPosition: Double { mean { $0.resolveMilliseconds } }

        /// What the `pread` achieved, which is what tells a disk read from a page-cache
        /// hit: 3.7-4.0 GB/s on a first read against 13-14 through the cache.
        var gigabytesPerSecond: Double {
            let seconds = records.map { $0.preadMilliseconds }.reduce(0, +) / 1000
            return seconds > 0 ? Double(bytes) / 1_073_741_824 / seconds : 0
        }
    }

    /// Walks `tokens` once with the hand-off recorder on for every position, and checks
    /// the records tile the positions they came from before returning them.
    private func liveFillArm(
        runtime: DeepSeekV41Runtime,
        executor: DeepSeekV41MetalForward,
        tokens: [Int],
        resetArena: Bool,
        label: String
    ) throws -> LiveFillArm {
        executor.reset()
        executor.recordsWalkTiming = false
        executor.recordsWalkStageTiming = false
        if resetArena { runtime.arena.reset() }

        let missesBefore = runtime.arena.acquisitions - runtime.arena.hits
        let firstBefore = runtime.arena.firstTouchMisses
        let refetchBefore = runtime.arena.refetchMisses
        let evictionsBefore = runtime.arena.evictions
        let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0

        executor.recordsLayerGaps = true
        var records: [LiveFillPosition] = []
        var logits: [Float] = []
        for (position, token) in tokens.enumerated() {
            logits = try executor.forwardToken(token, position: position)
            let gaps = executor.layerGaps
            let wall = executor.layerGapPositions.last?.wallMilliseconds ?? 0
            records.append(
                LiveFillPosition(
                    position: position,
                    wallMilliseconds: wall,
                    denseMilliseconds: gaps.map { $0.frontWindowMilliseconds }.reduce(0, +),
                    handoffMilliseconds: gaps
                        .map { $0.hostGapMilliseconds + $0.idleMilliseconds }
                        .reduce(0, +),
                    mixtureMilliseconds: gaps.map { $0.mixtureWindowMilliseconds }.reduce(0, +),
                    routingMilliseconds: 1e-6
                        * Double(gaps.map { $0.routingNanoseconds }.reduce(0, +)),
                    idleMilliseconds: gaps.map { $0.idleMilliseconds }.reduce(0, +),
                    encodeMilliseconds: 1e-6
                        * Double(gaps.map { $0.mixtureEncodeNanoseconds }.reduce(0, +)),
                    misses: gaps.map { $0.fillMisses }.reduce(0, +),
                    bytes: gaps.map { $0.fillBytes }.reduce(0, +),
                    preadMilliseconds: 1e3 * gaps.map { $0.fillReadSeconds }.reduce(0, +),
                    placementMilliseconds: 1e3
                        * gaps.map { $0.fillPlacementSeconds }.reduce(0, +),
                    prepareMilliseconds: 1e3
                        * gaps.map { $0.fillPrepareSeconds }.reduce(0, +),
                    allocateMilliseconds: 1e3
                        * gaps.map { $0.fillAllocateSeconds }.reduce(0, +),
                    wireMilliseconds: 1e3
                        * gaps.map { $0.fillWireSeconds }.reduce(0, +)
                )
            )
            XCTAssertEqual(
                gaps.count, executor.plan.count,
                "\(label) position \(position) recorded \(gaps.count) layer(s), not one per layer"
            )
            XCTAssertTrue(
                gaps.allSatisfy { $0.hasWindow },
                "\(label) position \(position) left an expert half with no device window"
            )
        }
        executor.recordsLayerGaps = false

        let misses = (runtime.arena.acquisitions - runtime.arena.hits) - missesBefore
        let firstTouch = runtime.arena.firstTouchMisses - firstBefore
        let refetch = runtime.arena.refetchMisses - refetchBefore
        let bytes = (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore

        // The instruments' own contracts, so a broken recorder cannot be read as a
        // result: the misses decompose, the layers' fills are the arena's misses, and the
        // four intervals tile the positions rather than overlapping them.
        XCTAssertEqual(
            firstTouch + refetch, misses,
            "\(label): the miss composition does not cover the arena's misses"
        )
        XCTAssertEqual(
            records.map { $0.misses }.reduce(0, +), misses,
            "\(label): the layers' recorded misses differ from the arena's"
        )
        let wall = records.map { $0.wallMilliseconds }.reduce(0, +)
        let tiled = records.map { $0.tiledMilliseconds }.reduce(0, +)
        XCTAssertLessThanOrEqual(
            tiled, wall * 1.02,
            "\(label): the layer records exceed the positions they came from"
        )
        XCTAssertGreaterThanOrEqual(
            tiled, wall * 0.8,
            "\(label): the layer records account for too little of the positions"
        )

        return LiveFillArm(
            label: label,
            records: records,
            misses: misses,
            firstTouch: firstTouch,
            refetch: refetch,
            bytes: bytes,
            evictions: runtime.arena.evictions - evictionsBefore,
            resident: runtime.arena.residentCount,
            capacity: runtime.arena.capacity,
            lastLogits: logits
        )
    }

    /// One arm's budget, its hand-off decomposed, then its per-position curve: a miss
    /// rate that decays with the position is a cold-start fill a longer turn would not
    /// pay, and one that plateaus is a steady state no sequence length improves.
    private func reportLiveFill(_ arm: LiveFillArm) {
        let positions = arm.records.count
        print(
            String(
                format: "🔎 [dsv41 real] live fill (%@): %.1f ms/position = dense %.1f + "
                    + "hand-off %.1f + expert halves %.1f; reads %.0f MB in %.1f miss(es) "
                    + "at %.1f GB/s",
                arm.label,
                arm.wallPerPosition,
                arm.densePerPosition,
                arm.handoffPerPosition,
                arm.mixturePerPosition,
                arm.megabytesPerPosition,
                arm.missesPerPosition,
                arm.gigabytesPerSecond
            )
        )
        let prepare = arm.preparePerPosition
        let fill = arm.preparePerPosition + arm.preadPerPosition + arm.placementPerPosition
        let named = arm.routingPerPosition + fill + arm.idlePerPosition
            + arm.encodePerPosition
        print(
            String(
                format: "🔎 [dsv41 real]   %@ hand-off %.1f ms = routing %.1f + fill %.1f "
                    + "(prepare %.1f = allocate %.1f + wire %.1f + resolve %.1f, "
                    + "pread %.1f, place %.1f) + idle %.1f + encode %.1f "
                    + "-> %.1f ms unattributed",
                arm.label,
                arm.handoffPerPosition,
                arm.routingPerPosition,
                fill,
                prepare,
                arm.allocatePerPosition,
                arm.wirePerPosition,
                arm.resolvePerPosition,
                arm.preadPerPosition,
                arm.placementPerPosition,
                arm.idlePerPosition,
                arm.encodePerPosition,
                arm.handoffPerPosition - named
            )
        )
        // Resolving a miss is host bookkeeping that a faster disk would not shorten, so
        // it is worth a number per miss rather than a bucket: allocate and wire are paid
        // once per NEW slot, and whatever preparation is left tracks the miss count.
        print(
            String(
                format: "🔎 [dsv41 real]   %@: %.3f ms of resolve per miss, %.3f ms of "
                    + "allocate+wiring per miss, %.1f miss(es)/position",
                arm.label,
                arm.missesPerPosition > 0
                    ? arm.resolvePerPosition / arm.missesPerPosition : 0,
                arm.missesPerPosition > 0
                    ? (arm.allocatePerPosition + arm.wirePerPosition)
                        / arm.missesPerPosition : 0,
                arm.missesPerPosition
            )
        )
        print(
            String(
                format: "🔎 [dsv41 real]   %@: %d miss(es) over %d position(s), %d first "
                    + "touch (%.1f%%), %d re-fetch (%.1f%%), %d eviction(s); arena "
                    + "%d/%d resident",
                arm.label,
                arm.misses,
                positions,
                arm.firstTouch,
                positions > 0 ? 100 * Double(arm.firstTouch) / Double(max(1, arm.misses)) : 0,
                arm.refetch,
                positions > 0 ? 100 * Double(arm.refetch) / Double(max(1, arm.misses)) : 0,
                arm.evictions,
                arm.resident,
                arm.capacity
            )
        )
        for record in arm.records where record.position % 32 == 0 || record.position == positions - 1 {
            print(
                String(
                    format: "🔎 [dsv41 real]   %@ position %3d: wall %.1f ms "
                        + "(dense %.1f, hand-off %.1f, expert %.1f), %d miss(es), "
                        + "%.0f MB, pread %.1f ms, idle %.1f ms",
                    String(arm.label.prefix(5)),
                    record.position,
                    record.wallMilliseconds,
                    record.denseMilliseconds,
                    record.handoffMilliseconds,
                    record.mixtureMilliseconds,
                    record.misses,
                    Double(record.bytes) / 1_048_576,
                    record.preadMilliseconds,
                    record.idleMilliseconds
                )
            )
        }
    }

    // MARK: - What acceptance a draft would have to reach

    /// The widths the bar is priced at: the two a chain's acceptance curve can fill,
    /// the checkpoint's declared block, and the eight `#22` measured a batched verify.
    private static let verifyBarWidths = [2, 3, 5, 8]

    /// The two numbers `#26` measured for the checkpoint's own drafter, per width: how
    /// often its chain agrees with the target across both continuations, and what one
    /// round of drafting costs. They are quoted rather than re-run because they are a
    /// property of the drafter, not of the walk's regime, and re-measuring them needs a
    /// runtime per width (`dsparkBlockSize` sizes buffers at construction).
    private static let dsparkMeasured: [Int: (acceptance: Double, draftMilliseconds: Double)] = [
        2: (0.525, 73), 3: (0.385, 78.5), 5: (0.31, 154.5)
    ]

    /// What a drafter's acceptance would have to be for its verify to pay for itself,
    /// at every width, against the walk a live turn actually pays -- and the one
    /// drafter that costs the model nothing, measured against the same bar.
    ///
    /// `#26` closed a narrow verify with both sides measured: acceptance is 50-55% at
    /// width two against a verify at 0.83-0.89x a walk, which loses, and 37-40% at
    /// width three against 1.08x, which loses worse. `#22`'s batched block of eight is
    /// 2.16x cheaper per position, but it needs ~47% acceptance over eight drafts and
    /// the checkpoint's own chain dies at its second step.
    ///
    /// Both were priced against a walk of 153.3-194.8 ms/position. `#67` then measured
    /// the walk a live turn pays -- 121.8 ms/position cold, 82 warm, 55 once the fill
    /// is served from the page cache -- and the arithmetic runs the wrong way with a
    /// faster walk: a verify's ENTIRE advantage is batching the expert fill, so the
    /// less fill there is to batch, the more acceptance a round needs to break even.
    /// `#26`'s conclusion therefore holds with more room than it could show, and the
    /// only question left open is whether any drafter can be cheap enough to live
    /// above the bar.
    ///
    /// That is what this case prices. In the regime `#67` established -- the arena
    /// reset before every arm, so every arm is compulsory first touches -- it measures
    /// a WALK arm and a VERIFY arm at widths 2, 3, 5 and 8 on one runtime, rotated and
    /// interleaved, and turns them into the bar
    ///
    ///     a_required = draft/walk + width * verify_ratio - 1
    ///
    /// -- the expected number of accepted drafts a round needs, from
    /// `draft + width*verify*W <= (1 + a)*W`. Every width also reports the bar with a
    /// FREE drafter, which is the floor no drafter can go below, and beside it the
    /// acceptance `#26` measured for the one this checkpoint ships.
    ///
    /// Then it measures the only drafter whose cost is genuinely zero: an n-gram
    /// lookup over the turn's own history, which needs no model. It generates a real
    /// greedy continuation and scores that drafter offline, exactly rather than
    /// approximately, because acceptance always stops at the first disagreement and
    /// every position before it is the model's own path.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_verify_bar`, plus the class's own marker).
    func testWhatAcceptanceADraftWouldHaveToReachToPayForItsVerify() throws {
        let markerPath = "/tmp/dynamoe_dsv41_verify_bar"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in verify-bar probe; touch \(markerPath) to run"
        )
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        guard let tokenizer = try? DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        ) else {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let widths = Self.verifyBarWidths
        // A live turn walks ONE position per round and verifies a short run, so an arm
        // of 48 positions is already longer than the thing being priced. Every arm and
        // every round draws a DIFFERENT slice of the pool, because a token this process
        // already read is a token the page cache already holds, and `#67` showed how
        // much a warm read flatters a fill.
        let positions = 48
        let rounds = 3
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the model, \
            and the fraction is chosen by a router rather than by the layout. The router \
            is small and is trained jointly with everything else, so which experts a \
            token reaches is a property of the weights and not of a schedule somebody \
            wrote down. That is why the same sentence decoded twice reads the same rows, \
            and why a different sentence does not, and why the disk's throughput is the \
            number that decides how fast a decode step can be.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 32, "the pool text tokenized to too little")
        let arms = widths.count + 1
        let pool = (0..<(positions * arms * rounds)).map { index in
            stream[(index * 37) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: 256,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        runtime.arena.victimPolicy = .leastFrequentlyUsed
        runtime.arena.recordsMissComposition = true
        let executor = runtime.forward()
        executor.recordsWalkTiming = false
        executor.recordsWalkStageTiming = false
        executor.recordsLayerGaps = false
        print(
            String(
                format: "🔎 [dsv41 real] verify bar: %d positions per arm, %d round(s) "
                    + "after a discarded warm-up, widths %@, arena %d slots",
                positions, rounds - 1,
                widths.map(String.init).joined(separator: "/"),
                runtime.arena.capacity
            )
        )

        var walks: [VerifyBarArm] = []
        var verifies: [Int: [VerifyBarArm]] = [:]
        for round in 0..<rounds {
            for offset in 0..<arms {
                // Rotate, so no width always inherits the previous arm's leftovers and
                // no width is always the first to pay for a cold pipeline.
                let slot = (offset + round) % arms
                let index = round * arms + slot
                let tokens = Array(pool[(index * positions)..<((index + 1) * positions)])
                let width: Int? = slot == 0 ? nil : widths[slot - 1]
                let arm = try verifyBarArm(
                    runtime: runtime,
                    executor: executor,
                    tokens: tokens,
                    chunkWidth: width,
                    label: width.map { "w\($0)" } ?? "walk"
                )
                // The warm-up round is discarded whole: it pays the pipeline encoders,
                // the slot wiring of whatever it touches and the shard mappings.
                guard round > 0 else { continue }
                if let width { verifies[width, default: []].append(arm) } else { walks.append(arm) }
            }
        }
        XCTAssertEqual(walks.count, rounds - 1, "a round did not record exactly one walk")
        for width in widths {
            XCTAssertEqual(
                verifies[width]?.count, rounds - 1,
                "width \(width) did not record one arm per measured round"
            )
        }
        // Every arm resets the arena, so every arm must be compulsory: a miss that is a
        // first touch is what a live turn's new positions pay, and a warm arm would
        // read like the regime `#67` spent a slice debunking.
        let everyArm = walks + widths.flatMap { verifies[$0] ?? [] }
        for arm in everyArm {
            XCTAssertGreaterThan(
                arm.firstTouchShare, 0.8,
                "\(arm.label) was not compulsory (first-touch share \(arm.firstTouchShare))"
            )
        }

        func median(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }
        let walk = median(walks.map(\.msPerPosition))
        for arm in walks {
            print(
                String(
                    format: "🔎 [dsv41 real]   %@: %.1f ms/position, %.0f MB/position, "
                        + "%.1f miss(es)/position, %.0f%% first touch",
                    arm.label, arm.msPerPosition, arm.megabytesPerPosition,
                    arm.missesPerPosition, 100 * arm.firstTouchShare
                )
            )
        }
        // The ratio is taken ROUND BY ROUND and then reduced, which is `#26`'s fix: the
        // absolute level of this machine moves between runs by more than any effect
        // being chased, so only a verify and a walk measured seconds apart can be
        // divided.
        for width in widths {
            let arms = verifies[width] ?? []
            for arm in arms {
                print(
                    String(
                        format: "🔎 [dsv41 real]   %@: %.1f ms/position, %.0f MB/position, "
                            + "%.1f miss(es)/position, %.0f%% first touch",
                        arm.label, arm.msPerPosition, arm.megabytesPerPosition,
                        arm.missesPerPosition, 100 * arm.firstTouchShare
                    )
                )
            }
            let ratios = zip(arms, walks).map { $0.msPerPosition / max(1e-9, $1.msPerPosition) }
            let ratio = median(ratios)
            let barFree = Self.verifyBarRequiredRate(width: width, ratio: ratio, draft: 0, walk: walk)
            var measured = ", and nothing was ever measured at this width"
            if let draft = Self.dsparkMeasured[width] {
                let barDraft = Self.verifyBarRequiredRate(
                    width: width, ratio: ratio, draft: draft.draftMilliseconds, walk: walk
                )
                measured = String(
                    format: ", or %.0f%% at #26's %.0f ms of drafting, which delivers %.0f%%",
                    100 * barDraft, draft.draftMilliseconds, 100 * draft.acceptance
                )
            }
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: verify %.2fx a walk (rounds %@), so "
                        + "one expert union replaces %d per-position sets and saves %.1f "
                        + "ms/position; break-even needs %.1f of %d drafts = %.0f%% FREE%@",
                    width,
                    ratio,
                    ratios.map { String(format: "%.2f", $0) }.joined(separator: "/"),
                    width,
                    (1 - ratio) * walk,
                    barFree * Double(width),
                    width,
                    100 * barFree,
                    measured
                )
            )
        }

        // What batching actually saves, which is not what `#22` said it did. That slice
        // priced a block of eight at 7.6x fewer read bytes on the reasoning that "one
        // expert union" replaces "one set per position" -- but the arena IS that union
        // already: a walked position hits ~80% of the time because the positions before
        // it in the same turn read the same experts, so the union a block fills up front
        // is very nearly the set the walks would have read anyway. The byte column is
        // what says so, and the per-layer host round trip is all a block removes.
        let walkBytes = median(walks.map(\.megabytesPerPosition))
        let bytes = widths.map { width -> String in
            let value = median((verifies[width] ?? []).map(\.megabytesPerPosition))
            return String(format: "w%d %.0f MB (%.2fx)", width, value, value / max(1, walkBytes))
        }
        print(
            String(
                format: "🔎 [dsv41 real] read bytes per position against the walk's "
                    + "%.0f MB: %@",
                walkBytes, bytes.joined(separator: ", ")
            )
        )

        // The one drafter that costs no model at all, scored against a real greedy
        // continuation. `#22` measured kernels instead of the fill before; this is the
        // same trap's mirror image, so the continuation is a genuine generation from a
        // rendered prompt rather than any text the fixture chose.
        executor.reset()
        let dialect = DeepSeekV41PromptDialect.resolve(snapshotDir: snapshot)
        let prompt = dialect.render(
            turns: [.init(isUser: true, text: Self.verifyBarQuestion)],
            cleanSystem: "",
            thinkingEnabled: false,
            effort: nil
        )
        let promptIds = try tokenizer.encode(text: prompt).map { Int($0) }
        let continuation = 96
        var path = promptIds
        var logits = try executor.prefillChunk(promptIds).last
        XCTAssertFalse(logits.isEmpty, "the prompt prefill returned no logits")
        for _ in 0..<continuation {
            let token = Int(Self.argmax(logits))
            path.append(token)
            logits = try executor.forwardToken(token, position: path.count - 1)
        }
        let generated = Array(path[promptIds.count...])
        print(
            "🔎 [dsv41 real] n-gram target: \(promptIds.count) prompt token(s) and "
                + "\(generated.count) generated greedily; \(generated.prefix(16))"
        )
        for order in [2, 3] {
            let scored = ngramAcceptance(
                path: path, promptCount: promptIds.count, widths: widths, order: order
            )
            let lines = widths.map { width -> String in
                let entry = scored[width] ?? (0, 0, 0, 0)
                let rate = entry.drafted > 0
                    ? Double(entry.accepted) / Double(entry.drafted) : 0
                let armsForWidth = verifies[width] ?? []
                let ratio = median(
                    zip(armsForWidth, walks).map { $0.msPerPosition / max(1e-9, $1.msPerPosition) }
                )
                let bar = Self.verifyBarRequiredRate(width: width, ratio: ratio, draft: 0, walk: walk)
                return String(
                    format: "w%d %d/%d (%.0f%%) in %d of %d round(s) vs a %.0f%% bar%@",
                    width, entry.accepted, entry.drafted, 100 * rate,
                    entry.draftedRounds, entry.rounds, 100 * bar,
                    rate >= bar && entry.drafted > 0 ? " -- CLEARS" : ""
                )
            }
            print(
                "🔎 [dsv41 real] n-gram drafter, \(order)-token lookup, "
                    + "\(widths.count) width(s) over \(continuation) drafted position(s): "
                    + lines.joined(separator: ", ")
            )
        }
        print(String(format: "🔎 [dsv41 real] %@", runtime.describe()))
        runtime.shutdown()
        withExtendedLifetime(engine) {}
    }

    /// A question whose answer is ordinary prose, so the continuation the n-gram
    /// drafter is scored on is the kind of text a turn actually produces rather than a
    /// list or a template that repeats itself.
    private static let verifyBarQuestion =
        "Explain in a short paragraph why reading from a solid state drive is slower "
        + "than reading from memory."

    /// One arm's per-position cost, what it read, and whether that read was compulsory.
    private struct VerifyBarArm {
        let label: String
        let msPerPosition: Double
        let megabytesPerPosition: Double
        let missesPerPosition: Double
        let firstTouchShare: Double
    }

    /// Runs `tokens` once, from an empty cache and an empty arena, either as a walk or
    /// as `tokens.count / width` blocks of `width`.
    private func verifyBarArm(
        runtime: DeepSeekV41Runtime,
        executor: DeepSeekV41MetalForward,
        tokens: [Int],
        chunkWidth: Int?,
        label: String
    ) throws -> VerifyBarArm {
        executor.reset()
        runtime.arena.reset()
        let missesBefore = runtime.arena.acquisitions - runtime.arena.hits
        let firstBefore = runtime.arena.firstTouchMisses
        let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
        let began = DispatchTime.now().uptimeNanoseconds
        if let width = chunkWidth {
            var offset = 0
            while offset < tokens.count {
                let end = min(offset + width, tokens.count)
                _ = try executor.prefillChunk(Array(tokens[offset..<end]))
                offset = end
            }
        } else {
            for (position, token) in tokens.enumerated() {
                _ = try executor.forwardToken(token, position: position)
            }
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - began) / 1_000_000
        let misses = (runtime.arena.acquisitions - runtime.arena.hits) - missesBefore
        let firstTouch = runtime.arena.firstTouchMisses - firstBefore
        let bytes = (runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore
        let count = Double(tokens.count)
        return VerifyBarArm(
            label: label,
            msPerPosition: elapsed / count,
            megabytesPerPosition: Double(bytes) / 1_048_576 / count,
            missesPerPosition: Double(misses) / count,
            firstTouchShare: misses > 0 ? Double(firstTouch) / Double(misses) : 0
        )
    }

    /// The acceptance a round needs, as a rate per drafted token, from
    /// `draft + width*ratio*walk <= (1 + a)*walk`.
    ///
    /// A negative bar is a round that wins with nothing accepted, which happens once a
    /// block is cheaper per position than `1/width` of a walk.
    private static func verifyBarRequiredRate(
        width: Int, ratio: Double, draft: Double, walk: Double
    ) -> Double {
        guard width > 0, walk > 0 else { return 1 }
        let required = draft / walk + Double(width) * ratio - 1
        return required / Double(width)
    }

    /// Every width's agreement, for a drafter that costs nothing: the tokens that
    /// followed the most recent earlier occurrence of the history's last `order`
    /// tokens.
    ///
    /// The round advances by `width` and not by `accepted + 1`, because that is what
    /// `DeepSeekV41DSpark`'s protocol does (`dsparkAcceptance` walks `steps` positions
    /// per round whatever the drafts did), so the two drafters are scored against the
    /// same target path and their rates can be compared.
    private func ngramAcceptance(
        path: [Int],
        promptCount: Int,
        widths: [Int],
        order: Int
    ) -> [Int: (accepted: Int, drafted: Int, rounds: Int, draftedRounds: Int)] {
        var result: [Int: (accepted: Int, drafted: Int, rounds: Int, draftedRounds: Int)] = [:]
        for width in widths {
            var accepted = 0
            var drafted = 0
            var roundCount = 0
            var draftedRounds = 0
            var position = promptCount
            while position + width <= path.count {
                let draft = ngramDraft(
                    history: path[0..<position], width: width, order: order
                )
                accepted += DeepSeekV41DSpark.acceptedCount(
                    draft: draft,
                    targetChoices: Array(path[position..<(position + width)])
                )
                drafted += draft.count
                roundCount += 1
                if !draft.isEmpty { draftedRounds += 1 }
                position += width
            }
            result[width] = (accepted, drafted, roundCount, draftedRounds)
        }
        return result
    }

    /// The tokens that followed the most recent earlier occurrence of the history's
    /// last `order` tokens, or nothing when there is none -- a round that drafts
    /// nothing and therefore accepts nothing.
    private func ngramDraft(history: ArraySlice<Int>, width: Int, order: Int) -> [Int] {
        guard history.count > order else { return [] }
        let suffix = Array(history.suffix(order))
        var start = history.count - order - 1
        while start >= 0 {
            let candidate = Array(history[start..<(start + order)])
            if candidate == suffix, start + order + width <= history.count {
                return Array(history[(start + order)..<(start + order + width)])
            }
            start -= 1
        }
        return []
    }

    // MARK: - #69: the whole turn, end to end

    /// One prefill report as it arrived: how many positions the chunk held, and the
    /// wall clock when the executor reported it.
    struct TurnPrefillBlock: Sendable {
        var done: Int
        var seconds: Double
    }

    /// What one turn reported while it ran, beside what it returned.
    struct TurnTrace: Sendable {
        var output: DeepSeekV41TurnRunner.Output
        var prefillBlocks: [TurnPrefillBlock]
        var tokenSeconds: [Double]
    }

    /// What a live turn actually costs, start to finish.
    ///
    /// Every slice from `#60` on measured a piece of a decoded position: a stage of the
    /// dense half, the expert fill's floor, one kernel cell by cell. The only
    /// end-to-end figure in this journal is `#22`'s 205-324 ms/position, taken before
    /// the dense pin, the coalesced GEMV family, the arena's replacement rule and
    /// `#67`'s re-measurement of the same regime, so the numbers since are not
    /// comparable with it or with each other.
    ///
    /// This runs `DeepSeekV41TurnRunner` over the published checkpoint exactly as
    /// `CompletionEngine.launchTurn` does -- one chunked prefill from position 0, then
    /// a walk per drawn token -- and reports the wall the app's user waits on: prefill
    /// seconds divided into its blocks, the wait for the first token, decode seconds
    /// per position, and the host/device split of that decode from the executor's own
    /// counters. Each prompt gets a second turn that continues it, so the resume the
    /// app performs is measured rather than assumed: how much of the prompt the cache
    /// still holds, how many positions the turn really ran, and what those cost.
    ///
    /// Two prompt shapes, both the app's. The short thinking-off one keeps this
    /// comparable with every earlier slice. The app's DEFAULT (thinking on, effort
    /// high) is longer than one block, so its prefill shows the cadence rather than a
    /// single report.
    ///
    /// It also pins the composer's progress contract at the published sizes. A whole
    /// prompt is ONE call into the executor, and before the progress hook existed that
    /// call reported nothing until the last block drained: the status line stayed blank
    /// for the entire prefill and then jumped to a finished count. The assertions here
    /// are that contract -- reports climb, one per block when chunked and one per
    /// position when walked, and the last one is what the turn actually ran.
    ///
    /// The arena is WIRED (`#71`), which is the app's regime: `#69` first ran this arm
    /// with the harness's unwired default, where a fill costs twice as much, so its
    /// absolute walls are the slow regime's and not a turn a user waits on.
    func testWhatAWholeTurnActuallyCosts() async throws {
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
        func renderPrompt(thinking: Bool) -> [UInt32] {
            let text = dialect.render(
                turns: [.init(isUser: true, text: userText)],
                cleanSystem: "",
                thinkingEnabled: thinking,
                effort: thinking ? .high : nil
            )
            return (try? tokenizer.encode(text: text)) ?? []
        }
        let promptIds = renderPrompt(thinking: false)
        let thinkingIds = renderPrompt(thinking: true)
        XCTAssertFalse(promptIds.isEmpty, "the rendered prompt tokenized to nothing")
        XCTAssertFalse(thinkingIds.isEmpty, "the thinking-on prompt tokenized to nothing")
        print(
            "🔎 [dsv41 real] prompts: \(promptIds.count) token(s) thinking off, "
                + "\(thinkingIds.count) thinking on high"
        )

        let contextLength = max(promptIds.count, thinkingIds.count) + firstTurnTokens * 2 + 16
        let engine = try DynaMoeEngine(filePath: snapshot.path)
        // The SHIPPING arena, because this case exists to price the turn the app runs.
        // `makeRuntime` defaults to the unwired regime so the earlier slices compare with
        // each other, and that regime is not the app's: a fill into an unwired slot is a
        // page the compressor reconstructs, which `#56` measured at 4.6-6.7 GB/s against
        // 13.0 wired and `#71` confirmed on this very stream. The default is overridden
        // here and only here.
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: contextLength,
            allowEngramSkip: FileManager.default.fileExists(
                atPath: Self.engramOffMarkerPath
            ),
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let forward = runtime.forward()
        XCTAssertTrue(forward.isRunnable, forward.describe())
        // The walk's own buckets are what decompose the decode half. The block path
        // does not fill them, so a chunked prefill cannot pollute the columns; a WALKED
        // prefill adds its positions, which is why every print names the position count
        // it divided by.
        forward.recordsWalkTiming = true

        let settings = DeepSeekV41TurnRunner.Settings(
            temperature: 0,
            topP: 1,
            minP: 0,
            topK: 1,
            repetitionPenalty: 1,
            maxTokens: firstTurnTokens,
            eosTokenIds: config.eosTokenIDs.map { UInt32(clamping: $0) }
        )

        /// One turn, asked for exactly what the app asks for: the pin proposes every
        /// position but the last, and the executor decides how much it really holds.
        func runTurn(_ prompt: [UInt32]) throws -> TurnTrace {
            let reused = forward.reusablePrefix(
                proposedPrefix: max(0, prompt.count - 1),
                promptTokenIds: prompt.map { Int($0) }
            )
            forward.resetWalkTiming()
            let started = CFAbsoluteTimeGetCurrent()
            var blocks: [TurnPrefillBlock] = []
            var tokenSeconds: [Double] = []
            let output = try DeepSeekV41TurnRunner.run(
                forward: forward,
                promptTokenIds: prompt,
                prefixTokens: reused,
                settings: settings,
                sample: { logits, _ in Self.argmax(logits) },
                decode: { (try? tokenizer.decode(ids: $0)) ?? "" },
                onPrefill: { done, _ in
                    blocks.append(
                        TurnPrefillBlock(
                            done: done, seconds: CFAbsoluteTimeGetCurrent() - started
                        )
                    )
                },
                onText: { _, _, elapsed in tokenSeconds.append(elapsed) }
            )
            return TurnTrace(
                output: output, prefillBlocks: blocks, tokenSeconds: tokenSeconds
            )
        }

        /// Every column the app's user waits on, plus the split of the decode.
        func report(_ label: String, _ trace: TurnTrace, blockWidth: Int) {
            let out = trace.output
            let run = out.promptTokens - out.prefixTokensReused
            print(
                "🔎 [dsv41 real] \(label): \(out.promptTokens) prompt position(s), "
                    + "\(out.prefixTokensReused) reused, \(run) run"
            )
            print(
                "🔎 [dsv41 real]   prefill "
                    + String(
                        format: "%.2f s (%.1f ms/position over %d)", out.prefillSeconds,
                        out.prefillSeconds / Double(max(1, run)) * 1000, run
                    )
                    + ", \(trace.prefillBlocks.count) report(s)"
            )
            var previousSeconds = 0.0
            var previousDone = 0
            for block in trace.prefillBlocks {
                let positions = block.done - previousDone
                let seconds = block.seconds - previousSeconds
                print(
                    "🔎 [dsv41 real]     step \(block.done)/\(run): "
                        + String(
                            format: "%.2f s for %d position(s), %.1f ms each",
                            seconds, positions, seconds / Double(max(1, positions)) * 1000
                        )
                )
                previousSeconds = block.seconds
                previousDone = block.done
            }
            let timing = forward.walkTiming
            let hostNanoseconds = timing.frontEncodeNanoseconds + timing.frontWaitNanoseconds
                + timing.expertFillNanoseconds + timing.indexerWaitNanoseconds
                + timing.mixtureEncodeNanoseconds
            let deviceNanoseconds = timing.frontDeviceNanoseconds + timing.mixtureDeviceNanoseconds
            let walked = max(1, timing.positions)
            // A resumed turn walks its prefill, so the walk's own counters cover the
            // prefill positions too. Dividing the decode window by every walked
            // position would understate both, and once made the remainder negative.
            // The window these buckets belong to is prefill + decode when the prefill
            // walked, and the decode alone when it was one chunk.
            let walkedPrefill = out.prefixTokensReused == 0 ? 0 : run
            let wallMilliseconds = out.decodeSeconds * 1000
                + (walkedPrefill > 0 ? out.prefillSeconds * 1000 : 0)
            print(
                "🔎 [dsv41 real]   decode \(out.tokensGenerated) token(s) in "
                    + String(
                        format: "%.2f s, %.1f ms/position, %.1f tok/s",
                        out.decodeSeconds,
                        out.decodeSeconds * 1000 / Double(max(1, out.tokensGenerated)),
                        out.tokensPerSecond
                    )
                    + String(
                        format: "; user waits %.2f s for the first token",
                        out.prefillSeconds + (out.timeToFirstTokenSeconds ?? 0)
                    )
            )
            print(
                "🔎 [dsv41 real]   split over \(timing.positions) walked position(s) "
                    + "(\(walkedPrefill) prefill + \(timing.positions - walkedPrefill) decode): "
                    + String(
                        format: "%.1f ms wall, %.1f ms host (encode %.1f + front wait %.1f "
                            + "+ fill %.1f + indexer %.1f + mixture encode %.1f), "
                            + "%.1f ms of that host time is device windows, %.1f ms outside "
                            + "the walk",
                        wallMilliseconds / Double(walked),
                        Double(hostNanoseconds) / 1_000_000 / Double(walked),
                        Double(timing.frontEncodeNanoseconds) / 1_000_000 / Double(walked),
                        Double(timing.frontWaitNanoseconds) / 1_000_000 / Double(walked),
                        Double(timing.expertFillNanoseconds) / 1_000_000 / Double(walked),
                        Double(timing.indexerWaitNanoseconds) / 1_000_000 / Double(walked),
                        Double(timing.mixtureEncodeNanoseconds) / 1_000_000 / Double(walked),
                        Double(deviceNanoseconds) / 1_000_000 / Double(walked),
                        (wallMilliseconds - Double(hostNanoseconds) / 1_000_000)
                            / Double(walked)
                    )
            )
            print(
                "🔎 [dsv41 real]   turn wall "
                    + String(
                        format: "%.2f s (prefill %.2f + decode %.2f)",
                        out.prefillSeconds + out.decodeSeconds, out.prefillSeconds,
                        out.decodeSeconds
                    )
                    + String(
                        format: ", %.1f MB read",
                        Double(runtime.expertStreamer?.bytesRead ?? 0) / 1_048_576
                    )
            )
            print("🔎 [dsv41 real]   text: \(out.text.debugDescription)")

            // The composer's contract, at whatever size this turn ran: one report per
            // block when the prompt was chunked, one per position when it was walked.
            XCTAssertEqual(
                trace.prefillBlocks.map(\.done).last ?? -1, run,
                "\(label): the last prefill report must be the positions the turn ran"
            )
            XCTAssertEqual(
                trace.prefillBlocks.map(\.done), trace.prefillBlocks.map(\.done).sorted(),
                "\(label): progress must be monotonic"
            )
            let expected = out.prefixTokensReused == 0
                ? max(1, (run + blockWidth - 1) / blockWidth)
                : max(1, run)
            XCTAssertEqual(
                trace.prefillBlocks.count, expected,
                "\(label): one report per block when chunked, per position when walked"
            )
            XCTAssertGreaterThan(out.tokensGenerated, 0, "\(label): the turn produced no token")
            XCTAssertGreaterThan(out.prefillSeconds, 0, "\(label): prefill took no time")
            XCTAssertGreaterThan(out.decodeSeconds, 0, "\(label): decode took no time")
        }

        // The SHIPPED width, not the workspace's: since `#71` the rule is a constant four
        // at every prompt length, so it is resolved here for the widest prompt this case
        // runs and the cadence below is that many reports. Taking `chunkSlots` here read
        // the buffer size instead and asserted a report count the engine never produces.
        let blockWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: contextLength, promptTokens: 0
        )
        let first = try runTurn(promptIds)
        report(
            "turn 1 thinking off (\(userText.debugDescription))", first,
            blockWidth: blockWidth
        )
        XCTAssertEqual(first.output.prefixTokensReused, 0, "a cold cache cannot reuse anything")

        let continuation = promptIds + first.output.generatedTokenIds
        let second = try runTurn(continuation)
        report("turn 2 its continuation", second, blockWidth: blockWidth)
        XCTAssertEqual(second.output.promptTokens, continuation.count)
        if first.output.generatedTokenIds.isEmpty {
            XCTAssertEqual(second.output.prefixTokensReused, 0)
        } else {
            XCTAssertEqual(
                second.output.prefixTokensReused, continuation.count - 1,
                "a turn that just wrote these tokens can skip all but the last"
            )
        }

        let third = try runTurn(thinkingIds)
        report("turn 3 app default shape", third, blockWidth: blockWidth)
        XCTAssertEqual(
            third.output.prefixTokensReused, 0, "a different prompt is a full prefill"
        )
        XCTAssertGreaterThan(
            third.prefillBlocks.count, 1,
            "the app's default prompt must span more than one block, or it measures nothing"
        )

        let fourth = try runTurn(thinkingIds + third.output.generatedTokenIds)
        report("turn 4 its continuation", fourth, blockWidth: blockWidth)

        print(
            "🔎 [dsv41 real] arena after four turns: "
                + "\(runtime.arena.residentCount)/\(runtime.arena.capacity) resident, "
                + String(
                    format: "%.1f%% hit over %d acquisition(s)",
                    runtime.arena.hitRate * 100, runtime.arena.acquisitions
                )
        )
        print("🔎 [dsv41 real] \(runtime.expertStreamer?.describe() ?? "no streamer")")
        withExtendedLifetime(engine) {}
    }
    // MARK: - #70: where a text prefix's per-position cost goes

    /// One prefill arm's whole trace: the wall the user waits on, what it read, and
    /// every counter that can attribute it.
    struct PrefixArm: Sendable {
        var label: String
        var length: Int
        var width: Int
        var isWalk: Bool
        var seconds: Double
        var positions: Int
        var bytes: Double
        var misses: Int
        var dispatches: Int
        var dispatchTable: [String: Int]
        var blocks: Int
        var layerPasses: Int
        var blockMixtures: Int
        var perPositionMixtures: Int
        var blockTiming: DeepSeekV41BlockTiming?
        var walkTiming: DeepSeekV41WalkTiming?
        var lastLogits: [Float]

        var millisecondsPerPosition: Double {
            seconds * 1000 / Double(max(1, positions))
        }

        /// Milliseconds per LAYER PASS. A walked arm pays one per position; a block
        /// arm pays one per block however many positions it carries, so dividing it by
        /// the block width gives the per-position figure batching is supposed to lower.
        var millisecondsPerLayerPass: Double {
            seconds * 1000 / Double(max(1, layerPasses))
        }

        var millisecondsPerBlock: Double {
            seconds * 1000 / Double(max(1, blocks))
        }

        var dispatchesPerPosition: Double {
            Double(dispatches) / Double(max(1, positions))
        }
    }

    /// Where a text prefill's per-position cost goes, and whether a block amortises any
    /// of it.
    ///
    /// `#69` measured the app's own turn and found a prefill position costs what a
    /// decode position costs (187-281 against 138-183 ms). That is a statement about
    /// the shipped layout, which chunks a prompt into blocks of eight and, per layer,
    /// pays ONE dense-half wait, ONE union fill and ONE mixture submission for the whole
    /// block. Eight positions for the price of one ought to be cheap; the numbers say
    /// they are not, and no slice since `#53` has measured the SHAPE of that.
    ///
    /// So this runs the walk (the per-position baseline) and the block path at widths
    /// 1, 2, 4 and the shipped 8 over the same 128-token stream, in one process, with
    /// the arm order rotated every round so no width owns the young-process advantage.
    /// A width-1 block is the block path with nothing to batch, so the difference
    /// between it and the walk is the block path's own overhead, and the difference
    /// between it and width 8 is what batching is worth on this checkpoint. It then
    /// repeats the comparison at the app's own prompt length, where a prompt is four
    /// blocks at the shipped width and the hand-off count is what a change to that
    /// width would move.
    ///
    /// Three columns decide the question:
    ///
    /// - **milliseconds per layer pass**, divided by the block width. The walked arm
    ///   pays 40 per position; a block pays 40 per block. A flat column means the work
    ///   inside a layer scales with positions and no layout can amortise it.
    /// - **dispatches per position**, from the census. The block path dispatches the
    ///   stages that cannot batch once per SLOT, so if its count per position is not
    ///   below the walk's, host encode cannot amortise either. This is the census's
    ///   first run on the block path: `encodeBlockStage` never labelled its stages, and
    ///   `dispatchCensusPositions` never counted a block's positions.
    /// - **the buckets**, now comparable across the two layouts. The block's
    ///   `denseEncode` used to fold the indexer split (a GPU drain plus a host
    ///   readback) into an encode, which made a block read 28.6 ms of encode per
    ///   position against a walked position's ~3.5; it is `indexerWait` now, its own
    ///   column, exactly as the walk reports it.
    ///
    /// The arena is reset before every arm, so every arm is cold against the page cache
    /// and reads the same bytes: that is the regime a prompt meets, and two arms that
    /// read different bytes would not be comparable at all. Byte counts are printed and
    /// asserted stable across rounds for the same arm.
    ///
    /// This runs the harness's UNWIRED arena (`makeRuntime`'s default), which is how
    /// `#20`-`#69` compare like with like and NOT the regime the app runs. Every absolute
    /// figure here is therefore about twice the shipping one: `#71` re-ran these arms
    /// wired and got 132.9 / 114.0 / 113.6 ms per position at 128 positions and 207.5 /
    /// 182.5 / 187.0 at 31, where the fill reads at 12.2-13.7 GB/s instead of 7.7-8.5.
    /// What survives from this table is the ORDER and the fact that the fill was the
    /// largest bucket; what does not is the 31 ms/position "gap" it has left open.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_prefix_time`): two lengths, twenty-four passes of up
    /// to 128 positions, ~15 minutes at `-O`.
    /// `/tmp/dynamoe_dsv41_prefix_time_quick` runs one round of 32 positions instead.
    func testWhereATextPrefixsPerPositionCostGoesOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_prefix_time"
        let quickMarkerPath = "/tmp/dynamoe_dsv41_prefix_time_quick"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in prefill width attribution; touch \(markerPath) to run"
        )
        let quick = FileManager.default.fileExists(atPath: quickMarkerPath)
        let rounds = quick ? 1 : 3
        let longest = quick ? 32 : 128
        // The app's own prompt is 31 positions (its DEFAULT thinking-on shape, which
        // `#69` measured at four blocks). The long arm is the same stream `#52`-`#59`
        // all used, so its numbers are comparable with every prefill table in the
        // journal; the short arm is the one a user actually waits on.
        // The 64-position arm exists to bracket the crossover between the two widths:
        // a prompt's routed working set fits the arena at 31 positions and does not at
        // 128, and the width that wins follows that, not the length.
        let lengths = quick ? [32] : [128, 64, 31]

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
        // The paragraph repeated with a rotation per repetition, so the sequence is not
        // periodic: a plain repeat cycles the same routing and leaves the arena warm.
        func prompt(_ length: Int) -> [Int] {
            (0..<length).map { index -> Int in
                let repetition = index / stream.count
                return stream[(index + repetition * 13) % stream.count]
            }
        }

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
        let shippedWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: longest
        )
        print(
            "🔎 [dsv41 real] prefix attribution: lengths \(lengths), \(rounds) round(s), "
                + "shipped block width \(shippedWidth), arena \(runtime.arena.capacity) slot(s)"
        )

        /// One arm. `width` nil is the walk, which is the per-position baseline and the
        /// path a resumed turn takes; anything else is the block path truncating the
        /// stream into chunks of that many tokens, which is how a prompt is laid out.
        func run(width armWidth: Int?, tokens: [Int], warm: Bool = false) throws -> PrefixArm {
            executor.reset()
            // A cold arm drops the arena so it re-reads every expert; a warm arm leaves
            // what the previous pass put there, which is what a follow-up turn meets.
            if !warm { runtime.arena.reset() }
            executor.resetBlockTiming()
            executor.resetWalkTiming()
            executor.resetDispatchCensus()
            executor.recordsBlockTiming = armWidth != nil
            executor.recordsWalkTiming = armWidth == nil
            executor.countsDispatches = true
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let blockMixturesBefore = executor.blockMixtureSubmissions
            let perPositionMixturesBefore = executor.perPositionMixtureSubmissions
            let started = DispatchTime.now().uptimeNanoseconds
            var last: [Float] = []
            if let blockWidth = armWidth {
                var offset = 0
                while offset < tokens.count {
                    let end = min(offset + blockWidth, tokens.count)
                    last = try executor.prefillChunk(
                        Array(tokens[offset..<end]), blockWidth: blockWidth
                    ).last
                    offset = end
                }
            } else {
                for (position, token) in tokens.enumerated() {
                    last = try executor.forwardToken(token, position: position)
                }
            }
            let seconds = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000_000
            executor.recordsBlockTiming = false
            executor.recordsWalkTiming = false
            executor.countsDispatches = false
            let blockTiming = executor.blockTiming
            let walkTiming = executor.walkTiming
            let label = armWidth.map { "block \($0)" } ?? "walk"
            return PrefixArm(
                label: label,
                length: tokens.count,
                width: armWidth ?? 1,
                isWalk: armWidth == nil,
                seconds: seconds,
                positions: tokens.count,
                bytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                dispatches: executor.dispatchCensus.values.reduce(0, +),
                dispatchTable: executor.dispatchCensus,
                blocks: armWidth != nil ? blockTiming.blocks : walkTiming.positions,
                layerPasses: armWidth != nil ? blockTiming.layers : walkTiming.layers,
                blockMixtures: executor.blockMixtureSubmissions - blockMixturesBefore,
                perPositionMixtures: executor.perPositionMixtureSubmissions
                    - perPositionMixturesBefore,
                blockTiming: armWidth != nil ? blockTiming : nil,
                walkTiming: armWidth == nil ? walkTiming : nil,
                lastLogits: last
            )
        }

        /// Relative difference against the reference arm's own logit scale.
        func relativeDifference(_ reference: [Float], _ other: [Float]) -> Double {
            guard !reference.isEmpty, reference.count == other.count else {
                return .infinity
            }
            let scale = max(reference.map { abs(Double($0)) }.max() ?? 0, 1e-6)
            let worst = zip(reference, other)
                .map { abs(Double($0) - Double($1)) }
                .max() ?? 0
            return worst / scale
        }

        for length in lengths {
            let tokens = prompt(length)
            // The app's own length does not need the whole sweep, only the two widths a
            // decision would choose between; the long arm pays for the four.
            var widths: [Int?] = length == longest ? [nil, 1, 2, 4] : [nil, 4]
            // `shippedWidth` is four since `#71`, so it is usually already in the list;
            // appending it blindly measured the same width twice a round.
            if !widths.contains(shippedWidth) { widths.append(shippedWidth) }
            var steady: [String: [PrefixArm]] = [:]
            var bytesByLabel: [String: [Double]] = [:]

            for round in 0..<rounds {
                // Rotated, because a pass that runs first in a young process is 1.3-1.6x
                // faster in every bucket (`#53`, and `#55`/`#56` traced it to the
                // arena's destination pages rather than to any code path) and that
                // artefact would otherwise decide the sweep.
                let order = (0..<widths.count).map { widths[($0 + round) % widths.count] }
                var results: [PrefixArm] = []
                for armWidth in order {
                    let arm = try run(width: armWidth, tokens: tokens)
                    results.append(arm)
                    bytesByLabel[arm.label, default: []].append(arm.bytes)
                    if round > 0 { steady[arm.label, default: []].append(arm) }
                    print(
                        String(
                            format: "🔎 [dsv41 real]   %d pos, round %d %-7@ %6.2f s, "
                                + "%6.1f ms/position, %6.2f ms/layer/position, %.0f MB, "
                                + "%d misses, %.0f dispatch(es)/position, "
                                + "mixtures %d block + %d per position",
                            length,
                            round + 1,
                            arm.label as NSString,
                            arm.seconds,
                            arm.millisecondsPerPosition,
                            arm.millisecondsPerLayerPass / Double(arm.width),
                            arm.bytes,
                            arm.misses,
                            arm.dispatchesPerPosition,
                            arm.blockMixtures,
                            arm.perPositionMixtures
                        )
                    )
                    if let timing = arm.blockTiming {
                        print("🔎 [dsv41 real]     " + timing.describe())
                        XCTAssertGreaterThan(
                            timing.positionCount, 0, "\(arm.label) at \(length) measured nothing"
                        )
                    }
                    if let timing = arm.walkTiming {
                        print("🔎 [dsv41 real]     " + timing.describe())
                        XCTAssertGreaterThan(
                            timing.positions, 0, "\(arm.label) at \(length) measured nothing"
                        )
                    }
                }
                // Every width runs the same graph, so every width must land on the same
                // logits at the last position. The bar is loose on purpose: a width
                // change re-associates every reduction and can flip a routing tie, and
                // the legitimate widths measured 5.3e-3 to 1.1e-2 apart at a logit scale
                // of ~20 on this stream (128 and 31 positions), while a stage that was
                // skipped or a slot that read another position's rows is orders of
                // magnitude larger. The argmax must agree exactly, which is the check a
                // real defect fails.
                let reference = try XCTUnwrap(
                    results.first(where: { !$0.isWalk && $0.width == shippedWidth })?.lastLogits
                )
                let referenceChoice = Self.argmax(reference)
                for arm in results where arm.isWalk || arm.width != shippedWidth {
                    let difference = relativeDifference(reference, arm.lastLogits)
                    XCTAssertLessThan(
                        difference, 5e-2,
                        "\(arm.label) at \(length) disagreed with block \(shippedWidth) "
                            + "by \(difference) relative"
                    )
                    XCTAssertEqual(
                        Self.argmax(arm.lastLogits), referenceChoice,
                        "\(arm.label) at \(length) chose a different token"
                    )
                }
                XCTAssertGreaterThan(
                    results.first(where: \.isWalk)?.dispatches ?? 0, 0,
                    "the walk's census is empty"
                )
                for arm in results where !arm.isWalk {
                    XCTAssertGreaterThan(
                        arm.dispatches, 0,
                        "\(arm.label) at \(length) reported no dispatch, so its stages "
                            + "are unlabelled"
                    )
                }
            }

            // The byte counts are the harness's own determinism check: an arm that read
            // a different number of bytes between two rounds is not a stable regime.
            for (label, counts) in bytesByLabel.sorted(by: { $0.key < $1.key }) {
                let low = counts.min() ?? 0
                let high = counts.max() ?? 0
                XCTAssertLessThan(
                    high - low, max(high * 0.005, 1),
                    "\(length) \(label) read \(low)-\(high) MB across rounds, "
                        + "so the rounds are not comparable"
                )
            }

            // The steady state, one row per width: this is the table that says whether a
            // block amortises the per-pass work or pays it once per position.
            let walkRow: PrefixArm
            if let recorded = steady["walk"]?.first {
                walkRow = recorded
            } else {
                walkRow = try run(width: nil, tokens: tokens)
            }
            print("🔎 [dsv41 real] steady state at \(length) positions (rounds 2+):")
            for label in [walkRow.label] + widths.compactMap({ $0.map { "block \($0)" } }) {
                let arms = label == walkRow.label ? [walkRow] : (steady[label] ?? [])
                guard !arms.isEmpty else { continue }
                let positions = arms.map(\.millisecondsPerPosition).reduce(0, +)
                    / Double(arms.count)
                let passes = arms.map(\.millisecondsPerLayerPass).reduce(0, +)
                    / Double(arms.count)
                let blocks = arms.map(\.millisecondsPerBlock).reduce(0, +) / Double(arms.count)
                let bytes = arms.map(\.bytes).reduce(0, +) / Double(arms.count)
                let dispatches = arms.map(\.dispatchesPerPosition).reduce(0, +)
                    / Double(arms.count)
                print(
                    String(
                        format: "🔎 [dsv41 real]   %-7@ %7.1f ms/position, %6.2f ms/layer pass, "
                            + "%7.1f ms/block, %.0f MB, %.0f dispatch(es)/position, %.2fx the walk",
                        label as NSString,
                        positions,
                        passes,
                        blocks,
                        bytes,
                        dispatches,
                        walkRow.millisecondsPerPosition / max(positions, 0.001)
                    )
                )
            }

        // The same two widths against an arena the previous passes filled.
        //
        // This is the arm the decision turns on. Every arm above resets the arena, so
        // every one of them pays the full expert read, and in that regime the width
        // barely matters: at 31 positions four and eight are a percent apart because
        // both are dominated by 47 GB of routed experts. A live turn meets the other
        // regime - turn 3 of `#69` follows two turns that left two thousand experts
        // resident - and there the block's own per-layer overhead is what a width
        // changes. It is measured here rather than argued, because getting it wrong is
        // a 15% regression on the prompt a user actually waits on.
        for length in lengths {
            let tokens = prompt(length)
            for round in 0..<max(1, rounds - 1) {
                for armWidth in [4, shippedWidth] {
                    let arm = try run(width: armWidth, tokens: tokens, warm: true)
                    print(
                        String(
                            format: "🔎 [dsv41 real]   %d pos, WARM round %d %-7@ %6.2f s, "
                                + "%6.1f ms/position, %.0f MB, %d misses",
                            length,
                            round + 1,
                            arm.label as NSString,
                            arm.seconds,
                            arm.millisecondsPerPosition,
                            arm.bytes,
                            arm.misses
                        )
                    )
                    if let timing = arm.blockTiming {
                        print("🔎 [dsv41 real]     " + timing.describe())
                    }
                }
            }
        }
            // Which stage holds the block path's dispatch count, at the widest width: the
            // per-slot loop is the candidate that would have to become one dispatch for
            // host encode to amortise with width.
            if length == longest {
                let censusArm: PrefixArm
                if let recorded = steady["block \(shippedWidth)"]?.first {
                    censusArm = recorded
                } else {
                    censusArm = try run(width: shippedWidth, tokens: tokens)
                }
                let censusTotal = max(censusArm.dispatchTable.values.reduce(0, +), 1)
                print(
                    "🔎 [dsv41 real] block \(shippedWidth) dispatch census, "
                        + "\(censusArm.dispatchTable.count) stage(s), "
                        + "\(censusTotal) dispatch(es) total:"
                )
                for (stage, count) in censusArm.dispatchTable
                    .sorted(by: { $0.value > $1.value })
                    .prefix(12)
                {
                    print(
                        String(
                            format: "🔎 [dsv41 real]   %-38@ %6d  %5.1f%%",
                            stage as NSString,
                            count,
                            Double(count) / Double(censusTotal) * 100
                        )
                    )
                }
            }
        }

        withExtendedLifetime(engine) {}
    }
    // MARK: - #71: which arena regime a prefill is measured in

    /// The streamer's counters at one instant, so a fill can be billed from the inside.
    struct StreamerMarks: Sendable {
        var bytesRead: Int
        var misses: Int
        var reading: Double
        var allocating: Double
        var slots: Int
        var wiring: Double
        var placing: Double
        var placements: Int
        var preparing: Double
        var preparations: Int
    }

    /// One arm's fill, read two ways: the bucket the timing struct reports, and what the
    /// streamer measured while it was filling.
    ///
    /// `#70` priced a prefill's expert fill off the OUTER bucket — the wall around
    /// `ensureResident` — and got 7.7-8.5 GB/s against a 13-14 GB/s ceiling, and left the
    /// ~31 ms/position difference as the largest open number in the journal. That bucket
    /// is not only the read: it is arena placement, the slot's first allocation, `mlock`
    /// wiring, the resolve of six byte ranges per expert, and the `pread`. Only the last
    /// of those is the disk, and `readMisses` times it separately.
    struct FillSplit {
        /// Every fill of one size, kept together so a rate can be read per shape.
        struct Group {
            var fills: Int
            var ranges: Int
            var megabytes: Double
            var seconds: Double
        }

        var label: String
        var positions: Int
        var wallMilliseconds: Double
        /// The timing struct's own fill bucket, per position.
        var outerMilliseconds: Double
        /// `readMisses`'s wall, which is the `pread` fan-out and nothing else.
        var readMilliseconds: Double
        var placeMilliseconds: Double
        /// Everything else inside `placeResident`. It CONTAINS allocation and wiring, so
        /// the two are printed nested rather than added.
        var prepareMilliseconds: Double
        var allocateMilliseconds: Double
        var wiringMilliseconds: Double
        var misses: Int
        var fills: Int
        var ranges: Int
        /// What `bytesRead` counted, i.e. misses x slot size.
        var slotMegabytes: Double
        /// What the ranges actually asked the file for.
        var rangeMegabytes: Double
        var byMisses: [Int: Group]
        var lastLogits: [Float]

        var missesPerPosition: Double { Double(misses) / Double(max(1, positions)) }
        var missesPerFill: Double { Double(misses) / Double(max(1, fills)) }
        var millisecondsPerPosition: Double {
            wallMilliseconds / Double(max(1, positions))
        }

        /// GB/s over the slot bytes `bytesRead` counts, which is the convention `#70`
        /// used and therefore the one its 7.7-8.5 is comparable with. A megabyte per
        /// millisecond IS a gigabyte per second, so this is a plain division.
        var outerGigabytesPerSecond: Double {
            outerMilliseconds > 0
                ? slotMegabytes / Double(max(1, positions)) / outerMilliseconds : 0
        }

        /// The same over the ranges' true bytes, which is the number `#51`'s isolated
        /// fill quotes.
        var readGigabytesPerSecond: Double {
            readMilliseconds > 0
                ? rangeMegabytes / Double(max(1, positions)) / readMilliseconds : 0
        }

        /// What the outer bucket holds beyond the read, per position.
        var bookkeepingMilliseconds: Double {
            max(0, outerMilliseconds - readMilliseconds)
        }
    }

    /// What regime a prefill's expert fill is measured in, and whether the width rule
    /// `#70` shipped survives the one production actually runs.
    ///
    /// `#70` left one number open: at the best prefill width a 128-position prompt spends
    /// 81.1 ms/position filling the arena, 703 MB/position at 7.7-8.5 GB/s against this
    /// machine's 13-14 GB/s ceiling. The same journal had already measured a cold WALKED
    /// position, same bytes and same misses, at 15.1 GB/s (`#67`), and `#70` itself
    /// recorded the app's own turn moving 5.79 s to 8.57 s to 6.69 s on byte-identical
    /// reads. Two processes, one workload, 2x apart.
    ///
    /// The difference is not the layout and not the disk. It is the ARENA: `#56` measured
    /// the same bytes into the same slots at **4.6-6.7 GB/s unwired against 13.0 wired**,
    /// because a slot that is not `mlock`ed is a page the compressor reconstructs after
    /// every pass that rewrote it. Production wires the arena whenever the memory plan
    /// grants the budget (`DeepSeekV41Engine.swift`, `wireArena ?? !wireArenaDisabled`),
    /// and the harness's `makeRuntime` defaults to **`wireArena: false`** precisely so
    /// that `#20`-`#56` compare like with like. `#70`'s width sweep, its bucket table and
    /// the 31 ms gap all inherited that default, so the gap is the harness's regime and
    /// not the app's.
    ///
    /// That matters beyond the number: if a fill is twice as expensive in the regime the
    /// sweep ran in, the balance the width rule was derived from is the wrong balance. So
    /// this measures the fill AND the width curve in both regimes, shipping first:
    ///
    /// - the streamer's own clocks, with `recordsFillSamples` on, tile each fill into
    ///   placement, preparation, allocation, wiring and the `pread` itself;
    /// - a CALIBRATION runs the same `ensureResident` cold at 6 and 24 experts in the same
    ///   process, because the only honest ceiling for an in-situ rate is a rate this
    ///   process measured;
    /// - the walk and widths 4 and 8 at 128 positions (where `#70` chose four) and at 31
    ///   (where it chose eight), in the shipping regime only. An unwired arm was tried
    ///   and withdrawn: run second in one process it collapses to 0.9 GB/s, which is a
    ///   second 65.84 GB arena in a process that just released a wired one and not a
    ///   regime. The contrast is `#70`'s own table and `/tmp/test71_fill1.log`.
    ///
    /// Opt-in (`/tmp/dynamoe_dsv41_fill_regime`), two measured rounds and a discarded
    /// warm-up: ~13 minutes at `-O`.
    /// `/tmp/dynamoe_dsv41_fill_regime_quick` runs one round of 24 positions instead.
    func testWhatRegimeAPrefillsFillIsMeasuredInOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_fill_regime"
        let quickMarkerPath = "/tmp/dynamoe_dsv41_fill_regime_quick"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in prefill fill regime; touch \(markerPath) to run"
        )
        let quick = FileManager.default.fileExists(atPath: quickMarkerPath)
        // 64 is the crossover candidate: `#70` put its threshold there on the strength of a
        // working-set measurement, and a shipping-regime rule may not inherit a threshold
        // whose crossover was located in a different regime.
        let lengths = quick ? [24] : [128, 64, 31]
        let longest = lengths[0]
        // ONE regime, the app's. An unwired arm was tried here and withdrawn: run after
        // a wired one in the same process it collapses to 0.9 GB/s (a second 65.84 GB
        // arena in a process that just released one), so it measures memory pressure
        // rather than a regime. The unwired contrast comes from `#70`'s own table and
        // from a fresh process that ran nothing else (`/tmp/test71_fill1.log`: 5.9-6.9
        // GB/s), which is what `#56` said it would be.
        let shippingWiring: Bool? = nil
        let regime = (label: "shipping", wireArena: shippingWiring, rounds: quick ? 1 : 2)

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
        // The same paragraph and rotation `#52`-`#70` used, so this arm's bytes and misses
        // are comparable with every prefill table in the journal.
        let text = """
            A mixture of experts layer routes each token to a small subset of its \
            parameters, so the memory a decode step touches is a fraction of the \
            model, and the fraction is chosen by a router rather than by the layout.
            """
        let stream = try tokenizer.encode(text: text).map { Int($0) }
        XCTAssertGreaterThan(stream.count, 8)
        func prompt(_ length: Int) -> [Int] {
            (0..<length).map { index -> Int in
                let repetition = index / stream.count
                return stream[(index + repetition * 13) % stream.count]
            }
        }

        /// Every arm of one regime, in one runtime. The runtime is local to this function
        /// so the arena it holds — 65.84 GB of wired shared memory — is released before
        /// the next regime builds its own; two of them do not fit on this machine.
        func measure(regime label: String, wireArena: Bool?, rounds: Int) throws -> [FillSplit] {
            let engine = try DynaMoeEngine(filePath: snapshot.path)
            let runtime = try makeRuntime(
                snapshot: snapshot,
                engine: engine,
                config: config,
                contextLength: longest + 1,
                allowEngramSkip: false,
                wireArena: wireArena
            )
            XCTAssertTrue(runtime.isRunnable, runtime.describe())
            let executor = runtime.forward()
            let streamer = try XCTUnwrap(runtime.expertStreamer)
            streamer.recordsFillSamples = true
            let wiring = streamer.wiringState
            let slotMegabytes = Double(streamer.geometry.slotBytes) / 1_048_576
            print(
                "🔎 [dsv41 real] regime \(label): arena \(runtime.arena.capacity) slot(s) of "
                    + String(format: "%.2f", slotMegabytes) + " MB, wiring "
                    + (wiring.isEnabled ? "ENABLED" : (wiring.isRefused ? "REFUSED" : "off"))
                    + String(
                        format: " (%.2f of %.2f GB, %d slot(s))",
                        Double(wiring.bytes) / 1_073_741_824,
                        Double(wiring.budgetBytes) / 1_073_741_824,
                        wiring.slots
                    )
            )

            func mark() -> StreamerMarks {
                let timing = streamer.timing
                let placing = streamer.placementTiming
                let preparing = streamer.prepareTiming
                return StreamerMarks(
                    bytesRead: streamer.bytesRead,
                    misses: streamer.expertsLoaded,
                    reading: timing.reading,
                    allocating: timing.allocating,
                    slots: timing.slotsAllocated,
                    wiring: timing.wiring,
                    placing: placing.seconds,
                    placements: placing.placements,
                    preparing: preparing.seconds,
                    preparations: preparing.preparations
                )
            }

            func groups(_ samples: [DeepSeekV41FillSample]) -> [Int: FillSplit.Group] {
                var byMisses: [Int: FillSplit.Group] = [:]
                for sample in samples {
                    var entry = byMisses[sample.misses]
                        ?? FillSplit.Group(fills: 0, ranges: 0, megabytes: 0, seconds: 0)
                    entry.fills += 1
                    entry.ranges += sample.ranges
                    entry.megabytes += Double(sample.bytes) / 1_048_576
                    entry.seconds += sample.seconds
                    byMisses[sample.misses] = entry
                }
                return byMisses
            }

            /// One pass, with the arena dropped first so it re-reads every expert. A
            /// width of nil is the walk, which is the per-position baseline.
            func arm(tokens: [Int], width: Int?) throws -> FillSplit {
                executor.reset()
                runtime.arena.reset()
                streamer.clearFillSamples()
                executor.resetBlockTiming()
                executor.resetWalkTiming()
                executor.recordsBlockTiming = width != nil
                executor.recordsWalkTiming = width == nil
                let before = mark()
                let started = DispatchTime.now().uptimeNanoseconds
                var last: [Float] = []
                if let blockWidth = width {
                    var offset = 0
                    while offset < tokens.count {
                        let end = min(offset + blockWidth, tokens.count)
                        last = try executor.prefillChunk(
                            Array(tokens[offset..<end]), blockWidth: blockWidth
                        ).last ?? []
                        offset = end
                    }
                } else {
                    for (position, token) in tokens.enumerated() {
                        last = try executor.forwardToken(token, position: position)
                    }
                }
                let seconds = Double(
                    DispatchTime.now().uptimeNanoseconds - started
                ) / 1_000_000_000
                let after = mark()
                executor.recordsBlockTiming = false
                executor.recordsWalkTiming = false
                let samples = streamer.fillSamples
                let positions = Double(max(1, tokens.count))
                let scale = 1_000.0 / positions
                let outerNanoseconds = width != nil
                    ? executor.blockTiming.expertFillNanoseconds
                    : executor.walkTiming.expertFillNanoseconds
                return FillSplit(
                    label: width.map { "block \($0)" } ?? "walk",
                    positions: tokens.count,
                    wallMilliseconds: seconds * 1_000,
                    outerMilliseconds: Double(outerNanoseconds) / 1_000_000 / positions,
                    readMilliseconds: (after.reading - before.reading) * scale,
                    placeMilliseconds: (after.placing - before.placing) * scale,
                    prepareMilliseconds: (after.preparing - before.preparing) * scale,
                    allocateMilliseconds: (after.allocating - before.allocating) * scale,
                    wiringMilliseconds: (after.wiring - before.wiring) * scale,
                    misses: after.misses - before.misses,
                    fills: samples.count,
                    ranges: samples.reduce(0) { $0 + $1.ranges },
                    slotMegabytes: Double(after.bytesRead - before.bytesRead) / 1_048_576,
                    rangeMegabytes: samples.reduce(0) { $0 + Double($1.bytes) } / 1_048_576,
                    byMisses: groups(samples),
                    lastLogits: last
                )
            }

            /// The same fill a prefill runs, with nothing else on the machine, from a
            /// `firstPair` the caller rotates so no calibration re-reads another's pairs
            /// out of the page cache (which is what made the first cut of this probe
            /// report 86 GB/s).
            func calibrate(
                label: String, perFill: Int, fills: Int, firstPair: Int
            ) throws -> FillSplit {
                runtime.arena.reset()
                streamer.clearFillSamples()
                let before = mark()
                let started = DispatchTime.now().uptimeNanoseconds
                for index in 0..<fills {
                    let layer = [0, 7, 13, 21, 29, 33, 36, 39][index % 8]
                    let base = (firstPair + index * perFill) % max(config.numRoutedExperts - perFill, 1)
                    let experts = (0..<perFill).map { base + $0 }
                    _ = try streamer.ensureResident(layer: layer, experts: experts)
                }
                let seconds = Double(
                    DispatchTime.now().uptimeNanoseconds - started
                ) / 1_000_000_000
                let after = mark()
                let samples = streamer.fillSamples
                let positions = Double(max(1, fills))
                let scale = 1_000.0 / positions
                return FillSplit(
                    label: label,
                    positions: fills,
                    wallMilliseconds: seconds * 1_000,
                    outerMilliseconds: seconds * 1_000 / positions,
                    readMilliseconds: (after.reading - before.reading) * scale,
                    placeMilliseconds: (after.placing - before.placing) * scale,
                    prepareMilliseconds: (after.preparing - before.preparing) * scale,
                    allocateMilliseconds: (after.allocating - before.allocating) * scale,
                    wiringMilliseconds: (after.wiring - before.wiring) * scale,
                    misses: after.misses - before.misses,
                    fills: samples.count,
                    ranges: samples.reduce(0) { $0 + $1.ranges },
                    slotMegabytes: Double(after.bytesRead - before.bytesRead) / 1_048_576,
                    rangeMegabytes: samples.reduce(0) { $0 + Double($1.bytes) } / 1_048_576,
                    byMisses: groups(samples),
                    lastLogits: []
                )
            }

            func report(_ split: FillSplit, unit: String) {
                print(
                    "🔎 [dsv41 real]   " + split.label + " " + unit + "s: fill bucket "
                        + String(format: "%.2f", split.outerMilliseconds) + " ms = read "
                        + String(format: "%.2f", split.readMilliseconds) + " + place "
                        + String(format: "%.2f", split.placeMilliseconds) + " + prepare "
                        + String(format: "%.2f", split.prepareMilliseconds) + " (of it create "
                        + String(format: "%.2f", split.allocateMilliseconds) + ", wire "
                        + String(format: "%.2f", split.wiringMilliseconds) + ") + "
                        + String(
                            format: "%.2f unaccounted; %.1f misses/%@ in %.0f fill(s) of "
                                + "%.1f (%.0f range(s)); %.0f MB at %.1f GB/s slot, "
                                + "%.1f GB/s asked",
                            split.bookkeepingMilliseconds - split.placeMilliseconds
                                - split.prepareMilliseconds,
                            split.missesPerPosition,
                            unit,
                            Double(split.fills),
                            split.missesPerFill,
                            split.fills > 0 ? Double(split.ranges) / Double(split.fills) : 0,
                            split.slotMegabytes,
                            split.outerGigabytesPerSecond,
                            split.readGigabytesPerSecond
                        )
                )
                for (misses, entry) in split.byMisses.sorted(by: { $0.key < $1.key }) {
                    print(
                        "🔎 [dsv41 real]     fills of " + String(misses) + ": "
                            + String(format: "%3d fill(s), %5.1f range(s), %7.1f MB, "
                                + "%6.2f ms each, %5.1f GB/s",
                                entry.fills,
                                entry.fills > 0
                                    ? Double(entry.ranges) / Double(entry.fills) : 0,
                                entry.megabytes,
                                entry.seconds / Double(max(1, entry.fills)) * 1_000,
                                entry.seconds > 0
                                    ? entry.megabytes / entry.seconds / 1_000 : 0)
                    )
                }
            }

            // Two shapes, measured last of all so the arms' own churn cannot flatter
            // them: six experts is a walked layer's set, twenty-four is a width-four
            // block's union. Distinct pairs from the arms and from each other.
            var results: [FillSplit] = []
            var bytesByLabel: [String: [Double]] = [:]
            for length in lengths {
                let tokens = prompt(length)
                var steady: [FillSplit] = []
                for round in 0..<(rounds + 1) {
                    let order = [nil, 4, 8]
                    for offset in 0..<order.count {
                        let width = order[(offset + round) % order.count]
                        let split = try arm(tokens: tokens, width: width)
                        print(
                            "🔎 [dsv41 real]   " + label + " " + String(length)
                                + " pos round " + String(round + 1) + " "
                                + split.label + String(
                                    format: " %7.1f ms/position, %.0f MB, %d misses",
                                    split.millisecondsPerPosition,
                                    split.slotMegabytes,
                                    split.misses
                                )
                        )
                        report(split, unit: "position")
                        if width != nil {
                            print("🔎 [dsv41 real]     " + executor.blockTiming.describe())
                        } else {
                            print("🔎 [dsv41 real]     " + executor.walkTiming.describe())
                        }
                        // The first round runs in a young process, where every bucket is
                        // 1.3-1.6x faster (`#53`), so it is printed and never averaged.
                        guard round > 0 else { continue }
                        steady.append(split)
                        bytesByLabel[split.label + " @ " + String(length), default: []]
                            .append(split.slotMegabytes)
                    }
                }

                print(
                    "🔎 [dsv41 real]   " + label + " at " + String(length)
                        + " positions, " + String(rounds) + " measured round(s):"
                )
                var walkMilliseconds: Double?
                for armLabel in ["walk", "block 4", "block 8"] {
                    let splits = steady.filter { $0.label == armLabel }
                    guard !splits.isEmpty else { continue }
                    let count = Double(splits.count)
                    let wall = splits.map(\.millisecondsPerPosition).reduce(0, +) / count
                    let outer = splits.map(\.outerMilliseconds).reduce(0, +) / count
                    let read = splits.map(\.readMilliseconds).reduce(0, +) / count
                    let bytes = splits.map(\.slotMegabytes).reduce(0, +) / count
                    let misses = Double(splits.map(\.misses).reduce(0, +)) / count
                    if armLabel == "walk" { walkMilliseconds = wall }
                    print(
                        String(
                            format: "🔎 [dsv41 real]     %-7@ %7.1f ms/position, fill "
                                + "%6.1f (read %6.1f, %.1f GB/s), %.0f MB, %.0f misses, "
                                + "%.2fx the walk",
                            armLabel as NSString,
                            wall,
                            outer,
                            read,
                            read > 0 ? bytes / Double(splits[0].positions) / read : 0,
                            bytes,
                            misses,
                            (walkMilliseconds ?? wall) / max(wall, 0.001)
                        )
                    )
                }
                // The fill's own samples must cover every miss the bucket was charged
                // for, or the decomposition is describing a different set of fills than
                // the timer.
                for split in steady {
                    XCTAssertEqual(
                        split.byMisses.reduce(0) { $0 + $1.key * $1.value.fills },
                        split.misses,
                        "\(label) \(split.label) fill samples do not cover its misses"
                    )
                    XCTAssertLessThanOrEqual(
                        split.readMilliseconds, split.outerMilliseconds + 1,
                        "\(label) \(split.label) read longer than the bucket holding it"
                    )
                    XCTAssertLessThan(
                        abs(split.rangeMegabytes - split.slotMegabytes),
                        max(split.slotMegabytes * 0.05, 1),
                        "\(label) \(split.label) ranges and slots disagree on the bytes"
                    )
                }
                // The two layouts are different graphs, so this is not a parity check: it
                // is the probe's guard that an arm did not reuse another arm's cache.
                if let reference = steady.first(where: { $0.label == "walk" })?.lastLogits,
                   !reference.isEmpty {
                    let scale = max(reference.map { abs(Double($0)) }.max() ?? 0, 1e-6)
                    for split in steady where split.label != "walk" {
                        guard split.lastLogits.count == reference.count else {
                            XCTFail("\(split.label) returned no logits")
                            continue
                        }
                        let worst = zip(reference, split.lastLogits)
                            .map { abs(Double($0) - Double($1)) }
                            .max() ?? 0
                        XCTAssertLessThan(
                            worst / scale, 5e-2,
                            "\(label) \(split.label) disagreed with the walk"
                        )
                        XCTAssertEqual(
                            Self.argmax(split.lastLogits), Self.argmax(reference),
                            "\(label) \(split.label) chose a different token"
                        )
                    }
                }
            }

            do {
                let ceilings = [
                    try calibrate(label: "calib6", perFill: 6, fills: 8, firstPair: 96),
                    try calibrate(label: "calib24", perFill: 24, fills: 8, firstPair: 192)
                ]
                print("🔎 [dsv41 real]   " + label + " calibration, cold fills, same call:")
                for split in ceilings {
                    report(split, unit: "fill")
                    XCTAssertEqual(
                        split.misses,
                        split.fills * (split.label == "calib24" ? 24 : 6),
                        "\(label) \(split.label) did not read every pair it asked for"
                    )
                }
                results += ceilings
            }

            // The harness's own determinism check: an arm that read different bytes in
            // two rounds is not the same regime.
            for (armLabel, counts) in bytesByLabel.sorted(by: { $0.key < $1.key }) {
                let low = counts.min() ?? 0
                let high = counts.max() ?? 0
                XCTAssertLessThan(
                    high - low, max(high * 0.005, 1),
                    "\(label) \(armLabel) read \(low)-\(high) MB across rounds"
                )
            }
            streamer.recordsFillSamples = false
            return results
        }

        _ = try measure(
            regime: regime.label, wireArena: regime.wireArena, rounds: regime.rounds
        )
    }

    // MARK: - #72: what a block's mixture half is made of

    /// The mixture half is the largest bucket of a prefill position that is not the
    /// disk (19.3-20.0 ms/position at the shipped width against 18.0 walked), and it is
    /// the one bucket that does NOT fall with the block width. That is either a stage
    /// whose bytes per position are fixed by the graph or a stage that never batches,
    /// and nothing in this journal could say which: every block stage table (`#52`)
    /// covers the DENSE half, because `recordsStageTiming` only ever gave that half a
    /// buffer per stage.
    ///
    /// So this measures the four mixture stages at widths four and eight in the app's
    /// own regime (`wireArena: nil`, `#71`), with the shipped one-buffer layout as the
    /// control. The split is a different submission order (stage-major rather than
    /// position-major), so the two layouts must return the same answer, and the table is
    /// read as a ranking of stages rather than as the shipped layout's wall.
    func testWhereABlocksMixtureTimeGoesOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_mixture_stages"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in mixture attribution; touch \(markerPath) to run"
        )
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

        let length = 128
        let tokens = (0..<length).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 13) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: length + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let shippedWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: length
        )
        var widths = [shippedWidth]
        if !widths.contains(4) { widths.append(4) }
        // The dense half's stage names, so the mixture's can be told apart in a table
        // that is one dictionary.
        let denseStages = Set(DeepSeekV41DevicePlan.frontHalf.map(\.rawValue))
        let mixtureStages = DeepSeekV41DevicePlan.backHalf.map(\.rawValue)
        XCTAssertFalse(mixtureStages.isEmpty, "the plan has no mixture stages")
        let expertBytes = runtime.expertStreamer?.geometry.slotBytes ?? 0
        let sharedBytesPerLayer = 3 * executor.shapes.sharedIntermediate * config.hiddenSize
        let wiredGigabytes = Double(runtime.arena.capacity) * Double(expertBytes) / 1_073_741_824
        let expertMegabytes = Double(expertBytes) / 1_048_576
        let sharedMegabytes = Double(sharedBytesPerLayer) / 1_048_576
        let expertRatio = "\(config.numActivatedExperts)/\(config.numRoutedExperts)"
        print(
            String(
                format: "🔎 [dsv41 real] mixture attribution: %d positions, arena %d slot(s) "
                    + "wired (%.1f GB), widths %@, %@ experts of %.2f MB, shared expert "
                    + "%.1f MB/layer",
                length,
                runtime.arena.capacity,
                wiredGigabytes,
                String(describing: widths),
                expertRatio,
                expertMegabytes,
                sharedMegabytes
            )
        )
        print("🔎 [dsv41 real] mixture stages: " + mixtureStages.joined(separator: ", "))

        struct MixtureSample {
            var label: String
            var width: Int
            var millisecondsPerPosition: Double
            var stageMilliseconds: [String: Double]
            var denseDevice: Double
            var mixtureDevice: Double
            var megabytes: Double
            var misses: Int
            var last: [Float]
        }

        func arm(width: Int, split: Bool) throws -> MixtureSample {
            executor.reset()
            executor.resetBlockTiming()
            executor.resetStageTiming()
            executor.recordsBlockTiming = true
            executor.recordsStageTiming = split
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try executor.prefillChunk(tokens, blockWidth: width)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsStageTiming = false
            executor.recordsBlockTiming = false
            let per = executor.blockTiming.millisecondsPerPosition()
            let divisor = Double(length) * 1_000_000
            var stages: [String: Double] = [:]
            for (stage, nanoseconds) in executor.stageTiming {
                stages[stage] = Double(nanoseconds) / divisor
            }
            return MixtureSample(
                label: split ? "split" : "one buffer",
                width: width,
                millisecondsPerPosition: wall / Double(length),
                stageMilliseconds: stages,
                denseDevice: per.denseDevice,
                mixtureDevice: per.mixtureDevice,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                last: result.last
            )
        }

        // One discarded pass so no arm pays the slot allocations or a cold page cache.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        var averaged: [String: [MixtureSample]] = [:]
        for width in widths {
            for round in 0..<3 {
                // Rotated: whoever runs first reads the colder cache, and the shipped
                // layout is the one that must be compared like for like.
                let order = round % 2 == 0 ? [false, true] : [true, false]
                for split in order {
                    let sample = try arm(width: width, split: split)
                    let label = "\(sample.label) @ \(width)"
                    print(
                        String(
                            format: "🔎 [dsv41 real]   %-18@ round %d wall %7.1f ms/position, "
                                + "device %.1f (dense %.1f, mixture %.1f), %.0f MB, %d misses",
                            label as NSString,
                            round + 1,
                            sample.millisecondsPerPosition,
                            sample.denseDevice + sample.mixtureDevice,
                            sample.denseDevice,
                            sample.mixtureDevice,
                            sample.megabytes,
                            sample.misses
                        )
                    )
                    // A young process is 1.3-1.6x faster in every bucket (`#53`), so the
                    // first round is printed and never averaged.
                    guard round > 0 else { continue }
                    averaged[label, default: []].append(sample)
                }
            }
        }

        func mean(_ samples: [MixtureSample]) -> MixtureSample {
            let count = Double(max(1, samples.count))
            var stages: [String: Double] = [:]
            for sample in samples {
                for (stage, milliseconds) in sample.stageMilliseconds {
                    stages[stage, default: 0] += milliseconds / count
                }
            }
            return MixtureSample(
                label: samples[0].label,
                width: samples[0].width,
                millisecondsPerPosition: samples.map(\.millisecondsPerPosition).reduce(0, +) / count,
                stageMilliseconds: stages,
                denseDevice: samples.map(\.denseDevice).reduce(0, +) / count,
                mixtureDevice: samples.map(\.mixtureDevice).reduce(0, +) / count,
                megabytes: samples.map(\.megabytes).reduce(0, +) / count,
                misses: Int(Double(samples.map(\.misses).reduce(0, +)) / count),
                last: samples[0].last
            )
        }

        for width in widths {
            let oneBuffer = try XCTUnwrap(
                averaged["one buffer @ \(width)"].map(mean),
                "the one-buffer arm at width \(width) was never measured"
            )
            let split = try XCTUnwrap(
                averaged["split @ \(width)"].map(mean),
                "the split arm at width \(width) was never measured"
            )

            // The split is the same graph: same kernels, same bindings per slot, and one
            // in-order queue instead of one buffer. Anything but an identical answer is
            // a hazard the single-buffer layout was hiding.
            let scale = max(split.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
            let worst = zip(oneBuffer.last, split.last)
                .map { abs(Double($0) - Double($1)) }
                .max() ?? 0
            XCTAssertEqual(
                split.last.count, oneBuffer.last.count, "the two layouts returned no logits"
            )
            XCTAssertLessThan(
                worst / scale, 1e-6,
                "the stage-major mixture order changed the answer at width \(width)"
            )
            XCTAssertEqual(
                Self.argmax(split.last), Self.argmax(oneBuffer.last),
                "the stage-major mixture order chose a different token at width \(width)"
            )
            // Not a result, a guard: an arm that read different bytes is a different
            // regime and its device windows are not comparable.
            XCTAssertLessThan(
                abs(split.megabytes - oneBuffer.megabytes),
                max(oneBuffer.megabytes * 0.02, 1),
                "the two layouts at width \(width) read different bytes"
            )
            for stage in mixtureStages {
                XCTAssertNotNil(
                    split.stageMilliseconds[stage],
                    "the split layout never reported \(stage) at width \(width)"
                )
            }

            let mixtureTotal = mixtureStages.reduce(0.0) { $0 + (split.stageMilliseconds[$1] ?? 0) }
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, mixture device %.2f ms/position (one "
                        + "buffer %.2f), stages sum to %.2f of it; dense device %.2f; "
                        + "%.0f MB, %d misses",
                    width,
                    split.mixtureDevice,
                    oneBuffer.mixtureDevice,
                    mixtureTotal,
                    split.denseDevice,
                    split.megabytes,
                    split.misses
                )
            )
            for stage in mixtureStages {
                let milliseconds = split.stageMilliseconds[stage] ?? 0
                // The shared expert's bytes are the same for every position of the
                // block, which is what makes its rate the question: a block that reads
                // them once per position pays `width` times.
                let bytes = Double(config.numHiddenLayers) * Double(sharedBytesPerLayer)
                    / 1_073_741_824
                let rate = stage == "sharedExpert" && milliseconds > 0
                    ? bytes / (milliseconds / 1_000) : 0
                let suffix = rate > 0
                    ? String(
                        format: "  (%.2f GB/position of weight, %.0f GB/s)", bytes, rate
                    )
                    : ""
                print(
                    String(
                        format: "🔎 [dsv41 real]   %-20@ %7.3f ms/position  %5.1f%% of the "
                            + "mixture half%@",
                        stage as NSString,
                        milliseconds,
                        milliseconds / max(split.mixtureDevice, 1e-9) * 100,
                        suffix as NSString
                    )
                )
            }
            let denseTotal = split.stageMilliseconds
                .filter { denseStages.contains($0.key) }
                .values.reduce(0, +)
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, dense stages sum to %.2f of %.2f "
                        + "ms/position of dense device (%.0f%% over, the split pays its own "
                        + "pickup, `#60`)",
                    width, denseTotal, split.denseDevice,
                    split.denseDevice > 0 ? (denseTotal / split.denseDevice - 1) * 100 : 0
                )
            )
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: - #73: what a block's dense half is made of

    /// The dense half is 13.4-15.4 ms/position of a ~78-80 ms prefill position at the
    /// shipped width: the largest bucket that is neither the disk (`#71` closed the fill
    /// at the disk's own 13.6 GB/s) nor the routed mixture (`#72` closed that at 352
    /// GB/s with 6 of 384 experts). Every dense stage table this journal holds was taken
    /// on the SYNTHETIC checkpoint at published widths (`#16`-`#18`), where `writeLatent`
    /// was 45.5% of the half, `compressPool`/`attention` printed zero because the fixture
    /// had no compressed layers, and every weight was already resident. The published
    /// plan is half compressed, so the ranking has never been measured where it ships.
    ///
    /// `recordsStageTiming` gives the dense half one command buffer per stage (`#72` did
    /// the same for the mixture half), so this is a table rather than a change: widths
    /// eight and four in the app's regime (`wireArena: nil`, `#71`), every front-half
    /// stage in ms/position beside the weight bytes it reads, so a stage can be read as
    /// bandwidth or as not-a-GEMV instead of guessed at.
    func testWhereABlocksDenseHalfGoesOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_dense_stages"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in dense attribution; touch \(markerPath) to run"
        )
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

        let length = 128
        let tokens = (0..<length).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 13) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: length + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let shippedWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: length
        )
        var widths = [shippedWidth]
        if !widths.contains(4) { widths.append(4) }
        let denseStages = DeepSeekV41DevicePlan.frontHalf.map(\.rawValue)
        let mixtureStages = DeepSeekV41DevicePlan.backHalf.map(\.rawValue)

        // What each stage reads as a WEIGHT, per layer, resolved from the catalog the
        // runtime actually bound rather than from the config: a stage that reads none
        // (a norm, a rope, a mix, the attention itself) is priced at zero and prints
        // without a rate. The scale companion counts, because an FP8 projection reads
        // it alongside the values and that is what the pin and the kernel both touch.
        let stageRoles: [String: [DeepSeekV41TensorRole]] = [
            "queryDown": [.attnWqA],
            "queryUp": [.attnWqB],
            "latentProject": [.attnWkv],
            "outputDown": [.attnWoA],
            "outputUp": [.attnWoB],
            "engram": [.engramWkv],
            "indexer": [.indexerWqB, .indexerWk, .indexerWeights],
            "compressPool": [.compressorWkv, .compressorWgate]
        ]
        let layers = runtime.plan
        func weightBytes(_ stage: String) -> Int {
            guard let roles = stageRoles[stage] else { return 0 }
            var total = 0
            for layer in layers {
                for role in roles {
                    total += runtime.weights.tensor(role, layer: layer.index)?.byteLength ?? 0
                    total += runtime.weights.scaleTensor(role, layer: layer.index)?.byteLength ?? 0
                }
            }
            return total
        }

        let compressedLayers = layers.filter(\.ownsCompressedIndex).count
        print(
            String(
                format: "🔎 [dsv41 real] dense attribution: %d positions, %d layer(s) (%d with "
                    + "a compressed index), widths %@, arena %d slot(s)",
                length,
                layers.count,
                compressedLayers,
                String(describing: widths),
                runtime.arena.capacity
            )
        )

        struct DenseSample {
            var label: String
            var width: Int
            var millisecondsPerPosition: Double
            var stages: [String: Double]
            var denseDevice: Double
            var mixtureDevice: Double
            var frontDrain: Double
            var frontLateStart: Double
            var mixtureDrain: Double
            var megabytes: Double
            var misses: Int
            var last: [Float]
        }

        func arm(width: Int, split: Bool, slots: Bool) throws -> DenseSample {
            executor.reset()
            executor.resetBlockTiming()
            executor.resetStageTiming()
            executor.recordsBlockTiming = true
            executor.recordsStageTiming = split
            executor.enablesFP8SlotGemv = slots
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try executor.prefillChunk(tokens, blockWidth: width)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsStageTiming = false
            executor.recordsBlockTiming = false
            let timing = executor.blockTiming
            let per = timing.millisecondsPerPosition()
            let divisor = Double(length) * 1_000_000
            let positionDivisor = Double(max(1, timing.positionCount)) * 1_000_000
            var stages: [String: Double] = [:]
            for (stage, nanoseconds) in executor.stageTiming {
                stages[stage] = Double(nanoseconds) / divisor
            }
            let name: String
            if !split {
                name = "one buffer"
            } else {
                name = slots ? "split slots" : "split per position"
            }
            return DenseSample(
                label: name,
                width: width,
                millisecondsPerPosition: wall / Double(length),
                stages: stages,
                denseDevice: per.denseDevice,
                mixtureDevice: per.mixtureDevice,
                frontDrain: Double(timing.frontDrainNanoseconds) / positionDivisor,
                frontLateStart: Double(timing.frontLateStartNanoseconds) / positionDivisor,
                mixtureDrain: Double(timing.mixtureDrainNanoseconds) / positionDivisor,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore) / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                last: result.last
            )
        }

        // One discarded pass so no arm pays the slot allocations or a cold page cache.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        var averaged: [String: [DenseSample]] = [:]
        for width in widths {
            for round in 0..<3 {
                // Rotated for the reason `#72` rotates: whoever runs first reads the
                // colder cache, and the layouts must be compared like for like. The two
                // stage-split arms differ only in the FP8 projection's form (`#73`), so
                // the difference between their stage rows IS what that form is worth.
                let arms = [(split: false, slots: false), (split: true, slots: false),
                            (split: true, slots: true)]
                let offset = round % arms.count
                let order = Array(arms[offset...]) + Array(arms[..<offset])
                for (split, slots) in order {
                    let sample = try arm(width: width, split: split, slots: slots)
                    let label = "\(sample.label) @ \(width)"
                    print(
                        String(
                            format: "🔎 [dsv41 real]   %-18@ round %d wall %7.1f ms/position, "
                                + "device %.1f (dense %.1f, mixture %.1f), drained front %.1f "
                                + "(late start %.1f), %.0f MB, %d misses",
                            label as NSString,
                            round + 1,
                            sample.millisecondsPerPosition,
                            sample.denseDevice + sample.mixtureDevice,
                            sample.denseDevice,
                            sample.mixtureDevice,
                            sample.frontDrain,
                            sample.frontLateStart,
                            sample.megabytes,
                            sample.misses
                        )
                    )
                    // A young process is 1.3-1.6x faster in every bucket (`#53`).
                    guard round > 0 else { continue }
                    averaged[label, default: []].append(sample)
                }
            }
        }

        func mean(_ samples: [DenseSample]) -> DenseSample {
            let count = Double(max(1, samples.count))
            var stages: [String: Double] = [:]
            for sample in samples {
                for (stage, milliseconds) in sample.stages {
                    stages[stage, default: 0] += milliseconds / count
                }
            }
            func average(_ keyPath: KeyPath<DenseSample, Double>) -> Double {
                samples.map { $0[keyPath: keyPath] }.reduce(0, +) / count
            }
            return DenseSample(
                label: samples[0].label,
                width: samples[0].width,
                millisecondsPerPosition: average(\.millisecondsPerPosition),
                stages: stages,
                denseDevice: average(\.denseDevice),
                mixtureDevice: average(\.mixtureDevice),
                frontDrain: average(\.frontDrain),
                frontLateStart: average(\.frontLateStart),
                mixtureDrain: average(\.mixtureDrain),
                megabytes: average(\.megabytes),
                misses: Int(Double(samples.map(\.misses).reduce(0, +)) / count),
                last: samples[0].last
            )
        }

        for width in widths {
            let oneBuffer = try XCTUnwrap(
                averaged["one buffer @ \(width)"].map(mean),
                "the one-buffer arm at width \(width) was never measured"
            )
            let split = try XCTUnwrap(
                averaged["split slots @ \(width)"].map(mean),
                "the slot-form split arm at width \(width) was never measured"
            )
            let perPosition = try XCTUnwrap(
                averaged["split per position @ \(width)"].map(mean),
                "the per-position split arm at width \(width) was never measured"
            )
            // The split is the same graph in the same order on one in-order queue, so
            // anything but an identical answer is a hazard the single-buffer layout hid.
            // The slot form only changes which kernel reads the weight, so it has to
            // agree with the per-position form to rounding rather than to the bit: the
            // summation order over the row differs.
            let scale = max(perPosition.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
            let worstLayout = zip(oneBuffer.last, perPosition.last)
                .map { abs(Double($0) - Double($1)) }
                .max() ?? 0
            let worstForm = zip(perPosition.last, split.last)
                .map { abs(Double($0) - Double($1)) }
                .max() ?? 0
            XCTAssertEqual(
                split.last.count, oneBuffer.last.count, "the two layouts returned no logits"
            )
            XCTAssertLessThan(
                worstLayout / scale, 1e-6,
                "the stage-split dense order changed the answer at width \(width)"
            )
            // The form is rejected on TIME below; this records whether it is even the
            // same function. It is not bit-identical and cannot be: a different
            // summation order over every projection moves the router's near-ties, and 40
            // layers of that reach the logits. The bar is a sanity bound on that
            // amplification, not a parity claim -- the slot kernel is held to 1e-4
            // against the host rule on the published shapes with full-range bytes
            // (`DeepSeekV41FP8GemvKernelTests`).
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: the slot form's logits move %.1e "
                        + "relative, layout alone %.1e",
                    width, worstForm / scale, worstLayout / scale
                )
            )
            XCTAssertLessThan(
                worstForm / scale, 5e-2,
                "the slot-form projection changed the answer at width \(width)"
            )
            XCTAssertEqual(
                Self.argmax(split.last), Self.argmax(oneBuffer.last),
                "the stage-split dense order chose a different token at width \(width)"
            )
            XCTAssertEqual(
                Self.argmax(split.last), Self.argmax(perPosition.last),
                "the slot-form projection chose a different token at width \(width)"
            )
            XCTAssertLessThan(
                abs(split.megabytes - oneBuffer.megabytes),
                max(oneBuffer.megabytes * 0.02, 1),
                "the two layouts at width \(width) read different bytes"
            )
            for stage in denseStages {
                XCTAssertNotNil(
                    split.stages[stage],
                    "the split layout never reported the dense stage \(stage) at width \(width)"
                )
            }

            let denseTotal = denseStages.reduce(0.0) { $0 + (split.stages[$1] ?? 0) }
            let perPositionTotal = denseStages.reduce(0.0) {
                $0 + (perPosition.stages[$1] ?? 0)
            }
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: dense device %.2f ms/position (per "
                        + "position %.2f, one buffer %.2f), wall %.1f; mixture device %.2f",
                    width,
                    split.denseDevice,
                    perPosition.denseDevice,
                    oneBuffer.denseDevice,
                    split.millisecondsPerPosition,
                    split.mixtureDevice
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: dense stages sum to %.2f (slot form) "
                        + "of %.2f (per position)",
                    width, denseTotal, perPositionTotal
                )
            )
            // A guard, not a result: a table whose rows do not add up to the half is an
            // instrument that is not measuring it.
            XCTAssertGreaterThan(
                perPositionTotal, perPosition.denseDevice * 0.5,
                "the dense stages at width \(width) sum to less than half of the half"
            )
            for stage in denseStages {
                let milliseconds = split.stages[stage] ?? 0
                let previous = perPosition.stages[stage] ?? 0
                let bytes = Double(weightBytes(stage))
                var suffix = ""
                if bytes > 0, previous > 0 {
                    // Milliseconds in, GB/s out: one division by 1e9, matching
                    // `blockTiming.describe()`, which prices its fill the same way.
                    // The bytes are the weight's, i.e. what ONE pass over the row per
                    // block would be, so the rate is what the per-position form really
                    // asks the memory system for -- and the ratio beside it is what
                    // removing that repetition was worth.
                    let gigabytes = bytes / 1_073_741_824
                    let rate = gigabytes / (previous / 1_000)
                    let ratio = String(
                        format: "%4.2fx", milliseconds / max(previous, 1e-9)
                    )
                    suffix = String(
                        format: "  %.2f GB of weight, per-position form %.0f GB/s",
                        gigabytes,
                        rate
                    ) + ", the slot form costs \(ratio)"
                } else if stage == "writeLatent" {
                    // No weight: this stage reads the gathered key block, `heads` times
                    // over every reachable row, which is why its cost tracks row count
                    // rather than a matrix (`#17`).
                    suffix = "  reads the gathered key block, no weight"
                }
                print(
                    String(
                        format: "🔎 [dsv41 real]   dense %-24@ slots %7.3f  per position %7.3f"
                            + "  %5.1f%% of the half  %@",
                        stage as NSString,
                        milliseconds,
                        previous,
                        // Against the PER-POSITION half, because that is the shipped
                        // table; the slot arm's own half is 2.4x longer and would
                        // dilute every share.
                        previous / max(perPosition.denseDevice, 1e-9) * 100,
                        suffix as NSString
                    )
                )
            }
            let calibrationKey = split.stages.keys.first { $0.hasPrefix("calibration(") }
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: %d trivial dispatch(es) cost %.3f "
                        + "ms/position (%.0f%% of the half, so the split pays its own "
                        + "pickup, `#60`)",
                    width,
                    width,
                    calibrationKey.flatMap { split.stages[$0] } ?? 0,
                    split.denseDevice > 0
                        ? (calibrationKey.flatMap { split.stages[$0] } ?? 0)
                            / split.denseDevice * 100
                        : 0
                )
            )
            let mixtureTotal = mixtureStages.reduce(0.0) { $0 + (split.stages[$1] ?? 0) }
            print(
                String(
                    format: "🔎 [dsv41 real] width %d: mixture stages sum to %.2f of %.2f "
                        + "ms/position, front drained %.1f (late start %.1f), mixture "
                        + "drained %.1f, %.0f MB, %d misses",
                    width,
                    mixtureTotal,
                    split.mixtureDevice,
                    split.frontDrain,
                    split.frontLateStart,
                    split.mixtureDrain,
                    split.megabytes,
                    split.misses
                )
            )
        }
        withExtendedLifetime(engine) {}
    }

    // MARK: - #74: what the indexer stage is made of

    /// `#73` billed `.indexer` at 1.7 ms/position, 11.3% of a prefill position's dense
    /// half, and left it there: the stage is ONE command buffer holding three unrelated
    /// rails, so the number named no mechanism. Four of forty layers own a compressed KV
    /// cache and run the compressor, eight own a compressed index and run the query-side
    /// projections, and only the scores depend on how many rows a position can see.
    ///
    /// `DeepSeekV41IndexerSection` splits the stage so `recordsStageTiming` gives each
    /// rail its own buffer, and this prices each against the bytes it re-reads per
    /// position: the compressor's pair (once per slot, on four layers), the indexer's
    /// pair (once per slot, on eight, and `attn.indexer.wq_b` is 1280 x 4096 of E4M3),
    /// and the score kernel's row census. The split is an instrument, so the sum of its
    /// three windows is a little larger than the single-buffer stage `#73` printed.
    func testWhatTheIndexerStageIsMadeOfOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_indexer_stages"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in indexer attribution; touch \(markerPath) to run"
        )
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
        let length = 128
        let tokens = (0..<length).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 13) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: length + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let width = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: length
        )
        let layers = runtime.plan
        let kvOwners = layers.filter(\.ownsCompressedKV)
        let indexOwners = layers.filter(\.ownsCompressedIndex)

        /// One pass over the owners' tensors, scales included: an FP8 projection reads
        /// its E8M0 companion alongside the values, which is what the kernel binds.
        func weightBytes(
            _ roles: [DeepSeekV41TensorRole],
            over owners: [DeepSeekV41LayerPlan]
        ) -> Int {
            var total = 0
            for layer in owners {
                for role in roles {
                    total += runtime.weights.tensor(role, layer: layer.index)?.byteLength ?? 0
                    total += runtime.weights.scaleTensor(role, layer: layer.index)?.byteLength ?? 0
                }
            }
            return total
        }
        // Both projection pairs run once for EVERY slot, so their bytes are already a
        // per-position figure.
        let poolBytes = weightBytes([.compressorWkv, .compressorWgate], over: kvOwners)
        let projectBytes = weightBytes([.indexerWqB, .indexerWeights], over: indexOwners)
        // The scores read one key row per reachable row instead of a matrix, so their
        // traffic is the row census the cache itself reports rather than an average.
        var selectBytesTotal = 0
        for layer in indexOwners {
            let ratio = max(1, layer.compressRatio)
            for position in 0..<length {
                let produced = DeepSeekV41CacheAddressing.compressedRowCount(
                    afterPosition: position, ratio: ratio
                )
                selectBytesTotal += min(produced, config.indexTopK) * config.indexHeadDim * 4
            }
        }
        let selectBytes = selectBytesTotal / length

        struct IndexerSample {
            var label: String
            var stages: [String: Double]
            var denseDevice: Double
            var indexerWait: Double
            var megabytes: Double
            var last: [Float]
        }

        func attempt(split: Bool, forced: Bool) throws -> IndexerSample {
            executor.reset()
            executor.resetBlockTiming()
            executor.resetStageTiming()
            executor.recordsBlockTiming = true
            executor.recordsStageTiming = split
            executor.forcesFP8WideGrid = forced
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let result = try executor.prefillChunk(tokens, blockWidth: width)
            executor.recordsStageTiming = false
            executor.recordsBlockTiming = false
            executor.forcesFP8WideGrid = false
            let timing = executor.blockTiming
            let per = timing.millisecondsPerPosition()
            let divisor = Double(length) * 1_000_000
            var stages: [String: Double] = [:]
            for (stage, nanoseconds) in executor.stageTiming {
                stages[stage] = Double(nanoseconds) / divisor
            }
            return IndexerSample(
                label: !split ? "one buffer" : (forced ? "split wide" : "split"),
                stages: stages,
                denseDevice: per.denseDevice,
                indexerWait: per.indexerWait,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore)
                    / 1_048_576,
                last: result.last
            )
        }

        // One discarded pass so no arm pays the slot allocations or a cold page cache.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        var averaged: [String: [IndexerSample]] = [:]
        for round in 0..<3 {
            // Rotated for the reason `#72` and `#73` rotate: whoever runs first reads the
            // colder cache, and the two layouts have to be compared like for like.
            let arms = [(split: true, forced: false), (split: true, forced: true),
                        (split: false, forced: false)]
            let offset = round % arms.count
            let order = Array(arms[offset...]) + Array(arms[..<offset])
            for (split, forced) in order {
                let sample = try attempt(split: split, forced: forced)
                // The one-buffer arm reports no stage rows at all: it has one buffer per
                // layer, which is the layout whose answer and byte count this arm is here
                // to pin against the split.
                let detail = split
                    ? String(
                        format: "indexer %.3f ms/position (idle %.3f, pool %.3f, project "
                            + "%.3f, select %.3f), ",
                        sample.stages["indexer"] ?? 0,
                        sample.stages["indexer.idle"] ?? 0,
                        sample.stages["indexer.pool"] ?? 0,
                        sample.stages["indexer.project"] ?? 0,
                        sample.stages["indexer.select"] ?? 0
                    )
                    : ""
                print(
                    String(
                        format: "🔎 [dsv41 real]   %-10@ round %d: %@dense device %.2f, "
                            + "indexer wait %.2f, %.0f MB",
                        sample.label as NSString,
                        round + 1,
                        detail as NSString,
                        sample.denseDevice,
                        sample.indexerWait,
                        sample.megabytes
                    )
                )
                // A young process is 1.3-1.6x faster in every bucket (`#53`).
                guard round > 0 else { continue }
                averaged[sample.label, default: []].append(sample)
            }
        }

        func mean(_ samples: [IndexerSample]) -> IndexerSample {
            let count = Double(max(1, samples.count))
            var stages: [String: Double] = [:]
            for sample in samples {
                for (stage, milliseconds) in sample.stages {
                    stages[stage, default: 0] += milliseconds / count
                }
            }
            func average(_ keyPath: KeyPath<IndexerSample, Double>) -> Double {
                samples.map { $0[keyPath: keyPath] }.reduce(0, +) / count
            }
            return IndexerSample(
                label: samples[0].label,
                stages: stages,
                denseDevice: average(\.denseDevice),
                indexerWait: average(\.indexerWait),
                megabytes: average(\.megabytes),
                last: samples[0].last
            )
        }

        let split = mean(try XCTUnwrap(averaged["split"], "the split arm was never measured"))
        let oneBuffer = mean(
            try XCTUnwrap(averaged["one buffer"], "the one-buffer arm was never measured")
        )
        print(
            String(
                format: "🔎 [dsv41 real] indexer attribution: %d positions at width %d, "
                    + "%d layer(s): %d own a compressed KV cache, %d a compressed index; "
                    + "dense device %.2f ms/position",
                length,
                width,
                layers.count,
                kvOwners.count,
                indexOwners.count,
                split.denseDevice
            )
        )
        let indexer = split.stages["indexer"] ?? 0
        let running = Set(kvOwners.map(\.index)).union(indexOwners.map(\.index)).count
        let rails: [(name: String, bytes: Int, owners: Int)] = [
            ("indexer.idle", 0, layers.count - running),
            ("indexer.pool", poolBytes, kvOwners.count),
            ("indexer.project", projectBytes, indexOwners.count),
            ("indexer.select", selectBytes, indexOwners.count)
        ]
        var railTotal = 0.0
        for rail in rails {
            let milliseconds = split.stages[rail.name] ?? 0
            railTotal += milliseconds
            let megabytes = Double(rail.bytes) / 1_048_576
            let rate = milliseconds > 0 ? megabytes / 1_024 / (milliseconds / 1_000) : 0
            print(
                String(
                    format: "🔎 [dsv41 real]   %-16@ %7.3f ms/position = %5.1f%% of the stage, "
                        + "%5.1f%% of the dense half, %.3f per layer that runs it, %6.2f MB "
                        + "read per position (%3.0f GB/s)",
                    rail.name as NSString,
                    milliseconds,
                    indexer > 0 ? milliseconds / indexer * 100 : 0,
                    split.denseDevice > 0 ? milliseconds / split.denseDevice * 100 : 0,
                    milliseconds / Double(max(1, rail.owners)),
                    megabytes,
                    rate
                )
            )
        }
        print(
            String(
                format: "🔎 [dsv41 real]   %-16@ %7.3f ms/position = %5.1f%% of the dense half; "
                    + "host wait on the split %.2f ms/position (the round trip, not the stage)",
                "indexer (sum)" as NSString,
                railTotal,
                split.denseDevice > 0 ? railTotal / split.denseDevice * 100 : 0,
                split.indexerWait
            )
        )

        // The one candidate the split names: the indexer's `wq_b` is 4,096 rows over
        // 1,280 columns and the plan rule keeps it on the one-thread form because
        // 1,280 < 2,048. That clause is a lane-per-block argument measured on shapes
        // whose `outDim` alone fills the machine; here it is 4,096 threads. Forcing the
        // coalesced form prices the question instead of arguing it, and only the
        // `indexer.project` row is a clean comparison: it is the only FP8 projection the
        // stage encodes, and the other stages' rows move as a side effect.
        let wide = mean(
            try XCTUnwrap(averaged["split wide"], "the forced-wide-grid arm was never measured")
        )
        let shippedProject = split.stages["indexer.project"] ?? 0
        let wideProject = wide.stages["indexer.project"] ?? 0
        print(
            String(
                format: "🔎 [dsv41 real] the coalesced form on the indexer's 4096x1280: "
                    + "project %.3f -> %.3f ms/position (%.2fx), dense device %.2f -> %.2f, "
                    + "%.0f MB either way",
                shippedProject,
                wideProject,
                wideProject / max(shippedProject, 1e-9),
                split.denseDevice,
                wide.denseDevice,
                wide.megabytes
            )
        )
        let wideScale = max(oneBuffer.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
        let wideWorst = zip(wide.last, oneBuffer.last)
            .map { abs(Double($0) - Double($1)) }
            .max() ?? 0
        print(
            String(
                format: "🔎 [dsv41 real] the forced form's logits move %.1e relative",
                wideWorst / wideScale
            )
        )
        XCTAssertLessThan(
            wideWorst / wideScale, 5e-2,
            "forcing the coalesced form on every FP8 projection changed the answer"
        )
        XCTAssertLessThan(
            abs(wide.megabytes - oneBuffer.megabytes),
            max(oneBuffer.megabytes * 0.02, 1),
            "the forced-wide-grid arm read different bytes"
        )

        // The instrument's own guard: the three section entries are the same three
        // windows the stage row sums, so anything but agreement means one of them is
        // billed twice or dropped.
        XCTAssertEqual(railTotal, indexer, accuracy: max(indexer * 1e-6, 1e-9))
        for rail in rails {
            XCTAssertNotNil(
                split.stages[rail.name],
                "the split layout never reported \(rail.name)"
            )
        }
        // The split adds submissions and changes nothing else: same graph, same order on
        // one in-order queue, one encoder per stage either way.
        let scale = max(oneBuffer.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
        let worst = zip(split.last, oneBuffer.last)
            .map { abs(Double($0) - Double($1)) }
            .max() ?? 0
        print(
            String(
                format: "🔎 [dsv41 real] the section split's logits move %.1e relative; "
                    + "%.0f MB either way",
                worst / scale,
                split.megabytes
            )
        )
        XCTAssertLessThan(worst / scale, 1e-6, "the indexer split changed the answer")
        XCTAssertLessThan(
            abs(split.megabytes - oneBuffer.megabytes),
            max(oneBuffer.megabytes * 0.02, 1),
            "the two layouts read different bytes"
        )
        withExtendedLifetime(engine) {}
    }

    // MARK: - #75: what the front wait is made of

    /// `#71`'s bucket table reads `front wait` as 43.7 of a 99.7 ms/position prefill, the
    /// largest term that is not the disk, and every attempt to fix a bucket that size has
    /// to know whether it is the GPU, the driver, or the host. It is one
    /// `settlePending`, and the queue is in order, so the mixture the PREVIOUS layer
    /// committed is drained inside it: the bucket is a settle, not a stall.
    ///
    /// This decomposes it on the published checkpoint. `recordsBlockGaps` keeps one
    /// clock record per layer pass with that layer's own buffers' windows attached, so
    /// each layer's wait splits exactly into
    ///
    ///     busy     the previous layer's mixture still on the queue (already billed)
    ///   + pickup   the queue's own cost to start the dense half once nothing is running
    ///   + window   the dense half's own device time
    ///   + tail     the host noticing the last buffer had finished
    ///
    /// and `recordDeviceTiming` unions every window so the whole block can be split into
    /// busy and idle rather than into host intervals that overlap. Three arms (width 8,
    /// width 4, width 8 again) so the width question is answered rather than inferred.
    func testWhatTheFrontWaitIsMadeOfOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_front_wait"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in front-wait attribution; touch \(markerPath) to run"
        )
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
        let length = 128
        let tokens = (0..<length).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 13) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: length + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let shippedWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: length
        )
        XCTAssertGreaterThanOrEqual(shippedWidth, 2, "a width of one is the walk")
        print(
            String(
                format: "🔎 [dsv41 real] front wait attribution: %d positions, shipped "
                    + "width %d, %d layer(s), %d of them compressed, arena %d slot(s)",
                length,
                shippedWidth,
                runtime.plan.count,
                runtime.plan.filter(\.ownsCompressedIndex).count,
                runtime.arena.capacity
            )
        )

        struct Arm {
            var label: String
            var width: Int
            var wall: Double
            var timing: DeepSeekV41BlockTiming
            var gaps: [DeepSeekV41BlockGap]
            var megabytes: Double
            var misses: Int
            var last: [Float]
        }

        func measure(label: String, width: Int) throws -> Arm {
            executor.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            executor.recordsBlockGaps = true
            // Two blocks' worth, because the FIRST block of a prefill is the one that
            // allocates every arena slot it fills and reads a cold page cache. It is
            // recorded (so its layer zero has a neighbour to compare against) and then
            // dropped from the partition, which reads the second block instead.
            executor.blockGapLimit = 80
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try executor.prefillChunk(tokens, blockWidth: width)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsBlockTiming = false
            executor.recordsBlockGaps = false
            return Arm(
                label: label,
                width: width,
                wall: wall / Double(length),
                timing: executor.blockTiming,
                gaps: executor.blockGaps,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore)
                    / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                last: result.last
            )
        }

        func perPosition(_ nanoseconds: UInt64) -> Double {
            Double(nanoseconds) / Double(length) / 1_000_000
        }

        // One discarded pass so no arm pays slot creation or a cold page cache.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        let arms = [(label: "shipped", width: shippedWidth),
                    (label: "half", width: max(2, shippedWidth / 2)),
                    (label: "shipped again", width: shippedWidth)]
        var measured: [Arm] = []
        for round in 1...3 {
            let offset = (round - 1) % arms.count
            let order = Array(arms[offset...]) + Array(arms[..<offset])
            for arm in order {
                let sample = try measure(label: arm.label, width: arm.width)
                print(
                    String(
                        format: "🔎 [dsv41 real]   %-14@ width %2d round %d wall %6.1f "
                            + "ms/position, %.0f MB, %d misses",
                        sample.label as NSString,
                        sample.width,
                        round,
                        sample.wall,
                        sample.megabytes,
                        sample.misses
                    )
                )
                print("🔎 [dsv41 real]     \(sample.timing.describe())")
                // A young process is 1.3-1.6x faster in every bucket (`#53`), and the
                // first arm of the first round also pays the coldest page cache.
                guard round > 1 else { continue }
                measured.append(sample)
            }
        }

        // The decomposition, read back off the per-layer records rather than off the
        // totals: `busy` is the previous layer's mixture still running, and it is the
        // term that makes the front wait look like idle when it is device time.
        for arm in measured where arm.label == "shipped again" {
            let divisor = Double(length) * 1_000_000
            // The second block: the first pays every slot allocation and a cold page
            // cache, so its per-layer busy is a quarter high.
            let records = arm.gaps.count > 40 ? Array(arm.gaps.suffix(40)) : arm.gaps
            var busy: UInt64 = 0
            var pickup: UInt64 = 0
            var span: UInt64 = 0
            var windows: UInt64 = 0
            var tail: UInt64 = 0
            var mixtureIdle: UInt64 = 0
            var mixtureLate: UInt64 = 0
            var layersReported = 0
            var previousMixtureEnd: UInt64?
            for (index, gap) in records.enumerated() {
                // The first layer of a block has an upstream the record cannot see: the
                // mixture of the PREVIOUS block's last layer. It is left out of the
                // partition rather than folded into `pickup`, which is what would happen
                // if it were counted with no upstream to compare against.
                if index > 0 {
                    busy += gap.priorWorkNanoseconds(upstream: previousMixtureEnd)
                    pickup += gap.pickupNanoseconds(upstream: previousMixtureEnd)
                    span += gap.frontSpanNanoseconds
                    windows += gap.frontWindowNanoseconds
                    tail += gap.frontTailNanoseconds()
                    mixtureIdle += gap.mixtureIdleNanoseconds()
                    mixtureLate += gap.mixtureLateStartNanoseconds()
                    layersReported += 1
                }
                if gap.mixtureWindowEndNanoseconds > 0 {
                    previousMixtureEnd = gap.mixtureWindowEndNanoseconds
                }
            }
            XCTAssertGreaterThan(layersReported, 2, "too few layers to partition")
            // The totals are per position over the whole span, so the partition is put in
            // per-layer-pass units before it can be compared with them.
            let owned = Double(layersReported) / Double(max(1, arm.gaps.count))
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, one block's %d layer pass(es), "
                        + "ms/position: wait %.2f = busy %.2f + pickup %.2f + window "
                        + "%.2f + tail %.2f; of the dense span, %.2f is the gap between "
                        + "two submissions",
                    arm.width,
                    layersReported,
                    Double(busy + pickup + span + tail) / divisor,
                    Double(busy) / divisor,
                    Double(pickup) / divisor,
                    Double(span) / divisor,
                    Double(tail) / divisor,
                    Double(span - min(span, windows)) / divisor
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, the same layers: mixture gate idle "
                        + "%.2f ms/position (%.2f of it after its own commit), and the "
                        + "front wait the totals report for them is %.2f (%.0f%% of the "
                        + "block's %.2f)",
                    arm.width,
                    Double(mixtureIdle) / divisor,
                    Double(mixtureLate) / divisor,
                    Double(arm.timing.frontWaitNanoseconds) / divisor * owned,
                    owned * 100,
                    Double(arm.timing.frontWaitNanoseconds) / divisor
                )
            )
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, the block: device union %.1f of a "
                        + "%.1f wall, %.1f idle; frontBusy %.2f against the records' "
                        + "%.2f; %d/%d buffers reported no window",
                    arm.width,
                    Double(arm.timing.deviceUnionNanoseconds) / divisor,
                    arm.wall,
                    arm.wall - Double(arm.timing.deviceUnionNanoseconds) / divisor,
                    Double(arm.timing.frontBusyNanoseconds) / divisor,
                    Double(busy) / divisor,
                    arm.timing.frontUnwindowedBuffers,
                    arm.timing.mixtureUnwindowedBuffers
                )
            )

            // The classification the slice rests on: most of the wait is another
            // bucket's device time, and the dense half's own share still has room to be
            // the larger of the two only if the mixture had already retired.
            XCTAssertGreaterThan(
                arm.timing.frontBusyNanoseconds, 0,
                "no mixture drain was attributed to a front wait"
            )
            XCTAssertLessThan(
                arm.timing.frontBusyNanoseconds, arm.timing.frontWaitNanoseconds,
                "the previous layer's mixture cannot outlast the whole front wait"
            )
            // The hand-off itself is closed: the queue starts the dense half in well
            // under a millisecond of a several-millisecond wait, and the host notices its
            // last buffer in a fraction of one (#48, #50, and now here).
            XCTAssertLessThan(
                Double(pickup) / Double(max(1, layersReported)) / 1_000_000, 0.5,
                "the dense half's pickup is not the immediate start #48 measured"
            )
            XCTAssertLessThan(
                Double(tail) / Double(max(1, layersReported)) / 1_000_000, 0.5,
                "the host did not notice the dense half finishing promptly"
            )
            // `busy` is the mixture's share of the same settles, so it cannot exceed the
            // mixture drain the totals report.
            XCTAssertGreaterThanOrEqual(
                arm.timing.mixtureDrainNanoseconds, busy,
                "the front wait absorbed more mixture drain than the mixture half reports"
            )
            // A block's windows cannot add up to more wall than the block took.
            let union = Double(arm.timing.deviceUnionNanoseconds) / divisor
            XCTAssertGreaterThanOrEqual(
                arm.wall, union,
                "the device union is larger than the block's own wall"
            )
            XCTAssertGreaterThanOrEqual(
                Double(arm.timing.denseDeviceNanoseconds + arm.timing.mixtureDeviceNanoseconds)
                    / divisor,
                union,
                "a union of overlapping windows is larger than their sum"
            )
            // The two routes to the same number: `frontBusy` is accumulated at the
            // settle, the records are read back per layer. Both are per layer pass here.
            let settlePerLayer = Double(arm.timing.frontBusyNanoseconds)
                / Double(max(1, arm.timing.layers))
            let recordsPerLayer = Double(busy) / Double(max(1, layersReported))
            print(
                String(
                    format: "🔎 [dsv41 real] width %d, frontBusy per layer pass: settle "
                        + "%.2f ms, the block's records %.2f ms",
                    arm.width,
                    settlePerLayer / 1_000_000,
                    recordsPerLayer / 1_000_000
                )
            )
            XCTAssertEqual(
                settlePerLayer, recordsPerLayer,
                accuracy: max(settlePerLayer * 0.35, 500_000),
                "the settle-level `frontBusy` disagrees with the per-layer records"
            )
        }

        // Width must not move the answer, and the two arms that differ only in width do
        // read different bytes, so this is a guard rather than a result.
        let first = try XCTUnwrap(measured.first)
        let repeatArm = try XCTUnwrap(measured.last)
        let scale = max(first.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
        let drift = zip(first.last, repeatArm.last)
            .map { abs(Double($0) - Double($1)) }
            .max() ?? 0
        XCTAssertLessThan(drift / scale, 1e-6, "two prefills of one stream disagree")
        withExtendedLifetime(engine) {}
    }

    // MARK: - #76: what a block's mixture pickup is made of

    /// `#75`'s idle tiling left exactly one term it could not place: a block's mixture is
    /// committed with the GPU idle, and it does not start for ~2.4 ms per layer pass, i.e.
    /// 11.8-12.2 ms of every prefill position. `#22` measured a walked position's hand-off
    /// at ~1.2 ms/layer after a 2 ms fill gap and `#65` closed the deferred bind that could
    /// have moved it, but a block's gap is 34 ms and its pickup is twice as expensive, which
    /// is the opposite of what `#22`'s isolated idle-gap sweep predicts. So the question is
    /// narrow: is the queue refusing to start ANY work at that moment, or is it refusing to
    /// start THIS buffer?
    ///
    /// Four things are measured, all on the published checkpoint in the app's own regime:
    ///
    ///   * a trivial one-dispatch buffer committed immediately before the mixture, the same
    ///     probe `#50` used on the walk (shared code, so the two regimes are comparable);
    ///   * a depth-three arm, because three commits in a row separate a cadence from a
    ///     buffer;
    ///   * a probe binding 16 MB of shared memory the HOST has just written, which is the
    ///     one state an arena slot is in that a scratch buffer is not;
    ///   * the layer's own fill bytes beside its own pickup, so the term can be priced per
    ///     byte (`#50`'s 0.088-0.117 ms of idle per MB) rather than per layer, and a
    ///     width-one arm, which hands the same code eight times less to read.
    func testWhatABlocksMixturePickupIsMadeOfOnThePublishedCheckpoint() async throws {
        let markerPath = "/tmp/dynamoe_dsv41_mixture_pickup"
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: markerPath),
            "opt-in mixture-pickup attribution; touch \(markerPath) to run"
        )
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
        let length = 128
        let tokens = (0..<length).map { index in
            let repetition = index / stream.count
            return stream[(index + repetition * 13) % stream.count]
        }

        let engine = try DynaMoeEngine(filePath: snapshot.path)
        let runtime = try makeRuntime(
            snapshot: snapshot,
            engine: engine,
            config: config,
            contextLength: length + 1,
            allowEngramSkip: false,
            wireArena: nil
        )
        XCTAssertTrue(runtime.isRunnable, runtime.describe())
        let executor = runtime.forward()
        let shippedWidth = DeepSeekV41DevicePlan.prefillBlockWidth(
            config: config, contextLength: runtime.contextLength, promptTokens: length
        )
        XCTAssertGreaterThanOrEqual(shippedWidth, 2, "a width of one is the walk")
        print(
            String(
                format: "🔎 [dsv41 real] mixture pickup: %d positions, shipped width %d, "
                    + "%d layer(s), arena %d slot(s)",
                length,
                shippedWidth,
                runtime.plan.count,
                runtime.arena.capacity
            )
        )

        struct Arm {
            var label: String
            var width: Int
            var wall: Double
            var gaps: [DeepSeekV41BlockGap]
            var megabytes: Double
            var misses: Int
            var last: [Float]
        }

        struct Probe {
            var depth: Int
            var binding: Int
        }

        func measure(label: String, width: Int, probe: Probe?) throws -> Arm {
            executor.reset()
            executor.resetBlockTiming()
            executor.recordsBlockTiming = true
            executor.recordsBlockGaps = true
            executor.recordsMixtureStartProbe = probe != nil
            executor.mixtureStartProbeDepth = probe?.depth ?? 1
            executor.mixtureStartProbeBinding = probe?.binding ?? 0
            executor.mixtureStartProbeDispatches = 1
            executor.mixtureStartProbeDirtyBytes = 16 << 20
            // 80 records is two blocks at the shipped width, so the second one can be
            // partitioned while the cold first is still recorded (`#75`). At width one it is
            // the first eighty positions, which is the same cold-to-warm ramp.
            executor.blockGapLimit = 80
            let bytesBefore = runtime.expertStreamer?.bytesRead ?? 0
            let missesBefore = runtime.expertStreamer?.expertsLoaded ?? 0
            let started = DispatchTime.now().uptimeNanoseconds
            let result = try executor.prefillChunk(tokens, blockWidth: width)
            let wall = Double(DispatchTime.now().uptimeNanoseconds - started) / 1_000_000
            executor.recordsBlockTiming = false
            executor.recordsBlockGaps = false
            executor.recordsMixtureStartProbe = false
            return Arm(
                label: label,
                width: width,
                wall: wall / Double(length),
                gaps: executor.blockGaps,
                megabytes: Double((runtime.expertStreamer?.bytesRead ?? 0) - bytesBefore)
                    / 1_048_576,
                misses: (runtime.expertStreamer?.expertsLoaded ?? 0) - missesBefore,
                last: result.last
            )
        }

        /// One block's layer records: the second one recorded, which is the warm arm.
        func lastBlock(_ arm: Arm) -> [DeepSeekV41BlockGap] {
            let layers = max(1, runtime.plan.count)
            if arm.width == 1 {
                // One position per block, so the warm block is the last one recorded.
                return Array(arm.gaps.suffix(layers))
            }
            return arm.gaps.count > layers ? Array(arm.gaps.suffix(layers)) : arm.gaps
        }

        func mean(_ values: [Double]) -> Double {
            values.isEmpty ? 0 : values.reduce(0, +) / Double(values.count)
        }

        func median(_ values: [Double]) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[sorted.count / 2]
        }

        /// Pearson's correlation, so "does this layer's own read predict its own pickup"
        /// is a number rather than an impression (`#50` reported +0.05 on the walk).
        func correlation(_ a: [Double], _ b: [Double]) -> Double {
            guard a.count == b.count, a.count > 2 else { return 0 }
            let meanA = mean(a)
            let meanB = mean(b)
            var covariance = 0.0
            var varianceA = 0.0
            var varianceB = 0.0
            for index in a.indices {
                let da = a[index] - meanA
                let db = b[index] - meanB
                covariance += da * db
                varianceA += da * da
                varianceB += db * db
            }
            let denominator = (varianceA * varianceB).squareRoot()
            return denominator > 0 ? covariance / denominator : 0
        }

        // One discarded pass so no arm pays slot creation or a cold page cache.
        executor.reset()
        runtime.arena.reset()
        _ = try executor.prefillChunk(Array(stream.prefix(8)))

        let arms: [(label: String, width: Int, probe: Probe?)] = [
            (label: "control", width: shippedWidth, probe: nil),
            (label: "probe", width: shippedWidth, probe: Probe(depth: 1, binding: 0)),
            (label: "probe x3", width: shippedWidth, probe: Probe(depth: 3, binding: 0)),
            (label: "probe dirty", width: shippedWidth, probe: Probe(depth: 1, binding: 4)),
            (label: "width 1", width: 1, probe: nil)
        ]
        var measured: [Arm] = []
        for round in 1...4 {
            let offset = (round - 1) % arms.count
            let order = Array(arms[offset...]) + Array(arms[..<offset])
            for arm in order {
                let sample = try measure(label: arm.label, width: arm.width, probe: arm.probe)
                print(
                    String(
                        format: "🔎 [dsv41 real]   %-12@ width %2d round %d wall %6.1f "
                            + "ms/position, %.0f MB, %d misses",
                        sample.label as NSString,
                        sample.width,
                        round,
                        sample.wall,
                        sample.megabytes,
                        sample.misses
                    )
                )
                guard round > 1 else { continue }
                measured.append(sample)
            }
        }

        for arm in measured {
            let records = lastBlock(arm)
            XCTAssertGreaterThan(records.count, 2, "too few layers to partition")
            let pickup = records.map(\.mixturePickupMilliseconds)
            let bytes = records.map(\.fillMegabytes)
            let windows = records.filter { $0.mixtureBufferCount > 0 }
            XCTAssertEqual(
                windows.count, records.count,
                "every recorded layer pass must report a mixture window"
            )
            let read = bytes.reduce(0, +)
            // Per position as well as per layer pass: the totals bill a position, and a
            // block's layer pass covers `count` of them.
            let perPosition = mean(pickup) * Double(records.count)
                / Double(max(1, records.first?.count ?? 1))
            print(
                String(
                    format: "🔎 [dsv41 real] %-12@ mixture pickup %.2f ms/layer pass "
                        + "(median %.2f, %.2f ms/position), fill %.1f MB/layer pass "
                        + "(%d misses), %.4f ms per MB",
                    arm.label as NSString,
                    mean(pickup),
                    median(pickup),
                    perPosition,
                    read / Double(records.count),
                    records.map(\.fillMisses).reduce(0, +) / records.count,
                    read > 0 ? mean(pickup) * Double(records.count) / read : 0
                )
            )
            // Whether the queue would start anything at that moment. `#50` found a walked
            // position's probe starting in a tenth of a millisecond while the expert half
            // behind it waited a millisecond and a half; if a block's probe starts at once
            // too, the pickup belongs to the mixture buffer and not to the gap.
            if !arm.gaps.isEmpty, let probeCount = records.first?.probes.count, probeCount > 0 {
                var pickups: [Double] = []
                var started: [Double] = []
                for record in records {
                    for probe in record.probes {
                        pickups.append(probe.pickupMilliseconds)
                        XCTAssertTrue(probe.hasWindow, "a probe reported no window")
                    }
                    let first = record.probes[0]
                    if first.windowStartNanoseconds > 0,
                       record.mixtureWindowStartNanoseconds > 0 {
                        started.append(
                            Double(
                                Int64(record.mixtureWindowStartNanoseconds)
                                    - Int64(first.windowStartNanoseconds)
                            ) / 1_000_000
                        )
                    }
                }
                print(
                    String(
                        format: "🔎 [dsv41 real] %-12@ probes: %d per layer pass, pickup "
                            + "mean %.2f / median %.2f ms, and the mixture starts %.2f ms "
                            + "after the first probe (mean)",
                        arm.label as NSString,
                        probeCount,
                        mean(pickups),
                        median(pickups),
                        mean(started)
                    )
                )
                XCTAssertLessThan(
                    median(pickups),
                    max(0.5, median(pickup)),
                    "a trivial buffer committed at the same moment waited as long as the mixture"
                )
            }
            // The mechanism question: is the stall proportional to what the layer read?
            let previous = [0.0] + Array(bytes.dropLast())
            print(
                String(
                    format: "🔎 [dsv41 real] %-12@ pickup against bytes: r = %+.2f own, "
                        + "%+.2f previous layer's",
                    arm.label as NSString,
                    correlation(bytes, pickup),
                    correlation(previous, pickup)
                )
            )
        }

        // The guard every arm shares: the pickup must not have moved the answer.
        let control = try XCTUnwrap(measured.first { $0.label == "control" })
        for arm in measured where arm.width == control.width {
            let scale = max(control.last.map { abs(Double($0)) }.max() ?? 0, 1e-6)
            let drift = zip(control.last, arm.last)
                .map { abs(Double($0) - Double($1)) }
                .max() ?? 0
            XCTAssertLessThan(
                drift / scale, 1e-6,
                "\(arm.label) moved the logits, so a probe is not inert"
            )
        }
        withExtendedLifetime(engine) {}
    }
}
