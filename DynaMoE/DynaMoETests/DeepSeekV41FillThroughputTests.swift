//
//  DeepSeekV41FillThroughputTests.swift
//  DynaMoETests
//
//  Why a walked position's expert fill runs at 5.5-7.3 GB/s on a disk that serves
//  13.8 (`#50`), measured rather than argued about.
//
//  The walk's fill is not the isolated fill the arena was tuned against: a layer
//  that misses ONE expert asks the disk for that expert's six ranges in one shard,
//  so the queue is six deep however large those ranges are, and 17.9 MB arrives in
//  3.8-4.7 ms where the disk would take 1.3. The candidate explanations are the
//  fan-out's depth (`chunked` splits each range, so the same bytes are in far more
//  requests), the destination (a Metal shared slot against a plain host buffer),
//  and the machine's state during GPU work, which no isolated arm can reproduce.
//
//  So this case is a decomposition, not a benchmark: the same fill shapes over the
//  same distinct `(layer, expert)` pairs, interleaved across rounds so the first
//  arm is not simply the cold one, plus a second family that reads the SAME ranges
//  through the SAME reader into a malloc'd buffer. That second family is the
//  control the isolated fill never had, and it is what separates "the disk is slow
//  here" from "this fill is slow here".
//
//  Opt-in: it reads tens of gigabytes of a 475 GB checkpoint.
//
//      touch /tmp/dynamoe_dsv41_fill_measure
//      xcodebuild test-without-building ... \
//          -only-testing:DynaMoETests/DeepSeekV41FillThroughputTests
//

import XCTest
import Metal
@testable import DynaMoE

final class DeepSeekV41FillThroughputTests: XCTestCase {

    /// Touching this file opts the run in, for the runs where `xcodebuild` has not
    /// forwarded `DYNAMOE_DSV41_FILL_MEASURE`.
    static let markerPath = "/tmp/dynamoe_dsv41_fill_measure"

    override func setUpWithError() throws {
        let enabled = ProcessInfo.processInfo.environment["DYNAMOE_DSV41_FILL_MEASURE"] == "1"
            || FileManager.default.fileExists(atPath: Self.markerPath)
        try XCTSkipUnless(
            enabled,
            "opt-in published-checkpoint run; touch \(Self.markerPath) (or set "
                + "DYNAMOE_DSV41_FILL_MEASURE=1) to run"
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

    /// The same loading sequence the app performs, so the fill measured here is the
    /// fill that ships rather than one a fixture built.
    ///
    /// The engine comes back with the runtime and the caller has to keep it alive for
    /// as long as the runtime is used: the shard buffers are `bytesNoCopy` views of the
    /// engine's mappings, and an engine that goes out of scope unmaps them under a
    /// buffer the graph will happily read. That failure is a SIGSEGV at an address in
    /// the middle of nowhere (measured: 484 GB) with a stack that points at the gain
    /// vector being widened, which names nothing useful.
    private func makeRuntime(
        snapshot: URL,
        config: DeepSeekV41Config,
        contextLength: Int = 8
    ) throws -> (runtime: DeepSeekV41Runtime, engine: DynaMoeEngine) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("no Metal device")
        }
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
        guard let runtime = DeepSeekV41Runtime(
            snapshotDir: snapshot,
            summary: summary,
            config: config,
            shardBuffers: buffers,
            contextLength: contextLength,
            allowEngramSkip: false,
            // These cases measure the fill's FORM, and every number `#48`-`#52`
            // logged was taken with the arena unwired. The shipping residency policy
            // (`#57`) would put every destination on the disk's ceiling and hide the
            // differences the arms exist to show, so the regime is pinned.
            wireArena: false
        ) else {
            throw XCTSkip(
                "the runtime did not build: "
                    + (DeepSeekV41Runtime.lastInitializationError ?? "no reason recorded")
            )
        }
        // `#58` made the arena's frequency rule the shipping default; `#48`-`#52`'s
        // tables were taken under LRU, so the regime is pinned here for the same reason
        // as `wireArena`.
        runtime.arena.victimPolicy = .leastRecentlyUsed
        return (runtime, engine)
    }

    // MARK: - The decomposition

    /// One fill observed end to end: what the streamer reported inside it, and what
    /// the caller paid for it including placement.
    private struct FillObservation: Sendable {
        let misses: Int
        let shards: Int
        let ranges: Int
        let bytes: Int
        let fillMilliseconds: Double
        let totalMilliseconds: Double

        var gigabytesPerSecond: Double {
            fillMilliseconds > 0
                ? Double(bytes) / (fillMilliseconds / 1000) / 1_000_000_000 : 0
        }

        var totalGigabytesPerSecond: Double {
            totalMilliseconds > 0
                ? Double(bytes) / (totalMilliseconds / 1000) / 1_000_000_000 : 0
        }
    }

    func testWhatAFillsShapeCostsOnThePublishedCheckpoint() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        let (runtime, engine) = try makeRuntime(snapshot: snapshot, config: config)
        let report = try await Task.detached { [runtime, snapshot] () -> String in
            guard let streamer = runtime.expertStreamer else {
                return "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
            }
            let geometry = streamer.geometry
            let weights = runtime.weights
            let layers = runtime.plan.map { $0.index }
            let expertsPerLayer = config.numRoutedExperts

            let forms: [(name: String, form: DeepSeekV41FillForm)] = [
                ("shard-batched", .shardBatched),
                ("serial", .serial),
                ("chunk 8 MB", .chunked(bytes: 8 << 20)),
                ("chunk 2 MB", .chunked(bytes: 2 << 20)),
                ("chunk 512 KB", .chunked(bytes: 512 << 10)),
                ("chunk 128 KB", .chunked(bytes: 128 << 10))
            ]
            let counts = [1, 3, 6]
            // One round per arm so every arm is measured at every position in the
            // order, which is what removes the page-cache advantage of being early.
            let rounds = 6

            print(
                String(
                    format: "🔎 [dsv41 fill] slot %.2f MB (%@), arena %d slots, %d layers, "
                        + "%d experts/layer",
                    Double(geometry.slotBytes) / 1_048_576,
                    geometry.format.rawValue,
                    streamer.capacity,
                    layers.count,
                    expertsPerLayer
                )
            )

            // Every `(layer, expert)` this run asks for is distinct, so no fill reads
            // a slot the arena already holds and no page is served twice.
            var used = Set<Int>()
            var layerCursor = 0
            var expertCursor = 0
            func nextLayer() -> Int {
                defer { layerCursor += 1 }
                return layers[layerCursor % layers.count]
            }
            func nextExpert(in layer: Int) -> Int {
                while true {
                    let expert = (expertCursor * 11 + layerCursor) % expertsPerLayer
                    expertCursor += 1
                    let key = layer * expertsPerLayer + expert
                    if used.insert(key).inserted { return expert }
                }
            }

            // The same ranges through the same reader into a plain host buffer.
            //
            // The shard's own file, so the offsets a range carries are the offsets
            // this descriptor serves: the reader refuses a range past the end, which
            // is what makes a mis-resolved file show up as a missing arm rather than
            // as a fast one.
            let shardFiles = ((try? FileManager.default.contentsOfDirectory(
                atPath: snapshot.path
            )) ?? []).filter { $0.hasSuffix(".safetensors") }.sorted()
            var readers: [UInt32: DeepSeekV41ExpertShardReader] = [:]
            func reader(for shard: UInt32) -> DeepSeekV41ExpertShardReader? {
                if let existing = readers[shard] { return existing }
                guard Int(shard) < shardFiles.count else { return nil }
                let url = snapshot.appendingPathComponent(shardFiles[Int(shard)])
                guard let opened = try? DeepSeekV41ExpertShardReader(url: url) else {
                    return nil
                }
                readers[shard] = opened
                return opened
            }

            func ranges(
                layer: Int,
                expert: Int,
                into base: UnsafeMutableRawPointer
            ) -> (shard: UInt32, ranges: [DeepSeekV41ExpertShardReader.Range])? {
                guard let gate = weights.tensor(.expertW1, layer: layer, expert: expert),
                      let up = weights.tensor(.expertW3, layer: layer, expert: expert),
                      let down = weights.tensor(.expertW2, layer: layer, expert: expert) else {
                    return nil
                }
                let gateScale = weights.scaleTensor(.expertW1, layer: layer, expert: expert)
                let upScale = weights.scaleTensor(.expertW3, layer: layer, expert: expert)
                let downScale = weights.scaleTensor(.expertW2, layer: layer, expert: expert)
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
                    range(up, offset: geometry.upWeightOffset, length: geometry.upWeightBytes),
                    range(upScale, offset: geometry.upScaleOffset, length: geometry.upScaleBytes),
                    range(down, offset: geometry.downWeightOffset, length: geometry.downWeightBytes),
                    range(downScale, offset: geometry.downScaleOffset, length: geometry.downScaleBytes)
                ].compactMap { $0 }
                return (gate.shardIndex, pieces)
            }

            /// The same pairs read into host memory. One destination per pair, laid
            /// out back to back, so the bytes are disjoint exactly as a slot's are.
            func hostFill(
                layer: Int,
                experts: [Int],
                chunkBytes: Int,
                sequential: Bool
            ) -> FillObservation? {
                let slot = geometry.slotBytes
                let buffer = UnsafeMutableRawPointer.allocate(
                    byteCount: slot * max(1, experts.count), alignment: 4096
                )
                defer { buffer.deallocate() }
                var byShard: [UInt32: [DeepSeekV41ExpertShardReader.Range]] = [:]
                for (index, expert) in experts.enumerated() {
                    guard let made = ranges(
                        layer: layer,
                        expert: expert,
                        into: buffer.advanced(by: index * slot)
                    ) else { return nil }
                    byShard[made.shard, default: []].append(contentsOf: made.ranges)
                }
                let ranged = byShard.values.reduce(0) { $0 + $1.count }
                let started = DispatchTime.now().uptimeNanoseconds
                for (shard, pieces) in byShard {
                    guard let opened = reader(for: shard) else { return nil }
                    do {
                        try opened.read(pieces, chunkBytes: chunkBytes, sequential: sequential)
                    } catch {
                        print("🔎 [dsv41 fill] host arm refused: \(error)")
                        return nil
                    }
                }
                let elapsed = Double(
                    DispatchTime.now().uptimeNanoseconds - started
                ) / 1_000_000
                return FillObservation(
                    misses: experts.count,
                    shards: byShard.count,
                    ranges: ranged,
                    bytes: experts.count * slot,
                    fillMilliseconds: elapsed,
                    totalMilliseconds: elapsed
                )
            }

            var observed: [String: [Int: [FillObservation]]] = [:]

            // One round of every arm, with the ORDER rotated by the round.
            //
            // That rotation is load-bearing, not tidiness: the first fill of a round
            // runs against a colder page cache than the ones behind it, and a fixed
            // order made the shipping form look 2x slower than `.chunk 8 MB` — which
            // is the SAME code path, because every published range is smaller than
            // 8 MB and splitting an unsplittable range is the identity. That pair is
            // kept in the table as the harness's own control: if the two disagree
            // after rotation, nothing else in the table means anything.
            func oneRound(_ round: Int, printing: Bool) {
                let rotated = (0..<forms.count).map { forms[($0 + round) % forms.count] }
                for (name, form) in rotated {
                    streamer.fillForm = form
                    streamer.recordsFillSamples = true
                    for count in counts {
                        let layer = nextLayer()
                        var experts: [Int] = []
                        for _ in 0..<count { experts.append(nextExpert(in: layer)) }
                        streamer.clearFillSamples()
                        let started = DispatchTime.now().uptimeNanoseconds
                        _ = try? streamer.ensureResident(layer: layer, experts: experts)
                        let total = Double(
                            DispatchTime.now().uptimeNanoseconds - started
                        ) / 1_000_000
                        let samples = streamer.fillSamples
                        let observation = FillObservation(
                            misses: samples.reduce(0) { $0 + $1.misses },
                            shards: samples.reduce(0) { $0 + $1.shards },
                            ranges: samples.reduce(0) { $0 + $1.ranges },
                            bytes: samples.reduce(0) { $0 + $1.bytes },
                            fillMilliseconds: samples.reduce(0) { $0 + $1.milliseconds },
                            totalMilliseconds: total
                        )
                        observed[name, default: [:]][count, default: []].append(observation)
                        if printing {
                            print(
                                String(
                                    format: "🔎 [dsv41 fill] %-14@ misses %d shards %d ranges %2d "
                                        + "%6.1f MB  fill %6.2f ms (%5.1f GB/s)  total %6.2f ms",
                                    name,
                                    observation.misses,
                                    observation.shards,
                                    observation.ranges,
                                    Double(observation.bytes) / 1_048_576,
                                    observation.fillMilliseconds,
                                    observation.gigabytesPerSecond,
                                    observation.totalMilliseconds
                                )
                            )
                        }
                    }
                }

                // The control: the same six-expert shape into host memory. Each arm
                // gets its OWN pairs, because three arms over one set of experts left
                // the second and third reading pages the first had just cached (an
                // 88 GB/s "disk" is the tell).
                let hostArms: [(name: String, chunk: Int, sequential: Bool)] = [
                    ("host fan-out", 0, false),
                    ("host serial", 0, true),
                    ("host chunk 512 KB", 512 << 10, false)
                ]
                for offset in 0..<hostArms.count {
                    let arm = hostArms[(offset + round) % hostArms.count]
                    let layer = nextLayer()
                    var experts: [Int] = []
                    for _ in 0..<6 { experts.append(nextExpert(in: layer)) }
                    guard let observation = hostFill(
                        layer: layer,
                        experts: experts,
                        chunkBytes: arm.chunk,
                        sequential: arm.sequential
                    ) else { continue }
                    observed[arm.name, default: [:]][6, default: []].append(observation)
                    if printing {
                        print(
                            String(
                                format: "🔎 [dsv41 fill] %-14@ misses %d shards %d ranges %2d "
                                    + "%6.1f MB  read  %6.2f ms (%5.1f GB/s)",
                                arm.name,
                                observation.misses,
                                observation.shards,
                                observation.ranges,
                                Double(observation.bytes) / 1_048_576,
                                observation.totalMilliseconds,
                                observation.totalGigabytesPerSecond
                            )
                        )
                    }
                }
            }

            for round in 0..<rounds { oneRound(round, printing: round == 0) }

            streamer.recordsFillSamples = false
            streamer.fillForm = .shardBatched

            // Best and median of the rounds per arm: a fill moves by ~20% with the
            // machine's load, so one arm is not a measurement.
            var lines: [String] = []
            let order = forms.map(\.name)
                + ["host fan-out", "host serial", "host chunk 512 KB"]
            for name in order {
                guard let byCount = observed[name] else { continue }
                for count in byCount.keys.sorted() {
                    let samples = byCount[count]!
                    let useFill = name.hasPrefix("host") == false
                    func milliseconds(_ observation: FillObservation) -> Double {
                        useFill ? observation.fillMilliseconds : observation.totalMilliseconds
                    }
                    let best = samples.min { milliseconds($0) < milliseconds($1) }!
                    let median = samples.map(milliseconds).sorted()[samples.count / 2]
                    let bytes = Double(best.bytes) / 1_048_576
                    lines.append(String(
                        format: "   %-17@ misses %d  best %6.2f ms (%5.1f GB/s)  "
                            + "median %6.2f ms (%5.1f GB/s)",
                        name,
                        best.misses,
                        milliseconds(best),
                        best.gigabytesPerSecond,
                        median,
                        bytes / median
                    ))
                }
            }
            for opened in readers.values { opened.closeNow() }
            return lines.joined(separator: "\n")
        }.value
        print("🔎 [dsv41 fill] best-of-\(6):\n\(report)")

        XCTAssertTrue(report.contains("misses"), "the sweep produced no arms")
        let streamer = try XCTUnwrap(runtime.expertStreamer)
        print("🔎 [dsv41 fill] \(streamer.describe())")
        // The engine's mappings are what the runtime's buffers point at.
        withExtendedLifetime(engine) {}
    }

    // MARK: - Inside a walk

    /// The same question inside a walk, which is the only place the answer matters.
    ///
    /// An isolated fill runs on a quiet machine with a warm page cache; a walk's runs
    /// while the GPU is reading tens of gigabytes of arena out of the same unified
    /// memory, which is what `#49` and `#50` could not separate. So this case walks
    /// real positions with the fill accounting on and prices each form by what it
    /// actually moved: bytes over fill seconds, plus the position's own wall time.
    ///
    /// The arms alternate and each takes its own token stream, because the arena
    /// serves the second walk over the same tokens for free (`#26`) and a fill rate
    /// measured on zero misses is a rate of nothing.
    func testWhatAFillFormIsWorthInAWalk() async throws {
        let snapshot = try snapshotDirectory()
        let config = try DeepSeekV41Config.decode(
            contentsOf: snapshot.appendingPathComponent("config.json")
        )
        // Real text through the checkpoint's own tokenizer, not invented ids: an
        // id no prompt can carry is not a position this graph was built for, and the
        // first run of this case crashed the process inside the Engram stage on one.
        guard let tokenizer = try? DynaMoeTokenizer(
            tokenizerPath: snapshot.appendingPathComponent("tokenizer.json").path
        ) else {
            throw XCTSkip("the checkpoint's tokenizer could not be opened")
        }
        let texts = [
            "The capital of France is Paris, and the river that runs through the city is "
                + "the Seine. Its museums hold the work of centuries, and its streets hold "
                + "the arguments of the afternoon. A visitor who arrives by train sees the "
                + "same skyline the painters saw, only with better coffee and worse weather.",
            "A sparse mixture of experts routes every token to a few of its many "
                + "feed-forward blocks, so the machine reads a small part of a large model "
                + "for each position. The saving is real only when the routed weights are "
                + "cheap to reach, which is a property of the storage, not of the maths.",
            "func fibonacci(_ n: Int) -> Int { if n < 2 { return n } return fibonacci(n - 1) "
                + "+ fibonacci(n - 2) } // the naive form is exponential, and the memoised "
                + "one is linear in both time and space, which is the trade the article "
                + "spends nine paragraphs describing before it shows the table.",
            "Die Bibliothek öffnet um neun Uhr, und der Lesesaal bleibt bis zum Abend "
                + "geöffnet. Wer ein Buch braucht, das im Magazin lagert, bestellt es am "
                + "Schalter und wartet ungefähr zwanzig Minuten, bis es oben ankommt.",
            "The measured numbers were 5.5 to 7.3 gigabytes per second against a disk "
                + "that serves 13.8, which is a factor of two nobody had priced, and the "
                + "report says so in the third paragraph after the table of arms and their "
                + "medians, minimums and the correlations that failed to explain them.",
            "Reading a file with positional reads needs no seek, so disjoint ranges can "
                + "be read at the same time with nothing shared but the descriptor. The "
                + "lock is held for the whole batch so a close cannot recycle the number "
                + "under an in-flight read, which would silently return another file.",
            "In the summer the market sells tomatoes, peaches, basil and bread, and the "
                + "queue at the cheese stall is longer than the queue everywhere else. "
                + "Nobody minds, because the cheese is worth it and the morning is cool.",
            "A command buffer is not a thread and a queue is not a worker. Work is "
                + "submitted in order and begun whenever the device gets to it, which is "
                + "why a probe that binds nothing can start in a tenth of a millisecond "
                + "while the buffer behind it waits for a machine that is reading.",
            "The temperature outside was minus four degrees, the wind came from the "
                + "north-east, and the snow on the roof had not moved in three days. "
                + "Everyone agreed it was colder than last year, and nobody checked."
        ]
        let streams = texts.compactMap { try? tokenizer.encode(text: $0) }
        XCTAssertEqual(streams.count, texts.count, "a walk text did not tokenize")
        let positions = streams.map(\.count).min() ?? 0
        XCTAssertGreaterThan(positions, 8, "a walk text tokenized to almost nothing")
        let (runtime, engine) = try makeRuntime(
            snapshot: snapshot, config: config, contextLength: positions + 1
        )
        let report = try await Task.detached { [runtime, streams] () -> String in
            guard let streamer = runtime.expertStreamer else {
                return "no expert streamer: \(runtime.expertStreamerProblem ?? "no reason")"
            }
            let arms: [(name: String, form: DeepSeekV41FillForm)] = [
                ("shard-batched", .shardBatched),
                ("chunk 2 MB", .chunked(bytes: 2 << 20)),
                ("chunk 512 KB", .chunked(bytes: 512 << 10))
            ]
            let positions = streams.map(\.count).min() ?? 0
            guard positions > 0 else { return "no token stream" }

            struct ArmResult {
                var fills = 0
                var misses = 0
                var bytes = 0
                var fillSeconds = 0.0
                var wallSeconds = 0.0
            }
            var results: [String: ArmResult] = [:]
            // One executor for every arm, reset between them: what is under test is
            // the fill, and building the graph per arm makes the comparison carry a
            // second variable for no reason.
            let executor = runtime.forward()
            let rounds = arms.count
            let printing = true
            for round in 0..<rounds {
                for offset in 0..<arms.count {
                    let arm = arms[(offset + round) % arms.count]
                    executor.reset()
                    streamer.fillForm = arm.form
                    streamer.recordsFillSamples = true
                    streamer.clearFillSamples()
                    let stream = streams[(round * arms.count + offset) % streams.count]
                    let started = DispatchTime.now().uptimeNanoseconds
                    for (position, token) in stream.prefix(positions).enumerated() {
                        _ = try executor.forwardToken(Int(token), position: position)
                    }
                    let wall = Double(
                        DispatchTime.now().uptimeNanoseconds - started
                    ) / 1_000_000_000
                    let samples = streamer.fillSamples
                    let before = results[arm.name] ?? ArmResult()
                    var result = before
                    result.fills += samples.count
                    result.misses += samples.reduce(0) { $0 + $1.misses }
                    result.bytes += samples.reduce(0) { $0 + $1.bytes }
                    result.fillSeconds += samples.reduce(0) { $0 + $1.seconds }
                    result.wallSeconds += wall
                    results[arm.name] = result
                    streamer.clearFillSamples()
                    // This walk's own numbers, not the arm's running total: the table
                    // is per round, so the arms can be read against each other.
                    let roundBytes = result.bytes - before.bytes
                    let roundFills = result.fills - before.fills
                    let roundMisses = result.misses - before.misses
                    let roundFill = result.fillSeconds - before.fillSeconds
                    if printing {
                        print(
                            String(
                                format: "🔎 [dsv41 walk fill] round %d %-14@ %d positions  "
                                    + "%6.1f MB in %d fills, %4.2f misses/fill  fill %6.3f s  "
                                    + "%5.1f GB/s  %6.1f ms/position",
                                round,
                                arm.name,
                                positions,
                                Double(roundBytes) / 1_048_576,
                                roundFills,
                                Double(roundMisses) / Double(max(1, roundFills)),
                                roundFill,
                                roundFill > 0
                                    ? Double(roundBytes) / roundFill / 1_000_000_000 : 0,
                                wall * 1000 / Double(positions)
                            )
                        )
                    }
                }
            }
            streamer.recordsFillSamples = false
            streamer.fillForm = .shardBatched

            var lines: [String] = []
            for arm in arms {
                guard let result = results[arm.name] else { continue }
                lines.append(String(
                    format: "   %-14@ %6.1f GB read in %d fills, %d misses  "
                        + "fill %6.3f s (%5.1f GB/s)  walk %6.1f s (%5.1f ms/position)",
                    arm.name,
                    Double(result.bytes) / 1_000_000_000,
                    result.fills,
                    result.misses,
                    result.fillSeconds,
                    result.fillSeconds > 0
                        ? Double(result.bytes) / result.fillSeconds / 1_000_000_000 : 0,
                    result.wallSeconds,
                    result.wallSeconds * 1000 / Double(max(1, positions * rounds))
                ))
            }
            return lines.joined(separator: "\n")
        }.value
        print("🔎 [dsv41 walk fill] totals:\n\(report)")
        XCTAssertTrue(report.contains("GB/s"), "the walk arms produced no fills")
        // The engine's mappings are what the runtime's buffers point at.
        withExtendedLifetime(engine) {}
    }
}
