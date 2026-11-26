//
//  Inkling2ProfileTests.swift
//  DynaMoETests
//
//  Phase 6, slice 2: the instrument that says where a visit's time goes.
//
//  Slice 1 measured a decode position on the published checkpoint at 0.296 s
//  and the output head at 0.9% of it, which left one sentence: the position IS
//  its 48 visits at about 6.2 ms each. Nothing split those 6.2 ms. One visit is
//  thirteen command buffers, each committed and then waited on, so the four
//  questions a reader actually has are: how much host time built them, how long
//  the queue sat before starting them, how much arithmetic they were, and what
//  is left over (read-backs, host projections, the KV staging copy).
//
//  Three things are pinned here:
//
//    * the arithmetic of a stage record, including the subtraction that must
//      not go negative when the bills are larger than the wall;
//    * a committed buffer's own window, on a real GPU dispatch, because
//      `gpuStartTime`/`gpuEndTime` are the only honest device figure and a
//      conversion bug there would silently make every device total a floor;
//    * on the published checkpoint, that the instrument is INERT until it is
//      turned on, then bills exactly one record per visit and covers the whole
//      visit wall -- and that its two big structural claims hold: one expert
//      buffer per routed expert, and no stage whose device window is missing.
//
//  The whole-position measurement is opt-in: `touch
//  /tmp/dynamoe_inkling2_profile`, and run it from an `-O` build (a Debug build
//  of this port is ~60x slower and would measure the compiler).
//

import Metal
import XCTest
@testable import DynaMoE

// MARK: - The arithmetic

final class Inkling2StageTimingTests: XCTestCase {

    private func timing(
        wall: UInt64, encode: UInt64, drain: UInt64, queue: UInt64 = 0,
        device: UInt64 = 0, commits: Int = 0
    ) -> Inkling2StageTiming {
        var record = Inkling2StageTiming()
        record.wallNanoseconds = wall
        record.encodeNanoseconds = encode
        record.drainNanoseconds = drain
        record.queueWaitNanoseconds = queue
        record.deviceNanoseconds = device
        record.commits = commits
        return record
    }

    /// The host residue is what neither encode nor drain billed.
    func testTheHostResidueIsWhatEncodeAndDrainDidNotBill() {
        let record = timing(wall: 1_000, encode: 300, drain: 500, commits: 1)
        XCTAssertEqual(record.hostNanoseconds, 200)
    }

    /// A record whose bills overrun its wall answers zero rather than wrapping:
    /// a subtraction that went negative on a `UInt64` would read as a
    /// near-`UInt64.max` host bucket and swamp every table it appeared in.
    func testAResidueIsZeroRatherThanWrappingWhenTheBillsOverrun() {
        let record = timing(wall: 100, encode: 300, drain: 500)
        XCTAssertEqual(record.hostNanoseconds, 0)
        let exact = timing(wall: 800, encode: 300, drain: 500)
        XCTAssertEqual(exact.hostNanoseconds, 0)
    }

    /// One buffer bills encode, queue wait, device and drain once, and an empty
    /// window counts rather than being dropped.
    func testAddingABufferBillsEachBucketOnce() {
        var record = Inkling2StageTiming()
        let clock = Inkling2BufferClock(
            encodeNanoseconds: 10, queueWaitNanoseconds: 20, deviceNanoseconds: 30, windowed: true
        )
        record.add(clock, drainNanoseconds: 50)
        record.add(clock, drainNanoseconds: 50)
        XCTAssertEqual(record.encodeNanoseconds, 20)
        XCTAssertEqual(record.queueWaitNanoseconds, 40)
        XCTAssertEqual(record.deviceNanoseconds, 60)
        XCTAssertEqual(record.drainNanoseconds, 100)
        XCTAssertEqual(record.commits, 2)
        XCTAssertEqual(record.emptyWindows, 0)

        let unwindowed = Inkling2BufferClock(
            encodeNanoseconds: 1, queueWaitNanoseconds: 0, deviceNanoseconds: 0, windowed: false
        )
        record.add(unwindowed, drainNanoseconds: 1)
        XCTAssertEqual(record.emptyWindows, 1)
        XCTAssertEqual(record.commits, 3)
    }

    /// A recorder accumulates across visits and forgets them on `reset`, which
    /// is what keeps a second turn from reporting the first one's visits.
    func testTheRecorderAccumulatesVisitsAndForgetsThemOnReset() {
        let profile = Inkling2VisitProfile()
        for visit in 0..<3 {
            var buffers = Inkling2StageTiming()
            buffers.add(
                Inkling2BufferClock(
                    encodeNanoseconds: 10, queueWaitNanoseconds: 20,
                    deviceNanoseconds: 30, windowed: true
                ),
                drainNanoseconds: 50
            )
            profile.record(stage: .expertMixture, wallNanoseconds: UInt64(100 + visit), buffers: buffers)
            profile.closeVisit(wallNanoseconds: 200, attentionQueries: 4)
            profile.recordExperts(count: 8, bytes: 1_024)
        }
        XCTAssertEqual(profile.visits, 3)
        XCTAssertEqual(profile.wallNanoseconds, 600)
        XCTAssertEqual(profile.attentionQueries, 12)
        XCTAssertEqual(profile.experts, 24)
        XCTAssertEqual(profile.expertBytes, 3_072)
        // 100 + 101 + 102, which is the wall each visit actually billed.
        XCTAssertEqual(profile.billedWallNanoseconds, 303)
        let mixture = profile.timing(.expertMixture)
        XCTAssertEqual(mixture.commits, 3)
        XCTAssertEqual(mixture.deviceNanoseconds, 90)
        XCTAssertEqual(mixture.drainNanoseconds, 150)
        XCTAssertEqual(mixture.encodeNanoseconds, 30)
        // 303 of wall, 30 encode and 150 drain billed, so the rest is host.
        XCTAssertEqual(mixture.hostNanoseconds, 303 - 30 - 150)

        profile.reset()
        XCTAssertEqual(profile.visits, 0)
        XCTAssertEqual(profile.wallNanoseconds, 0)
        XCTAssertTrue(profile.stages.isEmpty)
        XCTAssertEqual(profile.billedWallNanoseconds, 0)
        XCTAssertEqual(profile.experts, 0)
        XCTAssertEqual(profile.timing(.expertMixture).commits, 0)
    }

    /// The summary names the stages it measured, omits the ones it did not, and
    /// reports per visit rather than a total: a total is what a reader divides
    /// wrongly.
    func testTheSummaryReportsPerVisitAndOmitsUnmeasuredStages() {
        let profile = Inkling2VisitProfile()
        var buffers = Inkling2StageTiming()
        buffers.add(
            Inkling2BufferClock(
                encodeNanoseconds: 1_000_000, queueWaitNanoseconds: 0,
                deviceNanoseconds: 2_000_000, windowed: true
            ),
            drainNanoseconds: 3_000_000
        )
        profile.record(
            stage: .expertMixture, wallNanoseconds: 10_000_000, buffers: buffers
        )
        profile.closeVisit(wallNanoseconds: 20_000_000, attentionQueries: 4)
        profile.closeVisit(wallNanoseconds: 20_000_000, attentionQueries: 4)
        profile.recordExperts(count: 16, bytes: 3 * 1_048_576)

        let line = profile.describe(label: "unit")
        XCTAssertTrue(line.contains("2 visits"), line)
        XCTAssertTrue(line.contains("at 20.00 ms"), line)
        XCTAssertTrue(line.contains("expertMixture 5.00"), line)
        XCTAssertTrue(line.contains("16 experts"), line)
        XCTAssertTrue(line.contains("3.0 MB"), line)
        // Committed once, so the per-visit table cannot report it as two.
        XCTAssertTrue(line.contains("1 commit(s)"), line)
        // Never measured, so it must not appear with a zero as if it were free.
        XCTAssertFalse(line.contains("hostPrep"), line)
        XCTAssertTrue(line.contains("device 1.00"), line)
    }

    /// The arena's counters reach the same line as the stage table, and "no
    /// arena" is distinct from "an arena with no hits": a reader must not see a
    /// 0% hit rate where no residency policy exists at all.
    func testTheArenaCountersReachTheProfileLine() {
        let profile = Inkling2VisitProfile()
        profile.closeVisit(wallNanoseconds: 1_000_000, attentionQueries: 1)
        let without = profile.describe(label: "unit")
        XCTAssertNil(profile.arena)
        XCTAssertFalse(without.contains("arena"), without)

        var counters = Inkling2ArenaCounters()
        counters.hits = 7
        counters.misses = 3
        counters.evictions = 1
        counters.bytesRead = 4 * 1_048_576
        counters.preadNanoseconds = 2_000_000
        profile.recordArena(counters)
        let with = profile.describe(label: "unit")
        XCTAssertTrue(with.contains("7 hit(s)"), with)
        XCTAssertTrue(with.contains("3 miss(es)"), with)
        XCTAssertTrue(with.contains("1 eviction(s)"), with)
        XCTAssertTrue(with.contains("4.0 MB read"), with)
        XCTAssertTrue(with.contains("70.0% hit"), with)

        // Cumulative on the arena's side: the latest snapshot wins, it does not sum.
        profile.recordArena(counters)
        XCTAssertEqual(profile.arena?.hits, 7)

        // An expert the slot layout cannot hold never entered the arena, so it
        // must be named rather than counted as a miss.
        counters.incompatible = 2
        profile.recordArena(counters)
        let refused = profile.describe(label: "unit")
        XCTAssertTrue(refused.contains("2 refused"), refused)
    }

    /// The stage list is the visit's own structure, and the three stages that
    /// commit nothing are marked as such: a caller summing `commits` over every
    /// stage must get four per visit, not seven.
    func testTheStageTableSaysWhichStagesCommit() {
        let committing = Inkling2VisitStage.allCases.filter(\.commits)
        XCTAssertEqual(
            Set(committing),
            [.attentionFront, .attentionTail,
             .mlpFront, .expertMixture]
        )
        // Nine until slice 5 folded the post-attention convolution into the
        // attention tail and slice 6 folded the post-MLP one into the mixture:
        // a stage that can never be recorded is a lie in the table, so each was
        // removed rather than left behind.
        XCTAssertEqual(Inkling2VisitStage.allCases.count, 7)
    }
}

// MARK: - A real buffer's window

/// `Inkling2BufferClock` against a dispatch the GPU actually ran. Model-free:
/// it uses the same trivial kernel the executor dispatches, on a 64-value
/// buffer, so it costs one command buffer and no weights.
final class Inkling2BufferWindowTests: XCTestCase {

    private func environment() throws -> (MTLDevice, MTLLibrary, MTLCommandQueue) {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal GPU device")
        }
        guard let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no default library")
        }
        guard let queue = device.makeCommandQueue() else {
            throw XCTSkip("no compute queue")
        }
        return (device, library, queue)
    }

    func testACommittedDispatchesWindowIsInsideItsOwnDrain() throws {
        let (device, library, queue) = try environment()
        let function = try XCTUnwrap(library.makeFunction(name: "inkling_sigmoid_inplace_f32"))
        let pipeline = try device.makeComputePipelineState(function: function)
        let vector = try XCTUnwrap(device.makeBuffer(length: 64 * 4, options: .storageModeShared))
        var count: UInt32 = 64
        let countBuffer = try XCTUnwrap(
            device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared)
        )
        let buffer = try XCTUnwrap(queue.makeCommandBuffer())
        let encoder = try XCTUnwrap(buffer.makeComputeCommandEncoder())
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(countBuffer, offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: 64, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
        encoder.endEncoding()

        let encodeStart = DispatchTime.now().uptimeNanoseconds
        let commit = DispatchTime.now().uptimeNanoseconds
        buffer.commit()
        let waitBegan = DispatchTime.now().uptimeNanoseconds
        buffer.waitUntilCompleted()
        let waitEnded = DispatchTime.now().uptimeNanoseconds
        XCTAssertNil(buffer.error)

        let clock = Inkling2BufferClock(
            encodeStart: encodeStart, commit: commit, buffer: buffer
        )
        print(
            "🔎 [inkling2 profile] one dispatch: encode \(clock.encodeNanoseconds) ns, "
                + "queue \(clock.queueWaitNanoseconds) ns, device \(clock.deviceNanoseconds) ns"
        )
        XCTAssertTrue(clock.windowed, "a dispatch the GPU ran must report a window")
        XCTAssertGreaterThan(clock.deviceNanoseconds, 0)
        // The invariant the per-stage table is summed on: everything after the
        // commit is the drain's territory, and every bucket in it -- the wait
        // itself, the queue before the work began, and the window -- lies inside
        // that interval rather than beside it.
        let afterCommit = waitEnded - commit
        XCTAssertLessThanOrEqual(waitEnded - waitBegan, afterCommit)
        XCTAssertLessThanOrEqual(clock.queueWaitNanoseconds + clock.deviceNanoseconds, afterCommit)
    }

    /// A buffer that never ran has no window, and the clock says so instead of
    /// reporting a zero that a caller would read as "instantaneous".
    func testABufferThatNeverRanReportsNoWindow() throws {
        let (_, _, queue) = try environment()
        let buffer = try XCTUnwrap(queue.makeCommandBuffer())
        let now = DispatchTime.now().uptimeNanoseconds
        let clock = Inkling2BufferClock(encodeStart: now, commit: now, buffer: buffer)
        XCTAssertFalse(clock.windowed)
        XCTAssertEqual(clock.deviceNanoseconds, 0)
        XCTAssertEqual(clock.queueWaitNanoseconds, 0)
        XCTAssertEqual(clock.encodeNanoseconds, 0)
    }
}

// MARK: - The published checkpoint

final class Inkling2VisitProfileTests: XCTestCase {

    private var engine: DynaMoeEngine?
    private var retained: [AnyObject] = []

    private func snapshot() throws -> URL {
        let path = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(Inkling2PublishedTensorTests.snapshotRelativePath)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw XCTSkip("Inkling 2 snapshot is not downloaded at \(path.path)")
        }
        return path
    }

    private struct Harness {
        let config: Inkling2Config
        let forward: Inkling2CPUForward
        let metal: Inkling2MetalForward
        let workspace: Inkling2DeviceWorkspace
        /// The segmented summary the buffers were made from: the arena install
        /// maps a bank's shard index back to the FILE its `pread` opens.
        let summary: ModelSummary
    }

    /// The engine's own arrangement: the executor's `host` IS the forward the
    /// turn runs on, because `runVisit` reads and appends that graph's states.
    /// A harness built with two forwards would run a valid position against the
    /// wrong sequence, and only a timing number would come out.
    private func harness(stagedPositions: Int = 8) throws -> Harness {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let rawSummary = try engine.getSummary()
        let summary = try inkling2MetalReadySummary(rawSummary)
        let config = try Inkling2Config.decode(
            contentsOf: directory.appendingPathComponent("config.json")
        )
        let catalog = Inkling2WeightCatalog(
            mainFile: summary.tensors.map { Inkling2TensorRef(metadata: $0) },
            layerCount: config.numHiddenLayers
        )
        let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
        guard let queue = device.makeCommandQueue(), let library = device.makeDefaultLibrary() else {
            throw XCTSkip("no Metal queue or default library")
        }
        var buffers: [UInt32: MTLBuffer] = [:]
        buffers = try inkling2MetalShardBuffers(summary, device: device)
        let source = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        let forward = try Inkling2CPUForward(
            config: config, weights: source,
            options: Inkling2ForwardOptions(maxPositions: stagedPositions)
        )
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: buffers,
            // A segmented shard's offsets are segment-relative while the arena
            // reads the FILE: without these origins its slots would hold the
            // wrong bytes and no error would be raised.
            fileOrigins: InferenceEngine.shardFileOrigins(original: rawSummary, segmented: summary)
        )
        let metal = try Inkling2MetalForward(
            device: device, library: library, queue: queue, weights: deviceWeights, host: forward
        )
        let workspace = try Inkling2DeviceWorkspace(
            device: device, config: config, stagedPositions: stagedPositions
        )
        forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            try metal.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
        }
        forward.headRunner = { hidden in
            try metal.headLogits(hidden: hidden, workspace: workspace)
        }
        return Harness(
            config: config, forward: forward, metal: metal, workspace: workspace, summary: summary
        )
    }

    /// The instrument is inert until it is turned on, then bills exactly one
    /// record set per visit, and those records cover the visit's wall.
    func testTheInstrumentIsInertUntilItIsTurnedOnAndThenBillsEveryVisit() throws {
        let harness = try harness()
        let profile = harness.metal.visitProfile
        XCTAssertFalse(harness.metal.recordsVisitTiming)
        XCTAssertEqual(profile.visits, 0)

        // Arm one: off. A position runs and the table stays empty.
        _ = try harness.forward.forwardToken(token: 0, position: 0)
        XCTAssertEqual(profile.visits, 0)
        XCTAssertTrue(profile.stages.isEmpty)
        XCTAssertEqual(profile.wallNanoseconds, 0)

        // Arm two: on. The same shape of position, one further along, and every
        // visit the schedule ran is billed.
        //
        // The visits are counted at the RUNNER, not through `onVisitComplete`:
        // the host `runVisit` is the one that fires that hook, and an installed
        // runner replaces it -- which is exactly the substitution under test.
        var visits = 0
        var queriesFromRunner = 0
        let metal = harness.metal
        let workspace = harness.workspace
        harness.forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            let result = try metal.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
            visits += 1
            queriesFromRunner += result.attentionQueries
            return result
        }
        harness.metal.recordsVisitTiming = true
        profile.reset()
        _ = try harness.forward.forwardToken(token: 0, position: 1)
        print("🔎 [inkling2 profile] \(profile.describe(label: "position 1"))")

        XCTAssertGreaterThan(visits, 40, "the 48-visit schedule")
        XCTAssertEqual(profile.visits, visits)
        XCTAssertEqual(profile.attentionQueries, queriesFromRunner)
        XCTAssertGreaterThan(profile.attentionQueries, 0)

        // Every stage that runs is billed, and each one that commits did.
        for stage in Inkling2VisitStage.allCases {
            let time = profile.timing(stage)
            XCTAssertGreaterThan(time.wallNanoseconds, 0, "\(stage) was never billed")
            if stage.commits {
                XCTAssertGreaterThan(time.commits, 0, "\(stage) billed no buffer")
                XCTAssertEqual(time.emptyWindows, 0, "\(stage) reported a buffer with no window")
                XCTAssertLessThanOrEqual(time.deviceNanoseconds, time.wallNanoseconds, "\(stage)")
            } else {
                XCTAssertEqual(time.commits, 0, "\(stage) committed a buffer")
            }
        }
        // The whole mixture is ONE buffer per visit, whatever the routed expert
        // count: twelve commits a visit, and `topK` expert slices inside the
        // mixture's single one (it read thirteen before the folds were folded).
        XCTAssertEqual(profile.timing(.expertMixture).commits, profile.visits)
        XCTAssertEqual(profile.experts, profile.visits * harness.config.numExpertsPerToken)
        XCTAssertGreaterThan(profile.expertBytes, 0)

        // The stage walls tile the visit. The residue is the function-call seams
        // between stages plus whatever sits outside every stage's clock (the
        // read-backs and history pushes a fold moved out of a helper), so it is
        // allowed to be small but never large, and the sum can never exceed the
        // visit it is cut out of.
        //
        // The bar is 90%, not 95%: the two convolution folds took it from >=95%
        // to a measured 94.4% (three runs), and the per-stage assertions above
        // are what prove no stage stopped billing itself -- this one is the
        // coarse backstop, so it should not be the thing that fails first when
        // a fold legitimately moves a clock.
        XCTAssertLessThanOrEqual(profile.billedWallNanoseconds, profile.wallNanoseconds)
        XCTAssertGreaterThanOrEqual(
            profile.billedWallNanoseconds * 100, profile.wallNanoseconds * 90
        )
        // And each bucket is inside the interval that contained it. A drain only
        // covers the wait, while the encode and the queue wait are measured from
        // the commit, so the bound is the stage's wall less its encode.
        for stage in Inkling2VisitStage.allCases {
            let time = profile.timing(stage)
            XCTAssertLessThanOrEqual(time.encodeNanoseconds, time.wallNanoseconds, "\(stage)")
            let afterCommit = time.wallNanoseconds - time.encodeNanoseconds
            XCTAssertLessThanOrEqual(time.drainNanoseconds, afterCommit, "\(stage)")
            XCTAssertLessThanOrEqual(
                time.queueWaitNanoseconds + time.deviceNanoseconds, afterCommit, "\(stage)"
            )
        }

        // A reset is what keeps a second turn's table its own.
        profile.reset()
        XCTAssertEqual(profile.visits, 0)
    }

    /// Where a decode position's time goes, per stage, on the published
    /// checkpoint. `-O` only, and opt-in because it reads the checkpoint's
    /// weights once per visit.
    func testWhereADecodePositionsTimeGoesOnThePublishedCheckpoint() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_profile"),
            "opt-in profile measurement; touch /tmp/dynamoe_inkling2_profile to run"
        )
        let harness = try harness()
        harness.metal.recordsVisitTiming = true
        let profile = harness.metal.visitProfile

        // Visits are counted at the runner, because an installed runner replaces
        // the host `runVisit` that fires `onVisitComplete`.
        var visits = 0
        let metal = harness.metal
        let workspace = harness.workspace
        harness.forward.visitRunner = { layer, pass, hidden, blockStates, messages, position in
            visits += 1
            return try metal.runVisit(
                layer: layer, pass: pass, hidden: hidden, blockStates: &blockStates,
                messages: messages, position: position, workspace: workspace
            )
        }

        // One discarded position warms the clock and the caches, then the
        // measured ones: a first pass on an idle GPU reads several times slow
        // (`Benchmarks/PERF_FINDINGS.md`, "DSV4.1 #53").
        _ = try harness.forward.forwardToken(token: 0, position: 0)

        // The prefault question, alternated: does touching the expert banks on
        // the host before the mixture dispatch move its device window? The arm
        // order flips every round because a first pass is faster for being first
        // ("DSV4.1 #53"), so a fixed order would decide it by itself. Each arm
        // starts from a RESET sequence: the same position twice would advance
        // the KV and convolution state past what the window can hold.
        var arms = [Double](repeating: 0, count: 4)
        for (round, prefaults) in [false, true, true, false].enumerated() {
            metal.prefaultsExpertBanks = prefaults
            harness.forward.reset()
            profile.reset()
            visits = 0
            _ = try harness.forward.forwardToken(token: 0, position: 0)
            arms[round] = Double(profile.timing(.expertMixture).wallNanoseconds) / Double(max(1, visits)) / 1_000_000
        }
        metal.prefaultsExpertBanks = false
        print(
            "🔎 [inkling2 profile] mixture ms/visit by arm (off, on, on, off): "
                + arms.map { String(format: "%.3f", $0) }.joined(separator: " ")
        )

        harness.forward.reset()
        profile.reset()
        visits = 0
        let positions = 4
        let start = CFAbsoluteTimeGetCurrent()
        for position in 0..<positions {
            _ = try harness.forward.forwardToken(token: 0, position: position)
        }
        let seconds = CFAbsoluteTimeGetCurrent() - start
        print("🔎 [inkling2 profile] \(positions) positions in \(String(format: "%.3f", seconds)) s")
        print("🔎 [inkling2 profile] \(profile.describe(label: "\(positions) positions"))")
        print(
            "🔎 [inkling2 profile] billed \(profile.billedWallNanoseconds) of "
                + "\(profile.wallNanoseconds) ns "
                + "(\(String(format: "%.2f", 100 * Double(profile.billedWallNanoseconds) / Double(max(1, profile.wallNanoseconds))))%)"
        )

        XCTAssertEqual(profile.visits, visits)
        XCTAssertEqual(visits % positions, 0)
        let visitsPerPosition = max(1, visits / positions)
        let perPosition = Double(profile.wallNanoseconds) / Double(positions) / 1_000_000
        let perVisit = perPosition / Double(visitsPerPosition)
        print(
            "🔎 [inkling2 profile] \(String(format: "%.2f", perPosition)) ms per position, "
                + "\(String(format: "%.2f", perVisit)) ms per visit"
        )
        for stage in Inkling2VisitStage.allCases {
            let time = profile.timing(stage)
            let ms = Double(time.wallNanoseconds) / Double(profile.visits) / 1_000_000
            let device = Double(time.deviceNanoseconds) / Double(profile.visits) / 1_000_000
            let queue = Double(time.queueWaitNanoseconds) / Double(profile.visits) / 1_000_000
            let host = Double(time.hostNanoseconds) / Double(profile.visits) / 1_000_000
            let share = perVisit > 0 ? 100 * ms / perVisit : 0
            print(
                "🔎 [inkling2 profile] \(stage.rawValue): "
                    + "\(String(format: "%.3f", ms)) ms/visit "
                    + "(\(String(format: "%.1f", share))% of a visit) = "
                    + "host \(String(format: "%.3f", host)), "
                    + "encode \(String(format: "%.3f", Double(time.encodeNanoseconds) / Double(profile.visits) / 1_000_000)), "
                    + "queue \(String(format: "%.3f", queue)), "
                    + "device \(String(format: "%.3f", device)), "
                    + "\(time.commits / max(profile.visits, 1)) commit(s)"
            )
        }
        XCTAssertGreaterThan(profile.wallNanoseconds, 0)
    }

    // MARK: The cold/warm arena comparison


    /// Does a bounded resident expert set pay on the published checkpoint?
    ///
    /// The pipeline streams ~1.75 GB of routed expert banks per position
    /// (384 expert slices), and on a 16 GB machine the expert banks do not fit
    /// the page cache beside the model, so the mmap path is re-reading them from
    /// SSD. A resident arena trades RAM for those reads: a hit costs a slot
    /// lookup, a miss costs the same `pread` the mmap path pays as page faults
    /// plus a `memcpy` into the slot.
    ///
    /// `-O` only, and opt-in (`touch /tmp/dynamoe_inkling2_arena`), because it
    /// reads gigabytes of banks per position. It REPORTS the comparison and
    /// asserts only that its own numbers are not lies: the arena arm filled and
    /// hit slots, and its logits are not degenerate. Fidelity is pinned where it
    /// can be exact -- `testAPublishedBankFillsItsSlotWithTheBanksOwnBytes`
    /// compares the slot against the HOST rule.
    func testTheExpertArenaColdWarmComparisonOnThePublishedCheckpoint() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_arena"),
            "opt-in arena measurement; touch /tmp/dynamoe_inkling2_arena to run"
        )
        let harness = try harness()
        let config = harness.config
        let metal = harness.metal

        // The geometry is the checkpoint's own: every stride is taken from the
        // stored bank shapes, so the slot is exactly the four ranges the mixture
        // reads out of the mmap.
        let gateUp = try metal.weights.expertBank(.expertW13, layer: 0, expert: 0)
        let down = try metal.weights.expertBank(.expertW2, layer: 0, expert: 0)
        let geometry = try XCTUnwrap(Inkling2ExpertSlotGeometry.make(
            gateUpRows: 2 * config.moeIntermediateSize,
            gateUpRowStride: gateUp.rowStride,
            gateUpScaleRowStride: gateUp.scaleRowStride,
            downRows: config.hiddenSize,
            downRowStride: down.rowStride,
            downScaleRowStride: down.scaleRowStride
        ))
        print(
            "🔎 [inkling2 arena] one NVFP4 slot is "
                + "\(String(format: "%.2f", Double(geometry.slotBytes) / 1_048_576)) MB"
        )

        /// One position from a RESET sequence: the logits, the wall, and the
        /// mixture's own wall per visit.
        ///
        /// The arena arm installs the arenas a TURN installs -- one per packed
        /// layout, sized by the memory plan -- through the app's own helper, so
        /// this measures the path that ships rather than a one-layout stand-in.
        func position(
            _ index: Int, arena: Bool
        ) throws -> (logits: [Float], ms: Double, mixtureMs: Double, stats: Inkling2ArenaCounters?) {
            if arena {
                if metal.expertArenas.isEmpty {
                    let line = try inkling2InstallExpertArenas(
                        executor: metal, weights: metal.weights, summary: harness.summary,
                        snapshot: try snapshot(), config: config, catalog: metal.weights.catalog,
                        positions: 8
                    )
                    print("🔎 [inkling2 arena] arenas:\(line)")
                }
            } else {
                metal.removeExpertArena()
            }
            harness.forward.reset()
            metal.recordsVisitTiming = true
            metal.visitProfile.reset()
            let start = CFAbsoluteTimeGetCurrent()
            let logits = try harness.forward.forwardToken(token: 0, position: index)
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1_000
            let visits = max(1, metal.visitProfile.visits)
            let mixture = Double(metal.visitProfile.timing(.expertMixture).wallNanoseconds)
                / Double(visits) / 1_000_000
            let stats = metal.expertArenas.isEmpty
                ? nil
                : Inkling2ArenaCounters.merged(metal.expertArenas.map { $0.arena.statistics })
            return (logits, ms, mixture, stats)
        }
        // Whether the mmap arm can run at all here. A mixture binds the layer's
        // fused bank and its scales as two WHOLE segment buffers; past what the
        // device will wire that dispatch returns zeros with no error, so the
        // guard refuses it by name and the arm becomes a refusal to record
        // rather than a wall to time.
        let layerZero = try metal.weights.expertBank(.expertW13, layer: 0, expert: 0)
        let layerZeroDown = try metal.weights.expertBank(.expertW2, layer: 0, expert: 0)
        var mmapBinding = Inkling2MixtureBinding()
        mmapBinding.add(layerZero.buffer, label: "the w13 bank of layer 0")
        mmapBinding.add(layerZero.scaleBuffer, label: "the w13 scales of layer 0")
        mmapBinding.add(layerZeroDown.buffer, label: "the w2 bank of layer 0")
        mmapBinding.add(layerZeroDown.scaleBuffer, label: "the w2 scales of layer 0")
        let mmapRuns = mmapBinding.bytes <= metal.mixtureBindingBudget
        print(
            "🔎 [inkling2 arena] the mmap mixture binds "
                + "\(Inkling2MixtureBinding.describe(mmapBinding.bytes)) against a "
                + "\(Inkling2MixtureBinding.describe(metal.mixtureBindingBudget)) budget: "
                + (mmapRuns ? "it can run" : "it is refused by name")
        )

        // The arm order FLIPS every round: a first pass is slow for being cold
        // (the page cache holds none of the banks) and a later one is fast for
        // being warm, so a fixed order would decide the comparison by itself.
        var mmapWalls: [Double] = []
        var arenaWalls: [Double] = []
        var mmapRefusals = 0
        for (round, useArena) in [false, true, true, false].enumerated() {
            if !useArena && !mmapRuns {
                XCTAssertThrowsError(try position(round, arena: false)) { error in
                    XCTAssertTrue("\(error)".contains("returns zeros"), "\(error)")
                }
                mmapRefusals += 1
                print(
                    "🔎 [inkling2 arena] round \(round) (mmap): refused by name -- the layer's "
                        + "two bank segments are past what this device will wire"
                )
                continue
            }
            let measured = try position(round, arena: useArena)
            var line = "🔎 [inkling2 arena] round \(round) (\(useArena ? "arena" : "mmap")): "
                + "\(String(format: "%.1f", measured.ms)) ms, "
                + "mixture \(String(format: "%.2f", measured.mixtureMs)) ms/visit"
            if let stats = measured.stats {
                line += String(
                    format: ", %d hit(s) / %d miss(es) / %d eviction(s), %.2f GB read in %.0f ms (%.1f%% hit)",
                    stats.hits, stats.misses, stats.evictions,
                    Double(stats.bytesRead) / 1_073_741_824,
                    Double(stats.preadNanoseconds) / 1_000_000, 100 * stats.hitRate
                )
                arenaWalls.append(measured.ms)
            } else {
                mmapWalls.append(measured.ms)
            }
            print(line)
        }
        XCTAssertFalse(arenaWalls.isEmpty, "no arena arm ran")
        if mmapRuns {
            XCTAssertEqual(mmapWalls.count, 2)
            print(
                "🔎 [inkling2 arena] median: mmap "
                    + "\(String(format: "%.1f", mmapWalls.sorted()[mmapWalls.count / 2])) ms, arena "
                    + "\(String(format: "%.1f", arenaWalls.sorted()[arenaWalls.count / 2])) ms"
            )
        } else {
            XCTAssertEqual(mmapRefusals, 2, "both mmap rounds must have been refused")
            print(
                "🔎 [inkling2 arena] median: arena "
                    + "\(String(format: "%.1f", arenaWalls.sorted()[arenaWalls.count / 2])) ms "
                    + "per position, with no mmap arm to compare against on this device"
            )
        }

        // Correctness, on a POSITION both arms can share when they both exist:
        // the same token at the same index, after the timed rounds have warmed
        // the page cache.
        var streamedLogits: [Float]?
        if mmapRuns {
            streamedLogits = try position(4, arena: false).logits
        }
        let resident = try position(4, arena: true)
        metal.removeExpertArena()
        XCTAssertEqual(resident.logits.count, config.unpaddedVocabSize)
        // The two arms are REPORTED, not required to agree: on this machine the
        // mmap arm is the suspect one (the mixture binds the layer's two 9+ GB
        // bank segments at once, and that binding returns zeros here while each
        // bank alone reads correctly -- see `testAPublishedBankFillsItsSlotWith
        // TheBanksOwnBytes`, which pins the arena path against the HOST rule).
        // An equality assertion here would therefore be asserting that the
        // arena reproduces the broken arm.
        if let streamedLogits {
            let scale = streamedLogits.reduce(Float(0)) { max($0, abs($1)) }
            let delta = zip(streamedLogits, resident.logits)
                .reduce(Float(0)) { max($0, abs($1.0 - $1.1)) }
            print(
                "🔎 [inkling2 arena] arena logits vs mmap: "
                    + "\(scale > 0 ? delta / scale : delta) relative, worst element"
            )
        }
        // What IS required: the arena arm is not degenerate (it read real expert
        // bytes through its slots), and its mixture contributed.
        let spread = (resident.logits.max() ?? 0) - (resident.logits.min() ?? 0)
        XCTAssertGreaterThan(spread, 1, "the arena arm's logits collapsed")
        XCTAssertGreaterThan(resident.logits.reduce(Float(0)) { max($0, abs($1)) }, 1)
    }
}

// MARK: - What a device KV store kernel would buy

/// Phase 7's last open item is to dispatch `store_kv_cache_fp8` instead of
/// quantizing a visited row on the host, and this measures the thing it would
/// replace BEFORE it is built.
///
/// The saving is bounded from both sides. What the host does per visit is two
/// rows of `kvHeads * headDim` through the quantizer -- 2,048 floats at the
/// published width. And the row still has to reach the host either way: the
/// device attention reads the convolved K and V back to push the convolution
/// history and to append (the host graph appends at exactly that point), so a
/// store kernel could remove the QUANTIZE, not the read-back.
///
/// Model-free and cheap on purpose: 48 appends per arm at the published width,
/// which is what one position's schedule does. Opt-in behind the profile marker
/// because a printed share is machine-specific; the assertions are behavioural
/// (every row lands, the compact form is the smaller one) and the timing is the
/// diagnostic that decides whether a kernel is worth a dispatch.
final class Inkling2KvStoreCostTests: XCTestCase {

    func testWhatTheHostKvStoreCostsAPositionAtThePublishedWidth() throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: "/tmp/dynamoe_inkling2_profile"),
            "opt-in measurement; touch /tmp/dynamoe_inkling2_profile to run"
        )
        // The published geometry: 8 KV heads of 128, 48 visits a position.
        let kvWidth = 8 * 128
        let heads = 8
        let visits = 48
        let window = 1_024
        // The measured wall of a device visit with the mixture folded
        // (`testWhereADecodePositionsTimeGoesOnThePublishedCheckpoint`), which is
        // what the share is taken of.
        let measuredVisitMilliseconds = 2.46
        func row(seed: Int) -> [Float] {
            (0..<kvWidth).map { index in
                let mixed = (index &* 37 &+ seed &* 101) % 29
                let sign: Float = (index &+ seed) % 4 == 0 ? -1 : 1
                return sign * (Float(mixed) + 0.5) / 8
            }
        }
        let rows = (0..<visits).map { row(seed: $0) }

        func position(storage: Inkling2KVStorage) -> (best: Double, states: [Inkling2VisitState]) {
            var best = Double.greatestFiniteMagnitude
            var last: [Inkling2VisitState] = []
            // Best of three: the first loop pays for allocation and the last one
            // is what a turn's own loop would see.
            for _ in 0..<3 {
                let states = (0..<visits).map { _ in
                    Inkling2VisitState(
                        isFullAttention: true, window: window,
                        stateWidths: [kvWidth, kvWidth, 64, 64], historyDepth: 3,
                        storage: storage, heads: heads
                    )
                }
                let started = CFAbsoluteTimeGetCurrent()
                for (visit, state) in states.enumerated() {
                    _ = state.append(key: rows[visit], value: rows[visit])
                }
                best = min(best, (CFAbsoluteTimeGetCurrent() - started) * 1_000)
                last = states
            }
            return (best, last)
        }

        let exact = position(storage: .float32)
        let compact = position(storage: .int8)
        for state in exact.states + compact.states {
            XCTAssertEqual(state.committedPositions, 1)
            XCTAssertEqual(state.rowCount, 1)
            XCTAssertEqual(state.rowWidth, kvWidth)
        }
        let exactBytes = exact.states.reduce(0) { $0 + $1.residentBytes }
        let compactBytes = compact.states.reduce(0) { $0 + $1.residentBytes }
        XCTAssertLessThan(compactBytes, exactBytes / 2, "the compact form must hold fewer bytes")
        // The rows survive the round trip at the width the kernel reads.
        let compactRow = try XCTUnwrap(compact.states[0].quantizedKey(0))
        XCTAssertEqual(compactRow.stored.count, kvWidth)
        XCTAssertEqual(compactRow.scales.count, heads)
        XCTAssertEqual(compact.states[0].keys[0].count, kvWidth)

        let perVisit = measuredVisitMilliseconds * Double(visits)
        func share(_ milliseconds: Double) -> String {
            String(format: "%.3f%% of a %.1f ms position", 100 * milliseconds / perVisit, perVisit)
        }
        print(
            "\u{1F50E} [inkling2 kv store] 48 appends at \(kvWidth) wide: FP32 "
                + String(format: "%.3f", exact.best) + " ms (" + share(exact.best) + "), INT8 "
                + String(format: "%.3f", compact.best) + " ms (" + share(compact.best) + "), "
                + "\(exactBytes) B vs \(compactBytes) B resident"
        )
        print(
            "\u{1F50E} [inkling2 kv store] a device store kernel could remove the quantize "
                + "and not the read-back: the convolved rows reach the host either way for "
                + "the convolution history"
        )
    }
}

