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

    /// The stage list is the visit's own structure, and the three stages that
    /// commit nothing are marked as such: a caller summing `commits` over all
    /// nine must get thirteen per visit, not nine.
    func testTheStageTableSaysWhichStagesCommit() {
        let committing = Inkling2VisitStage.allCases.filter(\.commits)
        XCTAssertEqual(
            Set(committing),
            [.attentionFront, .attentionTail, .postAttentionConvolution,
             .mlpFront, .expertMixture, .mlpTail]
        )
        XCTAssertEqual(Inkling2VisitStage.allCases.count, 9)
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
    }

    /// The engine's own arrangement: the executor's `host` IS the forward the
    /// turn runs on, because `runVisit` reads and appends that graph's states.
    /// A harness built with two forwards would run a valid position against the
    /// wrong sequence, and only a timing number would come out.
    private func harness(stagedPositions: Int = 8) throws -> Harness {
        let directory = try snapshot()
        let engine = try DynaMoeEngine(filePath: directory.path)
        self.engine = engine
        let summary = try engine.getSummary()
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
        for shard in summary.shards {
            guard shard.length > 0,
                  let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
            guard shard.length <= UInt64(device.maxBufferLength) else { continue }
            if let buffer = device.makeBuffer(
                bytesNoCopy: pointer, length: Int(shard.length),
                options: .storageModeShared, deallocator: nil
            ) { buffers[shard.index] = buffer }
        }
        let source = Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: buffers
        )
        let forward = try Inkling2CPUForward(
            config: config, weights: source,
            options: Inkling2ForwardOptions(maxPositions: stagedPositions)
        )
        let deviceWeights = Inkling2DeviceWeights(
            config: config, catalog: catalog, shardBuffers: buffers
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
        return Harness(config: config, forward: forward, metal: metal, workspace: workspace)
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
        // One expert buffer per routed expert: thirteen commits a visit, and the
        // expert stage is the one that is `topK` of them.
        XCTAssertEqual(
            profile.timing(.expertMixture).commits,
            profile.visits * harness.config.numExpertsPerToken
        )
        XCTAssertEqual(profile.experts, profile.visits * harness.config.numExpertsPerToken)
        XCTAssertGreaterThan(profile.expertBytes, 0)

        // The stage walls tile the visit. The residue is the function-call seams
        // between stages, so it is allowed to be small but never large, and the
        // sum can never exceed the visit it is cut out of.
        XCTAssertLessThanOrEqual(profile.billedWallNanoseconds, profile.wallNanoseconds)
        XCTAssertGreaterThanOrEqual(
            profile.billedWallNanoseconds * 100, profile.wallNanoseconds * 95
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
        profile.reset()
        visits = 0
        let positions = 4
        let start = CFAbsoluteTimeGetCurrent()
        for position in 1...positions {
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
}
