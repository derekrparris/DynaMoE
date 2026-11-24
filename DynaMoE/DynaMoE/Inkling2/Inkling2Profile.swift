//
//  Inkling2Profile.swift
//  DynaMoE
//
//  Phase 6 instrumentation: where one Inkling 2 visit's time goes.
//
//  The previous slice's measurement is the reason this file exists. A decoded
//  position at position 0 costs about 0.296 s and the output head is 0.9% of
//  it, so the position IS its 48 visits at about 6.2 ms each -- and one visit
//  is thirteen command buffers, each committed and then waited on: the
//  attention front, the attention tail, the post-attention convolution, the MLP
//  norm plus router, one buffer per routed expert, and the MLP tail. A single
//  per-visit total would leave a reader exactly where that slice left them.
//
//  So a stage is billed four ways, and the four are chosen because they are the
//  four different fixes:
//
//  - `encode` is host time building the command buffer. Encoder overhead, the
//    arena's scalar buffers and the dispatch setup live here, and only fewer
//    dispatches or a cheaper encode removes it.
//  - `queueWait` is how long the buffer sat committed before the GPU began it.
//    On a single in-order queue behind a previous wait this is the previous
//    stage's tail; a large value here is a scheduling question.
//  - `device` is the buffer's own `gpuStartTime`...`gpuEndTime` window, which
//    is the only honest answer to "how much arithmetic was this".
//  - `host` is what is left of the stage's wall: read-backs, host matrix
//    lookups, the KV staging copy, the relative-state and sink projections.
//    For `attentionState` that residue is most of the stage, and it is the one
//    bucket a kernel change cannot touch.
//
//  Off by default and off in every production path: a nil profile makes every
//  call site a branch that is never taken. `dynamoe_inkling2_profile` is the
//  seam a developer turns it on with, and it is read once per turn.
//

import Foundation
import Metal

// MARK: - Stages

/// The stages one visit is cut into, in the order they run.
///
/// The cut is the visit's OWN structure, not an invented one: each case is one
/// command buffer or one host interval between two of them, so a stage's wall
/// maps to a place in `Inkling2MetalForward.runVisit` a reader can open.
nonisolated public enum Inkling2VisitStage: String, Sendable, CaseIterable {
    /// The attention reads: the `attn_res` block reads, the four message gates
    /// and the three input norms. Host only, no command buffer.
    case hostPrep
    /// The Q/K/V projections, the two K/V convolutions and the two L2 clips,
    /// in one buffer.
    case attentionFront
    /// Between the two attention buffers: the K/V read-back and cache append,
    /// the relative-state and sink projections, and the staging copy of every
    /// KV row this position attends over. Host only, and the copy grows with
    /// the context.
    case attentionState
    /// The attention kernel, the output gate, the grouped output projection and
    /// -- since slice 5 -- the post-attention convolution, which shares this
    /// buffer rather than paying its own commit.
    case attentionTail
    /// The MLP block read, its message gate and the residual bookkeeping.
    case mlpRead
    /// The MLP norm, the router and its top-k selection, in one buffer.
    case mlpFront
    /// Every routed expert, one command buffer each: gate/up, the softcapped
    /// SwiGLU, the down projection and the weighted fold.
    case expertMixture
    /// The post-MLP convolution and the residual add.
    case mlpTail

    /// The stages that commit at least one command buffer.
    public var commits: Bool {
        switch self {
        case .hostPrep, .attentionState, .mlpRead: return false
        case .attentionFront, .attentionTail,
             .mlpFront, .expertMixture, .mlpTail: return true
        }
    }
}

// MARK: - What one buffer reports about itself

/// The clocks one committed command buffer reports about itself.
///
/// Taken only after `waitUntilCompleted`, because `gpuStartTime`/`gpuEndTime`
/// are documented as meaningful then. A window that came back empty is not
/// silently dropped: it counts as one empty window so a caller can tell a
/// device total that is a floor from one that is a measurement.
nonisolated public struct Inkling2BufferClock: Sendable, Equatable {
    /// Host time between the stage's encode stamp and the commit.
    public var encodeNanoseconds: UInt64
    /// Time the committed buffer waited before the GPU began it.
    public var queueWaitNanoseconds: UInt64
    /// The buffer's own device window.
    public var deviceNanoseconds: UInt64
    /// Whether the window was usable.
    public var windowed: Bool

    public init(
        encodeNanoseconds: UInt64,
        queueWaitNanoseconds: UInt64,
        deviceNanoseconds: UInt64,
        windowed: Bool
    ) {
        self.encodeNanoseconds = encodeNanoseconds
        self.queueWaitNanoseconds = queueWaitNanoseconds
        self.deviceNanoseconds = deviceNanoseconds
        self.windowed = windowed
    }

    /// - Parameters:
    ///   - encodeStart: when this stage's host encode began.
    ///   - commit: when the buffer was committed, from the same clock.
    ///   - buffer: the buffer, after `waitUntilCompleted`.
    public init(encodeStart: UInt64, commit: UInt64, buffer: MTLCommandBuffer) {
        encodeNanoseconds = commit >= encodeStart ? commit - encodeStart : 0
        let start = buffer.gpuStartTime
        let end = buffer.gpuEndTime
        windowed = end > start && start > 0
        deviceNanoseconds = windowed ? UInt64((end - start) * 1_000_000_000) : 0
        // `gpuStartTime` is host time in seconds since boot, the same clock the
        // stamps above come from, so the two are subtractable.
        let startedSeconds = max(0, start)
        let committedSeconds = Double(commit) / 1_000_000_000
        queueWaitNanoseconds = windowed && startedSeconds > committedSeconds
            ? UInt64((startedSeconds - committedSeconds) * 1_000_000_000)
            : 0
    }
}

// MARK: - One stage's totals

/// One stage's time over however many visits and buffers it was measured on.
nonisolated public struct Inkling2StageTiming: Sendable, Equatable {
    /// The stage's whole interval, both halves included.
    public var wallNanoseconds: UInt64 = 0
    public var encodeNanoseconds: UInt64 = 0
    /// Host time blocked in `waitUntilCompleted`.
    public var drainNanoseconds: UInt64 = 0
    /// The part of `drainNanoseconds` the queue spent before starting the work.
    public var queueWaitNanoseconds: UInt64 = 0
    /// Summed device windows.
    public var deviceNanoseconds: UInt64 = 0
    public var commits: Int = 0
    /// Buffers whose device window came back empty, so `deviceNanoseconds` is a
    /// floor for this stage.
    public var emptyWindows: Int = 0

    public init() {}

    /// What is left of the wall once encode and drain are billed: read-backs,
    /// host projections, staging copies, matrix lookups.
    public var hostNanoseconds: UInt64 {
        let billed = encodeNanoseconds + drainNanoseconds
        return wallNanoseconds > billed ? wallNanoseconds - billed : 0
    }

    /// Bills one buffer of this stage.
    public mutating func add(_ clock: Inkling2BufferClock, drainNanoseconds drain: UInt64) {
        encodeNanoseconds += clock.encodeNanoseconds
        queueWaitNanoseconds += clock.queueWaitNanoseconds
        deviceNanoseconds += clock.deviceNanoseconds
        drainNanoseconds += drain
        commits += 1
        if !clock.windowed { emptyWindows += 1 }
    }
}

// MARK: - The recorder

/// Accumulates a stream of visits into one per-stage table.
///
/// A class rather than a value because the visit's stage helpers all need to
/// bill the SAME table while `runVisit`'s locals stay untouched, and threading
/// an `inout` through three functions that each own a `defer` is worse than one
/// reference. It is confined to the thread walking the sequence -- the engine is
/// single-flight -- so the unchecked conformance is the same one the executor's
/// other mutable state carries.
nonisolated public final class Inkling2VisitProfile: @unchecked Sendable {
    public private(set) var stages: [Inkling2VisitStage: Inkling2StageTiming] = [:]
    /// Visits closed, which is what a per-visit figure divides by.
    public private(set) var visits: Int = 0
    /// The visits' own wall time, from `runVisit`'s first statement to its
    /// return. It is recorded separately from the stage walls because a stage
    /// that forgot to bill itself would otherwise vanish; the difference is
    /// what `Inkling2VisitProfileTests` checks.
    public private(set) var wallNanoseconds: UInt64 = 0
    /// Routed experts the visits ran, summed.
    public private(set) var experts: Int = 0
    /// Bytes those experts' banks occupy, from the published strides. This
    /// family has no expert arena yet, so it is the WHOLE read volume a
    /// residency policy would have to move -- not a measured fault count.
    public private(set) var expertBytes: UInt64 = 0
    /// `attn_res` queries the visits asked for, summed.
    public private(set) var attentionQueries: Int = 0

    public init() {}

    public func reset() {
        stages.removeAll(keepingCapacity: true)
        visits = 0
        wallNanoseconds = 0
        experts = 0
        expertBytes = 0
        attentionQueries = 0
    }

    /// Bills one stage of one visit.
    public func record(
        stage: Inkling2VisitStage, wallNanoseconds wall: UInt64, buffers: Inkling2StageTiming
    ) {
        var total = stages[stage] ?? Inkling2StageTiming()
        total.wallNanoseconds += wall
        total.encodeNanoseconds += buffers.encodeNanoseconds
        total.drainNanoseconds += buffers.drainNanoseconds
        total.queueWaitNanoseconds += buffers.queueWaitNanoseconds
        total.deviceNanoseconds += buffers.deviceNanoseconds
        total.commits += buffers.commits
        total.emptyWindows += buffers.emptyWindows
        stages[stage] = total
    }

    /// Closes one visit.
    public func closeVisit(wallNanoseconds wall: UInt64, attentionQueries queries: Int) {
        visits += 1
        wallNanoseconds += wall
        attentionQueries += queries
    }

    /// Bills the routed experts one visit ran, and the bytes their banks occupy.
    public func recordExperts(count: Int, bytes: UInt64) {
        experts += count
        expertBytes += bytes
    }

    public func timing(_ stage: Inkling2VisitStage) -> Inkling2StageTiming {
        stages[stage] ?? Inkling2StageTiming()
    }

    /// The stage walls summed, which should be the visit wall less the
    /// function-call seams between stages.
    public var billedWallNanoseconds: UInt64 {
        stages.values.reduce(0) { $0 + $1.wallNanoseconds }
    }

    /// One line, because a per-stage table in a log line is unreadable and a
    /// per-stage line would be nine of them.
    ///
    /// Every figure is per visit. `device` and `queue` are the sums over the
    /// visit's thirteen buffers, so `device` can exceed a stage's wall only if
    /// two of its buffers overlapped on the queue -- which they do not, and
    /// that is what the test asserts.
    public func describe(label: String = "turn") -> String {
        func rate(_ nanoseconds: UInt64) -> String {
            guard visits > 0 else { return "0.00" }
            return String(format: "%.2f", Double(nanoseconds) / Double(visits) / 1_000_000)
        }
        var parts: [String] = []
        for stage in Inkling2VisitStage.allCases {
            guard let time = stages[stage] else { continue }
            parts.append("\(stage.rawValue) \(rate(time.wallNanoseconds))")
        }
        let device = stages.values.reduce(0) { $0 + $1.deviceNanoseconds }
        let queueWait = stages.values.reduce(0) { $0 + $1.queueWaitNanoseconds }
        let drain = stages.values.reduce(0) { $0 + $1.drainNanoseconds }
        let commits = stages.values.reduce(0) { $0 + $1.commits }
        let empty = stages.values.reduce(0) { $0 + $1.emptyWindows }
        var line = "inkling2 profile [\(label)]: \(visits) visits at \(rate(wallNanoseconds)) ms"
        line += " (\(commits) commit(s), \(experts) experts, "
        line += String(format: "%.1f", Double(expertBytes) / 1_048_576) + " MB of expert banks)"
        line += "; " + parts.joined(separator: " | ")
        line += "; drain \(rate(drain)) of which queue \(rate(queueWait)), device \(rate(device))"
        if empty > 0 { line += ", \(empty) buffer(s) with no device window" }
        return line
    }
}
