//
//  DeepSeekV41SlotWiring.swift
//  DynaMoE
//
//  Keeps the expert arena's slot pages out of the machine's compressor.
//
//  `#55` measured the same warm expert bytes writing into resident slots at 90-97
//  GB/s and into fresh ones at 0.8-3.4 GB/s on a machine with no free memory, with
//  the process's and the machine's compressor counters moving in between. `#56`
//  made the compressor the variable by wiring destinations with `mlock`, which a
//  compressed page can never be: a wired destination read 79-100 GB/s with ZERO
//  decompressions at every phase of the process's life, while an unwired resident
//  one read 9-17 under pressure with ~1 decompression per 16 KB page. Wiring the
//  arena took a 128-position prefill of the published checkpoint from 330-375 to
//  171-174 ms/position and put the expert fill back on the disk's own ceiling.
//
//  `#56` also found the shape of the policy that this file implements: a FRACTION
//  of the arena wired is worse than none of it (8 GB wired measured 412-427
//  ms/position against 324-371 unwired), because a pass writes every slot and every
//  still-compressible slot then costs a decompression. So this policy is
//  all-or-nothing. It is granted a budget (the arena's own bytes, from the memory
//  plan), it wires each slot as it is allocated, and the first refusal takes the
//  whole policy back off rather than leaving an arena that is half of each regime.
//
//  A wired page can be neither evicted nor compressed, so the budget has to leave
//  the rest of the machine somewhere to work. The memory plan prices that, because
//  it is the same arithmetic that sized the arena in the first place.
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

// MARK: - Policy

/// What the arena may wire, and what wiring it actually did.
///
/// The policy is a value the streamer owns under its own lock; every method here is
/// called from the streamer's slot allocation or from its diagnostic accessor.
nonisolated public struct DeepSeekV41SlotWiring: Sendable {

    /// What the policy has done so far.
    ///
    /// A snapshot rather than a live view, so a caller prints one line without the
    /// streamer's lock and a test can compare two moments.
    public struct Status: Sendable, Equatable {
        /// True once a budget was granted. False before that, and after a refusal.
        public var isEnabled: Bool = false
        /// True when a slot refused to be wired and the policy withdrew.
        ///
        /// Latched: `#56` measured a partly wired arena as worse than none, so a
        /// policy that could not have all of it stays off for the rest of the run
        /// rather than retrying per slot.
        public var isRefused: Bool = false
        /// The highest budget ever granted, in bytes. A later grant can only raise it.
        public var budgetBytes: Int = 0
        /// Bytes currently wired: slots already counted, nothing double-counted.
        public var bytes: Int = 0
        /// Slots currently wired.
        public var slots: Int = 0
        /// How many `mlock` calls the kernel refused.
        public var failures: Int = 0

        public init() {}
    }

    /// One wired span: a slot's `contents()` and the length that begins there.
    ///
    /// The pointer is never dereferenced here; it is handed straight back to the
    /// page primitives that locked it, which is why the policy can be a value that
    /// carries it without becoming a reference to the slot.
    private struct Region: @unchecked Sendable {
        let pointer: UnsafeMutableRawPointer
        let length: Int
    }

    private var state = Status()
    private var regions: [Region] = []
    private let lockPages: @Sendable (UnsafeMutableRawPointer, Int) -> Bool
    private let unlockPages: @Sendable (UnsafeMutableRawPointer, Int) -> Bool

    /// The kernel's own page lock, which is what the policy exists to use.
    public static let system = DeepSeekV41SlotWiring(
        lockPages: { pointer, length in mlock(pointer, length) == 0 },
        unlockPages: { pointer, length in munlock(pointer, length) == 0 }
    )

    /// Builds a policy over a given pair of page primitives.
    ///
    /// The seam exists so the refusal path is testable at all: `mlock` refuses when
    /// the kernel cannot account the range, which a test cannot ask for on purpose
    /// without first exhausting the machine.
    public init(
        lockPages: @escaping @Sendable (UnsafeMutableRawPointer, Int) -> Bool,
        unlockPages: @escaping @Sendable (UnsafeMutableRawPointer, Int) -> Bool
    ) {
        self.lockPages = lockPages
        self.unlockPages = unlockPages
    }

    public var status: Status { state }

    /// Grants a budget, up to the highest one ever granted.
    ///
    /// A budget of zero or less is a no-op rather than an error: the caller derived
    /// "this machine should not wire its arena", and the policy stays off. A grant
    /// after a refusal is ignored, because the refusal already handed back the
    /// slots and the regime it would return to is the one `#56` measured as worse
    /// than being unwired.
    public mutating func enable(budgetBytes: Int) {
        guard budgetBytes > 0, !state.isRefused else { return }
        state.budgetBytes = max(state.budgetBytes, budgetBytes)
        state.isEnabled = true
    }

    /// Wires one slot's pages.
    ///
    /// Returns true when the pages are wired, including when this exact slot was
    /// wired by an earlier call. A refused `mlock` withdraws the WHOLE policy: the
    /// slots already wired are handed back, so the arena is either entirely wired
    /// or entirely not. The slot's memory is still valid either way; the fill simply
    /// runs in the unwired regime.
    @discardableResult
    public mutating func wire(_ pointer: UnsafeMutableRawPointer, length: Int) -> Bool {
        guard state.isEnabled, !state.isRefused, length > 0 else { return false }
        if regions.contains(where: { $0.pointer == pointer }) { return true }
        guard state.bytes + length <= state.budgetBytes else { return false }
        guard lockPages(pointer, length) else {
            state.failures += 1
            withdraw()
            return false
        }
        state.bytes += length
        state.slots += 1
        regions.append(Region(pointer: pointer, length: length))
        return true
    }

    /// Hands every wired page back and turns the policy off.
    ///
    /// Used when the owning streamer is closed: the pages are about to be freed
    /// with it, and until they are the machine should not have to treat them as
    /// pinned. It is also what a test observes to prove a withdrawal released
    /// rather than merely forgot.
    public mutating func disable() {
        for region in regions { _ = unlockPages(region.pointer, region.length) }
        regions.removeAll(keepingCapacity: false)
        state.bytes = 0
        state.slots = 0
        state.isEnabled = false
    }

    private mutating func withdraw() {
        for region in regions { _ = unlockPages(region.pointer, region.length) }
        regions.removeAll(keepingCapacity: false)
        state.bytes = 0
        state.slots = 0
        state.isEnabled = false
        state.isRefused = true
    }
}
