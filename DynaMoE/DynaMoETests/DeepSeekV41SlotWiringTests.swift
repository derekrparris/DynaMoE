//
//  DeepSeekV41SlotWiringTests.swift
//  DynaMoE
//
//  The arena's residency policy (`#57`) and the budget it is granted.
//
//  `#56` measured the win and the hazard: a wired destination reads 79-100 GB/s
//  with zero decompressions at every pressure level, and wiring a FRACTION of the
//  arena is worse than wiring none. So the policy has three properties worth
//  pinning without a checkpoint: a slot is wired once, a budget is never exceeded,
//  and a refusal hands back everything already wired.
//
//  The page primitives are injected so the refusal path can be exercised at all:
//  `mlock` refuses when the kernel cannot account a range, which a test cannot ask
//  for on purpose. One case still drives the real `mlock`/`munlock` pair, because
//  the injected form proves the state machine and not the syscall.
//

import XCTest
@testable import DynaMoE

// MARK: - Policy

final class DeepSeekV41SlotWiringTests: XCTestCase {

    private let page = max(4096, Int(vm_kernel_page_size))

    /// Stands in for the kernel's page lock: records every call, and can refuse the
    /// nth one so a withdrawal has a moment to happen.
    private final class PageLocker: @unchecked Sendable {
        var locked: [UnsafeMutableRawPointer] = []
        var unlocked: [UnsafeMutableRawPointer] = []
        var refusals = 0
        /// 1-based ordinal of the `lock` call to refuse, counted across all slots.
        var refuseAt: Int?

        func lock(_ pointer: UnsafeMutableRawPointer, _ length: Int) -> Bool {
            if let refuseAt, locked.count + refusals + 1 == refuseAt {
                refusals += 1
                return false
            }
            locked.append(pointer)
            return true
        }

        func unlock(_ pointer: UnsafeMutableRawPointer, _ length: Int) -> Bool {
            unlocked.append(pointer)
            return true
        }
    }

    /// Four page-aligned slots the policy can point at without a Metal device.
    private func withPages<T>(
        _ body: (UnsafeMutableRawPointer, [Int]) throws -> T
    ) rethrows -> T {
        let base = UnsafeMutableRawPointer.allocate(
            byteCount: 4 * page, alignment: page
        )
        defer { base.deallocate() }
        return try body(base, (0..<4).map { $0 * page })
    }

    func testEachSlotIsWiredOnceAndTheBudgetIsNeverExceeded() {
        withPages { base, offsets in
            let locker = PageLocker()
            var policy = DeepSeekV41SlotWiring(
                lockPages: locker.lock, unlockPages: locker.unlock
            )
            policy.enable(budgetBytes: 2 * page)
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[0]), length: page))
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[0]), length: page))
            XCTAssertEqual(policy.status.bytes, page, "a slot already wired is not wired twice")
            XCTAssertEqual(policy.status.slots, 1)
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[1]), length: page))
            XCTAssertEqual(policy.status.bytes, 2 * page)
            // A third slot is outside the grant. It is skipped, not refused: the
            // policy still covers the arena it was priced for.
            XCTAssertFalse(policy.wire(base.advanced(by: offsets[2]), length: page))
            XCTAssertEqual(policy.status.bytes, 2 * page)
            XCTAssertTrue(policy.status.isEnabled)
            XCTAssertFalse(policy.status.isRefused)
            XCTAssertEqual(locker.locked.count, 2)
        }
    }

    func testARefusalHandsBackTheWholeArenaAndStaysOff() {
        withPages { base, offsets in
            let locker = PageLocker()
            locker.refuseAt = 3
            var policy = DeepSeekV41SlotWiring(
                lockPages: locker.lock, unlockPages: locker.unlock
            )
            policy.enable(budgetBytes: 4 * page)
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[0]), length: page))
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[1]), length: page))
            XCTAssertFalse(policy.wire(base.advanced(by: offsets[2]), length: page))

            // `#56`: 8 GB wired against 90 GB of destinations measured WORSE than
            // wiring none, because every still-compressible slot a pass rewrites
            // costs a decompression. So the third refusal takes the first two back.
            XCTAssertEqual(policy.status.bytes, 0)
            XCTAssertEqual(policy.status.slots, 0)
            XCTAssertFalse(policy.status.isEnabled)
            XCTAssertTrue(policy.status.isRefused)
            XCTAssertEqual(policy.status.failures, 1)
            XCTAssertEqual(
                Set(locker.unlocked.map { UInt(bitPattern: $0) }),
                Set([offsets[0], offsets[1]].map { UInt(bitPattern: base.advanced(by: $0)) }),
                "a refusal must unlock exactly what it had wired"
            )

            // A refused policy is latched: a later, larger grant does not turn it
            // back on, and the next slot is not attempted.
            policy.enable(budgetBytes: 16 * page)
            XCTAssertFalse(policy.wire(base.advanced(by: offsets[3]), length: page))
            XCTAssertEqual(policy.status.bytes, 0)
            XCTAssertEqual(locker.locked.count, 2)
        }
    }

    func testDisablingReleasesWhatItWiredWithoutBeingARefusal() {
        withPages { base, offsets in
            let locker = PageLocker()
            var policy = DeepSeekV41SlotWiring(
                lockPages: locker.lock, unlockPages: locker.unlock
            )
            policy.enable(budgetBytes: 4 * page)
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[0]), length: page))
            XCTAssertTrue(policy.wire(base.advanced(by: offsets[1]), length: page))
            policy.disable()
            XCTAssertEqual(policy.status.bytes, 0)
            XCTAssertEqual(policy.status.slots, 0)
            XCTAssertFalse(policy.status.isEnabled)
            XCTAssertFalse(policy.status.isRefused)
            XCTAssertEqual(locker.unlocked.count, 2)
            XCTAssertFalse(policy.wire(base.advanced(by: offsets[2]), length: page))
        }
    }

    func testAZeroBudgetIsOffRatherThanARefusal() {
        withPages { base, offsets in
            let locker = PageLocker()
            var policy = DeepSeekV41SlotWiring(
                lockPages: locker.lock, unlockPages: locker.unlock
            )
            policy.enable(budgetBytes: 0)
            XCTAssertFalse(policy.wire(base.advanced(by: offsets[0]), length: page))
            XCTAssertFalse(policy.status.isEnabled)
            XCTAssertFalse(policy.status.isRefused)
            XCTAssertEqual(locker.locked.count, 0)
        }
    }

    func testTheSystemPolicyWiresARealPageAndHandsItBack() {
        withPages { base, offsets in
            var policy = DeepSeekV41SlotWiring.system
            policy.enable(budgetBytes: page)
            XCTAssertTrue(
                policy.wire(base.advanced(by: offsets[0]), length: page),
                "mlock refused a page of anonymous memory; the machine has no room"
            )
            XCTAssertEqual(policy.status.bytes, page)
            XCTAssertEqual(policy.status.slots, 1)
            XCTAssertEqual(policy.status.failures, 0)
            // `munlock` reports `EINVAL` for a range it does not hold, so the call
            // has to succeed for a handout that actually released the page.
            policy.disable()
            XCTAssertEqual(policy.status.bytes, 0)
        }
    }
}

// MARK: - Budget

final class DeepSeekV41ArenaWiringBudgetTests: XCTestCase {

    /// A plan whose non-arena claims are fixed, so the budget arithmetic is the only
    /// thing that moves. Sizes are the published checkpoint's order of magnitude.
    private let plan = DeepSeekV41MemoryPlan(
        contextLength: 32_768,
        useFP4MainCache: true,
        compressedCacheBytes: 1 << 30,
        windowRingBytes: 1 << 20,
        indexerBytes: 1 << 20,
        candidateBytes: 1 << 20,
        bytesPerExpert: 18_800_640,
        residentWeightBytes: 10 << 30,
        engramStagingBytes: 1 << 20
    )

    func testAWholeArenaIsGrantedWhenTheMachineHasRoomForEverythingElse() {
        // 60 GB + ~11 GB of plan claims + the 16 GB floor is 87 GB of 128: granted,
        // and as ONE number. `#56` measured that a partial grant loses to none, so
        // there is no fraction to hand back.
        let arena = 60 << 30
        XCTAssertEqual(
            plan.wireBudgetBytes(arenaBytes: arena, physicalGB: 128),
            arena
        )
    }

    func testAnArenaThatWouldEatTheFloorIsRefusedEntirely() {
        let arena = 110 << 30
        XCTAssertEqual(plan.wireBudgetBytes(arenaBytes: arena, physicalGB: 128), 0)
        // The same arena fits once the floor is lowered, which is what proves the
        // refusal is the floor and not some other claim.
        XCTAssertEqual(
            plan.wireBudgetBytes(arenaBytes: arena, physicalGB: 128, floorGB: 0),
            arena
        )
    }

    func testNothingToWireIsNoGrant() {
        XCTAssertEqual(plan.wireBudgetBytes(arenaBytes: 0, physicalGB: 128), 0)
        XCTAssertEqual(plan.wireBudgetBytes(arenaBytes: 1 << 30, physicalGB: 0), 0)
        XCTAssertEqual(plan.wireBudgetBytes(arenaBytes: 1 << 30, physicalGB: 8), 0)
    }

    func testTheBudgetDoesNotMoveTheArenaItPrices() {
        // A wiring grant is an upper bound, not an allocation: the plan's own
        // required bytes are unchanged by asking the question.
        let before = plan.requiredBytes(expertSlots: 4_000)
        _ = plan.wireBudgetBytes(arenaBytes: 4_000 * plan.bytesPerExpert, physicalGB: 128)
        XCTAssertEqual(plan.requiredBytes(expertSlots: 4_000), before)
    }
}
