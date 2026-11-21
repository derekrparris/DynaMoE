//
//  SubagentRelayWait.swift
//  DynaMoE
//
//  The teardown wait that lets in-flight delegated runs land their reports
//  before a coordinator turn ends. See SUBAGENTS.md §5 and §11.
//

import Foundation

/// Waits, bounded, for a conversation's delegated runs to reach a terminal
/// state.
///
/// This is a Stop-sensitive wait: the caller is a generation task that the
/// operator can cancel at any moment, and a wait that ignored cancellation
/// kept a stopped turn alive for up to its whole timeout. For a wait of two
/// minutes that is not a nuance: the relay that follows it generated into the
/// conversation the operator had stopped.
///
/// The predicate is injected because the manager it reads is main-actor state
/// and a test drives the wait without spawning a real run.
nonisolated enum SubagentRelayWait {
    static func wait(
        sessionId: UUID,
        timeout: TimeInterval,
        isActive: @escaping @Sendable @MainActor (UUID) -> Bool
    ) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if Task.isCancelled { return }
            let stillRunning = await isActive(sessionId)
            if !stillRunning { return }
            try? await Task.sleep(nanoseconds: 250_000_000)
        }
    }
}
