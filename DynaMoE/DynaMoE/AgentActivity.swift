//
//  AgentActivity.swift
//  DynaMoE
//
//  Phase 7: Model Context Protocol client.
//  "Is an agent operation running right now?" for subsystems that must not
//  interrupt one, such as the MCP config reload.
//
//  `CompletionEngine.isActive` only covers decoding and prefill. An agent turn
//  spends the rest of its life outside the engine: waiting for a human
//  approval, or running a tool call that can take minutes. A config reload
//  during that window closes the server's transport and aborts the call, so
//  those phases count here too.
//

import Foundation

/// Refcounted activity for the parts of an agent operation the engine does not
/// own. Pairs with `CompletionEngine.isActive`, which covers the other half.
public nonisolated final class AgentActivity: @unchecked Sendable {
    public static let shared = AgentActivity()

    public enum Kind: String, Sendable, CaseIterable {
        /// A human-in-the-loop decision is pending.
        case approval
        /// A tool call has been dispatched and has not returned.
        case toolCall
    }

    private let lock = NSLock()
    private var counts: [Kind: Int] = [:]

    private init() {}

    /// True while any counted operation is in flight.
    public var isBusy: Bool {
        lock.lock()
        defer { lock.unlock() }
        return counts.values.contains { $0 > 0 }
    }

    public func begin(_ kind: Kind) {
        lock.lock()
        counts[kind, default: 0] += 1
        lock.unlock()
    }

    public func end(_ kind: Kind) {
        lock.lock()
        let remaining = (counts[kind] ?? 0) - 1
        if remaining > 0 {
            counts[kind] = remaining
        } else {
            counts[kind] = nil
        }
        lock.unlock()
    }

    /// Current counts, for tests and diagnostics.
    public var snapshot: [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return counts.reduce(into: [:]) { $0[$1.key.rawValue] = $1.value }
    }
}
