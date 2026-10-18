//
//  DiagLog.swift
//  DynaMoE
//
//  Minimal structured unified-log facade over `os.Logger`. Messages are
//  composed eagerly as plain Strings (OSLog's interpolation closure cannot
//  capture) and emitted `.public`, so call sites must pass metadata only —
//  never message content, credentials, or authorization headers. Modeled on
//  the RemoteAgentRunLog conventions: always-on info/error levels, one
//  category per subsystem so Console.app can filter without debug flags.
//
//  Engine code can migrate its ad-hoc `print()` diagnostics onto this facade
//  incrementally; nothing here is MainActor-bound.
//

import Foundation
import os

enum DiagLog {
    /// Matches PRODUCT_BUNDLE_IDENTIFIER so Console.app queries join with
    /// the app's own log entries.
    static let subsystem = "DRP.DynaMoE"

    /// `Logger` handles are cached by the system per (subsystem, category), so
    /// per-call construction is cheap and avoids shared mutable state.
    nonisolated private static func logger(category: String) -> Logger {
        Logger(subsystem: subsystem, category: category)
    }

    nonisolated static func info(_ category: String, _ message: String) {
        logger(category: category).info("\(message, privacy: .public)")
    }

    nonisolated static func error(_ category: String, _ message: String) {
        logger(category: category).error("\(message, privacy: .public)")
    }

    nonisolated static func warning(_ category: String, _ message: String) {
        logger(category: category).warning("\(message, privacy: .public)")
    }

    /// High-volume trails (heartbeat probes and the like); visible in
    /// Console.app only when the debug/include-debug level is enabled.
    nonisolated static func debug(_ category: String, _ message: String) {
        logger(category: category).debug("\(message, privacy: .public)")
    }
}
