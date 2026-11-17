//
//  TurboModeSettings.swift
//  DynaMoE
//
//  One owner for the Agent Turbo Mode preference.
//
//  The key used to be spelled as a bare string at every read and write, so a
//  flip left no trace: the only evidence that Turbo had turned itself off was
//  the composer's "Safe Mode" label, and "which call site did that" had to be
//  answered by reading the whole target. Every programmatic read and write now
//  goes through here and lands in the unified log, so
//  `log stream --predicate 'subsystem == "DRP.DynaMoE"'` names the writer.
//
//  Nothing in this type decides policy. The approval rules live on
//  `AgentHarness` (`requiresApproval` / `turboModeMayBypassApproval`); this is
//  storage plus attribution.
//

import Foundation

nonisolated enum TurboModeSettings {
    /// The `@AppStorage` wrappers in the UI bind to this same string. It must
    /// stay byte-identical to what a shipped build wrote, or an upgrade reads
    /// the user's setting as absent and silently drops them to Safe Mode.
    static let key = "dynamoe_agent_turbo_mode"

    /// The current value. An absent key reads `false` (Safe Mode), which is the
    /// documented default: Turbo is opt-in.
    static var isEnabled: Bool {
        let enabled = UserDefaults.standard.bool(forKey: key)
        DiagLog.debug("settings.turbo", "read -> \(enabled)")
        return enabled
    }

    /// Writes the switch and records the transition.
    ///
    /// `callsite` is captured by the default argument, so the log line names the
    /// function that changed Turbo rather than this helper.
    static func set(_ enabled: Bool, from callsite: StaticString = #function) {
        let previous = UserDefaults.standard.object(forKey: key)
        UserDefaults.standard.set(enabled, forKey: key)
        DiagLog.info("settings.turbo", "write \(describe(previous)) -> \(enabled) from \(callsite)")
    }

    /// Puts back a value captured with `rawValue`. `nil` removes the key, which
    /// is the honest restore when the operator had never set it: `RemoveObject`
    /// and "set false" look the same to `bool(forKey:)` but only one of them
    /// leaves the defaults domain as it found it.
    static func restore(_ previous: Any?, from callsite: StaticString = #function) {
        if let previous {
            UserDefaults.standard.set(previous, forKey: key)
        } else {
            UserDefaults.standard.removeObject(forKey: key)
        }
        DiagLog.info("settings.turbo", "restore -> \(describe(previous)) from \(callsite)")
    }

    /// The raw stored value, for a caller that intends to restore it.
    static var rawValue: Any? {
        UserDefaults.standard.object(forKey: key)
    }

    /// Records a change the UI made through `@AppStorage`. Those writes cannot
    /// route through `set(_:)` without dropping SwiftUI's observation, so the
    /// views report the transition themselves and the log stays complete.
    static func logObservedChange(from previous: Bool, to current: Bool) {
        guard previous != current else { return }
        DiagLog.info("settings.turbo", "observed \(previous) -> \(current)")
    }

    private static func describe(_ value: Any?) -> String {
        guard let value else { return "nil" }
        return "\(value)"
    }
}
