//
//  SubagentNotifier.swift
//  DynaMoE
//
//  A finish notice for a delegated run. See SUBAGENTS.md §11.
//
//  A subagent works while the user reads something else, so a run that ends is
//  only useful if it can be noticed without watching the inspector. The Dock
//  bounce is unconditional (it is not a permission and does not survive the
//  moment); the banner is opt-out and only requested when the operator has left
//  the app.
//

import AppKit
import Foundation
import UserNotifications

@MainActor
enum SubagentNotifier {
    static let enabledKey = "dynamoe_subagent_notifications"

    private static var requestedAuthorization = false
    private static var isAuthorized = false

    static var isEnabled: Bool {
        get {
            let stored = UserDefaults.standard.object(forKey: enabledKey) as? Bool
            return stored ?? true
        }
        set { UserDefaults.standard.set(newValue, forKey: enabledKey) }
    }

    /// Called once per finished run. The summary is the same text the
    /// coordinator receives, so a notice never claims more than the report.
    static func runFinished(
        runId: UUID,
        role: String,
        status: SubagentStatus,
        summary: String,
        parentSessionId: UUID?
    ) {
        guard status == .completed || status == .failed else { return }

        if !NSApp.isActive {
            NSApp.requestUserAttention(.informationalRequest)
        }

        guard isEnabled, !isRunningTests, Bundle.main.bundleIdentifier != nil else { return }

        let body = summary
            .replacingOccurrences(of: "\n", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { return }
        let content = UNMutableNotificationContent()
        content.title = status == .completed
            ? "\(role) finished"
            : "\(role) failed"
        content.body = String(body.prefix(180))
        content.sound = .default
        var info: [String: String] = ["runId": runId.uuidString]
        if let parentSessionId { info["sessionId"] = parentSessionId.uuidString }
        content.userInfo = info

        let request = UNNotificationRequest(
            identifier: "dynamoe.subagent.\(runId.uuidString)",
            content: content,
            trigger: nil
        )
        requestAuthorizationIfNeeded { granted in
            guard granted else { return }
            UNUserNotificationCenter.current().add(request) { error in
                if let error {
                    print("⚠️ [SubagentNotifier] could not post \(role): \(error.localizedDescription)")
                }
            }
        }
    }

    /// Delivering into a test run would pop a system permission dialog under a
    /// headless xcodebuild process and fail the suite for a reason unrelated to
    /// the code under test.
    private static var isRunningTests: Bool {
        ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil
            || ProcessInfo.processInfo.environment["XCTestBundlePath"] != nil
    }

    private static func requestAuthorizationIfNeeded(_ completion: @escaping (Bool) -> Void) {
        if requestedAuthorization {
            completion(isAuthorized)
            return
        }
        requestedAuthorization = true
        // Every static above is main-actor state, so the answer comes back to
        // the main actor before it is recorded. The completion itself runs on
        // whichever queue the notification framework used; posting a request
        // does not care, but reading `isAuthorized` does.
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            DispatchQueue.main.async {
                switch settings.authorizationStatus {
                case .authorized, .provisional, .ephemeral:
                    isAuthorized = true
                    completion(true)
                case .denied:
                    isAuthorized = false
                    completion(false)
                case .notDetermined:
                    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
                        DispatchQueue.main.async {
                            isAuthorized = granted
                            completion(granted)
                        }
                    }
                @unknown default:
                    isAuthorized = false
                    completion(false)
                }
            }
        }
    }
}
