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

    /// Requests queued while authorization is being resolved. A second run
    /// finishing in that window used to be dropped because the cached answer
    /// was still `false`.
    private static var pendingRequests: [UNNotificationRequest] = []
    private static var isRequestingAuthorization = false
    private static let maximumQueuedNotices = 8

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
        pendingRequests.append(request)
        if pendingRequests.count > maximumQueuedNotices {
            pendingRequests.removeFirst(pendingRequests.count - maximumQueuedNotices)
        }
        deliverPendingNotifications()
    }

    /// Reads the CURRENT authorization on every notice, then posts or queues.
    /// The old code cached the first answer forever, so a permission granted in
    /// System Settings afterwards could never take effect, and a second finish
    /// during the first check saw `false` and vanished.
    private static func deliverPendingNotifications() {
        guard !isRequestingAuthorization, !pendingRequests.isEmpty else { return }
        UNUserNotificationCenter.current().getNotificationSettings { settings in
            DispatchQueue.main.async {
                switch settings.authorizationStatus {
                case .authorized, .provisional, .ephemeral:
                    postPendingNotifications()
                case .denied:
                    pendingRequests.removeAll()
                case .notDetermined:
                    guard !isRequestingAuthorization else { return }
                    isRequestingAuthorization = true
                    UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in
                        DispatchQueue.main.async {
                            isRequestingAuthorization = false
                            deliverPendingNotifications()
                        }
                    }
                @unknown default:
                    pendingRequests.removeAll()
                }
            }
        }
    }

    private static func postPendingNotifications() {
        let requests = pendingRequests
        pendingRequests.removeAll()
        for request in requests {
            UNUserNotificationCenter.current().add(request) { error in
                if let error {
                    print("⚠️ [SubagentNotifier] could not post: \(error.localizedDescription)")
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

}

/// Makes a banner click mean something. Without a delegate, macOS delivers the
/// click to nobody and the run the operator tapped stays unopened in a window
/// they were not looking at.
@MainActor
final class SubagentNotificationDelegate: NSObject, UNUserNotificationCenterDelegate {
    static let shared = SubagentNotificationDelegate()

    private override init() {
        super.init()
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler:
            @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .sound])
    }

    nonisolated func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        if let raw = response.notification.request.content.userInfo["runId"] as? String,
           let runId = UUID(uuidString: raw) {
            Task { @MainActor in
                NSApp.activate(ignoringOtherApps: true)
                SubagentManager.shared.selectedSubagentId = runId
                SubagentManager.shared.isDrawerOpen = true
            }
        }
        completionHandler()
    }
}
