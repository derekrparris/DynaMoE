//
//  SubagentNotifier.swift
//  DynaMoE
//
//  A finish notice for a delegated run. See SUBAGENTS.md §11.
//
//  A subagent works while the user reads something else, so a run that ends is
//  only useful if it can be noticed without watching the inspector. The Dock
//  bounce is unconditional (it is not a permission and does not survive the
//  moment); the banner is opt-IN and only posted when the operator has left
//  the app. SUBAGENTS.md §11 states both.
//

import AppKit
import Foundation
import UserNotifications

@MainActor
enum SubagentNotifier {
    static let enabledKey = "dynamoe_subagent_notifications"

    /// Posted by the notification delegate when the operator taps a banner, so
    /// the window can select the CONVERSATION the run belongs to before the
    /// inspector opens: the inspector is session-scoped, so opening the drawer
    /// without switching chats landed the tap on a list the run is not in.
    static let revealRunNotification = Notification.Name("dynamoe.subagentRevealRun")

    /// Requests queued while authorization is being resolved. A second run
    /// finishing in that window used to be dropped because the cached answer
    /// was still `false`.
    private static var pendingRequests: [UNNotificationRequest] = []
    private static var isRequestingAuthorization = false
    private static let maximumQueuedNotices = 8

    static var isEnabled: Bool {
        get {
            let stored = UserDefaults.standard.object(forKey: enabledKey) as? Bool
            // Opt-in: an absent key means no banner until the operator turns
            // it on in Settings.
            return stored ?? false
        }
        set { UserDefaults.standard.set(newValue, forKey: enabledKey) }
    }

    /// Called once per finished run.
    ///
    /// The banner deliberately carries no summary. A notification is rendered
    /// on the lock screen and in Notification Centre, outside the app's own
    /// privacy boundary, while a run's report can quote source code, paths and
    /// tool output. The role and outcome are enough to decide whether to open
    /// the inspector, which is where the report belongs.
    static func runFinished(
        runId: UUID,
        role: String,
        status: SubagentStatus,
        parentSessionId: UUID?
    ) {
        // `.blocked` is a loop guard's terminal decision, not a crash: the run
        // stopped because continuing would not help. It is exactly the outcome
        // an operator away from the app needs told, and it used to end in
        // silence (no banner, no Dock bounce).
        guard status == .completed || status == .failed || status == .blocked else { return }

        // A finish notice exists for a run the operator is not watching. While
        // DynaMoE is frontmost the inspector is one click away, and a banner
        // over the app is noise the doc never promised. The Dock bounce shares
        // that inactivity gate but NOT the banner's opt-in: it is not a
        // permission and does not survive the moment, so it is the only signal
        // a run gets when notifications are off (the default). It is skipped
        // under test because an app-hosted test run has a real NSApp and would
        // bounce the Dock once per finished run.
        // `NSApp` is an implicitly-unwrapped optional and is nil in a
        // non-app-hosted XCTest process, so the test guard is evaluated FIRST:
        // reading `NSApp.isActive` before it trapped exactly where the guard
        // was meant to protect.
        guard !isRunningTests, Bundle.main.bundleIdentifier != nil else { return }
        let isInactive = !NSApp.isActive
        guard isInactive else { return }
        NSApp.requestUserAttention(.informationalRequest)

        // The banner is opt-IN, so everything below is the operator's choice.
        guard isEnabled else { return }

        let content = UNMutableNotificationContent()
        // The role is model-supplied text on a surface outside the app. A
        // newline would break the one-line title and let a report-shaped
        // string render as a title; the marker pass and the bounds keep it a
        // label.
        let label = Self.roleLabel(role)
        switch status {
        case .completed:
            content.title = "\(label) finished"
            content.body = "Open DynaMoE to read the report."
        case .blocked:
            content.title = "\(label) was blocked"
            content.body = "Open DynaMoE to see why its loop guard stopped it."
        default:
            content.title = "\(label) failed"
            content.body = "Open DynaMoE to see what went wrong."
        }
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
                    DiagLog.warning("subagent.notifier", "could not post: \(error.localizedDescription)")
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

    /// One bounded line, so a model-authored role cannot shape the banner.
    nonisolated static func roleLabel(_ role: String) -> String {
        let singleLine = role
            .replacingOccurrences(of: "\r", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
        let cleaned = AgentHarness.sanitizeText(singleLine)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty ? "Subagent run" : String(cleaned.prefix(80))
    }

    /// Withdraws a run's delivered and pending banners.
    ///
    /// A notice outlives the record it names: `clearCompleted`, a discard and a
    /// conversation delete all remove the run while its banner sits in
    /// Notification Centre, and the tap then selects a run no inspector row
    /// can show. Nonisolated because the store's delete path runs on its IO
    /// queue, and the notification centre's own API is thread-safe.
    nonisolated static func withdraw(runId: UUID) {
        // The guard comes FIRST. `UNUserNotificationCenter.current()` and its
        // removals are the same calls `post` refuses to make under a test run
        // or an unbundled build; reaching them here contradicted the rule
        // stated above and could touch the notification daemon from a
        // headless suite that never posted anything.
        guard !isRunningTestsInThisProcess, Bundle.main.bundleIdentifier != nil else { return }
        let identifier = "dynamoe.subagent.\(runId.uuidString)"
        let center = UNUserNotificationCenter.current()
        center.removeDeliveredNotifications(withIdentifiers: [identifier])
        center.removePendingNotificationRequests(withIdentifiers: [identifier])
        // The in-memory queue is the window between a finish and the
        // authorization answer. A banner still queued there was posted after
        // the record had already been deleted, and its tap selected a dead id.
        //
        // No `Task { @MainActor }`: a queued hop can land AFTER a delivery
        // already drained the queue, and then it would drop a request posted in
        // between. `deliverPendingNotifications` mutates the same array from a
        // `DispatchQueue.main.async`, so this stays on that queue — ordered
        // with it.
        DispatchQueue.main.async {
            pendingRequests.removeAll { $0.identifier == identifier }
        }
    }

    /// The test check as a `nonisolated` fact, for callers that are not on the
    /// main actor (`withdraw` runs from the store's IO queue).
    nonisolated static var isRunningTestsInThisProcess: Bool {
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
            let sessionId = (response.notification.request.content.userInfo["sessionId"] as? String)
                .flatMap(UUID.init(uuidString:))
            Task { @MainActor in
                NSApp.activate(ignoringOtherApps: true)
                // The banner is only cancelled when its run is DELETED, so a
                // notice already sitting in Notification Centre (delivered
                // before an update, or while the app was closed) can outlive
                // the record. Pinning the selection to an id with no row left
                // the inspector on a dead tap that never cleared.
                guard SubagentManager.shared.getSubagent(byId: runId) != nil
                    || AgentRunStore.shared.hasRecordFile(runId: runId) else {
                    SubagentNotifier.withdraw(runId: runId)
                    return
                }
                SubagentManager.shared.selectedSubagentId = runId
                SubagentManager.shared.isDrawerOpen = true
                // The inspector renders the CURRENT conversation's runs, so a
                // run belonging to a different chat is invisible until that
                // chat is selected. The delegate cannot reach the window's
                // selection state directly; the notification carries the
                // session and ContentView picks it up.
                var info: [String: Any] = ["runId": runId.uuidString]
                if let sessionId { info["sessionId"] = sessionId.uuidString }
                NotificationCenter.default.post(
                    name: SubagentNotifier.revealRunNotification,
                    object: nil,
                    userInfo: info
                )
            }
        }
        completionHandler()
    }
}
