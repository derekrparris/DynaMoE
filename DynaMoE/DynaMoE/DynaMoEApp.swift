//
//  DynaMoEApp.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.
//

import SwiftUI
import SwiftData
import Combine

final class AppZoomManager: ObservableObject {
    static let shared = AppZoomManager()

    @Published var zoomScale: CGFloat {
        didSet {
            UserDefaults.standard.set(Double(zoomScale), forKey: "dynamoe_ui_zoom_scale")
            showZoomHUD()
        }
    }

    @Published var hudText: String? = nil
    private var hudDismissTask: Task<Void, Never>? = nil

    init() {
        let saved = UserDefaults.standard.double(forKey: "dynamoe_ui_zoom_scale")
        self.zoomScale = (saved >= 0.5 && saved <= 3.0) ? CGFloat(saved) : 1.0
    }

    func zoomIn() {
        if zoomScale < 2.5 {
            zoomScale = min(2.5, (zoomScale + 0.15).roundedScale())
        }
    }

    func zoomOut() {
        if zoomScale > 0.6 {
            zoomScale = max(0.6, (zoomScale - 0.15).roundedScale())
        }
    }

    func resetZoom() {
        zoomScale = 1.0
    }

    private func showZoomHUD() {
        let percent = Int(round(zoomScale * 100))
        hudText = "\(percent)%"
        hudDismissTask?.cancel()
        hudDismissTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            if !Task.isCancelled {
                self.hudText = nil
            }
        }
    }

    var dynamicTypeSize: DynamicTypeSize {
        if zoomScale <= 0.75 { return .xSmall }
        if zoomScale <= 0.85 { return .small }
        if zoomScale <= 0.95 { return .medium }
        if zoomScale <= 1.05 { return .large }
        if zoomScale <= 1.15 { return .xLarge }
        if zoomScale <= 1.25 { return .xxLarge }
        if zoomScale <= 1.45 { return .xxxLarge }
        if zoomScale <= 1.75 { return .accessibility1 }
        if zoomScale <= 2.05 { return .accessibility2 }
        if zoomScale <= 2.35 { return .accessibility3 }
        return .accessibility4
    }
}

private extension CGFloat {
    func roundedScale() -> CGFloat {
        return (self * 100.0).rounded() / 100.0
    }
}

@main
struct DynaMoEApp: App {
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @Environment(\.openWindow) private var openWindow

    var sharedModelContainer: ModelContainer = {
        let schema = Schema([
            Item.self,
        ])
        let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

        do {
            return try ModelContainer(for: schema, configurations: [modelConfiguration])
        } catch {
            fatalError("Could not create ModelContainer: \(error)")
        }
    }()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(zoomManager)
                .dynamicTypeSize(zoomManager.dynamicTypeSize)
        }
        .modelContainer(sharedModelContainer)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About DynaMoE") {
                    openWindow(id: "about-dynamoe")
                }
            }
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    NotificationCenter.default.post(name: .openDynaMoESettings, object: nil)
                }
                .keyboardShortcut(",", modifiers: .command)
            }
            SidebarCommands()
            CommandGroup(after: .sidebar) {
                Divider()
                Button("Actual Size") {
                    AppZoomManager.shared.resetZoom()
                }
                .keyboardShortcut("0", modifiers: .command)

                Button("Zoom In") {
                    AppZoomManager.shared.zoomIn()
                }
                .keyboardShortcut("=", modifiers: .command)

                Button("Zoom Out") {
                    AppZoomManager.shared.zoomOut()
                }
                .keyboardShortcut("-", modifiers: .command)
            }
            CommandGroup(replacing: .help) {
                Button("DynaMoE Help & Settings Guide") {
                    openWindow(id: "dynamoe-help")
                }
                .keyboardShortcut("?", modifiers: .command)
            }
        }

        Window("About DynaMoE", id: "about-dynamoe") {
            AboutDynaMoEView()
        }
        .windowResizability(.contentSize)

        Window("DynaMoE Help & Settings Guide", id: "dynamoe-help") {
            HelpAndSettingsGuideView()
        }
        .defaultSize(width: 880, height: 680)
    }
}
