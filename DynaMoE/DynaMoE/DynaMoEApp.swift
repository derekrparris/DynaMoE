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
        }
    }

    init() {
        let saved = UserDefaults.standard.double(forKey: "dynamoe_ui_zoom_scale")
        self.zoomScale = (saved >= 0.5 && saved <= 3.0) ? CGFloat(saved) : 1.0
    }

    func zoomIn() {
        if zoomScale < 2.5 {
            zoomScale = (zoomScale + 0.1).roundedScale()
        }
    }

    func zoomOut() {
        if zoomScale > 0.6 {
            zoomScale = (zoomScale - 0.1).roundedScale()
        }
    }

    func resetZoom() {
        zoomScale = 1.0
    }
}

private extension CGFloat {
    func roundedScale() -> CGFloat {
        return (self * 10.0).rounded() / 10.0
    }
}

@main
struct DynaMoEApp: App {
    @StateObject private var zoomManager = AppZoomManager.shared
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
            GeometryReader { geo in
                ContentView()
                    .frame(
                        width: max(100, geo.size.width / zoomManager.zoomScale),
                        height: max(100, geo.size.height / zoomManager.zoomScale)
                    )
                    .scaleEffect(zoomManager.zoomScale, anchor: .topLeading)
                    .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
                    .clipped()
            }
            .environmentObject(zoomManager)
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
                    zoomManager.resetZoom()
                }
                .keyboardShortcut("0", modifiers: .command)

                Button("Zoom In") {
                    zoomManager.zoomIn()
                }
                .keyboardShortcut("+", modifiers: .command)

                Button("Zoom Out") {
                    zoomManager.zoomOut()
                }
                .keyboardShortcut("-", modifiers: .command)
            }
        }

        Window("About DynaMoE", id: "about-dynamoe") {
            AboutDynaMoEView()
        }
        .windowResizability(.contentSize)
    }
}
