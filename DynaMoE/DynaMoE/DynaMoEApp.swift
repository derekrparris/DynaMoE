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

final class WindowZoomView: NSView {
    var zoomScale: CGFloat = 1.0 {
        didSet {
            applyZoom()
        }
    }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        if let window = window {
            NotificationCenter.default.removeObserver(self, name: NSWindow.didResizeNotification, object: nil)
            NotificationCenter.default.addObserver(self, selector: #selector(windowDidResize), name: NSWindow.didResizeNotification, object: window)
            DispatchQueue.main.async { [weak self] in
                self?.applyZoom()
            }
        }
    }

    @objc private func windowDidResize() {
        applyZoom()
    }

    private func applyZoom() {
        guard let cv = window?.contentView else { return }
        let fSize = cv.frame.size
        guard fSize.width > 0, fSize.height > 0, zoomScale > 0 else { return }

        let targetBounds = NSSize(width: fSize.width / zoomScale, height: fSize.height / zoomScale)
        if cv.bounds.size != targetBounds {
            cv.setBoundsSize(targetBounds)
            cv.needsLayout = true
            cv.needsDisplay = true
        }
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

struct WindowZoomHelper: NSViewRepresentable {
    let zoomScale: CGFloat

    func makeNSView(context: Context) -> WindowZoomView {
        let v = WindowZoomView()
        v.zoomScale = zoomScale
        return v
    }

    func updateNSView(_ nsView: WindowZoomView, context: Context) {
        nsView.zoomScale = zoomScale
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
            ContentView()
                .background(WindowZoomHelper(zoomScale: zoomManager.zoomScale))
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
