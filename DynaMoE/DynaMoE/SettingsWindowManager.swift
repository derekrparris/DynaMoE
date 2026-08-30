//
//  SettingsWindowManager.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/30/26.
//

import SwiftUI
import AppKit

extension Notification.Name {
    static let openDynaMoESettings = Notification.Name("openDynaMoESettings")
}

final class SettingsWindowManager: NSObject, NSWindowDelegate {
    static let shared = SettingsWindowManager()

    private var window: NSWindow?
    private var hostingController: NSHostingController<AnyView>?

    override private init() {
        super.init()
    }

    func show<Content: View>(title: String = "Settings & Diagnostics", content: @escaping () -> Content) {
        let view = content()

        if let existingWindow = window {
            if let hc = hostingController {
                hc.rootView = AnyView(view)
            } else {
                let hc = NSHostingController(rootView: AnyView(view))
                self.hostingController = hc
                existingWindow.contentViewController = hc
            }

            if existingWindow.isMiniaturized {
                existingWindow.deminiaturize(nil)
            }
            existingWindow.makeKeyAndOrderFront(nil)
            NSApplication.shared.activate(ignoringOtherApps: true)
            return
        }

        let hc = NSHostingController(rootView: AnyView(view))
        self.hostingController = hc

        let newWindow = NSWindow(
            contentRect: NSRect(x: 120, y: 120, width: 820, height: 680),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        newWindow.title = title
        newWindow.minSize = NSSize(width: 680, height: 540)
        newWindow.center()
        newWindow.setFrameAutosaveName("DynaMoESettingsWindowFrame")
        newWindow.contentViewController = hc
        newWindow.isReleasedWhenClosed = false
        newWindow.tabbingMode = .disallowed
        newWindow.delegate = self

        self.window = newWindow
        newWindow.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    func update<Content: View>(content: @escaping () -> Content) {
        guard let window = window, window.isVisible || window.isMiniaturized else { return }
        let view = content()
        hostingController?.rootView = AnyView(view)
    }

    func close() {
        window?.close()
    }

    var isWindowOpen: Bool {
        guard let window = window else { return false }
        return window.isVisible || window.isMiniaturized
    }

    func windowWillClose(_ notification: Notification) {
        // Window closed by user
    }
}
