//
//  SidebarView.swift
//  DynaMoE
//

import SwiftUI

struct SidebarView: View {
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @Binding var sessions: [ChatSession]
    @Binding var selectedSessionId: UUID?
    @Binding var isSettingsPresented: Bool
    
    var modelName: String?
    var metalStatus: String
    var currentRssGB: Double
    var isGenerating: Bool
    var onNewChat: () -> Void
    var onDeleteSession: (UUID) -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Header: New Chat Button
            VStack(alignment: .leading, spacing: 12) {
                // New Chat Action Button
                Button(action: onNewChat) {
                    HStack(spacing: 8) {
                        Image(systemName: "square.and.pencil")
                            .font(.system(size: max(10, 13 * zoomManager.zoomScale), weight: .semibold))
                        Text("New Chat")
                            .font(.system(size: max(10, 13 * zoomManager.zoomScale), weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(Color.primary.opacity(0.06))
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.top, 14)
            }
            .padding(.bottom, 8)

            Divider()
                .padding(.horizontal, 8)

            // Conversations List
            ScrollView {
                LazyVStack(spacing: 2) {
                    if sessions.isEmpty {
                        VStack(spacing: 8) {
                            Image(systemName: "bubble.left.and.bubble.right")
                                .font(.system(size: max(14, 18 * zoomManager.zoomScale)))
                                .foregroundColor(.secondary.opacity(0.6))
                            Text("No Conversations Yet")
                                .font(.system(size: max(9, 11 * zoomManager.zoomScale)))
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 40)
                    } else {
                        ForEach(sessions) { session in
                            SidebarSessionRow(
                                session: session,
                                isSelected: session.id == selectedSessionId,
                                zoomScale: zoomManager.zoomScale,
                                onSelect: {
                                    selectedSessionId = session.id
                                },
                                onRename: { newTitle in
                                    if let idx = sessions.firstIndex(where: { $0.id == session.id }) {
                                        sessions[idx].title = newTitle
                                        sessions[idx].titleOrigin = .user
                                    }
                                },
                                onDelete: {
                                    onDeleteSession(session.id)
                                }
                            )
                        }
                    }
                }
                .padding(.vertical, 8)
                .padding(.horizontal, 8)
            }

            Spacer(minLength: 0)

            Divider()
                .padding(.horizontal, 8)

            // Bottom Left: Settings Button & System Status
            VStack(spacing: 6) {
                // Model Status Pill (if loaded)
                if let model = modelName {
                    let attributedGB = WorkingSetManager.shared.attributedMemoryGB
                    HStack(spacing: 6) {
                        Circle()
                            .fill(isGenerating ? Color.green : Color.purple)
                            .frame(width: max(5, 7 * zoomManager.zoomScale), height: max(5, 7 * zoomManager.zoomScale))
                        Text(model)
                            .font(.system(size: max(9, 11 * zoomManager.zoomScale), weight: .medium))
                            .lineLimit(1)
                            .foregroundColor(.secondary)
                        Spacer()
                        if attributedGB > 0 {
                            Text(String(format: "%.1f GB", attributedGB))
                                .font(.system(size: max(8, 10 * zoomManager.zoomScale), design: .monospaced))
                                .foregroundColor(.secondary.opacity(0.8))
                                .help(String(format: "DynaMoE memory: %.1f GB\nProcess footprint (Activity Monitor / Xcode): %.1f GB\nWeight pages in the Unified Memory Cache: %.1f GB (reclaimable)\nResident pages (RSIZE): %.1f GB\n\nTotal = what this process is holding in RAM for the model: the Darwin footprint (dirty heap, compressed memory, IOKit/GPU — including the FlashMoE pinned backbone) plus the clean file-backed model weights Darwin's Unified Memory Buffer Cache is holding on its behalf. Activity Monitor's 'Memory' column shows only the footprint, so it excludes those clean pages. The cache share is reclaimable under memory pressure; the footprint is not.", attributedGB, getActivityMonitorFootprintGB(), WorkingSetManager.shared.residentShardBytesGB, currentRssGB))
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(6)
                }

                // Settings Button
                SidebarSettingsButton(zoomScale: zoomManager.zoomScale) {
                    isSettingsPresented = true
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
        }
        .frame(minWidth: max(180, 220 * zoomManager.zoomScale), idealWidth: max(210, 250 * zoomManager.zoomScale), maxWidth: max(260, 320 * zoomManager.zoomScale))
        .background(Color(NSColor.windowBackgroundColor))
    }
}

struct SidebarSettingsButton: View {
    var zoomScale: CGFloat = 1.0
    let action: () -> Void
    @State private var isHovered: Bool = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: max(11, 14 * zoomScale)))
                    .foregroundColor(isHovered ? .primary : .secondary)
                Text("Settings")
                    .font(.system(size: max(10, 13 * zoomScale), weight: .medium))
                    .foregroundColor(isHovered ? .primary : .secondary)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(isHovered ? Color.primary.opacity(0.06) : Color.clear)
            .contentShape(Rectangle())
            .cornerRadius(6)
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovered = hovering
        }
    }
}

struct SidebarSessionRow: View {
    let session: ChatSession
    let isSelected: Bool
    var zoomScale: CGFloat = 1.0
    let onSelect: () -> Void
    let onRename: (String) -> Void
    let onDelete: () -> Void
    @State private var isHovered = false
    @State private var isRenaming = false
    @State private var draftTitle = ""
    @FocusState private var isTitleFieldFocused: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "bubble.left")
                .font(.system(size: max(9.5, 12 * zoomScale)))
                .foregroundColor(isSelected ? .purple : .secondary)

            if isRenaming {
                TextField("Session name", text: $draftTitle)
                    .textFieldStyle(.plain)
                    .font(.system(size: max(10, 13 * zoomScale)))
                    .focused($isTitleFieldFocused)
                    .onSubmit { commitRename() }
                    .onExitCommand { isRenaming = false }
                    .onAppear { isTitleFieldFocused = true }
            } else {
                Text(session.displayTitle)
                    .font(.system(size: max(10, 13 * zoomScale)))
                    .foregroundColor(isSelected ? .primary : .secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }

            Spacer()

            if isHovered || isSelected {
                Button(action: beginRename) {
                    Image(systemName: "pencil")
                        .font(.system(size: max(8.5, 11 * zoomScale)))
                        .foregroundColor(.secondary.opacity(0.8))
                }
                .buttonStyle(.plain)
                .help("Rename Conversation")

                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: max(8.5, 11 * zoomScale)))
                        .foregroundColor(.secondary.opacity(0.8))
                }
                .buttonStyle(.plain)
                .help("Delete Conversation")
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(
            isSelected
                ? Color.purple.opacity(0.12)
                : (isHovered ? Color.primary.opacity(0.04) : Color.clear)
        )
        .cornerRadius(7)
        .contentShape(Rectangle())
        .onTapGesture {
            if !isRenaming {
                onSelect()
            }
        }
        .onHover { hovering in
            isHovered = hovering
        }
    }

    private func beginRename() {
        draftTitle = session.title
        isRenaming = true
    }

    private func commitRename() {
        let trimmed = draftTitle.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty {
            onRename(trimmed)
        }
        isRenaming = false
    }
}
