//
//  SidebarView.swift
//  DynaMoE
//

import SwiftUI

struct SidebarView: View {
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
            // Header: Branding & New Chat Button
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 8) {
                    Image(systemName: "circle.hexagongrid.circle.fill")
                        .font(.title2)
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.purple, .indigo, .blue],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                    
                    Text("DynaMoE")
                        .font(.headline)
                        .fontWeight(.bold)
                    
                    Spacer()
                }
                .padding(.horizontal, 14)
                .padding(.top, 14)

                // New Chat Action Button
                Button(action: onNewChat) {
                    HStack(spacing: 8) {
                        Image(systemName: "square.and.pencil")
                            .font(.system(size: 13, weight: .semibold))
                        Text("New Chat")
                            .font(.system(size: 13, weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(Color.primary.opacity(0.06))
                    .cornerRadius(8)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
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
                                .font(.title3)
                                .foregroundColor(.secondary.opacity(0.6))
                            Text("No Conversations Yet")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.top, 40)
                    } else {
                        ForEach(sessions) { session in
                            SidebarSessionRow(
                                session: session,
                                isSelected: session.id == selectedSessionId,
                                onSelect: {
                                    selectedSessionId = session.id
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
                    HStack(spacing: 6) {
                        Circle()
                            .fill(isGenerating ? Color.green : Color.purple)
                            .frame(width: 7, height: 7)
                        Text(model)
                            .font(.system(size: 11, weight: .medium))
                            .lineLimit(1)
                            .foregroundColor(.secondary)
                        Spacer()
                        if currentRssGB > 0 {
                            Text(String(format: "%.1f GB", currentRssGB))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundColor(.secondary.opacity(0.8))
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(6)
                }

                // Settings Button
                Button(action: {
                    isSettingsPresented = true
                }) {
                    HStack(spacing: 8) {
                        Image(systemName: "gearshape.fill")
                            .font(.system(size: 14))
                            .foregroundColor(.secondary)
                        Text("Settings")
                            .font(.system(size: 13, weight: .medium))
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.clear)
                    .contentShape(Rectangle())
                    .cornerRadius(6)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
        }
        .frame(minWidth: 220, idealWidth: 250, maxWidth: 300)
        .background(Color(NSColor.windowBackgroundColor))
    }
}

struct SidebarSessionRow: View {
    let session: ChatSession
    let isSelected: Bool
    let onSelect: () -> Void
    let onDelete: () -> Void
    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "bubble.left")
                .font(.system(size: 12))
                .foregroundColor(isSelected ? .purple : .secondary)

            Text(session.title)
                .font(.system(size: 13))
                .foregroundColor(isSelected ? .primary : .secondary)
                .lineLimit(1)
                .truncationMode(.tail)

            Spacer()

            if isHovered || isSelected {
                Button(action: onDelete) {
                    Image(systemName: "trash")
                        .font(.system(size: 11))
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
            onSelect()
        }
        .onHover { hovering in
            isHovered = hovering
        }
    }
}
