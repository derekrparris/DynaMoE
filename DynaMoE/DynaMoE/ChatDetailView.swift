//
//  ChatDetailView.swift
//  DynaMoE
//

import SwiftUI
import AppKit

struct ChatDetailView: View {
    @Binding var session: ChatSession?
    @Binding var promptText: String
    
    var isGenerating: Bool
    var generationSpeed: Double
    var generationTokens: Int
    var modelName: String?
    var onSendMessage: (String) -> Void
    var onStopGeneration: () -> Void
    var onSelectPromptStarter: (String) -> Void

    @FocusState private var isInputFocused: Bool
    @State private var isReasoningExpanded: [UUID: Bool] = [:]

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(session?.title ?? "Chat")
                        .font(.headline)
                        .fontWeight(.semibold)
                        .lineLimit(1)
                    
                    if let model = modelName {
                        HStack(spacing: 4) {
                            Text(model)
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            if isGenerating {
                                Text("•")
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.1f tok/s", generationSpeed))
                                    .font(.caption2)
                                    .fontWeight(.bold)
                                    .foregroundColor(.purple)
                            }
                        }
                    }
                }
                
                Spacer()
                
                if isGenerating {
                    Button(action: onStopGeneration) {
                        HStack(spacing: 4) {
                            Image(systemName: "stop.fill")
                                .font(.system(size: 10))
                            Text("Stop")
                                .font(.caption)
                                .fontWeight(.semibold)
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(Color.red.opacity(0.12))
                        .foregroundColor(.red)
                        .cornerRadius(6)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()

            // Main Message Content or Empty State
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 20) {
                        if let session = session, !session.messages.isEmpty {
                            ForEach(session.messages) { message in
                                ChatMessageView(
                                    message: message,
                                    isExpanded: Binding(
                                        get: { isReasoningExpanded[message.id] ?? true },
                                        set: { isReasoningExpanded[message.id] = $0 }
                                    )
                                )
                                .id(message.id)
                            }
                        } else {
                            EmptyWelcomeView(
                                modelName: modelName,
                                onSelectStarter: onSelectPromptStarter
                            )
                            .padding(.top, 40)
                        }
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 20)
                }
                .onChange(of: session?.messages.last?.content) { _ in
                    if let lastId = session?.messages.last?.id {
                        withAnimation(.easeOut(duration: 0.15)) {
                            proxy.scrollTo(lastId, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: session?.messages.last?.thinkingContent) { _ in
                    if let lastId = session?.messages.last?.id {
                        proxy.scrollTo(lastId, anchor: .bottom)
                    }
                }
            }

            // Floating Bottom Input Bar (Antigravity Style)
            VStack(spacing: 0) {
                Divider()
                
                HStack(alignment: .bottom, spacing: 10) {
                    // Multi-line Expanding Input
                    MacTextEditor(
                        text: $promptText,
                        placeholder: "Ask DynaMoE anything... (Enter to send, Shift+Enter for newline)",
                        onCommit: {
                            let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty && !isGenerating {
                                onSendMessage(trimmed)
                                promptText = ""
                            }
                        }
                    )
                    .frame(minHeight: 36, maxHeight: 120)
                    .padding(.horizontal, 4)
                    
                    // Send / Stop Button
                    if isGenerating {
                        Button(action: onStopGeneration) {
                            Image(systemName: "stop.circle.fill")
                                .font(.system(size: 28))
                                .foregroundColor(.red)
                        }
                        .buttonStyle(.plain)
                        .help("Stop generating")
                    } else {
                        Button(action: {
                            let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
                            if !trimmed.isEmpty {
                                onSendMessage(trimmed)
                                promptText = ""
                            }
                        }) {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 28))
                                .foregroundColor(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .secondary.opacity(0.4) : .purple)
                        }
                        .buttonStyle(.plain)
                        .disabled(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        .help("Send Message")
                    }
                }
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(14)
                .overlay(
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                )
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
            }
            .background(Color(NSColor.windowBackgroundColor))
        }
    }
}

// Custom Mac NSTextView Wrapper to support Enter-to-Submit and Shift+Enter for newlines
struct MacTextEditor: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String
    var onCommit: () -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        guard let textView = scrollView.documentView as? NSTextView else { return scrollView }

        textView.delegate = context.coordinator
        textView.font = .systemFont(ofSize: 13.5)
        textView.isRichText = false
        textView.drawsBackground = false
        textView.allowsUndo = true
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false

        scrollView.hasVerticalScroller = false
        scrollView.drawsBackground = false
        return scrollView
    }

    func updateNSView(_ nsView: NSScrollView, context: Context) {
        guard let textView = nsView.documentView as? NSTextView else { return }
        if textView.string != text {
            textView.string = text
        }
    }

    class Coordinator: NSObject, NSTextViewDelegate {
        var parent: MacTextEditor

        init(_ parent: MacTextEditor) {
            self.parent = parent
        }

        func textDidChange(_ notification: Notification) {
            guard let textView = notification.object as? NSTextView else { return }
            self.parent.text = textView.string
        }

        func textView(_ textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                if let event = NSApp.currentEvent, event.modifierFlags.contains(.shift) {
                    textView.insertNewlineIgnoringFieldEditor(nil)
                    return true
                } else {
                    parent.onCommit()
                    return true
                }
            }
            return false
        }
    }
}

// Single Message Bubble View
struct ChatMessageView: View {
    let message: ChatMessage
    @Binding var isExpanded: Bool
    @State private var isCopied = false

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if message.role == .assistant {
                // Assistant Avatar
                Image(systemName: "circle.hexagongrid.fill")
                    .font(.system(size: 16))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 28, height: 28)
                    .background(Color.purple.opacity(0.12))
                    .clipShape(Circle())
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                // User Message Bubble
                if message.role == .user {
                    HStack {
                        Spacer(minLength: 40)
                        Text(message.content)
                            .font(.system(size: 14))
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.purple.opacity(0.18))
                            .foregroundColor(.primary)
                            .cornerRadius(12)
                            .textSelection(.enabled)
                    }
                } else {
                    // Assistant Thinking / Reasoning Accordion
                    if let thinking = message.thinkingContent, !thinking.isEmpty {
                        VStack(alignment: .leading, spacing: 6) {
                            Button(action: {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    isExpanded.toggle()
                                }
                            }) {
                                HStack(spacing: 6) {
                                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(.secondary)
                                    
                                    Image(systemName: "brain.head.profile")
                                        .font(.system(size: 12))
                                        .foregroundColor(.purple)
                                    
                                    Text(message.isThinking ? "Thinking..." : "Thought Process")
                                        .font(.system(size: 12, weight: .medium))
                                        .foregroundColor(.secondary)
                                    
                                    if message.isThinking {
                                        ProgressView()
                                            .scaleEffect(0.6)
                                            .frame(width: 12, height: 12)
                                    }
                                    
                                    Spacer()
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.06))
                                .cornerRadius(6)
                            }
                            .buttonStyle(.plain)

                            if isExpanded {
                                Text(thinking)
                                    .font(.system(size: 12.5, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.secondary.opacity(0.04))
                                    .cornerRadius(8)
                                    .textSelection(.enabled)
                            }
                        }
                    }

                    // Main Response Text
                    if !message.content.isEmpty {
                        Text(message.content)
                            .font(.system(size: 14))
                            .lineSpacing(4)
                            .foregroundColor(.primary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else if message.isThinking {
                        // Empty placeholder while purely thinking
                        HStack(spacing: 6) {
                            Text("Generating response...")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Action & Metrics Footer
                    if !message.isThinking && !message.content.isEmpty {
                        HStack(spacing: 12) {
                            Button(action: {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(message.content, forType: .string)
                                isCopied = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                                    isCopied = false
                                }
                            }) {
                                HStack(spacing: 4) {
                                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                                        .font(.system(size: 11))
                                    Text(isCopied ? "Copied" : "Copy")
                                        .font(.system(size: 11))
                                }
                                .foregroundColor(.secondary)
                            }
                            .buttonStyle(.plain)

                            if message.tokenCount > 0 {
                                Text("•")
                                    .foregroundColor(.secondary.opacity(0.5))
                                Text("\(message.tokenCount) tokens")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }

                            if message.tokensPerSec > 0 {
                                Text("•")
                                    .foregroundColor(.secondary.opacity(0.5))
                                Text(String(format: "%.1f tok/s", message.tokensPerSec))
                                    .font(.system(size: 11, design: .monospaced))
                                    .foregroundColor(.purple)
                            }

                            Spacer()
                        }
                        .padding(.top, 4)
                    }
                }
            }

            if message.role == .user {
                // User Avatar
                Image(systemName: "person.crop.circle.fill")
                    .font(.system(size: 24))
                    .foregroundColor(.secondary)
            }
        }
    }
}

// Welcome / Empty State View
struct EmptyWelcomeView: View {
    var modelName: String?
    var onSelectStarter: (String) -> Void

    let promptStarters = [
        "Tell me a clever computer related joke",
        "Explain how Mixture-of-Experts (MoE) routing works",
        "Write a Swift function for parallel async tasks",
        "What is the difference between BF16 and FP8?"
    ]

    var body: some View {
        VStack(spacing: 24) {
            VStack(spacing: 12) {
                Image(systemName: "circle.hexagongrid.circle.fill")
                    .font(.system(size: 54))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.purple, .indigo, .blue],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )

                Text("How can I help you today?")
                    .font(.title2)
                    .fontWeight(.bold)

                if let model = modelName {
                    Text("Powered by \(model)")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                } else {
                    Text("Load model weights in Settings (bottom left) to start generating.")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
            }

            // Starter Prompt Chips
            VStack(spacing: 10) {
                ForEach(promptStarters, id: \.self) { starter in
                    Button(action: {
                        onSelectStarter(starter)
                    }) {
                        HStack {
                            Text(starter)
                                .font(.system(size: 13))
                                .foregroundColor(.primary)
                            Spacer()
                            Image(systemName: "arrow.up.right")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(10)
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: 480)
                }
            }
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity)
    }
}
