//
//  ChatDetailView.swift
//  DynaMoE
//

import SwiftUI
import AppKit

struct ChatDetailView: View {
    @Binding var session: ChatSession?
    @Binding var promptText: String
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    
    var isGenerating: Bool
    var isStreamingOffDisk: Bool = false
    var generationSpeed: Double
    var generationTokens: Int
    var modelName: String?
    var supportsThinking: Bool = false
    var isThinkingEnabled: Bool = true
    var onSendMessage: (String) -> Void
    var onStopGeneration: () -> Void
    var onSelectPromptStarter: (String) -> Void
    var onSelectDiscoveredModel: ((DiscoveredModel) -> Void)? = nil
    var onOpenSettings: (() -> Void)? = nil
    var onToggleThinking: ((Bool) -> Void)? = nil

    @FocusState private var isInputFocused: Bool
    @State private var isReasoningExpanded: [UUID: Bool] = [:]

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar (Antigravity breadcrumb style)
            HStack(spacing: 8) {
                Text("DynaMoE")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundColor(.secondary)
                
                Text("/")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundColor(.secondary.opacity(0.4))
                
                Text(session?.title ?? "App")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundColor(.primary)
                    .lineLimit(1)
                
                Spacer()
                
                if let model = modelName {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(isGenerating ? (isStreamingOffDisk ? Color.orange : Color.green) : Color.purple)
                            .frame(width: 6, height: 6)
                        Text(model)
                            .font(.caption)
                            .foregroundColor(.secondary)
                        if isGenerating {
                            Text("•")
                                .foregroundColor(.secondary)
                            HStack(spacing: 3) {
                                Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                Text(String(format: "%.1f tok/s", generationSpeed))
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                            }
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.secondary.opacity(0.06))
                    .cornerRadius(8)
                }
                
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
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()

            // Main Message Content or Empty State (Centered & Constrained to Max Width for Readability)
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                        if let session = session, !session.messages.isEmpty {
                            LazyVStack(spacing: 24) {
                                ForEach(session.messages) { message in
                                    ChatMessageView(
                                        message: message,
                                        isGenerating: isGenerating && message.id == session.messages.last?.id,
                                        isStreamingOffDisk: isStreamingOffDisk,
                                        isExpanded: Binding(
                                            get: { isReasoningExpanded[message.id] ?? true },
                                            set: { isReasoningExpanded[message.id] = $0 }
                                        )
                                    )
                                    .id(message.id)
                                }
                            }
                            .padding(.top, 24)
                            .padding(.bottom, 24)
                        } else {
                            EmptyWelcomeView(
                                modelName: modelName,
                                isStreamingOffDisk: isStreamingOffDisk,
                                onSelectStarter: onSelectPromptStarter
                            )
                            .padding(.top, 48)
                            .padding(.bottom, 32)
                        }
                    }
                    .frame(maxWidth: 800)
                    .padding(.horizontal, 24)
                    .frame(maxWidth: .infinity)
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

            // Floating Bottom Composer (Antigravity Centered Constrained Width)
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 10) {
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
                    .frame(minHeight: 38, maxHeight: 130)
                    .padding(.horizontal, 4)
                    .padding(.top, 2)
                    
                    // Bottom Controls Bar inside the floating card
                    HStack(spacing: 8) {
                        // Attachment / Plus Button
                        Button(action: {}) {
                            Image(systemName: "plus")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.secondary)
                                .frame(width: 24, height: 24)
                                .background(Color.secondary.opacity(0.08))
                                .clipShape(Circle())
                        }
                        .buttonStyle(.plain)
                        .help("Attach context or files")

                        // Antigravity-Style Model Selector Menu
                        Menu {
                            if localModelManager.discoveredModels.isEmpty {
                                Text("No models found in ~/.cache/huggingface/hub")
                            } else {
                                Section("Discovered Hugging Face Models") {
                                    ForEach(localModelManager.discoveredModels) { dm in
                                        let isCurrent = (session?.selectedModelId == dm.id) ||
                                                        (session?.selectedModelPath == dm.snapshotPath) ||
                                                        (modelName == dm.displayName) ||
                                                        (session?.selectedModelName == dm.displayName)
                                        Button(action: {
                                            onSelectDiscoveredModel?(dm)
                                        }) {
                                            HStack {
                                                if isCurrent {
                                                    Image(systemName: "checkmark")
                                                }
                                                Text(dm.displayName)
                                                Text("(\(dm.formattedSize))")
                                                if dm.isMoE {
                                                    Text("• MoE")
                                                }
                                                if dm.supportsThinking {
                                                    Text("• 🧠 Thinking")
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            Divider()

                            if let onOpenSettings = onOpenSettings {
                                Button(action: onOpenSettings) {
                                    Label("Manage Models in Settings...", systemImage: "gearshape")
                                }
                            }
                        } label: {
                            HStack(spacing: 5) {
                                Image(systemName: "cube.fill")
                                    .font(.system(size: 10))
                                    .foregroundColor(.purple)
                                Text(session?.selectedModelName ?? modelName ?? "Select Model")
                                    .font(.system(size: 11.5, weight: .medium))
                                    .foregroundColor(.primary)
                                    .lineLimit(1)
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.system(size: 7.5, weight: .semibold))
                                    .foregroundColor(.secondary.opacity(0.7))
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 4.5)
                            .background(Color.secondary.opacity(0.08))
                            .cornerRadius(10)
                        }
                        .menuStyle(.borderlessButton)
                        .fixedSize()

                        // Thinking On/Off Dropdown beside Model Selector (shown only if loaded model supports thinking)
                        if supportsThinking {
                            Menu {
                                Button(action: {
                                    onToggleThinking?(true)
                                }) {
                                    HStack {
                                        if isThinkingEnabled {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking On (Reasoning Process)", systemImage: "brain.head.profile")
                                    }
                                }

                                Button(action: {
                                    onToggleThinking?(false)
                                }) {
                                    HStack {
                                        if !isThinkingEnabled {
                                            Image(systemName: "checkmark")
                                        }
                                        Label("Thinking Off (Direct Response)", systemImage: "bolt.slash")
                                    }
                                }
                            } label: {
                                HStack(spacing: 5) {
                                    Image(systemName: isThinkingEnabled ? "brain.head.profile" : "bolt.slash")
                                        .font(.system(size: 10))
                                        .foregroundColor(isThinkingEnabled ? .purple : .secondary)
                                    Text(isThinkingEnabled ? "Thinking On" : "Thinking Off")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .foregroundColor(isThinkingEnabled ? .primary : .secondary)
                                    Image(systemName: "chevron.up.chevron.down")
                                        .font(.system(size: 7.5, weight: .semibold))
                                        .foregroundColor(.secondary.opacity(0.7))
                                }
                                .padding(.horizontal, 9)
                                .padding(.vertical, 4.5)
                                .background(isThinkingEnabled ? Color.purple.opacity(0.12) : Color.secondary.opacity(0.08))
                                .cornerRadius(10)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(isThinkingEnabled ? Color.purple.opacity(0.3) : Color.clear, lineWidth: 1)
                                )
                            }
                            .menuStyle(.borderlessButton)
                            .fixedSize()
                            .help(isThinkingEnabled ? "Thinking / Reasoning is enabled for this model" : "Direct response without thinking is enabled")
                            .transition(.opacity.combined(with: .scale))
                        }

                        Spacer()

                        // Mic Button
                        Button(action: {}) {
                            Image(systemName: "mic")
                                .font(.system(size: 13))
                                .foregroundColor(.secondary)
                                .frame(width: 24, height: 24)
                        }
                        .buttonStyle(.plain)

                        // Send / Stop Button
                        if isGenerating {
                            Button(action: onStopGeneration) {
                                Image(systemName: "stop.circle.fill")
                                    .font(.system(size: 26))
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
                                Image(systemName: "arrow.right.circle.fill")
                                    .font(.system(size: 26))
                                    .foregroundColor(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .secondary.opacity(0.3) : .purple)
                            }
                            .buttonStyle(.plain)
                            .disabled(promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            .help("Send Message")
                        }
                    }
                }
                .padding(12)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(18)
                .overlay(
                    RoundedRectangle(cornerRadius: 18)
                        .stroke(Color.primary.opacity(0.09), lineWidth: 1)
                )
                .shadow(color: Color.black.opacity(0.06), radius: 10, x: 0, y: 4)
                .frame(maxWidth: 800)
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .frame(maxWidth: .infinity)
            }
            .background(Color(NSColor.windowBackgroundColor))
        }
    }
}

// MARK: - Custom Mac NSTextView Wrapper
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

// MARK: - Single Message Bubble View
struct ChatMessageView: View {
    let message: ChatMessage
    var isGenerating: Bool = false
    var isStreamingOffDisk: Bool = false
    @Binding var isExpanded: Bool
    @State private var isCopied = false
    @State private var feedback: String? = nil

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            if message.role == .assistant {
                // Assistant Avatar
                Image(systemName: "circle.hexagongrid.fill")
                    .font(.system(size: 15))
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
                    .padding(.top, 2)
            }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 8) {
                // User Message Bubble (Constrained max width, elegant purple card)
                if message.role == .user {
                    HStack {
                        Spacer(minLength: 40)
                        Text(message.content)
                            .font(.system(size: 14))
                            .lineSpacing(3)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(Color.purple.opacity(0.18))
                            .foregroundColor(.primary)
                            .cornerRadius(14)
                            .textSelection(.enabled)
                    }
                } else {
                    // Assistant Thinking / Reasoning Accordion (Osaurus Style)
                    let hasThinkingContent = message.thinkingContent != nil && !message.thinkingContent!.isEmpty
                    if (message.isThinking && isGenerating) || hasThinkingContent {
                        let thinking = message.thinkingContent ?? ""
                        VStack(alignment: .leading, spacing: 8) {
                            Button(action: {
                                withAnimation(.easeInOut(duration: 0.2)) {
                                    isExpanded.toggle()
                                }
                            }) {
                                HStack(spacing: 8) {
                                    Image(systemName: "brain.head.profile")
                                        .font(.system(size: 12, weight: .semibold))
                                        .foregroundStyle(
                                            LinearGradient(
                                                colors: [.purple, .indigo],
                                                startPoint: .topLeading,
                                                endPoint: .bottomTrailing
                                            )
                                        )

                                    if message.isThinking && isGenerating {
                                        Text("Thinking...")
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundColor(.primary)
                                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                                    } else if let tTime = message.thinkingTimeSeconds {
                                        Text(String(format: "Thought for %.1fs", tTime))
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundColor(.secondary)
                                    } else {
                                        Text("Thought Process")
                                            .font(.system(size: 12, weight: .medium))
                                            .foregroundColor(.secondary)
                                    }

                                    Spacer()

                                    let charCount = thinking.count
                                    let charCountStr = charCount >= 1000 ? String(format: "%.1fk chars", Double(charCount) / 1000.0) : "\(charCount) chars"
                                    Text(charCountStr)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(.secondary.opacity(0.7))

                                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(.secondary.opacity(0.8))
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 7)
                                .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                                )
                            }
                            .buttonStyle(.plain)

                            if isExpanded {
                                VStack(alignment: .leading, spacing: 4) {
                                    if !thinking.isEmpty {
                                        Text(thinking)
                                            .font(.system(size: 12, design: .monospaced))
                                            .foregroundColor(.secondary)
                                            .lineSpacing(3)
                                            .padding(.horizontal, 14)
                                            .padding(.vertical, 10)
                                            .frame(maxWidth: .infinity, alignment: .leading)
                                            .background(Color.secondary.opacity(0.035))
                                            .cornerRadius(8)
                                            .overlay(
                                                RoundedRectangle(cornerRadius: 8)
                                                    .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                                            )
                                            .textSelection(.enabled)
                                    } else if message.isThinking && isGenerating {
                                        HStack(spacing: 8) {
                                            StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                                            Text(isStreamingOffDisk ? "Streaming MoE experts off SSD disk..." : "Generating thought process...")
                                                .font(.system(size: 12, design: .monospaced))
                                                .foregroundColor(.secondary.opacity(0.8))
                                        }
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 10)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Color.secondary.opacity(0.035))
                                        .cornerRadius(8)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 8)
                                                .stroke(Color.primary.opacity(0.05), lineWidth: 1)
                                        )
                                    }
                                }
                            }
                        }
                    }

                    // Main Response Text rendered via Rich Markdown Engine
                    if !message.content.isEmpty {
                        MarkdownMessageView(
                            content: message.content,
                            isStreaming: isGenerating && message.isThinking == false,
                            isStreamingOffDisk: isStreamingOffDisk
                        )
                    } else if isGenerating && !message.isThinking && !hasThinkingContent {
                        // Only for active generation while initial pre-fill occurs
                        HStack(spacing: 8) {
                            StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                            Text(isStreamingOffDisk ? "Streaming MoE experts off SSD disk..." : "Generating response...")
                                .font(.system(size: 13))
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    }

                    // Action & Metrics Footer (Osaurus Style Telemetry & Actions)
                    if (!message.isThinking || !isGenerating) && !message.content.isEmpty {
                        HStack(spacing: 12) {
                            // Telemetry
                            HStack(spacing: 6) {
                                if let ttft = message.timeToFirstTokenSeconds {
                                    Text(String(format: "TTFT %.2fs", ttft))
                                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                                        .foregroundColor(.secondary.opacity(0.8))

                                    Text("•")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary.opacity(0.4))
                                }

                                if message.tokensPerSec > 0 {
                                    HStack(spacing: 3) {
                                        Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                                            .font(.system(size: 9, weight: .bold))
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                        Text(String(format: "%.1f tok/s", message.tokensPerSec))
                                            .font(.system(size: 11, weight: .medium, design: .monospaced))
                                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                    }

                                    Text("•")
                                        .font(.system(size: 10))
                                        .foregroundColor(.secondary.opacity(0.4))
                                }

                                if message.tokenCount > 0 {
                                    Text("\(message.tokenCount.formatted()) tokens")
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(.secondary.opacity(0.8))
                                }
                            }

                            Spacer()

                            // Action Buttons
                            HStack(spacing: 8) {
                                // Copy button
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
                                        if isCopied {
                                            Text("Copied")
                                                .font(.system(size: 11))
                                        }
                                    }
                                    .foregroundColor(isCopied ? .green : .secondary)
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 3)
                                    .background(Color.secondary.opacity(0.06))
                                    .cornerRadius(4)
                                }
                                .buttonStyle(.plain)
                                .help("Copy full response")

                                // Thumbs up / down
                                Button(action: {
                                    feedback = feedback == "up" ? nil : "up"
                                }) {
                                    Image(systemName: feedback == "up" ? "hand.thumbsup.fill" : "hand.thumbsup")
                                        .font(.system(size: 11))
                                        .foregroundColor(feedback == "up" ? .purple : .secondary)
                                        .padding(4)
                                        .background(Color.secondary.opacity(0.06))
                                        .cornerRadius(4)
                                }
                                .buttonStyle(.plain)
                                .help("Good response")

                                Button(action: {
                                    feedback = feedback == "down" ? nil : "down"
                                }) {
                                    Image(systemName: feedback == "down" ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                                        .font(.system(size: 11))
                                        .foregroundColor(feedback == "down" ? .purple : .secondary)
                                        .padding(4)
                                        .background(Color.secondary.opacity(0.06))
                                        .cornerRadius(4)
                                }
                                .buttonStyle(.plain)
                                .help("Bad response")

                                // Audio speak
                                Button(action: {
                                    NSSpeechSynthesizer().startSpeaking(message.content)
                                }) {
                                    Image(systemName: "speaker.wave.2")
                                        .font(.system(size: 11))
                                        .foregroundColor(.secondary)
                                        .padding(4)
                                        .background(Color.secondary.opacity(0.06))
                                        .cornerRadius(4)
                                }
                                .buttonStyle(.plain)
                                .help("Read response aloud")
                            }
                        }
                        .padding(.top, 4)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
        }
    }
}

// MARK: - Welcome / Empty State View
struct EmptyWelcomeView: View {
    var modelName: String?
    var isStreamingOffDisk: Bool = false
    var onSelectStarter: (String) -> Void

    let promptStarters = [
        "Tell me a clever computer related joke",
        "Explain how Mixture-of-Experts (MoE) routing works",
        "Write a Swift function for parallel async tasks",
        "What is the difference between BF16 and FP8?"
    ]

    var body: some View {
        VStack(spacing: 28) {
            VStack(spacing: 12) {
                Image(systemName: "circle.hexagongrid.circle.fill")
                    .font(.system(size: 52))
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
                    HStack(spacing: 6) {
                        Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                            .font(.system(size: 12))
                            .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                        Text("Powered by \(model) • \(isStreamingOffDisk ? "Dynamic SSD Streaming" : "Fast Unified RAM Engine")")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
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
                                .font(.system(size: 13.5))
                                .foregroundColor(.primary)
                            Spacer()
                            Image(systemName: "arrow.up.right")
                                .font(.system(size: 11))
                                .foregroundColor(.secondary)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 12)
                        .background(Color.secondary.opacity(0.06))
                        .cornerRadius(12)
                    }
                    .buttonStyle(.plain)
                    .frame(maxWidth: 520)
                }
            }
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Markdown Table & Typography Models

public enum TextAlignmentType {
    case leading
    case center
    case trailing

    public var alignment: Alignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    public var textAlignment: TextAlignment {
        switch self {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

public struct MarkdownTableData: Identifiable, Equatable {
    public let id: String
    public let headers: [String]
    public let alignments: [TextAlignmentType]
    public let rows: [[String]]

    public init(headers: [String], alignments: [TextAlignmentType], rows: [[String]], id: String? = nil) {
        self.headers = headers
        self.alignments = alignments
        self.rows = rows
        self.id = id ?? "tbl-\(headers.joined(separator: "_"))-\(rows.count)"
    }
}

// MARK: - Markdown Block AST

public enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(text: String)
    case table(MarkdownTableData)
    case codeBlock(language: String, code: String)
    case blockquote(text: String)
    case listItem(number: Int?, text: String)
    case horizontalRule
}

public struct IdentifiableMarkdownBlock: Identifiable, Equatable {
    public let id: String
    public let block: MarkdownBlock
    public let isLast: Bool
}

// MARK: - Markdown Message View

struct MarkdownMessageView: View {
    let content: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false

    private var identifiableBlocks: [IdentifiableMarkdownBlock] {
        let rawBlocks = parseMarkdownBlocks(content)
        return rawBlocks.enumerated().map { index, block in
            let isLast = isStreaming && (index == rawBlocks.count - 1)
            let blockId: String
            switch block {
            case .heading(let lvl, _): blockId = "h-\(index)-\(lvl)"
            case .paragraph: blockId = "p-\(index)"
            case .table(let tbl): blockId = "tbl-\(index)-\(tbl.headers.count)"
            case .codeBlock(let lang, _): blockId = "code-\(index)-\(lang)"
            case .blockquote: blockId = "q-\(index)"
            case .listItem(let num, _): blockId = "li-\(index)-\(num ?? 0)"
            case .horizontalRule: blockId = "hr-\(index)"
            }
            return IdentifiableMarkdownBlock(id: blockId, block: block, isLast: isLast)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(identifiableBlocks) { item in
                switch item.block {
                case .heading(let level, let text):
                    renderHeading(level: level, text: text, isLast: item.isLast)
                case .paragraph(let text):
                    renderParagraph(text: text, isLast: item.isLast)
                case .table(let table):
                    MarkdownTableView(table: table, isLast: item.isLast, isStreamingOffDisk: isStreamingOffDisk)
                case .codeBlock(let language, let code):
                    CodeBlockCard(
                        language: language,
                        code: code,
                        isStreaming: item.isLast,
                        isStreamingOffDisk: isStreamingOffDisk
                    )
                case .blockquote(let text):
                    renderBlockquote(text: text, isLast: item.isLast)
                case .listItem(let number, let text):
                    renderListItem(number: number, text: text, isLast: item.isLast)
                case .horizontalRule:
                    Divider()
                        .padding(.vertical, 6)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Block Renderers

    @ViewBuilder
    private func renderHeading(level: Int, text: String, isLast: Bool) -> some View {
        let size: CGFloat = {
            switch level {
            case 1: return 19
            case 2: return 16.5
            case 3: return 15
            default: return 14
            }
        }()
        let weight: Font.Weight = level <= 2 ? .bold : .semibold
        let topPad: CGFloat = level == 1 ? 12 : (level == 2 ? 10 : 6)
        let botPad: CGFloat = level == 1 ? 4 : 2

        if isLast {
            (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: 11, weight: .bold)))
                .font(.system(size: size, weight: weight))
                .foregroundColor(.primary)
                .padding(.top, topPad)
                .padding(.bottom, botPad)
                .textSelection(.enabled)
        } else {
            Text(LocalizedStringKey(text))
                .font(.system(size: size, weight: weight))
                .foregroundColor(.primary)
                .padding(.top, topPad)
                .padding(.bottom, botPad)
                .textSelection(.enabled)
        }
    }

    @ViewBuilder
    private func renderParagraph(text: String, isLast: Bool) -> some View {
        if isLast {
            (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: 11, weight: .bold)))
                .font(.system(size: 14))
                .lineSpacing(4)
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        } else {
            Text(LocalizedStringKey(text))
                .font(.system(size: 14))
                .lineSpacing(4)
                .foregroundColor(.primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    @ViewBuilder
    private func renderBlockquote(text: String, isLast: Bool) -> some View {
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(
                    LinearGradient(
                        colors: [.purple, .indigo],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 3.5)

            if isLast {
                (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: 11, weight: .bold)))
                    .font(.system(size: 13.5))
                    .italic()
                    .foregroundColor(.secondary)
                    .lineSpacing(3.5)
                    .textSelection(.enabled)
            } else {
                Text(LocalizedStringKey(text))
                    .font(.system(size: 13.5))
                    .italic()
                    .foregroundColor(.secondary)
                    .lineSpacing(3.5)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(Color.purple.opacity(0.04))
        .cornerRadius(6)
    }

    @ViewBuilder
    private func renderListItem(number: Int?, text: String, isLast: Bool) -> some View {
        HStack(alignment: .top, spacing: 8) {
            if let num = number {
                Text("\(num).")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(minWidth: 20, alignment: .trailing)
            } else {
                Text("•")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.purple)
                    .frame(width: 14, alignment: .center)
            }

            if isLast {
                (Text(LocalizedStringKey(text)) + Text(" ") + Text(Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")).foregroundColor(isStreamingOffDisk ? .orange : .purple).font(.system(size: 11, weight: .bold)))
                    .font(.system(size: 14))
                    .lineSpacing(3)
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
            } else {
                Text(LocalizedStringKey(text))
                    .font(.system(size: 14))
                    .lineSpacing(3)
                    .foregroundColor(.primary)
                    .textSelection(.enabled)
            }
        }
        .padding(.vertical, 1.5)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

// MARK: - Native Markdown Table Card View (SwiftUI Grid)

struct MarkdownTableView: View {
    let table: MarkdownTableData
    var isLast: Bool = false
    var isStreamingOffDisk: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ScrollView(.horizontal, showsIndicators: true) {
                Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                    // Header Row
                    GridRow {
                        ForEach(0..<table.headers.count, id: \.self) { colIdx in
                            let header = table.headers[colIdx]
                            let align = table.alignments.indices.contains(colIdx) ? table.alignments[colIdx] : .leading
                            HStack(spacing: 0) {
                                Text(LocalizedStringKey(header))
                                    .font(.system(size: 13, weight: .bold))
                                    .foregroundColor(.secondary)
                                    .multilineTextAlignment(align.textAlignment)
                                    .padding(.horizontal, 14)
                                    .padding(.vertical, 9)
                                    .frame(minWidth: 90, alignment: align.alignment)
                                    .textSelection(.enabled)

                                if colIdx < table.headers.count - 1 {
                                    Divider()
                                        .opacity(0.35)
                                }
                            }
                        }
                    }
                    .background(Color.secondary.opacity(0.08))

                    // Header Divider
                    GridRow {
                        Divider()
                            .gridCellColumns(max(1, table.headers.count))
                    }

                    // Data Rows
                    ForEach(0..<table.rows.count, id: \.self) { rowIdx in
                        let row = table.rows[rowIdx]
                        GridRow {
                            ForEach(0..<table.headers.count, id: \.self) { colIdx in
                                let cell = colIdx < row.count ? row[colIdx] : ""
                                let align = table.alignments.indices.contains(colIdx) ? table.alignments[colIdx] : .leading
                                HStack(spacing: 0) {
                                    Text(LocalizedStringKey(cell))
                                        .font(.system(size: 13))
                                        .foregroundColor(.primary)
                                        .multilineTextAlignment(align.textAlignment)
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 8)
                                        .frame(minWidth: 90, alignment: align.alignment)
                                        .textSelection(.enabled)

                                    if colIdx < table.headers.count - 1 {
                                        Divider()
                                            .opacity(0.25)
                                    }
                                }
                            }
                        }
                        .background(rowIdx % 2 == 1 ? Color.secondary.opacity(0.03) : Color.clear)

                        if rowIdx < table.rows.count - 1 {
                            GridRow {
                                Divider()
                                    .opacity(0.25)
                                    .gridCellColumns(max(1, table.headers.count))
                            }
                        }
                    }
                }
                .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
                .cornerRadius(10)
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.primary.opacity(0.08), lineWidth: 1)
                )
            }

            if isLast {
                HStack(spacing: 6) {
                    StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    if isStreamingOffDisk {
                        Text("Streaming MoE experts off NVMe disk...")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.orange)
                    }
                }
                .padding(.top, 2)
            }
        }
        .padding(.vertical, 6)
    }
}

// MARK: - Native Multi-Language Syntax Highlighter (Pure Swift AST / Token Engine)

public struct NativeSyntaxHighlighter {
    public static func highlight(code: String, language: String) -> AttributedString {
        let lang = language.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        
        let kwList: Set<String>
        let isPythonLike = (lang == "python" || lang == "py" || lang == "sh" || lang == "bash" || lang == "yaml" || lang == "yml")
        
        switch lang {
        case "swift":
            kwList = ["func", "var", "let", "struct", "class", "enum", "protocol", "extension", "guard", "if", "else", "for", "in", "while", "return", "import", "public", "private", "fileprivate", "internal", "open", "static", "final", "override", "mutating", "async", "await", "throws", "try", "catch", "throw", "switch", "case", "default", "break", "continue", "where", "init", "deinit", "some", "any", "typealias", "nil", "true", "false", "self", "Self"]
        case "python", "py":
            kwList = ["def", "class", "import", "from", "as", "return", "if", "elif", "else", "for", "while", "in", "try", "except", "finally", "with", "raise", "pass", "break", "continue", "lambda", "yield", "async", "await", "assert", "global", "nonlocal", "True", "False", "None", "self"]
        case "rust", "rs":
            kwList = ["fn", "let", "mut", "struct", "enum", "impl", "trait", "pub", "use", "mod", "crate", "return", "if", "else", "match", "for", "in", "while", "loop", "break", "continue", "async", "await", "move", "ref", "type", "const", "static", "where", "true", "false", "unsafe"]
        case "javascript", "js", "typescript", "ts":
            kwList = ["function", "const", "let", "var", "class", "interface", "type", "import", "export", "from", "return", "if", "else", "for", "while", "switch", "case", "default", "break", "continue", "async", "await", "try", "catch", "throw", "new", "this", "typeof", "instanceof", "true", "false", "null", "undefined"]
        case "c", "cpp", "c++", "metal":
            kwList = ["kernel", "device", "constant", "threadgroup", "thread", "void", "int", "float", "double", "char", "bool", "struct", "class", "return", "if", "else", "for", "while", "switch", "case", "break", "continue", "template", "typename", "namespace", "include", "typedef"]
        default:
            kwList = ["func", "def", "fn", "function", "var", "let", "const", "struct", "class", "return", "import", "if", "else", "for", "while", "true", "false", "null", "nil", "None"]
        }
        
        let typeList: Set<String> = ["Int", "String", "Bool", "Float", "Double", "UInt32", "UInt64", "Int32", "Int64", "Data", "URL", "View", "Text", "Color", "Array", "Dictionary", "Set", "Optional", "Result", "Task", "MainActor", "MTLDevice", "MTLBuffer", "MTLCommandQueue"]
        
        var result = AttributedString()
        let lines = code.components(separatedBy: "\n")
        
        for (lineIdx, line) in lines.enumerated() {
            var lineAttr = AttributedString()
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            let isCommentLine = isPythonLike ? trimmed.hasPrefix("#") : (trimmed.hasPrefix("//") || trimmed.hasPrefix("/*") || trimmed.hasPrefix("*"))
            
            if isCommentLine {
                var commentAttr = AttributedString(line)
                commentAttr.foregroundColor = Color.secondary.opacity(0.8)
                lineAttr.append(commentAttr)
            } else {
                var idx = line.startIndex
                while idx < line.endIndex {
                    let remaining = String(line[idx...])
                    
                    // 1. Comment till end of line
                    if (!isPythonLike && remaining.hasPrefix("//")) || (isPythonLike && remaining.hasPrefix("#")) {
                        var cAttr = AttributedString(remaining)
                        cAttr.foregroundColor = Color.secondary.opacity(0.8)
                        lineAttr.append(cAttr)
                        break
                    }
                    
                    // 2. String literal ("..." or '...' or `...`)
                    if remaining.hasPrefix("\"") || remaining.hasPrefix("'") || remaining.hasPrefix("`") {
                        let quote = remaining.first!
                        var endIdx = line.index(after: idx)
                        var escaped = false
                        while endIdx < line.endIndex {
                            let ch = line[endIdx]
                            if escaped {
                                escaped = false
                            } else if ch == "\\" {
                                escaped = true
                            } else if ch == quote {
                                endIdx = line.index(after: endIdx)
                                break
                            }
                            endIdx = line.index(after: endIdx)
                        }
                        let strContent = String(line[idx..<endIdx])
                        var strAttr = AttributedString(strContent)
                        strAttr.foregroundColor = Color.green
                        lineAttr.append(strAttr)
                        idx = endIdx
                        continue
                    }
                    
                    // 3. Word / Identifier
                    if let wordMatch = remaining.range(of: #"^[a-zA-Z_][a-zA-Z0-9_]*"#, options: .regularExpression) {
                        let word = String(remaining[wordMatch])
                        var wAttr = AttributedString(word)
                        if kwList.contains(word) {
                            wAttr.foregroundColor = Color.purple
                            wAttr.font = .system(size: 12.5, weight: .bold, design: .monospaced)
                        } else if typeList.contains(word) {
                            wAttr.foregroundColor = Color.teal
                            wAttr.font = .system(size: 12.5, weight: .semibold, design: .monospaced)
                        }
                        lineAttr.append(wAttr)
                        idx = line.index(idx, offsetBy: word.count)
                        continue
                    }
                    
                    // 4. Number
                    if let numMatch = remaining.range(of: #"^(0x[0-9a-fA-F]+|\d+(\.\d+)?)"#, options: .regularExpression) {
                        let num = String(remaining[numMatch])
                        var nAttr = AttributedString(num)
                        nAttr.foregroundColor = Color.orange
                        lineAttr.append(nAttr)
                        idx = line.index(idx, offsetBy: num.count)
                        continue
                    }
                    
                    // 5. Plain character
                    let char = remaining.first!
                    let charAttr = AttributedString(String(char))
                    lineAttr.append(charAttr)
                    idx = line.index(after: idx)
                }
            }
            
            result.append(lineAttr)
            if lineIdx < lines.count - 1 {
                result.append(AttributedString("\n"))
            }
        }
        
        return result
    }
}

// MARK: - Dedicated Code Block Card with Header, Syntax Highlighting & Copy Button

struct CodeBlockCard: View {
    let language: String
    let code: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false
    @State private var isCopied: Bool = false
    @State private var highlighted: AttributedString? = nil

    private var displayLanguage: String {
        let clean = language.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if clean.isEmpty { return "CODE" }
        return clean.uppercased()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header Bar
            HStack {
                HStack(spacing: 6) {
                    Image(systemName: "chevron.left.forwardslash.chevron.right")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundColor(.purple)

                    Text(displayLanguage)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .foregroundColor(.secondary)

                    if isStreaming {
                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                    }
                }

                Spacer()

                Button(action: copyCode) {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 10))
                        Text(isCopied ? "Copied!" : "Copy")
                            .font(.system(size: 10, weight: .medium))
                    }
                    .foregroundColor(isCopied ? .green : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.secondary.opacity(0.08))
                    .cornerRadius(4)
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color(NSColor.controlBackgroundColor))

            Divider()

            // Code Text Canvas with Native Syntax Highlighting (Cached)
            ScrollView(.horizontal, showsIndicators: true) {
                Text(highlighted ?? NativeSyntaxHighlighter.highlight(code: code, language: language))
                    .font(.system(size: 12.5, design: .monospaced))
                    .lineSpacing(3)
                    .textSelection(.enabled)
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(NSColor.textBackgroundColor))
        }
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
        )
        .padding(.vertical, 4)
        .onAppear {
            highlighted = NativeSyntaxHighlighter.highlight(code: code, language: language)
        }
        .onChange(of: code) { newCode in
            highlighted = NativeSyntaxHighlighter.highlight(code: newCode, language: language)
        }
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        isCopied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            isCopied = false
        }
    }
}

// MARK: - Streaming Pace Indicator (Bunny vs. Tortoise)

struct StreamingPaceIndicatorView: View {
    let isOffDisk: Bool
    @State private var isFading: Bool = false

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: isOffDisk ? "tortoise.fill" : "hare.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(
                    isOffDisk
                        ? LinearGradient(colors: [.orange, .yellow], startPoint: .topLeading, endPoint: .bottomTrailing)
                        : LinearGradient(colors: [.purple, .indigo, .cyan], startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .opacity(isFading ? 1.0 : 0.25)
                .scaleEffect(isFading ? 1.08 : 0.92)
        }
        .padding(.horizontal, 4)
        .padding(.vertical, 1)
        .background(
            (isOffDisk ? Color.orange : Color.purple)
                .opacity(isFading ? 0.12 : 0.03)
        )
        .cornerRadius(6)
        .onAppear {
            withAnimation(
                Animation.easeInOut(duration: 0.85)
                    .repeatForever(autoreverses: true)
            ) {
                isFading = true
            }
        }
        .help(isOffDisk ? "Streaming MoE experts off NVMe SSD (Paging)" : "Fast Streaming in Unified RAM (Dense / Resident)")
    }
}

// MARK: - Markdown Pre-processing Normalizer (Pre-compiled Regex Cache)

private enum MarkdownRegexCache {
    static let br = try? NSRegularExpression(pattern: #"<br\s*/?>"# , options: [.caseInsensitive])
    static let p = try? NSRegularExpression(pattern: #"</?p\b[^>]*>"#, options: [.caseInsensitive])
    static let div = try? NSRegularExpression(pattern: #"</?div\b[^>]*>"#, options: [.caseInsensitive])
    static let b = try? NSRegularExpression(pattern: #"<(?:b|strong)\b[^>]*>(.*?)</(?:b|strong)>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let i = try? NSRegularExpression(pattern: #"<(?:i|em)\b[^>]*>(.*?)</(?:i|em)>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let code = try? NSRegularExpression(pattern: #"<code\b[^>]*>(.*?)</code>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let li = try? NSRegularExpression(pattern: #"<li\b[^>]*>(.*?)(?:</li>|$)"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let ul = try? NSRegularExpression(pattern: #"</?(?:ul|ol)\b[^>]*>"#, options: [.caseInsensitive])
    static let span = try? NSRegularExpression(pattern: #"<span\b[^>]*>(.*?)</span>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let sup = try? NSRegularExpression(pattern: #"<sup>(.*?)</sup>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let sub = try? NSRegularExpression(pattern: #"<sub>(.*?)</sub>"#, options: [.caseInsensitive, .dotMatchesLineSeparators])
    static let strayHtml = try? NSRegularExpression(pattern: #"</?[a-zA-Z][^>]*>"#, options: [])
    static let stuckHeading = try? NSRegularExpression(pattern: #"([^\n#])\s*(#{1,6}\s+)"#, options: [])
    static let headingGlue = try? NSRegularExpression(pattern: #"(#{1,6}\s+[^\n]+?)([a-z])([A-Z]{2,}|\b(?:BF16|FP8|FP16|FP32|INT8|GPU|CPU|NPU|TPU)\b)"#, options: [])
    static let inlineList = try? NSRegularExpression(pattern: #"([^\n])\s*(?:-\s+|•\s+|\*\s+)([A-Z0-9])"#, options: [])
    static let inlineNumList = try? NSRegularExpression(pattern: #"([.:;?!])\s*(\d{1,2}\.\s+[A-Z])"#, options: [])
    static let sentenceGlue = try? NSRegularExpression(pattern: #"([a-z0-9])\.([A-Z])"#, options: [])
    static let titleTable = try? NSRegularExpression(pattern: #"(?:^|\n)([^|\n#]+?)\s*\|\s*([A-Za-z0-9].*\|[^\n]*)"#, options: [])
    static let fixSepPipes = try? NSRegularExpression(pattern: #"(?:^|\n)(\|[|:\- ]+[-:])(?=\n|$)"#, options: [])
    static let multiSep = try? NSRegularExpression(pattern: #"(\|[|:\- ]+\|)\s*\n\s*(\|[|:\- ]+\|)"#, options: [])
    static let sectionHeader = try? NSRegularExpression(pattern: #"(?:^|\n)\s*--\s*([^-\n]+?)\s*--\s*\|?"#, options: [.anchorsMatchLines])
    static let hrHeader = try? NSRegularExpression(pattern: #"---+[\t ]*(#{1,6}\s*)"#, options: [])
    static let stuckBullet = try? NSRegularExpression(pattern: #"([^\n])(•\s+[A-Za-z0-9])"#, options: [])
    static let multiNl = try? NSRegularExpression(pattern: #"\n{3,}"#, options: [])
}

public func normalizeMarkdownText(_ raw: String) -> String {
    var text = raw
    text = text.replacingOccurrences(of: "\r\n", with: "\n")
    text = text.replacingOccurrences(of: "\r", with: "\n")
    text = text.replacingOccurrences(of: "\\n", with: "\n")

    // 1. Convert HTML Line breaks & Paragraphs to newlines
    if let regexBr = MarkdownRegexCache.br {
        text = regexBr.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n")
    }
    if let regexP = MarkdownRegexCache.p {
        text = regexP.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
    }
    if let regexDiv = MarkdownRegexCache.div {
        text = regexDiv.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
    }

    // 2. Convert HTML formatting tags to Markdown
    if let regexB = MarkdownRegexCache.b {
        text = regexB.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "**$1**")
    }
    if let regexI = MarkdownRegexCache.i {
        text = regexI.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "*$1*")
    }
    if let regexCode = MarkdownRegexCache.code {
        text = regexCode.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "`$1`")
    }

    // 3. Convert HTML Lists
    if let regexLi = MarkdownRegexCache.li {
        text = regexLi.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n- $1\n")
    }
    if let regexUl = MarkdownRegexCache.ul {
        text = regexUl.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n")
    }

    // 4. Strip span and convert sup/sub tags
    if let regexSpan = MarkdownRegexCache.span {
        text = regexSpan.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1")
    }
    if let regexSup = MarkdownRegexCache.sup {
        text = regexSup.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "^($1)")
    }
    if let regexSub = MarkdownRegexCache.sub {
        text = regexSub.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "_($1)")
    }
    if let regexStrayHtml = MarkdownRegexCache.strayHtml {
        text = regexStrayHtml.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "")
    }

    // 5. Clean double pipes || -> |
    while text.contains("||") {
        text = text.replacingOccurrences(of: "||", with: "|")
    }

    // 6. Fix heading embedded in paragraph or missing space: "text. #### 1. Heading" -> "text.\n\n#### 1. Heading"
    if let regexStuckHeading = MarkdownRegexCache.stuckHeading {
        text = regexStuckHeading.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n$2")
    }

    // 7. Fix heading title glued to paragraph text:
    if let regexHeadingGlue = MarkdownRegexCache.headingGlue {
        text = regexHeadingGlue.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1$2\n\n$3")
    }

    // 8. Fix inline list glued inside paragraph: "formats: - Total 16 bits" -> "formats:\n\n- Total 16 bits"
    if let regexInlineList = MarkdownRegexCache.inlineList {
        text = regexInlineList.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n- $2")
    }

    // 9. Fix inline numbered list glued inside paragraph: "computing. 1. Technical" -> "computing.\n\n1. Technical"
    if let regexInlineNumList = MarkdownRegexCache.inlineNumList {
        text = regexInlineNumList.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n\n$2")
    }

    // 10. Fix sentence glued to next sentence without space: "quantization.This" -> "quantization. This"
    if let regexSentenceGlue = MarkdownRegexCache.sentenceGlue {
        text = regexSentenceGlue.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1. $2")
    }

    // 11. Fix title stuck before table header on same line
    if let regexTitleTable = MarkdownRegexCache.titleTable {
        text = regexTitleTable.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n### $1\n| $2")
    }

    // 12. Fix unclosed table separator line (missing trailing pipe)
    if let regexFixSepPipes = MarkdownRegexCache.fixSepPipes {
        text = regexFixSepPipes.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n$1|")
    }

    // 13. Fix multiple table separator lines
    if let regexMultiSep = MarkdownRegexCache.multiSep {
        text = regexMultiSep.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1")
    }

    // 14. Fix table section headers like "-- Section Name -- |"
    if let regexSectionHeader = MarkdownRegexCache.sectionHeader {
        text = regexSectionHeader.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n#### $1\n")
    }

    // 15. Fix concatenated horizontal rules + headers: "---##" -> "\n\n---\n\n## "
    if let regexHrHeader = MarkdownRegexCache.hrHeader {
        text = regexHrHeader.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n---\n\n$1")
    }

    // 16. Fix bullet points stuck to preceding text: "text• Bullet" -> "text\n• Bullet"
    if let regexStuckBullet = MarkdownRegexCache.stuckBullet {
        text = regexStuckBullet.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "$1\n$2")
    }

    // 17. Collapse 3+ newlines to 2
    if let regexMultiNl = MarkdownRegexCache.multiNl {
        text = regexMultiNl.stringByReplacingMatches(in: text, options: [], range: NSRange(location: 0, length: text.utf16.count), withTemplate: "\n\n")
    }

    return text
}

// MARK: - Table Line Parser

private func parseMarkdownTable(lines: [String], startIndex: Int) -> (table: MarkdownTableData, nextIndex: Int)? {
    guard startIndex + 1 < lines.count else { return nil }
    let headerLine = lines[startIndex].trimmingCharacters(in: .whitespaces)
    let separatorLine = lines[startIndex + 1].trimmingCharacters(in: .whitespaces)

    guard headerLine.contains("|") else { return nil }

    let separatorChars = CharacterSet(charactersIn: "|- :—–â")
    guard separatorLine.contains("|") && separatorLine.contains("-") && separatorLine.unicodeScalars.allSatisfy({ separatorChars.contains($0) }) else {
        return nil
    }

    func splitCells(_ line: String) -> [String] {
        var trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("|") { trimmed.removeFirst() }
        if trimmed.hasSuffix("|") { trimmed.removeLast() }
        return trimmed.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
    }

    let headers = splitCells(headerLine)
    guard headers.count >= 2 else { return nil }
    let separatorCells = splitCells(separatorLine)

    var alignments: [TextAlignmentType] = []
    for colIdx in 0..<headers.count {
        if colIdx < separatorCells.count {
            let cell = separatorCells[colIdx]
            let hasLeadingColon = cell.hasPrefix(":")
            let hasTrailingColon = cell.hasSuffix(":")
            if hasLeadingColon && hasTrailingColon {
                alignments.append(.center)
            } else if hasTrailingColon {
                alignments.append(.trailing)
            } else {
                alignments.append(.leading)
            }
        } else {
            alignments.append(.leading)
        }
    }

    var rows: [[String]] = []
    var curIdx = startIndex + 2
    while curIdx < lines.count {
        let line = lines[curIdx].trimmingCharacters(in: .whitespaces)
        if line.isEmpty || !line.contains("|") {
            break
        }
        var cells = splitCells(line)
        if cells.count < headers.count {
            cells.append(contentsOf: Array(repeating: "", count: headers.count - cells.count))
        } else if cells.count > headers.count {
            cells = Array(cells.prefix(headers.count))
        }
        rows.append(cells)
        curIdx += 1
    }

    return (MarkdownTableData(headers: headers, alignments: alignments, rows: rows), curIdx)
}

// MARK: - Full Markdown Block Parsing Engine

public func parseMarkdownBlocks(_ raw: String) -> [MarkdownBlock] {
    let normalized = normalizeMarkdownText(raw)
    var blocks: [MarkdownBlock] = []
    let lines = normalized.components(separatedBy: "\n")

    var inCodeBlock = false
    var codeLang = ""
    var codeLines: [String] = []

    var paragraphLines: [String] = []

    func flushParagraph() {
        if !paragraphLines.isEmpty {
            let combined = paragraphLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !combined.isEmpty {
                blocks.append(.paragraph(text: combined))
            }
            paragraphLines.removeAll()
        }
    }

    var i = 0
    while i < lines.count {
        let line = lines[i]
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // 1. Code Block Fence
        if trimmed.hasPrefix("```") {
            if inCodeBlock {
                blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
                inCodeBlock = false
                codeLang = ""
                codeLines.removeAll()
            } else {
                flushParagraph()
                inCodeBlock = true
                codeLang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespacesAndNewlines)
                codeLines.removeAll()
            }
            i += 1
            continue
        }

        if inCodeBlock {
            codeLines.append(line)
            i += 1
            continue
        }

        // 2. Table detection
        if !inCodeBlock && trimmed.contains("|") {
            if let tableResult = parseMarkdownTable(lines: lines, startIndex: i) {
                flushParagraph()
                blocks.append(.table(tableResult.table))
                i = tableResult.nextIndex
                continue
            }
        }

        // 3. Headings (#, ##, ###, etc.)
        if trimmed.hasPrefix("#") {
            flushParagraph()
            let hashCount = trimmed.prefix(while: { $0 == "#" }).count
            let headerText = String(trimmed.dropFirst(hashCount)).trimmingCharacters(in: .whitespaces)
            blocks.append(.heading(level: hashCount, text: headerText))
            i += 1
            continue
        }

        // 4. Blockquotes
        if trimmed.hasPrefix("> ") || trimmed == ">" {
            flushParagraph()
            var quoteLines: [String] = []
            while i < lines.count {
                let qLine = lines[i].trimmingCharacters(in: .whitespaces)
                if qLine.hasPrefix("> ") {
                    quoteLines.append(String(qLine.dropFirst(2)))
                    i += 1
                } else if qLine == ">" {
                    quoteLines.append("")
                    i += 1
                } else {
                    break
                }
            }
            blocks.append(.blockquote(text: quoteLines.joined(separator: "\n")))
            continue
        }

        // 5. Horizontal Rule
        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            flushParagraph()
            blocks.append(.horizontalRule)
            i += 1
            continue
        }

        // 6. Bullet list item
        if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("+ ") || trimmed.hasPrefix("• ") {
            flushParagraph()
            let itemText: String
            if trimmed.hasPrefix("• ") {
                itemText = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            } else {
                itemText = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            }
            blocks.append(.listItem(number: nil, text: itemText))
            i += 1
            continue
        }

        // 7. Numbered list item
        if let match = trimmed.range(of: #"^\d+\.\s+"#, options: .regularExpression) {
            flushParagraph()
            let itemText = String(trimmed[match.upperBound...]).trimmingCharacters(in: .whitespaces)
            let numStr = String(trimmed[..<match.upperBound]).trimmingCharacters(in: .whitespaces.union(CharacterSet(charactersIn: ".")))
            blocks.append(.listItem(number: Int(numStr), text: itemText))
            i += 1
            continue
        }

        // 8. Empty line
        if trimmed.isEmpty {
            flushParagraph()
            i += 1
            continue
        }

        paragraphLines.append(line)
        i += 1
    }

    if inCodeBlock {
        blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
    } else {
        flushParagraph()
    }

    return blocks
}
