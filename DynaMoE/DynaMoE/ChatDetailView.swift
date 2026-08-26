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
    var onSendMessage: (String) -> Void
    var onStopGeneration: () -> Void
    var onSelectPromptStarter: (String) -> Void
    var onSelectDiscoveredModel: ((DiscoveredModel) -> Void)? = nil
    var onOpenSettings: (() -> Void)? = nil

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
                                        isGenerating: isGenerating,
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
                                        StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
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

                    // Main Response Text rendered via Markdown Engine
                    if !message.content.isEmpty {
                        MarkdownMessageView(
                            content: message.content,
                            isStreaming: isGenerating && message.isThinking == false,
                            isStreamingOffDisk: isStreamingOffDisk
                        )
                    } else if message.isThinking {
                        // Empty placeholder while purely thinking
                        HStack(spacing: 8) {
                            StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
                            Text(isStreamingOffDisk ? "Streaming MoE experts off SSD disk..." : "Thinking & generating response...")
                                .font(.system(size: 13))
                                .foregroundColor(.secondary)
                        }
                        .padding(.vertical, 4)
                    }

                    // Action & Metrics Footer (Antigravity Reactions + Metrics)
                    if !message.isThinking && !message.content.isEmpty {
                        HStack(spacing: 12) {
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
                            }
                            .buttonStyle(.plain)
                            .help("Good response")

                            Button(action: {
                                feedback = feedback == "down" ? nil : "down"
                            }) {
                                Image(systemName: feedback == "down" ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                                    .font(.system(size: 11))
                                    .foregroundColor(feedback == "down" ? .purple : .secondary)
                            }
                            .buttonStyle(.plain)
                            .help("Bad response")

                            if message.tokenCount > 0 {
                                Text("•")
                                    .foregroundColor(.secondary.opacity(0.4))
                                Text("\(message.tokenCount) tokens")
                                    .font(.system(size: 11))
                                    .foregroundColor(.secondary)
                            }

                            if message.tokensPerSec > 0 {
                                Text("•")
                                    .foregroundColor(.secondary.opacity(0.4))
                                HStack(spacing: 3) {
                                    Image(systemName: isStreamingOffDisk ? "tortoise.fill" : "hare.fill")
                                        .font(.system(size: 10, weight: .bold))
                                        .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                    Text(String(format: "%.1f tok/s", message.tokensPerSec))
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(isStreamingOffDisk ? .orange : .purple)
                                }
                            }

                            Spacer()
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

// MARK: - Markdown Rendering Engine (Osaurus Style)

enum MarkdownBlock: Identifiable {
    case heading(level: Int, text: String)
    case paragraph(text: String)
    case codeBlock(language: String, code: String)
    case blockquote(text: String)
    case listItem(number: Int?, text: String)
    case horizontalRule

    var id: String {
        switch self {
        case .heading(let lvl, let txt): return "h-\(lvl)-\(txt.hashValue)"
        case .paragraph(let txt): return "p-\(txt.hashValue)"
        case .codeBlock(let lang, let code): return "code-\(lang)-\(code.hashValue)"
        case .blockquote(let txt): return "quote-\(txt.hashValue)"
        case .listItem(let num, let txt): return "li-\(num ?? 0)-\(txt.hashValue)"
        case .horizontalRule: return "hr-\(UUID().uuidString)"
        }
    }
}

struct MarkdownMessageView: View {
    let content: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false

    private var blocks: [MarkdownBlock] {
        parseMarkdownBlocks(content)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { index, block in
                switch block {
                case .heading(let level, let text):
                    renderHeading(level: level, text: text)
                case .paragraph(let text):
                    renderParagraph(text: text, isLast: isStreaming && index == blocks.count - 1)
                case .codeBlock(let language, let code):
                    CodeBlockCard(
                        language: language,
                        code: code,
                        isStreaming: isStreaming && index == blocks.count - 1,
                        isStreamingOffDisk: isStreamingOffDisk
                    )
                case .blockquote(let text):
                    renderBlockquote(text: text)
                case .listItem(let number, let text):
                    renderListItem(number: number, text: text, isLast: isStreaming && index == blocks.count - 1)
                case .horizontalRule:
                    Divider()
                        .padding(.vertical, 4)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Block Renderers

    @ViewBuilder
    private func renderHeading(level: Int, text: String) -> some View {
        let size: CGFloat = level == 1 ? 18 : (level == 2 ? 16 : 14.5)
        let weight: Font.Weight = level <= 2 ? .bold : .semibold

        Text(LocalizedStringKey(text))
            .font(.system(size: size, weight: weight))
            .foregroundColor(.primary)
            .padding(.top, level == 1 ? 8 : 4)
            .padding(.bottom, 2)
            .textSelection(.enabled)
    }

    @ViewBuilder
    private func renderParagraph(text: String, isLast: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            Text(LocalizedStringKey(text))
                .font(.system(size: 14))
                .lineSpacing(4)
                .foregroundColor(.primary)
                .textSelection(.enabled)

            if isLast {
                StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
            }
        }
    }

    @ViewBuilder
    private func renderBlockquote(text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            RoundedRectangle(cornerRadius: 1.5)
                .fill(
                    LinearGradient(
                        colors: [.purple, .indigo],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
                .frame(width: 3)

            Text(LocalizedStringKey(text))
                .font(.system(size: 13.5))
                .italic()
                .foregroundColor(.secondary)
                .lineSpacing(3)
                .textSelection(.enabled)
        }
        .padding(.vertical, 4)
        .padding(.horizontal, 4)
    }

    @ViewBuilder
    private func renderListItem(number: Int?, text: String, isLast: Bool = false) -> some View {
        HStack(alignment: .top, spacing: 8) {
            if let num = number {
                Text("\(num).")
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundColor(.secondary)
                    .frame(minWidth: 20, alignment: .trailing)
            } else {
                Text("•")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.purple)
                    .frame(width: 14, alignment: .center)
            }

            Text(LocalizedStringKey(text))
                .font(.system(size: 14))
                .lineSpacing(3)
                .foregroundColor(.primary)
                .textSelection(.enabled)

            if isLast {
                StreamingPaceIndicatorView(isOffDisk: isStreamingOffDisk)
            }
        }
        .padding(.vertical, 1)
    }
}

// MARK: - Dedicated Code Block Card with Header & Copy Button

struct CodeBlockCard: View {
    let language: String
    let code: String
    var isStreaming: Bool = false
    var isStreamingOffDisk: Bool = false
    @State private var isCopied: Bool = false

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

            // Code Text Canvas
            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(.system(size: 12.5, design: .monospaced))
                    .lineSpacing(3)
                    .foregroundColor(.primary)
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

// MARK: - Markdown Block Parsing Engine

func parseMarkdownBlocks(_ raw: String) -> [MarkdownBlock] {
    var blocks: [MarkdownBlock] = []
    let lines = raw.components(separatedBy: "\n")

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

    for line in lines {
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        if trimmed.hasPrefix("```") {
            if inCodeBlock {
                // Close code block
                blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
                inCodeBlock = false
                codeLang = ""
                codeLines.removeAll()
            } else {
                // Open code block
                flushParagraph()
                inCodeBlock = true
                codeLang = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespacesAndNewlines)
                codeLines.removeAll()
            }
            continue
        }

        if inCodeBlock {
            codeLines.append(line)
            continue
        }

        if trimmed.hasPrefix("#") {
            flushParagraph()
            let hashCount = trimmed.prefix(while: { $0 == "#" }).count
            let headerText = String(trimmed.dropFirst(hashCount)).trimmingCharacters(in: .whitespaces)
            blocks.append(.heading(level: hashCount, text: headerText))
            continue
        }

        if trimmed.hasPrefix("> ") || trimmed == ">" {
            flushParagraph()
            let quoteText = trimmed.hasPrefix("> ") ? String(trimmed.dropFirst(2)) : ""
            blocks.append(.blockquote(text: quoteText))
            continue
        }

        if trimmed == "---" || trimmed == "***" || trimmed == "___" {
            flushParagraph()
            blocks.append(.horizontalRule)
            continue
        }

        // Bullet list item
        if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("+ ") {
            flushParagraph()
            let itemText = String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)
            blocks.append(.listItem(number: nil, text: itemText))
            continue
        }

        // Numbered list item
        if let match = trimmed.range(of: #"^\d+\.\s+"#, options: .regularExpression) {
            flushParagraph()
            let itemText = String(trimmed[match.upperBound...]).trimmingCharacters(in: .whitespaces)
            let numStr = String(trimmed[..<match.upperBound]).trimmingCharacters(in: .whitespaces.union(CharacterSet(charactersIn: ".")))
            blocks.append(.listItem(number: Int(numStr), text: itemText))
            continue
        }

        if trimmed.isEmpty {
            flushParagraph()
            continue
        }

        paragraphLines.append(line)
    }

    // Handle unclosed code block during active streaming
    if inCodeBlock {
        blocks.append(.codeBlock(language: codeLang, code: codeLines.joined(separator: "\n")))
    } else {
        flushParagraph()
    }

    return blocks
}
