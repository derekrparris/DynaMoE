//
//  HelpAndSettingsGuideView.swift
//  DynaMoE
//
//  Created by Derek Parris on 9/3/26.
//

import SwiftUI
import AppKit

enum HelpTopic: String, CaseIterable, Identifiable {
    case quickStart = "Quick Start & Overview"
    case models = "Models & Repackaging"
    case profiles = "Model Profiles (Coder vs. Assistant)"
    case generation = "Generation & Sampling"
    case memory = "Memory & SSD Streaming"
    case kvCache = "KV Cache Precision"
    case jetSpec = "JetSpec Acceleration"
    case agent = "Agent & Tool Execution"
    case diagnostics = "Advanced Diagnostics"
    case hardware = "Hardware Profiles (16GB - 128GB)"
    case faq = "Troubleshooting & FAQs"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .quickStart: return "sparkles"
        case .models: return "square.stack.3d.up.fill"
        case .profiles: return "person.crop.circle.badge.checkmark"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .kvCache: return "externaldrive.badge.icloud"
        case .jetSpec: return "bolt.fill"
        case .agent: return "wrench.and.screwdriver.fill"
        case .diagnostics: return "waveform.path.ecg"
        case .hardware: return "laptopcomputer"
        case .faq: return "questionmark.circle.fill"
        }
    }
}

struct HelpAndSettingsGuideView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTopic: HelpTopic = .quickStart
    @State private var searchText: String = ""
    @State private var copiedNotice: String? = nil

    var body: some View {
        NavigationSplitView {
            // Sidebar Topics
            List(selection: $selectedTopic) {
                Section("Help Topics") {
                    ForEach(filteredTopics) { topic in
                        NavigationLink(value: topic) {
                            Label(topic.rawValue, systemImage: topic.icon)
                                .font(.system(size: 13, weight: .medium))
                                .padding(.vertical, 2)
                        }
                    }
                }
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 240, ideal: 270, max: 320)
            .searchable(text: $searchText, placement: .sidebar, prompt: "Search Help...")
        } detail: {
            VStack(spacing: 0) {
                // Detail Header
                HStack(spacing: 12) {
                    Label(selectedTopic.rawValue, systemImage: selectedTopic.icon)
                        .font(.system(size: 17, weight: .bold))
                        .foregroundColor(.primary)

                    Spacer()

                    // Quick Action: Open Settings
                    Button(action: {
                        NotificationCenter.default.post(name: .openDynaMoESettings, object: nil)
                    }) {
                        Label("Open Settings…", systemImage: "gearshape")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                    .help("Open DynaMoE Settings Sheet (⌘,)")

                    // Quick Preset Copy Button
                    Menu {
                        Button("Copy Coder Preset JSON") {
                            copyPreset(type: .coder)
                        }
                        Button("Copy Assistant Preset JSON") {
                            copyPreset(type: .assistant)
                        }
                    } label: {
                        Label(copiedNotice ?? "Copy Preset", systemImage: copiedNotice != nil ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.6))

                Divider()

                // Topic Content Area
                ScrollView {
                    VStack(alignment: .leading, spacing: 20) {
                        topicView(for: selectedTopic)
                    }
                    .padding(28)
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .background(Color(NSColor.windowBackgroundColor))
        }
        .frame(minWidth: 800, idealWidth: 920, minHeight: 580, idealHeight: 700)
    }

    private var filteredTopics: [HelpTopic] {
        if searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return HelpTopic.allCases
        }
        return HelpTopic.allCases.filter {
            $0.rawValue.localizedCaseInsensitiveContains(searchText)
        }
    }

    private func copyPreset(type: ModelProfileType) {
        let preset = GenerationProfileSettings.defaultFor(type: type)
        let jsonStr = """
        {
          "profile": "\(type.rawValue)",
          "temperature": \(preset.temperature),
          "topP": \(preset.topP),
          "minP": \(preset.minP),
          "topK": \(preset.topK),
          "repetitionPenalty": \(preset.repetitionPenalty),
          "presencePenalty": \(preset.presencePenalty),
          "maxTokens": \(preset.maxNewTokens),
          "jetSpecEnabled": \(preset.jetSpecEnabled)
        }
        """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(jsonStr, forType: .string)
        withAnimation {
            copiedNotice = "Copied \(type.rawValue.capitalized)!"
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation {
                copiedNotice = nil
            }
        }
    }

    // MARK: - Topic Views
    @ViewBuilder
    private func topicView(for topic: HelpTopic) -> some View {
        switch topic {
        case .quickStart:
            quickStartSection
        case .models:
            modelsSection
        case .profiles:
            profilesSection
        case .generation:
            generationSection
        case .memory:
            memorySection
        case .kvCache:
            kvCacheSection
        case .jetSpec:
            jetSpecSection
        case .agent:
            agentSection
        case .diagnostics:
            diagnosticsSection
        case .hardware:
            hardwareSection
        case .faq:
            faqSection
        }
    }

    // MARK: - 1. Quick Start & Overview
    private var quickStartSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Getting Started", color: .blue)

            Text("Welcome to DynaMoE")
                .font(.title2).bold()

            Text("DynaMoE is a native macOS application, local inference engine, and Model Context Protocol (MCP) server engineered specifically for Apple Silicon's Unified Memory Architecture (UMA). It enables you to run models that exceed physical system RAM by dynamically memory-mapping and streaming weights directly from high-speed NVMe storage to the GPU.")
                .font(.body)
                .lineSpacing(4)

            infoBox(title: "Keyboard Shortcuts", icon: "command") {
                VStack(alignment: .leading, spacing: 6) {
                    shortcutRow("⌘,", "Open Settings & Diagnostics")
                    shortcutRow("⌘?", "Open this Help & Settings Guide")
                    shortcutRow("⌘N", "New Chat Session")
                    shortcutRow("⌘0", "Actual Size (Reset Zoom)")
                    shortcutRow("⌘+", "Zoom In")
                    shortcutRow("⌘-", "Zoom Out")
                }
            }

            Text("Standard Workflow")
                .font(.headline).bold()

            VStack(alignment: .leading, spacing: 10) {
                stepRow(1, "Select your Model", "Open Settings (⌘,) and select a model from your Hugging Face cache (~/.cache/huggingface/hub) or pick one from the chat dropdown.")
                stepRow(2, "Choose your Profile", "Select either the 'Coder' or 'Assistant' profile from the toolbar menu to apply tuned sampling parameters for your task.")
                stepRow(3, "Generate & Converse", "Enter your prompt. DynaMoE renders real-time collapsible thinking chains (<think>), formatted Markdown, and syntax-highlighted code.")
            }
        }
    }

    // MARK: - 2. Models & Repackaging
    private var modelsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Tab 1: Models", color: .purple)

            Text("Local Model Registry & Fast Repackaging")
                .font(.title2).bold()

            Text("DynaMoE automatically scans `~/.cache/huggingface/hub` on startup to detect downloaded models, snapshots, weight formats, and quantizations.")
                .font(.body)

            infoBox(title: "Contiguous Binary Repackaging (Flash-MoE)", icon: "shippingbox.fill") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Sparse MoE models (such as Qwen 3.8 Flash Next with 512 experts) distribute weights across dozens of multi-gigabyte SafeTensors shards. During SSD streaming, loading non-contiguous expert slices causes random file seeks, severely limiting NVMe throughput.")
                        .font(.system(size: 12))
                        .lineSpacing(2)

                    Text("Clicking 'Repack Model for Fast Streaming' restructures expert weights into contiguous per-layer binaries (`layer_XX.bin` + `layout.json`). An 8-thread POSIX `pread` pool can then stream active expert slices in a single contiguous I/O read (>3.5 GB/s on Apple internal SSDs).")
                        .font(.system(size: 12))
                        .lineSpacing(2)

                    Text("Note: Dense models (like Ornith 1.5 9B) fit into memory and do not require binary repackaging.")
                        .font(.system(size: 11, weight: .semibold))
                        .foregroundColor(.secondary)
                }
            }

            Text("Model Management Features")
                .font(.headline).bold()

            VStack(alignment: .leading, spacing: 8) {
                bulletPoint("Auto-Discovery", "Automatically parses config.json and SafeTensors headers to detect architecture, hidden dimensions, heads, and quantization.")
                bulletPoint("Default Model", "Click 'Make Default' on any discovered model to load it automatically on future launches.")
                bulletPoint("Last Used Retention", "DynaMoE remembers the last used model and automatically restores it across sessions.")
            }
        }
    }

    // MARK: - 3. Model Profiles (Coder vs. Assistant)
    private var profilesSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Model Profiles", color: .green)

            Text("Official Architecture Profiles: Ornith & Qwen")
                .font(.title2).bold()

            Text("Different tasks and model architectures require fundamentally different sampling dynamics. DynaMoE automatically defaults to and persists the official publisher-recommended generation profiles per model family:")
                .font(.body)

            // Ornith Profiles
            Text("Ornith-1.5-9B Recommended Profiles")
                .font(.headline)
                .padding(.top, 4)

            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "curlybraces")
                            .foregroundColor(.blue)
                        Text("Precise Coding & Tool Calling")
                            .font(.headline).bold()
                    }
                    Text("• Temperature: 0.60 (low entropy)\n• Top-P: 0.95 | Top-K: 20\n• Min-P: 0.00\n• Repetition Penalty: 1.00\n• Presence Penalty: 0.00 (strict syntax)\n• Max Tokens: 8,192\n• JetSpec: Disabled (linear recurrence)")
                        .font(.system(size: 12, design: .monospaced))
                        .lineSpacing(4)
                }
                .padding(14)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.blue.opacity(0.3), lineWidth: 1))

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "bubble.left.and.bubble.right.fill")
                            .foregroundColor(.orange)
                        Text("General Chat / Agent Loops")
                            .font(.headline).bold()
                    }
                    Text("• Temperature: 1.00 (creative pacing)\n• Top-P: 0.95 | Top-K: 20\n• Min-P: 0.00\n• Repetition Penalty: 1.00\n• Presence Penalty: 1.50 (fresh vocabulary)\n• Max Tokens: 4,096\n• JetSpec: Disabled (linear recurrence)")
                        .font(.system(size: 12, design: .monospaced))
                        .lineSpacing(4)
                }
                .padding(14)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.orange.opacity(0.3), lineWidth: 1))
            }

            // Qwen Profiles
            Text("Qwen 3.8 Flash Next Recommended Profiles")
                .font(.headline)
                .padding(.top, 4)

            HStack(alignment: .top, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "brain.head.profile")
                            .foregroundColor(.purple)
                        Text("Coding & Agentic (Thinking Mode)")
                            .font(.headline).bold()
                    }
                    Text("• Temperature: 1.00 (high entropy)\n• Top-P: 0.95 | Top-K: 20\n• Min-P: 0.00\n• Repetition Penalty: 1.00\n• Presence Penalty: 0.00 (code consistency)\n• Max Tokens: 8,192\n• JetSpec: Enabled (Speculative Draft)")
                        .font(.system(size: 12, design: .monospaced))
                        .lineSpacing(4)
                }
                .padding(14)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.purple.opacity(0.3), lineWidth: 1))

                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Image(systemName: "bolt.fill")
                            .foregroundColor(.green)
                        Text("General Assistant (Instruct Mode)")
                            .font(.headline).bold()
                    }
                    Text("• Temperature: 0.70 (balanced focus)\n• Top-P: 0.80 | Top-K: 20\n• Min-P: 0.00\n• Repetition Penalty: 1.00\n• Presence Penalty: 1.50 (fresh vocabulary)\n• Max Tokens: 4,096\n• JetSpec: Enabled (Speculative Draft)")
                        .font(.system(size: 12, design: .monospaced))
                        .lineSpacing(4)
                }
                .padding(14)
                .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
                .cornerRadius(10)
                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.green.opacity(0.3), lineWidth: 1))
            }

            infoBox(title: "Why Repetition Penalty is 1.00 for Coding", icon: "exclamationmark.triangle.fill") {
                Text("Programming languages require repetitive keywords, variable names, indentation, and structural syntax (e.g. braces). Setting repetition penalty > 1.0 causes the model to invent erroneous synonyms for identifiers and omit closing brackets. For code, keep repetition penalty at 1.00.")
                    .font(.system(size: 12))
            }
        }
    }

    // MARK: - 4. Generation & Sampling Hyperparameters
    private var generationSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Tab 2: Generation", color: .orange)

            Text("Sampling Mathematics & Parameters")
                .font(.title2).bold()

            Text("DynaMoE uses vectorized Apple Accelerate routines (vDSP_maxvi, vvexpf) to sample tokens with zero CPU-bound overhead.")
                .font(.body)

            VStack(alignment: .leading, spacing: 12) {
                paramCard("Temperature (T)", "Controls distribution sharpness over the vocabulary. Lower values (0.0-0.4) produce deterministic, highly focused outputs. Higher values (0.7-1.0) promote diversity.")
                paramCard("Top-P (Nucleus)", "Restricts candidate tokens to the smallest subset whose cumulative probability exceeds P (typically 0.90-0.95). Discards unlikely hallucinations.")
                paramCard("Min-P (Dynamic Truncation)", "Filters tokens whose probability is less than Min-P * P_max. Unlike Top-P, Min-P automatically prunes aggressively when confident, and widens the pool when uncertain.")
                paramCard("Top-K", "Caps the maximum number of candidates evaluated during sampling (e.g. 20 for code, 50 for prose).")
                paramCard("Repetition Penalty (Multiplicative)", "Divides positive logits and multiplies negative logits of previously generated tokens by factor r (1.0 = disabled, 1.1 = moderate penalty).")
                paramCard("Presence Penalty (Additive)", "Subtracts a flat penalty from the logit of any unique token appearing in recent context (last 256 tokens). Encourages topic transitions without syntax distortion.")
            }

            infoBox(title: "System Prompt Conjunction Merging", icon: "link") {
                Text("DynaMoE automatically merges your custom system prompt in Settings with mandatory instruct personas required by models (e.g. Nanbeige, Ornith, DeepSeek-R1) into a single cohesive instruction block during inference.")
                    .font(.system(size: 12))
            }
        }
    }

    // MARK: - 5. Memory & SSD Management
    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Tab 3: Memory & SSD", color: .cyan)

            Text("Unified Memory Architecture (UMA) & SSD Paging")
                .font(.title2).bold()

            Text("DynaMoE is designed to run models larger than physical RAM by pairing Apple Silicon's high-bandwidth unified memory with NVMe SSD streaming.")
                .font(.body)

            VStack(alignment: .leading, spacing: 10) {
                bulletPoint("Auto (Smart) Mode", "Evaluates model footprint against total unified RAM. If >=25% headroom exists, it pins the model in memory. Otherwise, it automatically engages SSD streaming.")
                bulletPoint("Full RAM Mode", "Pre-faults and locks all weights into physical memory via posix_madvise(POSIX_MADV_WILLNEED) for maximum tokens/second without disk reads.")
                bulletPoint("SSD Streaming Mode", "Keeps embeddings, attention, and routing in RAM while streaming active MoE experts on demand from SSD storage using LRU page eviction.")
                bulletPoint("Speculative Lookahead Prefetching", "Asynchronously predicts upcoming layer expert activations and warms pages from disk before GPU execution begins.")
            }

            infoBox(title: "Cache Maintenance", icon: "arrow.triangle.2.circlepath") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("• Flush Cache: Evicts unpinned SSD pages from system RAM via posix_madvise(POSIX_MADV_DONTNEED) to immediately reduce memory pressure.")
                    Text("• Pre-Fault All: Sequentially reads all weights into system memory cache ahead of time for benchmark preparation.")
                }
                .font(.system(size: 12))
            }
        }
    }

    // MARK: - 6. KV Cache Precision
    private var kvCacheSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("KV Cache", color: .indigo)

            Text("Quantized KV Cache: FP32, FP16, and FP8")
                .font(.title2).bold()

            Text("During long-context autoregressive generation (e.g. 16k to 32k tokens), the Key-Value (KV) cache memory footprint can rival or exceed the model weights.")
                .font(.body)

            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    precisionBadge("FP32", "Full Precision", "Highest fidelity, largest memory footprint (~4x of FP8).")
                    precisionBadge("FP16", "Half Precision", "Default standard. 50% memory reduction with zero noticeable quality loss.")
                    precisionBadge("FP8 (E4M3)", "8-Bit Quantized", "75% memory reduction with per-head dynamic scaling. Enables 32k+ contexts on 16GB Macs.")
                }
            }

            infoBox(title: "Hardware Recommendation", icon: "info.circle") {
                Text("For 16 GB Apple Silicon Macs running contexts longer than 4,000 tokens, selecting FP8 KV Cache prevents macOS memory compression and swap thrashing.")
                    .font(.system(size: 12))
            }
        }
    }

    // MARK: - 7. JetSpec Acceleration
    private var jetSpecSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("JetSpec Tree Engine", color: .yellow)

            Text("Speculative Tree Drafting & Tree-Causal Verification")
                .font(.title2).bold()

            Text("JetSpec accelerates token generation by drafting structured candidate trees in parallel and scoring all nodes simultaneously using custom Metal tree-causal attention shaders.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                bulletPoint("Tree Depth (D)", "How many speculative sequential steps ahead to project (typically 2-4).")
                bulletPoint("Branching Factor (B)", "How many alternative candidate paths to expand per node (typically 2).")
                bulletPoint("Max Active Expert Cap (E_max)", "For MoE models, prunes branches that would require streaming more than E_max unique experts in a single step, protecting NVMe read bandwidth.")
            }

            infoBox(title: "When to Enable JetSpec", icon: "bolt.fill") {
                Text("JetSpec provides substantial speedups on dense autoregressive models and sparse MoE models with draft heads. For models with linear recurrent attention (like Ornith 1.5's Gated DeltaNet), JetSpec is disabled by default because state updates are already O(1) and branching states adds memory overhead.")
                    .font(.system(size: 12))
            }
        }
    }

    // MARK: - 8. Agent & Tool Execution
    private var agentSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Tab 4: Agent & Tools", color: .teal)

            Text("Autonomous Agent Engine & Function Calling")
                .font(.title2).bold()

            Text("DynaMoE includes an autonomous tool-calling engine that detects JSON function calls and executes local operations within a secure sandbox directory.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                bulletPoint("Working Directory Sandbox", "All file read/write and shell execution operations are constrained to your designated working directory to protect system security.")
                bulletPoint("Safety Limits", "Configure 'Max Tool Output Length' (to avoid blowing context limits) and 'Max Agent Steps' (to prevent runaway execution loops).")
                bulletPoint("Headless Chrome Web Search", "Zero-config, real-time web search powered by a local headless Chrome/Chromium instance executing on your Mac with DOM extraction and JavaScript SPA rendering.")
                bulletPoint("Brave Search API", "Optional enterprise search integration for high-volume structured search queries.")
            }

            infoBox(title: "Supported Local Tools", icon: "wrench.fill") {
                Text("• shell_run: Runs terminal commands in the sandbox\n• file_read: Reads files line-by-line\n• file_write: Creates or modifies code files\n• file_edit: Precise anchor-based search-and-replace\n• find_files / grep_search: File discovery & regex pattern matching\n• codebase_search: GPU vector & BM25 hybrid semantic search\n• spawn_subagent / list_subagents: Background multi-agent swarm orchestration\n• git_status / git_diff / git_commit: Full Git version control workflow\n• lint_diagnostics: Native compiler self-healing checks\n• web_search: Live web search via Headless Chrome / Brave\n• web_fetch: Fetches and cleans web pages with DOM rendering")
                    .font(.system(size: 12, design: .monospaced))
                    .lineSpacing(3)
            }
        }
    }

    // MARK: - 9. Advanced Diagnostics
    private var diagnosticsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Tab 5: Advanced Diagnostics", color: .red)

            Text("Metal Shaders & Tensor Inspector")
                .font(.title2).bold()

            Text("The Advanced Diagnostics panel allows developers and researchers to inspect SafeTensors shards and execute Metal compute kernels in isolation.")
                .font(.body)

            VStack(alignment: .leading, spacing: 8) {
                bulletPoint("Tensor Catalog", "Search through hundreds of tensors. Filter by Embedding, Attention, Routing, Experts, Normalization, or LM Head to inspect shapes, data types, and byte offsets.")
                bulletPoint("Execute Router Top-K", "Runs the Top-K gating shader to verify softmax routing probability distribution and expert index selection.")
                bulletPoint("Execute Full Layer", "Computes a complete forward pass through one transformer layer to measure execution time.")
                bulletPoint("Multi-Layer Test", "Chains N consecutive layers through double-buffered command encoders to profile raw GPU latency in milliseconds.")
            }
        }
    }

    // MARK: - 10. Hardware Profiles
    private var hardwareSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("Hardware Recommendations", color: .mint)

            Text("Optimizing for Your Apple Silicon Mac")
                .font(.title2).bold()

            VStack(alignment: .leading, spacing: 14) {
                hardwareCard("16 GB Unified RAM (M1/M2/M3/M4)", "Recommended Model: Ornith 1.5 9B OptiQ-4bit (Dense Hybrid)\n• Mode: Auto or Full RAM (~5.4 GB resident)\n• KV Cache: FP16 (or FP8 for >8k context)\n• JetSpec: Disabled\n• Coder Profile: T=0.60, TopP=0.95, TopK=20, RepPen=1.00, PresPen=0.00")
                hardwareCard("24 GB - 36 GB Unified RAM", "Recommended Models: Ornith 1.5 9B (Full RAM) or Ornith 1.5 35B / Qwen 3.8 Flash Next (SSD Streaming)\n• Mode: Auto (Smart)\n• Budget: Balanced (16GB) or High Capacity (24GB)\n• KV Cache: FP16\n• Lookahead Prefetching: Enabled (Depth: 2)")
                hardwareCard("64 GB - 128 GB+ Unified RAM (M-Max / M-Ultra)", "Recommended Models: Full MoE models in RAM without disk reads\n• Mode: Full RAM\n• KV Cache: FP16\n• JetSpec: Enabled (Depth: 3, Branching: 2, Expert Cap: 16)")
            }
        }
    }

    // MARK: - 11. Troubleshooting & FAQs
    private var faqSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            headerBadge("FAQs & Troubleshooting", color: .pink)

            Text("Frequently Asked Questions")
                .font(.title2).bold()

            faqCard(
                "Why does token speed decrease over long thinking chains?",
                "This occurs primarily because of the model's hybrid architecture. For example, Ornith 1.5 9B has 24 Gated DeltaNet layers (which run in constant O(1) time) and 8 full Grouped-Query Attention (GQA) layers (which scan all previous tokens in the KV cache on every step, scaling O(T)). Additionally, the displayed speed is a cumulative running average from token 1, meaning as instantaneous speed naturally drops at 4,000+ tokens, the displayed average will gradually slide toward 1.5-2.0 tok/s."
            )

            faqCard(
                "Why is repetition penalty 1.00 for Coder?",
                "Programming languages require repeating variable names, structural brackets, and keywords. Setting repetition penalty > 1.0 forces the model to invent erroneous synonyms and omit closing brackets. Use Presence Penalty (0.0-0.1) instead if you want mild topic variety."
            )

            faqCard(
                "How do I reset my settings to default?",
                "Open Settings (⌘,) -> Models, click 'Profiles' on your active model, and click 'Reset to Defaults'. This restores factory-tuned parameters for both Coder and Assistant profiles."
            )
        }
    }

    // MARK: - Helper Views & Components
    private func headerBadge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 11, weight: .bold))
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(color.opacity(0.15))
            .foregroundColor(color)
            .cornerRadius(6)
    }

    private func infoBox<Content: View>(title: String, icon: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 12, weight: .bold))
                    .foregroundColor(.accentColor)
                Text(title)
                    .font(.system(size: 13, weight: .bold))
            }
            content()
        }
        .padding(14)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
        .cornerRadius(10)
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private func shortcutRow(_ keys: String, _ description: String) -> some View {
        HStack {
            Text(keys)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(4)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.primary.opacity(0.12), lineWidth: 1))
            Text(description)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
        }
    }

    private func stepRow(_ number: Int, _ title: String, _ description: String) -> some View {
        HStack(alignment: .top, spacing: 12) {
            ZStack {
                Circle()
                    .fill(Color.accentColor.opacity(0.15))
                    .frame(width: 24, height: 24)
                Text("\(number)")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundColor(.accentColor)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 13, weight: .semibold))
                Text(description)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
        }
    }

    private func bulletPoint(_ title: String, _ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(.accentColor)
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold))
                Text(text)
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
            }
        }
    }

    private func paramCard(_ name: String, _ description: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(name)
                .font(.system(size: 12, weight: .bold))
                .foregroundColor(.primary)
            Text(description)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.35))
        .cornerRadius(8)
    }

    private func precisionBadge(_ title: String, _ subtitle: String, _ description: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(.accentColor)
            Text(subtitle)
                .font(.system(size: 11, weight: .semibold))
            Text(description)
                .font(.system(size: 11))
                .foregroundColor(.secondary)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
        .cornerRadius(8)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private func hardwareCard(_ title: String, _ content: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 13, weight: .bold))
                .foregroundColor(.primary)
            Text(content)
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.secondary)
                .lineSpacing(3)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
        .cornerRadius(8)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }

    private func faqCard(_ question: String, _ answer: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "questionmark.circle.fill")
                    .foregroundColor(.accentColor)
                    .font(.system(size: 12))
                Text(question)
                    .font(.system(size: 13, weight: .bold))
            }
            Text(answer)
                .font(.system(size: 12))
                .foregroundColor(.secondary)
                .lineSpacing(2)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.4))
        .cornerRadius(8)
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.08), lineWidth: 1))
    }
}
