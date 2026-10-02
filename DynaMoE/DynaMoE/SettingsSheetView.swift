//
//  SettingsSheetView.swift
//  DynaMoE
//

import SwiftUI
import Metal

enum SettingsTab: String, CaseIterable, Identifiable {
    case general = "General"
    case models = "Models"
    case generation = "Generation"
    case memory = "Memory & SSD"
    case agent = "Agent & Tools"
    case advanced = "Advanced Diagnostics"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .general: return "gearshape"
        case .models: return "square.stack.3d.up.fill"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .agent: return "wrench.and.screwdriver.fill"
        case .advanced: return "waveform.path.ecg"
        }
    }

    /// Short description shown in the detail pane header banner.
    var bannerSubtitle: String {
        switch self {
        case .general:
            return "Manage your overall DynaMoE setup, including updates, chat history, and reasoning display preferences."
        case .models:
            return "Browse installed local models, set defaults, repack for FlashMoE, and configure per-model generation profiles."
        case .generation:
            return "Tune sampling hyperparameters, system prompts, and JetSpec speculative tree acceleration."
        case .memory:
            return "Control memory execution, RAM budgets, KV-cache precision, SSD expert paging, and live diagnostics."
        case .agent:
            return "Configure agent tools, workspace, safety limits, web search, codebase indexing, subagents, and developer tooling."
        case .advanced:
            return "Inspect sharded SafeTensors metadata and run direct Metal kernel execution harnesses."
        }
    }

    /// Accent color for the detail pane header banner icon tile.
    var bannerTint: Color {
        switch self {
        case .general: return .gray
        case .models: return .indigo
        case .generation: return .purple
        case .memory: return .teal
        case .agent: return .orange
        case .advanced: return .pink
        }
    }

    /// Extra search terms matched by the sidebar search field.
    var keywords: [String] {
        switch self {
        case .general:
            return ["updates", "version", "thinking", "reasoning", "chat", "history", "retention", "auto-delete"]
        case .models:
            return ["model", "weights", "safetensors", "flashmoe", "repack", "default", "tokenizer", "metal", "profile", "cache"]
        case .generation:
            return ["temperature", "top-p", "top-k", "min-p", "sampling", "penalty", "tokens", "jetspec", "speculative", "system prompt", "preset"]
        case .memory:
            return ["memory", "ram", "budget", "kv cache", "precision", "fp8", "fp16", "prefetch", "ssd", "paging", "diagnostics", "cache"]
        case .agent:
            return ["agent", "tools", "turbo", "grammar", "workspace", "directory", "search", "brave", "chrome", "index", "rag", "subagent", "git", "lint", "dogfood", "benchmark"]
        case .advanced:
            return ["diagnostics", "tensor", "inspector", "router", "metal", "kernel", "layer", "shard", "inspect"]
        }
    }
}

struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openWindow) private var openWindow
    @State private var selectedTab: SettingsTab = .models
    @State private var sidebarSearchText: String = ""
    @State private var isPromptSavedFeedback: Bool = false
    @State private var repackingModelId: String? = nil
    @State private var repackProgress: Double = 0.0
    @State private var repackStatus: String = ""
    @State private var repackError: String? = nil
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    @AppStorage("dynamoe_agent_tools_enabled") private var isAgentToolsGloballyEnabled: Bool = true
    @AppStorage("dynamoe_reasoning_expanded_by_default") private var reasoningExpandedByDefault: Bool = false
    @AppStorage("dynamoe_chat_auto_delete_enabled") private var chatAutoDeleteEnabled: Bool = true
    @AppStorage("dynamoe_chat_retention_limit") private var chatRetentionLimit: Int = 10
    @AppStorage("dynamoe_agent_turbo_mode") private var isTurboModeEnabled: Bool = false
    @AppStorage("dynamoe_agent_grammar_masking") private var isGrammarMaskingEnabled: Bool = true
    @AppStorage("dynamoe_agent_working_directory") private var agentWorkingDirectory: String = ""
    @AppStorage("dynamoe_max_tool_output_length") private var maxToolOutputLength: Int = 4000
    @AppStorage("dynamoe_max_agent_steps") private var maxAgentSteps: Int = 15
    @AppStorage("dynamoe_brave_search_api_key") private var braveApiKey: String = ""
    @AppStorage("dynamoe_chrome_binary_path") private var customChromeBinaryPath: String = ""
    @AppStorage("dynamoe_jetspec_enabled") private var jetSpecEnabled: Bool = false
    @AppStorage("dynamoe_jetspec_depth") private var jetSpecMaxDepth: Int = 3
    @AppStorage("dynamoe_jetspec_branching") private var jetSpecBranchingFactor: Int = 2
    @AppStorage("dynamoe_jetspec_expert_cap") private var jetSpecMaxExpertCap: Int = 8
    @ObservedObject private var indexer = CodebaseIndexer.shared
    @ObservedObject private var dogfoodRunner = ModelDogfoodBenchmarkRunner.shared
    @ObservedObject private var updaterViewModel = UpdaterViewModel.shared
    @State private var showDogfoodReportModal: Bool = false

    // Model Profile Management State
    @ObservedObject private var profileManager = ModelProfileManager.shared
    @State private var selectedModelForProfiles: String? = nil
    @State private var inspectingProfileType: ModelProfileType = .coder
    @State private var editingDraft: GenerationProfileSettings = GenerationProfileSettings.defaultFor(type: .coder)
    @State private var profileFeedbackText: String? = nil

    // Model & Tokenizer bindings
    var summary: ModelSummary?
    var modelConfig: ModelConfig? = nil
    var tokenizer: DynaMoeTokenizer?
    var activeModelPath: String? = nil
    var metalStatus: String
    var detectedArchitecture: ModelArchitectureType
    var onSelectModel: () -> Void
    var onSelectTokenizer: () -> Void
    var onLoadDiscoveredModel: (DiscoveredModel) -> Void

    // Generation Parameters bindings
    @Binding var temperature: Float
    @Binding var topP: Float
    @Binding var minP: Float
    @Binding var topK: Int
    @Binding var repetitionPenalty: Float
    @Binding var presencePenalty: Float
    @Binding var maxNewTokens: Int
    @Binding var systemPrompt: String
    @Binding var targetLayerCount: Int
    @Binding var activeProfile: ModelProfileType

    // Memory & Working Set bindings
    @Binding var memoryExecutionMode: MemoryExecutionMode
    @Binding var memoryBudgetMode: MemoryBudgetMode
    @Binding var kvCachePrecision: KVCachePrecision
    @Binding var speculativePrefetchEnabled: Bool
    @Binding var prefetchLookaheadDepth: Int
    var currentRssGB: Double
    var residentExpertCount: Int
    var totalExpertCount: Int
    var cacheHitRate: Double
    var prefetchEfficiency: Double
    var lastPagingLatencyMs: Double
    var pagingStatusMessage: String?
    var onFlushCache: () -> Void
    var onPreFaultAll: () -> Void

    // Advanced Diagnostics bindings & callbacks
    @Binding var searchText: String
    @Binding var selectedCategory: String
    var categoryFilters: [String]

    private var filteredTensors: [TensorMetadata] {
        guard let tensors = summary?.tensors else { return [] }
        if searchText.isEmpty && selectedCategory == "All" {
            return Array(tensors.prefix(100))
        }
        var matches: [TensorMetadata] = []
        for tensor in tensors {
            let matchesSearch = searchText.isEmpty || tensor.name.localizedCaseInsensitiveContains(searchText)
            let matchesCategory = (selectedCategory == "All") || tensor.category.contains(selectedCategory)
            if matchesSearch && matchesCategory {
                matches.append(tensor)
                if matches.count >= 100 {
                    break
                }
            }
        }
        return matches
    }
    @Binding var selectedTensorID: String?
    var selectedTensor: TensorMetadata?
    var onExecuteMoERouter: (Int) -> Void
    var onExecuteFullLayer: (Int) -> Void
    var onExecuteMultiLayer: (Int) -> Void
    var isExecutingMlp: Bool
    var isExecutingFullLayer: Bool
    var isExecutingMultiLayer: Bool

    var body: some View {
        NavigationSplitView(columnVisibility: .constant(.all)) {
            settingsSidebar
                .navigationSplitViewColumnWidth(min: 200, ideal: 224, max: 260)
        } detail: {
            settingsDetail
        }
        .navigationSplitViewStyle(.balanced)
        .toolbar(removing: .sidebarToggle)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: {
                    openWindow(id: "dynamoe-help")
                }) {
                    Label("Help & Guide", systemImage: "questionmark.circle")
                }
                .help("Open DynaMoE Help & Settings Guide (⌘?)")

                Button("Done") {
                    SettingsWindowManager.shared.close()
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
            }
        }
        .frame(minWidth: 780, minHeight: 560)
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Sidebar & Detail Shell
    private var settingsSidebar: some View {
        List(selection: $selectedTab) {
            ForEach(filteredSettingsTabs) { tab in
                Label {
                    Text(tab.rawValue)
                } icon: {
                    Image(systemName: tab.icon)
                        .foregroundColor(tab.bannerTint)
                }
                .font(.system(size: 13))
                .padding(.vertical, 2)
                .tag(tab)
            }
        }
        .listStyle(.sidebar)
        .searchable(text: $sidebarSearchText, placement: .sidebar, prompt: "Search Settings…")
    }

    private var filteredSettingsTabs: [SettingsTab] {
        let query = sidebarSearchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return SettingsTab.allCases }
        return SettingsTab.allCases.filter { tab in
            tab.rawValue.localizedCaseInsensitiveContains(query)
                || tab.keywords.contains { $0.localizedCaseInsensitiveContains(query) }
        }
    }

    private var settingsDetail: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                SettingsSectionBanner(
                    icon: selectedTab.icon,
                    title: selectedTab.rawValue,
                    subtitle: selectedTab.bannerSubtitle,
                    tint: selectedTab.bannerTint
                )

                switch selectedTab {
                case .general:
                    generalSettingsSection
                case .models:
                    modelsSettingsSection
                case .generation:
                    generationSettingsSection
                case .memory:
                    memorySettingsSection
                case .agent:
                    agentSettingsSection
                case .advanced:
                    advancedDiagnosticsSection
                }
            }
            .padding(.horizontal, 28)
            .padding(.vertical, 16)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Tab 0: General
    private var generalSettingsSection: some View {
        VStack(alignment: .leading, spacing: 22) {
            SettingsCard(header: "Updates") {
                SettingsRow(
                    title: "Enable Auto-Updates",
                    subtitle: "Automatically check for new DynaMoE releases in the background. Updates are verified and installed safely; you can always check manually with ⌘U."
                ) {
                    Toggle("", isOn: Binding(
                        get: { UpdaterViewModel.shared.automaticallyChecksForUpdates },
                        set: { UpdaterViewModel.shared.automaticallyChecksForUpdates = $0 }
                    ))
                    .toggleStyle(.switch)
                    .labelsHidden()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Check for Updates Now",
                    subtitle: "Manually query the update feed regardless of the automatic setting."
                ) {
                    Button(action: {
                        UpdaterViewModel.shared.checkForUpdates()
                    }) {
                        Label("Check Now", systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(!updaterViewModel.canCheckForUpdates)
                    .help("Check for updates now (⌘U)")
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Current Version",
                    subtitle: "The DynaMoE release currently installed."
                ) {
                    Text(versionDisplayString)
                        .font(.system(size: 13))
                        .foregroundStyle(.secondary)
                }
            }

            SettingsCard(header: "Chat & Reasoning") {
                SettingsRow(
                    title: "Expand Thinking by Default",
                    subtitle: "Show the thinking/reasoning accordion expanded when a message first appears. When off, reasoning starts collapsed and can be opened manually."
                ) {
                    Toggle("", isOn: $reasoningExpandedByDefault)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Auto-Delete Old Conversations",
                    subtitle: "Keep conversations saved between launches, removing the oldest automatically once the limit below is reached. Turn off to keep every conversation forever."
                ) {
                    Toggle("", isOn: $chatAutoDeleteEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }

                if chatAutoDeleteEnabled {
                    SettingsRowDivider()

                    SettingsRow(
                        title: "Conversations to Keep",
                        subtitle: "The most-recently-updated conversations retained before older ones are deleted. The conversation you have open is always kept."
                    ) {
                        HStack(spacing: 6) {
                            Text("\(chatRetentionLimit)")
                                .font(.system(size: 13, design: .monospaced))
                                .frame(minWidth: 24, alignment: .trailing)
                            Stepper("Conversations to keep", value: $chatRetentionLimit, in: 1...100)
                                .labelsHidden()
                        }
                    }
                }
            }
        }
    }

    private var versionDisplayString: String {
        let shortVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0.0"
        let buildVersion = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "1"
        return "v\(shortVersion) (\(buildVersion))"
    }

    /// Single-line summary of a model's author, architecture, quantization, and size.
    private func modelMetadataLine(_ model: DiscoveredModel) -> String {
        var parts: [String] = [model.author, model.architectureName]
        if let quant = model.quantization, !quant.isEmpty {
            parts.append(quant)
        }
        parts.append(model.formattedSize)
        return parts.joined(separator: "  •  ")
    }

    /// Bridges an `Int` binding into the `Double` numeric control.
    private func doubleBinding(_ source: Binding<Int>) -> Binding<Double> {
        Binding(
            get: { Double(source.wrappedValue) },
            set: { source.wrappedValue = Int($0.rounded()) }
        )
    }

    /// Bridges a `Float` binding into the `Double` numeric control.
    private func doubleBinding(_ source: Binding<Float>) -> Binding<Double> {
        Binding(
            get: { Double(source.wrappedValue) },
            set: { source.wrappedValue = Float($0) }
        )
    }

    /// Label-left settings row carrying a slider-plus-field numeric control,
    /// used across the Sampling and JetSpec parameter sets.
    private func samplingSliderRow(
        _ title: String,
        subtitle: String,
        value: Binding<Double>,
        range: ClosedRange<Double>,
        step: Double,
        digits: Int = 2
    ) -> some View {
        SettingsRow(title: title, subtitle: subtitle) {
            SettingsValueSlider(value: value, range: range, step: step, fractionDigits: digits)
        }
    }

    // MARK: - Tab 1: Models & Weights Management
    private var modelsSettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Section Header with Refresh Button
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Installed Local Models")
                        .font(.system(size: 12.5, weight: .semibold))
                    Text("Auto-discovered from Hugging Face cache (~/.cache/huggingface/hub)")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Spacer()

                Button(action: {
                    localModelManager.scanLocalModels()
                }) {
                    HStack(spacing: 5) {
                        if localModelManager.isScanning {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "arrow.clockwise")
                        }
                        Text(localModelManager.isScanning ? "Scanning..." : "Refresh Cache")
                    }
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(localModelManager.isScanning)
            }
            .padding(.horizontal, 8)

            // Discovered Models List
            if localModelManager.discoveredModels.isEmpty {
                VStack(spacing: 10) {
                    Image(systemName: "shippingbox.and.arrow.backward")
                        .font(.system(size: 32))
                        .foregroundColor(.secondary.opacity(0.5))
                    Text("No Hugging Face models found in cache")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Text("Download a safetensors model into ~/.cache/huggingface/hub or select a local folder below.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 32)
                .background(Color.secondary.opacity(0.04))
                .cornerRadius(12)
            } else {
                VStack(spacing: 10) {
                    ForEach(localModelManager.discoveredModels) { model in
                        let isDefault = localModelManager.defaultModelId == model.id || localModelManager.defaultModelId == model.repoId
                        let isCurrentActive = (activeModelPath != nil && (activeModelPath == model.snapshotPath || activeModelPath == model.weightsEntryPath || (summary != nil && (modelConfig?.modelType ?? "").localizedCaseInsensitiveContains(model.displayName))))
                        let isFlashMoEPacked = ExpertRepacker.isPackedFormat(dir: URL(fileURLWithPath: model.snapshotPath))

                        VStack(alignment: .leading, spacing: 8) {
                            HStack(alignment: .center, spacing: 14) {
                            // Leading Model Icon
                            ZStack {
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(isCurrentActive ? Color.purple.opacity(0.15) : Color.secondary.opacity(0.08))
                                    .frame(width: 44, height: 44)
                                Image(systemName: model.isMoE ? "circle.hexagongrid.circle.fill" : "cube.fill")
                                    .font(.system(size: 20))
                                    .foregroundColor(isCurrentActive ? .purple : .secondary)
                            }

                            // Model Info
                            VStack(alignment: .leading, spacing: 3) {
                                Text(model.displayName)
                                    .font(.system(size: 13.5, weight: .semibold))
                                    .foregroundColor(.primary)
                                    .lineLimit(1)
                                    .truncationMode(.tail)

                                if isDefault || isCurrentActive || (model.isMoE && isFlashMoEPacked) {
                                    HStack(spacing: 6) {
                                        if isDefault {
                                            SettingsStatusBadge(text: "Default", tint: .purple)
                                        }
                                        if isCurrentActive {
                                            SettingsStatusBadge(text: "Active", tint: .green)
                                        }
                                        if model.isMoE && isFlashMoEPacked {
                                            SettingsStatusBadge(text: "FlashMoE", systemImage: "bolt.fill", tint: .orange)
                                        }
                                    }
                                }

                                Text(modelMetadataLine(model))
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }

                            Spacer()

                            // Actions: secondary actions live in an overflow menu so the
                            // row stays uncluttered and text never competes with controls.
                            HStack(spacing: 8) {
                                if repackingModelId == model.id {
                                    VStack(alignment: .trailing, spacing: 2) {
                                        ProgressView(value: repackProgress)
                                            .progressViewStyle(.linear)
                                            .frame(width: 80)
                                        Text(repackStatus)
                                            .font(.system(size: 9))
                                            .foregroundColor(.secondary)
                                            .lineLimit(1)
                                    }
                                } else {
                                    Menu {
                                        Button {
                                            if isDefault {
                                                localModelManager.setDefaultModel(id: nil)
                                            } else {
                                                localModelManager.setDefaultModel(id: model.id)
                                            }
                                        } label: {
                                            Label(isDefault ? "Clear Default" : "Set as Default",
                                                  systemImage: isDefault ? "star.slash" : "star")
                                        }

                                        Button {
                                            withAnimation(.easeInOut(duration: 0.2)) {
                                                if selectedModelForProfiles == model.id {
                                                    selectedModelForProfiles = nil
                                                } else {
                                                    selectedModelForProfiles = model.id
                                                    loadProfileDraft(modelId: model.id, type: inspectingProfileType)
                                                }
                                            }
                                        } label: {
                                            Label(selectedModelForProfiles == model.id ? "Hide Profiles" : "Configure Profiles…",
                                                  systemImage: "slider.horizontal.3")
                                        }

                                        if model.isMoE && !isFlashMoEPacked {
                                            Divider()
                                            Button {
                                                repackModel(model)
                                            } label: {
                                                Label("FlashMoE Repack", systemImage: "bolt.badge.automatic.fill")
                                            }
                                        }
                                    } label: {
                                        Image(systemName: "ellipsis.circle")
                                            .font(.system(size: 15))
                                    }
                                    .menuStyle(.borderlessButton)
                                    .menuIndicator(.hidden)
                                    .fixedSize()
                                    .help("Model actions")
                                }

                                if isCurrentActive {
                                    Label("Loaded", systemImage: "checkmark.circle.fill")
                                        .font(.system(size: 11, weight: .semibold))
                                        .foregroundColor(.green)
                                        .padding(.horizontal, 10)
                                        .padding(.vertical, 5)
                                        .background(Color.green.opacity(0.12))
                                        .clipShape(Capsule())
                                } else {
                                    Button {
                                        onLoadDiscoveredModel(model)
                                    } label: {
                                        Text("Load")
                                            .font(.system(size: 12, weight: .semibold))
                                    }
                                    .buttonStyle(.borderedProminent)
                                    .controlSize(.small)
                                }
                            }
                        }
                        if selectedModelForProfiles == model.id {
                            modelProfileEditorSection(for: model)
                                .transition(.opacity.combined(with: .move(edge: .top)))
                        }
                    }
                    .padding(12)
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(10)
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .stroke(selectedModelForProfiles == model.id ? Color.purple.opacity(0.4) : (isCurrentActive ? Color.purple.opacity(0.3) : Color.primary.opacity(0.06)), lineWidth: selectedModelForProfiles == model.id ? 1.5 : 1)
                    )
                }
                }
            }

            SettingsCard(header: "Custom Paths & Fallbacks") {
                SettingsRow(
                    title: "Custom Folder or Safetensors File",
                    subtitle: "Load a model outside of Hugging Face cache (e.g. external SSD)"
                ) {
                    Button("Browse Folder / Index...", action: onSelectModel)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Custom Tokenizer (tokenizer.json)",
                    subtitle: tokenizer != nil ? "Tokenizer loaded and ready" : "No tokenizer loaded."
                ) {
                    Button(tokenizer == nil ? "Load tokenizer.json" : "Replace Tokenizer", action: onSelectTokenizer)
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Metal GPU Acceleration",
                    subtitle: metalStatus
                ) {
                    if let device = MTLCreateSystemDefaultDevice() {
                        Text(device.name)
                            .font(.system(size: 13))
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    // MARK: - Tab 2: Generation & Sampler
    private var generationSettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Active Profile Banner
            let currentModelIdentifier = activeModelPath ?? modelConfig?.modelType ?? localModelManager.defaultModelId
            let modelDisplayText: String = {
                if let path = activeModelPath,
                   let model = localModelManager.discoveredModels.first(where: { $0.snapshotPath == path || $0.weightsEntryPath == path }) {
                    return model.displayName
                } else if let modelType = modelConfig?.modelType {
                    return modelType
                } else {
                    return detectedArchitecture.rawValue
                }
            }()

            SettingsCard {
                SettingsRow(
                    title: "Active Profile: \(activeProfile.profileDisplayName(for: currentModelIdentifier))",
                    subtitle: modelDisplayText,
                    icon: activeProfile.icon,
                    iconTint: .purple
                ) {
                    HStack(spacing: 8) {
                        Image(systemName: "info.circle")
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                            .help(activeProfile.description(for: currentModelIdentifier))

                        Picker("Profile", selection: $activeProfile) {
                            ForEach(ModelProfileType.allCases) { profile in
                                Text(profile.rawValue).tag(profile)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(width: 190)
                        .onChange(of: activeProfile) { newProfile in
                            let modelKey = activeModelPath ?? localModelManager.defaultModelId
                            let settings = profileManager.getProfile(for: modelKey, type: newProfile)
                            temperature = settings.temperature
                            topP = settings.topP
                            minP = settings.minP
                            topK = settings.topK
                            repetitionPenalty = settings.repetitionPenalty
                            presencePenalty = settings.presencePenalty
                            maxNewTokens = settings.maxNewTokens
                            systemPrompt = settings.systemPrompt
                            jetSpecEnabled = settings.jetSpecEnabled
                            jetSpecMaxDepth = settings.jetSpecMaxDepth
                            jetSpecBranchingFactor = settings.jetSpecBranchingFactor
                            jetSpecMaxExpertCap = settings.jetSpecMaxExpertCap
                            profileManager.setActiveProfile(for: modelKey, type: newProfile)
                        }

                        Button(action: {
                            let modelKey = activeModelPath ?? localModelManager.defaultModelId
                            saveCurrentGenerationToProfile(modelId: modelKey)
                        }) {
                            Label("Save", systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .help("Save the current generation settings to the \(activeProfile.rawValue) profile")
                    }
                }
            }

            SettingsCard(header: "Sampling") {
                samplingSliderRow(
                    "Temperature",
                    subtitle: "Controls randomness. Lower values are more focused and deterministic.",
                    value: doubleBinding($temperature), range: 0.0...2.0, step: 0.05
                )

                SettingsRowDivider()

                samplingSliderRow(
                    "Top-P (Nucleus)",
                    subtitle: "Samples from the smallest set of tokens whose probabilities add up to P.",
                    value: doubleBinding($topP), range: 0.0...1.0, step: 0.05
                )

                SettingsRowDivider()

                samplingSliderRow(
                    "Min-P (Confidence)",
                    subtitle: "Ignores tokens below this fraction of the most likely token's probability.",
                    value: doubleBinding($minP), range: 0.0...0.5, step: 0.01
                )

                SettingsRowDivider()

                samplingSliderRow(
                    "Top-K",
                    subtitle: "Restricts sampling to the K most likely tokens.",
                    value: doubleBinding($topK), range: 1...100, step: 1, digits: 0
                )

                SettingsRowDivider()

                samplingSliderRow(
                    "Repetition Penalty",
                    subtitle: "Downweights tokens that have already appeared. 1.00 disables the penalty.",
                    value: doubleBinding($repetitionPenalty), range: 1.0...2.0, step: 0.05
                )

                SettingsRowDivider()

                samplingSliderRow(
                    "Presence Penalty",
                    subtitle: "Penalizes any token already present to steer toward new topics.",
                    value: doubleBinding($presencePenalty), range: 0.0...2.0, step: 0.05
                )

                SettingsRowDivider()

                // Max Output Tokens: slider, editable field, and quick presets.
                SettingsRow(
                    title: "Max Output Tokens",
                    subtitle: "Hard cap on tokens generated per response."
                ) {
                    SettingsValueSlider(
                        value: doubleBinding($maxNewTokens),
                        range: 32...10000,
                        step: 32,
                        fractionDigits: 0,
                        fieldWidth: 64
                    )
                }

                HStack(spacing: 6) {
                    Text("Presets")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)

                    ForEach([512, 1024, 2048, 4096, 8192, 10000], id: \.self) { preset in
                        Button("\(preset)") {
                            maxNewTokens = preset
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 10.5, weight: maxNewTokens == preset ? .bold : .regular, design: .monospaced))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2.5)
                        .background(maxNewTokens == preset ? Color.purple.opacity(0.18) : Color.secondary.opacity(0.08))
                        .foregroundColor(maxNewTokens == preset ? .purple : .primary)
                        .cornerRadius(4)
                    }
                }
                .padding(.vertical, 4)
            }

            // JetSpec Causal Speculative Tree Acceleration
            SettingsCard(header: "JetSpec Speculative Tree Acceleration") {
                    SettingsRow(
                        title: "Enable JetSpec",
                        subtitle: "Parallel tree drafting with dynamic MoE budget pruning for standard attention models. Gated DeltaNet recurrent models (like Ornith 1.5) and disk-streamed models use optimized direct execution automatically.",
                        icon: "arrow.triangle.branch",
                        iconTint: .purple
                    ) {
                        Toggle("", isOn: $jetSpecEnabled)
                            .toggleStyle(.switch)
                            .labelsHidden()
                    }

                    if jetSpecEnabled {
                        SettingsRowDivider()

                        samplingSliderRow(
                            "Max Tree Depth",
                            subtitle: "How far ahead the tree drafts tokens before verification.",
                            value: doubleBinding($jetSpecMaxDepth), range: 1...5, step: 1, digits: 0
                        )

                        SettingsRowDivider()

                        samplingSliderRow(
                            "Branching Factor",
                            subtitle: "Candidate continuations explored per tree node.",
                            value: doubleBinding($jetSpecBranchingFactor), range: 1...4, step: 1, digits: 0
                        )

                        SettingsRowDivider()

                        samplingSliderRow(
                            "Max Expert Cap",
                            subtitle: "Upper bound on MoE experts evaluated per drafted token.",
                            value: doubleBinding($jetSpecMaxExpertCap), range: 2...16, step: 1, digits: 0
                        )
                    }
            }

            // System Prompt Editor & Conjunction Combination
            SettingsCard(header: "System Prompt", spacing: 8) {
                    HStack {
                        Text("User Default System Prompt")
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        if isPromptSavedFeedback {
                            HStack(spacing: 4) {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                                Text("Saved as Default!")
                                    .foregroundColor(.green)
                            }
                            .font(.caption2)
                            .transition(.opacity)
                        }

                        Button("Reset to Default") {
                            systemPrompt = ModelConfig.getUserDefaultSystemPrompt()
                        }
                        .buttonStyle(.plain)
                        .font(.caption2)
                        .foregroundColor(.secondary)

                        Button("Save as Default") {
                            ModelConfig.setUserDefaultSystemPrompt(systemPrompt)
                            withAnimation {
                                isPromptSavedFeedback = true
                            }
                            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                                withAnimation {
                                    isPromptSavedFeedback = false
                                }
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .controlSize(.small)
                    }

                    TextEditor(text: $systemPrompt)
                        .font(.system(.caption, design: .monospaced))
                        .frame(height: 75)
                        .padding(6)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                        )

                    // Active Model Required Instruct Prompt Callout
                    let requiredPrompt = ModelConfig.resolveRequiredSystemPrompt(
                        config: modelConfig,
                        summary: summary,
                        modelName: summary != nil ? (modelConfig?.modelType ?? detectedArchitecture.shortName) : nil,
                        modelPath: activeModelPath
                    )
                    if !requiredPrompt.isEmpty {
                        VStack(alignment: .leading, spacing: 5) {
                            HStack(spacing: 5) {
                                Image(systemName: "shield.lefthalf.filled")
                                    .font(.system(size: 11))
                                    .foregroundColor(.indigo)
                                Text("Active Model Required Instruct Prompt:")
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .foregroundColor(.indigo)
                            }

                            Text(requiredPrompt)
                                .font(.system(size: 11, design: .monospaced))
                                .foregroundColor(.primary.opacity(0.85))
                                .padding(8)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .background(Color.indigo.opacity(0.08))
                                .cornerRadius(6)

                            Text("ℹ️ DynaMoE automatically sends this required prompt in conjunction (combined) with your user prompt above.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 2)
                    } else {
                        Text("This prompt is used as the default personality and instructions for all conversations.")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
            }
        }
    }

    // MARK: - Tab 3: Memory & Working Set
    private var memorySettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsCard(
                header: "Execution",
                footer: "FP16 (Half) cuts KV-cache memory by 50% with ~2x memory bandwidth during autoregressive GQA decoding. FP8 cuts memory footprint by 75% for ultra-long context windows."
            ) {
                SettingsRow(
                    title: "Memory Execution Mode",
                    subtitle: "How model weights are loaded: fully resident in RAM, or streamed from SSD on demand."
                ) {
                    Picker("Mode", selection: $memoryExecutionMode) {
                        ForEach(MemoryExecutionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "RAM Working Set Budget",
                    subtitle: "Maximum resident working set before expert weights are evicted."
                ) {
                    Picker("Budget", selection: $memoryBudgetMode) {
                        ForEach(MemoryBudgetMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()
                    .fixedSize()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "KV-Cache Precision",
                    subtitle: "Numeric precision used for the attention key/value history."
                ) {
                    HStack(spacing: 8) {
                        SettingsStatusBadge(
                            text: kvCachePrecision == .fp8 ? "75% VRAM Reduction" : (kvCachePrecision == .fp16 ? "50% VRAM Reduction" : "Full Precision"),
                            tint: .indigo
                        )
                        Picker("KV Precision", selection: $kvCachePrecision) {
                            ForEach(KVCachePrecision.allCases) { prec in
                                Text(prec.rawValue).tag(prec)
                            }
                        }
                        .pickerStyle(.menu)
                        .labelsHidden()
                    }
                }
            }

            // Speculative Prefetching & Layer Lookahead
            SettingsCard(header: "Speculative Prefetching") {
                SettingsRow(
                    title: "MoE Expert & Layer Prefetching",
                    subtitle: "Asynchronously warms next-layer backbone weights and predicted expert slices in the background before GPU execution."
                ) {
                    Toggle("", isOn: $speculativePrefetchEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }

                if speculativePrefetchEnabled {
                    SettingsRowDivider()

                    SettingsRow(
                        title: "Lookahead Depth",
                        subtitle: "How many layers ahead the prefetcher should warm."
                    ) {
                        Picker("Lookahead", selection: $prefetchLookaheadDepth) {
                            Text("1 Layer").tag(1)
                            Text("2 Layers").tag(2)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .fixedSize()
                    }
                }
            }

            // Telemetry Metrics Grid
            SettingsCard(header: "Live Diagnostics") {
                HStack(alignment: .top, spacing: 24) {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 3) {
                            Text("WORKING SET RAM")
                                .font(.system(size: 9, weight: .semibold))
                                .foregroundStyle(.secondary)
                            Image(systemName: "info.circle")
                                .font(.system(size: 8))
                                .foregroundStyle(.secondary)
                        }
                        Text(String(format: "%.2f GB", currentRssGB))
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.indigo)
                        Text(String(format: "Heap: %.2f GB", getProcessResidentMemoryGB()))
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundStyle(.tertiary)
                    }
                    .help(String(format: "Working Set RAM: %.2f GB\nProcess Heap (Activity Monitor): %.2f GB\nUnified Memory Cache: %.2f GB\n\nApple Silicon places clean zero-copy model weights in Darwin's Unified Memory Buffer Cache, which Activity Monitor excludes from process footprint.", currentRssGB, getProcessResidentMemoryGB(), max(0, currentRssGB - getProcessResidentMemoryGB())))

                    VStack(alignment: .leading, spacing: 3) {
                        Text("RESIDENT EXPERTS")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text("\(residentExpertCount) / \(totalExpertCount)")
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text("CACHE HIT RATE")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text(String(format: "%.1f%%", cacheHitRate))
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.green)
                    }

                    VStack(alignment: .leading, spacing: 3) {
                        Text("PREFETCH EFFICIENCY")
                            .font(.system(size: 9, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text(String(format: "%.1f%%", prefetchEfficiency))
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .foregroundStyle(.purple)
                    }

                    Spacer()
                }
                .padding(.vertical, 10)

                SettingsRowDivider()

                // Cache Actions
                HStack(spacing: 10) {
                    Button(action: onFlushCache) {
                        Label("Flush Expert Cache", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)

                    Button(action: onPreFaultAll) {
                        Label("Pre-Fault All Weights into RAM", systemImage: "bolt.fill")
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                }
                .padding(.vertical, 8)
            }
        }
    }

    // MARK: - Tab 4: Agent & Tool Execution Suite
    private var agentSettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            SettingsCard(header: "Agent Tools") {
                SettingsRow(
                    title: "Enable Agent Tools by Default",
                    subtitle: "Equips models with shell execution, file reading/writing, and semantic search via ChatML <tool_call> tags.",
                    icon: "terminal.fill"
                ) {
                    Toggle("", isOn: $isAgentToolsGloballyEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Agent Turbo Mode (Auto-Approve)",
                    subtitle: "When enabled, file writes, edits, and shell commands execute automatically without individual authorization prompts (similar to Antigravity). When disabled, actions require explicit approval.",
                    icon: "bolt.fill",
                    iconTint: .orange
                ) {
                    Toggle("", isOn: $isTurboModeEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }

                SettingsRowDivider()

                SettingsRow(
                    title: "Grammar-Constrained Tool Sampling",
                    subtitle: "Dynamically sets disallowed token logits to -Float.infinity during tool calls, mathematically preventing syntax and schema errors.",
                    icon: "checkmark.shield.fill",
                    iconTint: .green
                ) {
                    Toggle("", isOn: $isGrammarMaskingEnabled)
                        .toggleStyle(.switch)
                        .labelsHidden()
                }
            }

            // Working Directory Picker
            SettingsCard(header: "Workspace") {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Agent Working Directory (CWD)")
                        .font(.system(size: 13, weight: .medium))
                    Text("Commands like zsh, find, and ripgrep run relative to this base directory.")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        Image(systemName: "folder.fill")
                            .foregroundColor(agentWorkingDirectory.isEmpty ? .secondary : .accentColor)
                        if agentWorkingDirectory.isEmpty {
                            Text("No workspace selected (click Browse to choose your project folder)")
                                .font(.system(size: 12))
                                .foregroundColor(.secondary)
                                .italic()
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.06))
                                .cornerRadius(6)
                        } else {
                            Text(agentWorkingDirectory)
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.primary)
                                .lineLimit(1)
                                .truncationMode(.middle)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 6)
                                .background(Color.secondary.opacity(0.06))
                                .cornerRadius(6)
                        }

                        Button("Browse...") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = false
                            panel.canChooseDirectories = true
                            panel.allowsMultipleSelection = false
                            panel.canCreateDirectories = true
                            panel.prompt = "Select Workspace"
                            if panel.runModal() == .OK, let url = panel.url {
                                agentWorkingDirectory = url.path
                            }
                        }
                        .buttonStyle(.bordered)

                        if !agentWorkingDirectory.isEmpty {
                            Button("Reset") {
                                agentWorkingDirectory = ""
                            }
                            .buttonStyle(.borderless)
                            .foregroundColor(.secondary)
                        }
                    }
                }
            }

            // Output Length Slider
            SettingsCard(header: "Limits") {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tool Output (tokens)")
                                .font(.system(size: 13))
                            Text("Token budget per tool result. Observations are rendered as plain text and truncated head/tail at line boundaries.")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("\(maxToolOutputLength) tokens")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: Binding(
                        get: { Double(maxToolOutputLength) },
                        set: { maxToolOutputLength = Int($0) }
                    ), in: 1000...16000, step: 500)
                }
                .padding(.vertical, 8)

                SettingsRowDivider()

                // Max Agent Iteration Steps
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Multi-Step Agent Iterations")
                                .font(.system(size: 13))
                            Text("Maximum number of reasoning and tool execution turns per user prompt.")
                                .font(.system(size: 11))
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Text("\(maxAgentSteps) steps")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: Binding(
                        get: { Double(maxAgentSteps) },
                        set: { maxAgentSteps = Int($0) }
                    ), in: 1...30, step: 1)
                }
                .padding(.vertical, 8)
            }

            // Web Search Engine Configuration
            SettingsCard(header: "Web Search", spacing: 10) {
                HStack {
                    Image(systemName: "globe")
                        .foregroundColor(.blue)
                    Text("Web Search Engine")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()
                    let isChromeAvailable = HeadlessChromeSearchEngine.shared.resolveBinaryPath() != nil
                    let activeEngineName: String = {
                        if !braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            return "Brave Search API"
                        } else if isChromeAvailable {
                            return "Headless Chrome (Active Local Engine)"
                        } else {
                            return "Chrome Not Detected"
                        }
                    }()
                    Text(activeEngineName)
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(!braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.orange.opacity(0.12) : (isChromeAvailable ? Color.green.opacity(0.12) : Color.red.opacity(0.12)))
                        .foregroundColor(!braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .orange : (isChromeAvailable ? .green : .red))
                        .cornerRadius(6)
                }

                if let detectedPath = HeadlessChromeSearchEngine.shared.resolveBinaryPath() {
                    HStack(spacing: 6) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                            .font(.system(size: 11))
                        Text("Detected browser: \(detectedPath)")
                            .font(.system(size: 10.5, design: .monospaced))
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                } else {
                    HStack(spacing: 6) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                            .font(.system(size: 11))
                        Text("Chrome, Chromium, Brave, or Edge not found in standard paths. Specify custom binary below.")
                            .font(.system(size: 10.5))
                            .foregroundColor(.orange)
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("Custom Chrome / Chromium Binary Path (Optional)")
                        .font(.system(size: 12, weight: .medium))
                    TextField("Default: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome", text: $customChromeBinaryPath)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))
                    Text("Local Headless Chrome executes with modern sandboxing (--headless=new --dump-dom) for live web search and JavaScript Single-Page App rendering with zero API keys.")
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("Brave Search API Key (Optional Override)")
                        .font(.system(size: 12, weight: .medium))
                    SecureField("Paste Brave Search API token (e.g. BSA...)", text: $braveApiKey)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))

                    Text("Optionally provide a Brave Search API key if you prefer remote JSON queries over local headless browser execution.")
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }
            }

            // Semantic Codebase Indexing & Local RAG Configuration
            SettingsCard(header: "Codebase Indexing", spacing: 10) {
                HStack {
                    Image(systemName: "sparkle.magnifyingglass")
                        .foregroundColor(.purple)
                    Text("Semantic Codebase Indexing (Metal RAG)")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()
                    if indexer.isIndexing {
                        HStack(spacing: 6) {
                            ProgressView()
                                .controlSize(.small)
                            Text("Indexing...")
                                .font(.caption)
                                .foregroundColor(.purple)
                        }
                    } else {
                        Text(indexer.indexedChunkCount > 0 ? "\(indexer.indexedChunkCount) Chunks" : "Not Indexed")
                            .font(.caption)
                            .fontWeight(.medium)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(indexer.indexedChunkCount > 0 ? Color.purple.opacity(0.12) : Color.secondary.opacity(0.12))
                            .foregroundColor(indexer.indexedChunkCount > 0 ? .purple : .secondary)
                            .cornerRadius(6)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text(indexer.statusMessage)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)

                    if indexer.isIndexing {
                        ProgressView(value: indexer.indexingProgress, total: 1.0)
                            .progressViewStyle(.linear)
                            .tint(.purple)
                    }

                    HStack(spacing: 10) {
                        Button(action: {
                            if agentWorkingDirectory.isEmpty {
                                let panel = NSOpenPanel()
                                panel.canChooseFiles = false
                                panel.canChooseDirectories = true
                                panel.allowsMultipleSelection = false
                                panel.canCreateDirectories = true
                                panel.prompt = "Choose Project"
                                panel.message = "Select your project root folder to index for Local RAG"
                                if panel.runModal() == .OK, let url = panel.url {
                                    agentWorkingDirectory = url.path
                                    indexer.indexWorkspace(url: url, forceRebuild: true)
                                }
                            } else {
                                let targetUrl = URL(fileURLWithPath: agentWorkingDirectory)
                                indexer.indexWorkspace(url: targetUrl, forceRebuild: true)
                            }
                        }) {
                            Label(
                                agentWorkingDirectory.isEmpty ? "Choose & Index Project..." : (indexer.indexedChunkCount > 0 ? "Re-index Workspace" : "Index Workspace Now"),
                                systemImage: agentWorkingDirectory.isEmpty ? "folder.badge.plus" : "arrow.clockwise"
                            )
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.purple)
                        .disabled(indexer.isIndexing)

                        if indexer.isIndexing {
                            Button("Cancel", role: .cancel) {
                                indexer.cancelIndexing()
                            }
                            .buttonStyle(.bordered)
                        }

                        Text("Apple Silicon GPU Cosine Similarity + BM25 Hybrid Rank")
                            .font(.system(size: 10.5))
                            .foregroundColor(.secondary)
                    }
                }
            }

            // Subagent & Multi-Agent Delegation Card
            SettingsCard(header: "Subagents", spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "square.2.layers.3d")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.purple)
                    Text("Multi-Agent Task Orchestration (spawn_subagent)")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()

                    Button(action: {
                        SubagentManager.shared.isDrawerOpen.toggle()
                    }) {
                        HStack(spacing: 4) {
                            Image(systemName: "sidebar.trailing")
                            Text(SubagentManager.shared.isDrawerOpen ? "Close Task Manager" : "Open Task Manager")
                        }
                        .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.bordered)
                }

                Text("Enables the coordinator model to spawn dedicated child agents (e.g., Codebase Researcher, Test Runner, Shader Optimizer) with isolated context windows, inter-agent messaging, and unified summary synthesis.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(SubagentManager.shared.activeSubagentsCount > 0 ? Color.blue : Color.secondary.opacity(0.4))
                            .frame(width: 8, height: 8)
                        Text("\(SubagentManager.shared.activeSubagentsCount) Active Subagents")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundColor(.primary)
                    }

                    Text("•")
                        .foregroundColor(.secondary.opacity(0.5))

                    Text("Total Spawned: \(SubagentManager.shared.subagents.count)")
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
            }

            // Deep Developer Tooling (Native Git & SourceKit-LSP) Card
            SettingsCard(header: "Developer Tooling", spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.teal)
                    Text("Deep Developer Tooling (Git & Code Intelligence)")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()

                    Text("IDE Grade")
                        .font(.system(size: 10, weight: .bold))
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.teal.opacity(0.15))
                        .foregroundColor(.teal)
                        .cornerRadius(4)
                }

                Text("Direct git version control (status, diff, commit with safety rails), AST & SourceKit code intelligence (find_symbol_definition, find_references), and automatic self-healing compiler feedback on file edits.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    Label("Git Safety Rails Active", systemImage: "shield.lefthalf.filled")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundColor(.teal)
                    Text("•")
                        .foregroundColor(.secondary.opacity(0.5))
                    Label("Compiler Self-Healing Loop", systemImage: "stethoscope")
                        .font(.system(size: 10.5, weight: .medium))
                        .foregroundColor(.indigo)
                }
            }

            // Live Model Dogfooding & Benchmarking Card
            SettingsCard(header: "Dogfooding & Benchmarking", spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: "gauge.with.dots.needle.bottom.50percent")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundColor(.purple)
                    Text("Live Model Dogfooding & Multi-Turn Benchmark")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()
                }

                Text("Validates real-world speedups from KV-cache prefix pinning, measures exact tokens/sec across multi-step developer tool turns, and exercises Turbo Mode compiler self-healing.")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)

                HStack(spacing: 12) {
                    Button(action: {
                        Task {
                            let modelName = localModelManager.discoveredModels.first(where: { $0.snapshotPath == activeModelPath })?.displayName ?? "Ornith-1.5-35B-A3B-FP8"
                            _ = try? await dogfoodRunner.runBenchmark(modelName: modelName)
                        }
                    }) {
                        HStack(spacing: 6) {
                            if dogfoodRunner.isRunning {
                                ProgressView()
                                    .controlSize(.small)
                                Text("Running Turn \(dogfoodRunner.currentTurn)/3...")
                            } else {
                                Image(systemName: "play.circle.fill")
                                Text("Run Dogfood Benchmark")
                            }
                        }
                        .font(.system(size: 11, weight: .medium))
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.purple)
                    .disabled(dogfoodRunner.isRunning)

                    if let report = dogfoodRunner.latestReport {
                        Button(action: {
                            showDogfoodReportModal = true
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "doc.text.magnifyingglass")
                                Text("View Report")
                            }
                            .font(.system(size: 11))
                        }
                        .buttonStyle(.bordered)
                    }
                }

                if let report = dogfoodRunner.latestReport {
                    HStack(spacing: 8) {
                        Text(String(format: "⚡ Prefill Speedup: %.2fx", report.prefixPinningSpeedup))
                            .font(.system(size: 10, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.green.opacity(0.15))
                            .foregroundColor(.green)
                            .cornerRadius(4)

                        Text(String(format: "🚀 Avg Decode: %.1f tok/s", report.averageDecodeTokensPerSec))
                            .font(.system(size: 10, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.blue.opacity(0.15))
                            .foregroundColor(.blue)
                            .cornerRadius(4)

                        Text(report.turboModeVerified ? "⚡ Turbo: PASS" : "⚡ Turbo: FAIL")
                            .font(.system(size: 10, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(report.turboModeVerified ? Color.orange.opacity(0.15) : Color.red.opacity(0.15))
                            .foregroundColor(report.turboModeVerified ? .orange : .red)
                            .cornerRadius(4)

                        Text(report.selfHealingVerified ? "🩺 Self-Heal: PASS" : "🩺 Self-Heal: FAIL")
                            .font(.system(size: 10, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(report.selfHealingVerified ? Color.indigo.opacity(0.15) : Color.red.opacity(0.15))
                            .foregroundColor(report.selfHealingVerified ? .indigo : .red)
                            .cornerRadius(4)
                    }
                }
            }
            .sheet(isPresented: $showDogfoodReportModal) {
                if let report = dogfoodRunner.latestReport {
                    VStack(alignment: .leading, spacing: 14) {
                        HStack {
                            Text("Dogfood Benchmark Report")
                                .font(.headline)
                            Spacer()
                            Button("Done") {
                                showDogfoodReportModal = false
                            }
                        }

                        ScrollView {
                            Text(report.summaryMarkdown)
                                .font(.system(size: 11, design: .monospaced))
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .padding(20)
                    .frame(width: 650, height: 480)
                }
            }

            // Available Built-in Tools List
            SettingsCard(header: "Installed Tool Suite", spacing: 10) {
                Text("Installed Tool Suite (\(AgentHarness.shared.loadedTools.count) Loaded / \(AgentHarness.shared.tools.count) Installed)")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    toolSummaryCard(name: "shell_run", icon: "terminal.fill", desc: "Runs native /bin/zsh shell commands with timeout, stdout/stderr capture, and exit codes.")
                    toolSummaryCard(name: "file_read", icon: "doc.text.fill", desc: "Reads text files with optional 1-indexed line ranges (start_line, end_line).")
                    toolSummaryCard(name: "file_write", icon: "doc.badge.plus", desc: "Creates or overwrites files with automatic parent directory creation and compiler feedback.")
                    toolSummaryCard(name: "file_edit", icon: "square.and.pencil", desc: "Performs precise anchor string search-and-replace edits with self-healing compiler diagnostics.")
                    toolSummaryCard(name: "find_files", icon: "folder.badge.gearshape", desc: "Discovers files and directories using glob matching and max depth.")
                    toolSummaryCard(name: "grep_search", icon: "magnifyingglass", desc: "Fast regex and literal text pattern search across files using ripgrep or grep.")
                    toolSummaryCard(name: "codebase_search", icon: "sparkle.magnifyingglass", desc: "Metal GPU vector search & BM25 hybrid retrieval across indexed codebase AST chunks.")
                    toolSummaryCard(name: "spawn_subagent", icon: "person.2.badge.gearshape", desc: "Spawns isolated background subagents with dedicated roles (Codebase Researcher, Test Runner, Shader Optimizer).")
                    toolSummaryCard(name: "get_subagent_status", icon: "clock.arrow.2.circlepath", desc: "Queries live execution status, transcript steps, and completed summary of a subagent.")
                    toolSummaryCard(name: "send_subagent_message", icon: "bubble.left.and.bubble.right.fill", desc: "Sends instructions or updated directives to a running or completed child subagent.")
                    toolSummaryCard(name: "list_subagents", icon: "list.bullet.rectangle", desc: "Lists all child subagents, their execution durations, and statuses.")
                    toolSummaryCard(name: "git_status", icon: "point.topleft.down.curvedto.point.bottomright.up", desc: "Inspects working tree status, branch tracking, staged files, and unstaged modifications.")
                    toolSummaryCard(name: "git_diff", icon: "plus.forwardslash.minus", desc: "Inspects differences for working tree, staged changes, or across commit targets.")
                    toolSummaryCard(name: "git_commit", icon: "arrow.triangle.branch", desc: "Stages files and commits changes with message validation and safety rails.")
                    toolSummaryCard(name: "find_symbol_definition", icon: "character.textbox", desc: "Locates symbol definitions (class, struct, func, kernel) with exact file coordinates.")
                    toolSummaryCard(name: "find_references", icon: "arrow.triangle.swap", desc: "Locates all call sites, references, and usages of a symbol across project files.")
                    toolSummaryCard(name: "lint_diagnostics", icon: "stethoscope", desc: "Runs native swiftc/metal/clang compiler checks for instant self-healing error reporting.")
                    toolSummaryCard(name: "web_search", icon: "globe", desc: "Live web search via Headless Chrome / Brave. Returns titles, URLs, and real-time snippets.")
                    toolSummaryCard(name: "web_fetch", icon: "arrow.down.doc.fill", desc: "Fetches and reads web pages with automatic HTML stripping and markdown extraction.")
                    toolSummaryCard(name: "complete", icon: "checkmark.seal.fill", desc: "Signals task completion with final structured summary.")
                }
            }
        }
    }

    private func toolSummaryCard(name: String, icon: String, desc: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundColor(.indigo)
                .frame(width: 24, height: 24)
                .background(Color.indigo.opacity(0.1))
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 3) {
                Text(name)
                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                    .foregroundColor(.primary)
                Text(desc)
                    .font(.system(size: 10.5))
                    .foregroundColor(.secondary)
                    .lineLimit(3)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.7))
        .cornerRadius(8)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.primary.opacity(0.06), lineWidth: 1)
        )
    }

    // MARK: - Tab 5: Advanced Diagnostics & Inspector
    private var advancedDiagnosticsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            SettingsCard(spacing: 12) {
                // Search & Filter
                HStack(spacing: 8) {
                    TextField("Search tensors by name...", text: $searchText)
                        .textFieldStyle(.roundedBorder)

                    Picker("Category", selection: $selectedCategory) {
                        ForEach(categoryFilters, id: \.self) { cat in
                            Text(cat).tag(cat)
                        }
                    }
                    .frame(width: 160)
                }

                // Quick Diagnostics Execution Buttons
                HStack(spacing: 8) {
                    Button(action: { onExecuteMoERouter(0) }) {
                        Label("Test Layer 0 Router", systemImage: "point.3.connected.trianglepath.dotted")
                    }
                    .buttonStyle(.bordered)

                    Button(action: { onExecuteFullLayer(0) }) {
                        Label(isExecutingFullLayer ? "Running..." : "Test Block L0", systemImage: "arrow.triangle.merge")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isExecutingFullLayer)

                    Button(action: { onExecuteMultiLayer(targetLayerCount) }) {
                        Label(isExecutingMultiLayer ? "Running..." : "Test Backbone (40L)", systemImage: "square.stack.3d.forward.dottedline.fill")
                    }
                    .buttonStyle(.bordered)
                    .disabled(isExecutingMultiLayer)
                }

                Divider()

                // Tensors Table
                Text("Tensors (\(filteredTensors.count) matching)")
                    .font(.caption)
                    .fontWeight(.semibold)
                    .foregroundColor(.secondary)

                ScrollView {
                    LazyVStack(spacing: 4) {
                        ForEach(filteredTensors.prefix(100)) { tensor in
                            HStack {
                                Text(tensor.name)
                                    .font(.system(size: 11, design: .monospaced))
                                    .lineLimit(1)
                                Spacer()
                                Text("Shard #\(tensor.shardIndex)")
                                    .font(.system(size: 10))
                                    .foregroundColor(.secondary)
                                Text(tensor.dtype)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.purple)
                                Text(tensor.category)
                                    .font(.system(size: 10))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.secondary.opacity(0.1))
                                    .cornerRadius(4)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(tensor.name == selectedTensorID ? Color.purple.opacity(0.12) : Color.clear)
                            .cornerRadius(6)
                            .onTapGesture {
                                selectedTensorID = tensor.name
                            }
                        }
                    }
                }
                .frame(height: 220)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(8)
            }
        }
    }

    // MARK: - Model Profile Management Helpers & Views
    private func loadProfileDraft(modelId: String, type: ModelProfileType) {
        inspectingProfileType = type
        editingDraft = profileManager.getProfile(for: modelId, type: type)
    }

    private func saveProfileDraft(model: DiscoveredModel, type: ModelProfileType) {
        profileManager.saveProfile(for: model.id, type: type, settings: editingDraft)
        profileFeedbackText = "Saved \(type.rawValue) Profile!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            profileFeedbackText = nil
        }
        let isCurrentActive = (activeModelPath != nil && (activeModelPath == model.snapshotPath || activeModelPath == model.weightsEntryPath || (summary != nil && (modelConfig?.modelType ?? "").localizedCaseInsensitiveContains(model.displayName))))
        if isCurrentActive && type == activeProfile {
            applyDraftToActiveSession()
        }
    }

    private func resetProfileDraft(model: DiscoveredModel, type: ModelProfileType) {
        let defaults = GenerationProfileSettings.defaultFor(type: type, modelName: model.displayName)
        editingDraft = defaults
        profileManager.saveProfile(for: model.id, type: type, settings: defaults)
        profileFeedbackText = "Reset to Defaults!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            profileFeedbackText = nil
        }
    }

    private func applyDraftToActiveSession() {
        temperature = editingDraft.temperature
        topP = editingDraft.topP
        minP = editingDraft.minP
        topK = editingDraft.topK
        repetitionPenalty = editingDraft.repetitionPenalty
        presencePenalty = editingDraft.presencePenalty
        maxNewTokens = editingDraft.maxNewTokens
        systemPrompt = editingDraft.systemPrompt
        jetSpecEnabled = editingDraft.jetSpecEnabled
        jetSpecMaxDepth = editingDraft.jetSpecMaxDepth
        jetSpecBranchingFactor = editingDraft.jetSpecBranchingFactor
        jetSpecMaxExpertCap = editingDraft.jetSpecMaxExpertCap
    }

    private func saveCurrentGenerationToProfile(modelId: String?) {
        let settings = GenerationProfileSettings(
            temperature: temperature,
            topP: topP,
            minP: minP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            maxNewTokens: maxNewTokens,
            systemPrompt: systemPrompt,
            jetSpecEnabled: jetSpecEnabled,
            jetSpecMaxDepth: jetSpecMaxDepth,
            jetSpecBranchingFactor: jetSpecBranchingFactor,
            jetSpecMaxExpertCap: jetSpecMaxExpertCap
        )
        profileManager.saveProfile(for: modelId, type: activeProfile, settings: settings)
        profileFeedbackText = "Saved to \(activeProfile.rawValue) Profile!"
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            profileFeedbackText = nil
        }
    }

    @ViewBuilder
    private func modelProfileEditorSection(for model: DiscoveredModel) -> some View {
        let isCurrentActive = (activeModelPath != nil && (activeModelPath == model.snapshotPath || activeModelPath == model.weightsEntryPath || (summary != nil && (modelConfig?.modelType ?? "").localizedCaseInsensitiveContains(model.displayName))))

        VStack(alignment: .leading, spacing: 14) {
            Divider()

            // Header: Model Profile Switcher
            HStack(alignment: .center) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Image(systemName: inspectingProfileType.icon)
                            .foregroundColor(.purple)
                        Text("\(inspectingProfileType.profileDisplayName(for: model.displayName)) Settings")
                            .font(.system(size: 13, weight: .bold))
                        Text("•")
                            .foregroundColor(.secondary)
                        Text(model.displayName)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Text(inspectingProfileType.description(for: model.displayName))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Picker("Profile", selection: $inspectingProfileType) {
                    ForEach(ModelProfileType.allCases) { profile in
                        Label(profile.rawValue, systemImage: profile.icon).tag(profile)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
                .onChange(of: inspectingProfileType) { newType in
                    loadProfileDraft(modelId: model.id, type: newType)
                }
            }
            .padding(.bottom, 4)

            // Sliders Grid
            VStack(spacing: 12) {
                // Temperature
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Temperature")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(format: "%.2f", editingDraft.temperature))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $editingDraft.temperature, in: 0.0...2.0, step: 0.05)
                }

                // Top-P (Nucleus)
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-P (Nucleus)")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(format: "%.2f", editingDraft.topP))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $editingDraft.topP, in: 0.0...1.0, step: 0.05)
                }

                // Min-P
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Min-P (Dynamic Truncation)")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(format: "%.2f", editingDraft.minP))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $editingDraft.minP, in: 0.0...0.5, step: 0.01)
                }

                // Top-K
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-K")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text("\(editingDraft.topK)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: Binding(
                        get: { Float(editingDraft.topK) },
                        set: { editingDraft.topK = Int($0) }
                    ), in: 1...100, step: 1)
                }

                // Repetition Penalty
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Repetition Penalty")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(format: "%.2f", editingDraft.repetitionPenalty))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $editingDraft.repetitionPenalty, in: 1.0...2.0, step: 0.05)
                }

                // Presence Penalty
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Presence Penalty")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text(String(format: "%.2f", editingDraft.presencePenalty))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: $editingDraft.presencePenalty, in: 0.0...2.0, step: 0.05)
                }

                // Max Output Tokens
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Max Output Tokens")
                            .font(.caption)
                            .fontWeight(.medium)
                        Spacer()
                        Text("\(editingDraft.maxNewTokens) tokens")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                    Slider(value: Binding(
                        get: { Float(editingDraft.maxNewTokens) },
                        set: { editingDraft.maxNewTokens = Int($0) }
                    ), in: 32...10000, step: 32)

                    HStack(spacing: 5) {
                        Text("Presets:")
                            .font(.system(size: 9))
                            .foregroundColor(.secondary)
                        ForEach([512, 1024, 2048, 4096, 8192, 10000], id: \.self) { preset in
                            Button("\(preset)") {
                                editingDraft.maxNewTokens = preset
                            }
                            .buttonStyle(.plain)
                            .font(.system(size: 9.5, weight: editingDraft.maxNewTokens == preset ? .bold : .regular, design: .monospaced))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(editingDraft.maxNewTokens == preset ? Color.purple.opacity(0.18) : Color.secondary.opacity(0.08))
                            .foregroundColor(editingDraft.maxNewTokens == preset ? .purple : .primary)
                            .cornerRadius(3)
                        }
                    }
                }

                // JetSpec Toggle
                Toggle(isOn: $editingDraft.jetSpecEnabled) {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.triangle.branch")
                            .foregroundColor(.purple)
                        Text("JetSpec Speculative Tree Acceleration")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
                .toggleStyle(.switch)

                // System Prompt
                VStack(alignment: .leading, spacing: 4) {
                    Text("Profile System Prompt")
                        .font(.caption)
                        .fontWeight(.medium)
                    TextEditor(text: $editingDraft.systemPrompt)
                        .font(.system(.caption, design: .monospaced))
                        .frame(minHeight: 55, maxHeight: 90)
                        .padding(4)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2), lineWidth: 1))
                }
            }
            .padding(12)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(8)

            // Action Buttons
            HStack(spacing: 10) {
                Button(action: {
                    resetProfileDraft(model: model, type: inspectingProfileType)
                }) {
                    Text("Reset to Defaults")
                        .font(.caption)
                }
                .buttonStyle(.bordered)

                Spacer()

                if let feedback = profileFeedbackText {
                    HStack(spacing: 4) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundColor(.green)
                        Text(feedback)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundColor(.green)
                    }
                }

                if isCurrentActive {
                    Button(action: {
                        activeProfile = inspectingProfileType
                        applyDraftToActiveSession()
                        profileFeedbackText = "Applied to Session!"
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                            profileFeedbackText = nil
                        }
                    }) {
                        Label("Apply to Session", systemImage: "bolt.fill")
                            .font(.caption)
                    }
                    .buttonStyle(.bordered)
                }

                Button(action: {
                    saveProfileDraft(model: model, type: inspectingProfileType)
                }) {
                    Label("Save \(inspectingProfileType.rawValue) Profile", systemImage: "square.and.arrow.down")
                        .font(.caption)
                        .fontWeight(.semibold)
                }
                .buttonStyle(.borderedProminent)
            }
        }
        .padding(12)
        .background(Color.purple.opacity(0.03))
        .cornerRadius(10)
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.purple.opacity(0.18), lineWidth: 1)
        )
    }

    private func repackModel(_ model: DiscoveredModel) {
        repackingModelId = model.id
        repackProgress = 0.0
        repackStatus = "Starting repacking..."
        repackError = nil

        Task.detached(priority: .userInitiated) {
            do {
                let dir = URL(fileURLWithPath: model.snapshotPath)
                try ExpertRepacker.shared.repackSafetensors(sourceDir: dir, outputDir: dir) { prog, msg in
                    Task { @MainActor in
                        self.repackProgress = prog
                        self.repackStatus = msg
                    }
                }
                Task { @MainActor in
                    self.repackingModelId = nil
                    self.localModelManager.scanLocalModels()
                }
            } catch {
                Task { @MainActor in
                    self.repackingModelId = nil
                    self.repackError = error.localizedDescription
                }
            }
        }
    }
}
