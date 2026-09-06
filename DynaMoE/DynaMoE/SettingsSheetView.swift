//
//  SettingsSheetView.swift
//  DynaMoE
//

import SwiftUI
import Metal

enum SettingsTab: String, CaseIterable, Identifiable {
    case models = "Models"
    case generation = "Generation"
    case memory = "Memory & SSD"
    case agent = "Agent & Tools"
    case advanced = "Advanced Diagnostics"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .models: return "square.stack.3d.up.fill"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .agent: return "wrench.and.screwdriver.fill"
        case .advanced: return "waveform.path.ecg"
        }
    }
}

struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openWindow) private var openWindow
    @State private var selectedTab: SettingsTab = .models
    @State private var isPromptSavedFeedback: Bool = false
    @State private var repackingModelId: String? = nil
    @State private var repackProgress: Double = 0.0
    @State private var repackStatus: String = ""
    @State private var repackError: String? = nil
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    @AppStorage("dynamoe_agent_tools_enabled") private var isAgentToolsGloballyEnabled: Bool = true
    @AppStorage("dynamoe_agent_turbo_mode") private var isTurboModeEnabled: Bool = false
    @AppStorage("dynamoe_agent_grammar_masking") private var isGrammarMaskingEnabled: Bool = true
    @AppStorage("dynamoe_agent_working_directory") private var agentWorkingDirectory: String = ""
    @AppStorage("dynamoe_max_tool_output_length") private var maxToolOutputLength: Int = 4000
    @AppStorage("dynamoe_max_agent_steps") private var maxAgentSteps: Int = 15
    @AppStorage("dynamoe_brave_search_api_key") private var braveApiKey: String = ""
    @AppStorage("dynamoe_jetspec_enabled") private var jetSpecEnabled: Bool = false
    @AppStorage("dynamoe_jetspec_depth") private var jetSpecMaxDepth: Int = 3
    @AppStorage("dynamoe_jetspec_branching") private var jetSpecBranchingFactor: Int = 2
    @AppStorage("dynamoe_jetspec_expert_cap") private var jetSpecMaxExpertCap: Int = 8
    @ObservedObject private var indexer = CodebaseIndexer.shared

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
        VStack(spacing: 0) {
            // Header
            HStack(spacing: 12) {
                Text("Settings & Diagnostics")
                    .font(.headline)
                    .fontWeight(.bold)

                Spacer()

                Button(action: {
                    openWindow(id: "dynamoe-help")
                }) {
                    Label("Help & Guide", systemImage: "questionmark.circle")
                        .font(.system(size: 11, weight: .medium))
                }
                .buttonStyle(.bordered)
                .help("Open DynaMoE Help & Settings Guide (⌘?)")

                Button("Done") {
                    SettingsWindowManager.shared.close()
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .buttonStyle(.borderedProminent)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(Color(NSColor.windowBackgroundColor))

            Divider()

            // Tab Bar
            Picker("Settings Tab", selection: $selectedTab) {
                ForEach(SettingsTab.allCases) { tab in
                    Label(tab.rawValue, systemImage: tab.icon).tag(tab)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(NSColor.controlBackgroundColor).opacity(0.5))

            Divider()

            // Tab Content
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    switch selectedTab {
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
                .padding(20)
            }
        }
        .frame(minWidth: 680, minHeight: 560)
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Tab 1: Models & Weights Management
    private var modelsSettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            // Section Header with Refresh Button
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Installed Local Models")
                        .font(.headline)
                        .fontWeight(.bold)
                    Text("Auto-discovered from Hugging Face cache (~/.cache/huggingface/hub)")
                        .font(.caption)
                        .foregroundColor(.secondary)
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
                            .font(.caption)
                    }
                }
                .buttonStyle(.bordered)
                .disabled(localModelManager.isScanning)
            }

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
                            VStack(alignment: .leading, spacing: 4) {
                                HStack(spacing: 6) {
                                    Text(model.displayName)
                                        .font(.system(size: 13.5, weight: .semibold))
                                        .foregroundColor(.primary)

                                    if isDefault {
                                        Text("★ DEFAULT")
                                            .font(.system(size: 9.5, weight: .bold))
                                            .foregroundColor(.purple)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.purple.opacity(0.12))
                                            .cornerRadius(4)
                                    }

                                    if isCurrentActive {
                                        Text("● ACTIVE")
                                            .font(.system(size: 9.5, weight: .bold))
                                            .foregroundColor(.green)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.green.opacity(0.12))
                                            .cornerRadius(4)
                                    }

                                    let isFlashMoEPacked = ExpertRepacker.isPackedFormat(dir: URL(fileURLWithPath: model.snapshotPath))
                                    if model.isMoE && isFlashMoEPacked {
                                        HStack(spacing: 3) {
                                            Image(systemName: "bolt.fill")
                                                .font(.system(size: 8))
                                                .foregroundColor(.green)
                                            Text("⚡ FlashMoE")
                                                .font(.system(size: 9.5, weight: .bold))
                                                .foregroundColor(.green)
                                        }
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 2)
                                        .background(Color.green.opacity(0.12))
                                        .cornerRadius(4)
                                    }
                                }

                                HStack(spacing: 8) {
                                    Text(model.author)
                                        .font(.caption)
                                        .foregroundColor(.secondary)

                                    Text("•")
                                        .foregroundColor(.secondary.opacity(0.4))

                                    Text(model.architectureName)
                                        .font(.caption)
                                        .foregroundColor(.secondary)

                                    if let quant = model.quantization {
                                        Text("•")
                                            .foregroundColor(.secondary.opacity(0.4))
                                        Text(quant)
                                            .font(.system(size: 10, weight: .medium, design: .monospaced))
                                            .foregroundColor(.indigo)
                                    }

                                    Text("•")
                                        .foregroundColor(.secondary.opacity(0.4))

                                    Text(model.formattedSize)
                                        .font(.system(size: 11, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }
                            }

                            Spacer()

                            // Action Buttons: FlashMoE Repack, Default toggle & Load button
                            HStack(spacing: 8) {
                                let isFlashMoEPacked = ExpertRepacker.isPackedFormat(dir: URL(fileURLWithPath: model.snapshotPath))
                                if model.isMoE && !isFlashMoEPacked {
                                    if repackingModelId == model.id {
                                        VStack(alignment: .trailing, spacing: 2) {
                                            ProgressView(value: repackProgress)
                                                .progressViewStyle(.linear)
                                                .frame(width: 80)
                                            Text(repackStatus)
                                                .font(.system(size: 8))
                                                .foregroundColor(.secondary)
                                                .lineLimit(1)
                                        }
                                    } else {
                                        Button(action: {
                                            repackModel(model)
                                        }) {
                                            HStack(spacing: 4) {
                                                Image(systemName: "bolt.badge.automatic.fill")
                                                    .font(.system(size: 10))
                                                    .foregroundColor(.yellow)
                                                Text("FlashMoE Repack")
                                                    .font(.caption2)
                                                    .fontWeight(.medium)
                                            }
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 5)
                                            .background(Color.yellow.opacity(0.12))
                                            .cornerRadius(6)
                                        }
                                        .buttonStyle(.plain)
                                        .help("Losslessly repack weights into FlashMoE layout for ~100x faster generation")
                                    }
                                }

                                Button(action: {
                                    if isDefault {
                                        localModelManager.setDefaultModel(id: nil)
                                    } else {
                                        localModelManager.setDefaultModel(id: model.id)
                                    }
                                }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: isDefault ? "star.fill" : "star")
                                            .font(.system(size: 11))
                                            .foregroundColor(isDefault ? .orange : .secondary)
                                        Text(isDefault ? "Default" : "Set Default")
                                            .font(.caption)
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(isDefault ? Color.orange.opacity(0.1) : Color.secondary.opacity(0.06))
                                    .cornerRadius(6)
                                }
                                .buttonStyle(.plain)
                                .help("Make this model load by default for all new conversations")

                                Button(action: {
                                    withAnimation(.easeInOut(duration: 0.2)) {
                                        if selectedModelForProfiles == model.id {
                                            selectedModelForProfiles = nil
                                        } else {
                                            selectedModelForProfiles = model.id
                                            loadProfileDraft(modelId: model.id, type: inspectingProfileType)
                                        }
                                    }
                                }) {
                                    HStack(spacing: 4) {
                                        Image(systemName: selectedModelForProfiles == model.id ? "slider.horizontal.3" : "slider.horizontal.2.square")
                                            .font(.system(size: 11))
                                            .foregroundColor(.purple)
                                        Text("Profiles")
                                            .font(.caption)
                                            .fontWeight(selectedModelForProfiles == model.id ? .semibold : .regular)
                                    }
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 5)
                                    .background(selectedModelForProfiles == model.id ? Color.purple.opacity(0.18) : Color.purple.opacity(0.08))
                                    .cornerRadius(6)
                                }
                                .buttonStyle(.plain)
                                .help("Configure Coder and Assistant generation profiles for \(model.displayName)")

                                if isCurrentActive {
                                    HStack(spacing: 4) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .font(.system(size: 11))
                                            .foregroundColor(.green)
                                        Text("Loaded")
                                            .font(.caption)
                                            .fontWeight(.semibold)
                                            .foregroundColor(.green)
                                    }
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 5)
                                    .background(Color.green.opacity(0.1))
                                    .cornerRadius(6)
                                } else {
                                    Button(action: {
                                        onLoadDiscoveredModel(model)
                                    }) {
                                        Text("Load Model")
                                            .font(.caption)
                                            .fontWeight(.semibold)
                                    }
                                    .buttonStyle(.borderedProminent)
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

            Divider()
                .padding(.vertical, 4)

            // Other Model Actions / Custom Folder Selector
            VStack(alignment: .leading, spacing: 12) {
                Text("Custom Local Paths & Fallbacks")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Custom Folder or Safetensors File")
                            .font(.system(size: 13, weight: .medium))
                        Text("Load a model outside of Hugging Face cache (e.g. external SSD)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button("Browse Folder / Index...", action: onSelectModel)
                        .buttonStyle(.bordered)
                }
                .padding(12)
                .background(Color.secondary.opacity(0.04))
                .cornerRadius(8)

                // Tokenizer Card
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Custom Tokenizer (tokenizer.json)")
                            .font(.system(size: 13, weight: .medium))
                        Text(tokenizer != nil ? "Tokenizer Loaded & Ready ✅" : "No tokenizer loaded.")
                            .font(.caption)
                            .foregroundColor(tokenizer != nil ? .green : .secondary)
                    }
                    Spacer()
                    Button(tokenizer == nil ? "Load tokenizer.json" : "Replace Tokenizer", action: onSelectTokenizer)
                        .buttonStyle(.bordered)
                }
                .padding(12)
                .background(Color.secondary.opacity(0.04))
                .cornerRadius(8)

                // GPU & Metal Backend Status
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Metal GPU Acceleration")
                            .font(.system(size: 13, weight: .medium))
                        Text(metalStatus)
                            .font(.caption)
                            .foregroundColor(metalStatus.contains("✅") ? .green : .secondary)
                    }
                    Spacer()
                    if let device = MTLCreateSystemDefaultDevice() {
                        Text(device.name)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.purple.opacity(0.1))
                            .cornerRadius(6)
                    }
                }
                .padding(12)
                .background(Color.secondary.opacity(0.04))
                .cornerRadius(8)
            }
        }
    }

    // MARK: - Tab 2: Generation & Sampler
    private var generationSettingsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Inference & Sampling Parameters")
                .font(.headline)

            // Active Profile Banner
            HStack(spacing: 12) {
                Image(systemName: activeProfile.icon)
                    .font(.system(size: 20))
                    .foregroundColor(.purple)
                let currentModelIdentifier = activeModelPath ?? modelConfig?.modelType ?? localModelManager.defaultModelId
                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text("Active Profile: \(activeProfile.profileDisplayName(for: currentModelIdentifier))")
                            .font(.system(size: 13, weight: .bold))
                        if let path = activeModelPath,
                           let model = localModelManager.discoveredModels.first(where: { $0.snapshotPath == path || $0.weightsEntryPath == path }) {
                            Text("• \(model.displayName)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else if let modelType = modelConfig?.modelType {
                            Text("• \(modelType)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else {
                            Text("• \(detectedArchitecture.rawValue)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    Text(activeProfile.description(for: currentModelIdentifier))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                Spacer()

                Picker("Profile", selection: $activeProfile) {
                    ForEach(ModelProfileType.allCases) { profile in
                        Label(profile.rawValue, systemImage: profile.icon).tag(profile)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 220)
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
                    HStack(spacing: 4) {
                        Image(systemName: "square.and.arrow.down")
                        Text("Save to \(activeProfile.rawValue)")
                    }
                    .font(.caption)
                }
                .buttonStyle(.bordered)
            }
            .padding(12)
            .background(Color.purple.opacity(0.08))
            .cornerRadius(10)

            VStack(spacing: 16) {
                // Temperature
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Temperature")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", temperature))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: $temperature, in: 0.0...2.0, step: 0.05)
                }

                // Top-P (Nucleus Sampling)
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-P (Nucleus)")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", topP))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: $topP, in: 0.0...1.0, step: 0.05)
                }

                // Min-P (Dynamic Truncation)
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Min-P (Confidence Truncation)")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", minP))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: $minP, in: 0.0...0.5, step: 0.01)
                }

                // Top-K Filtering
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-K")
                            .font(.subheadline)
                        Spacer()
                        Text("\(topK)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: Binding(
                        get: { Float(topK) },
                        set: { topK = Int($0) }
                    ), in: 1...100, step: 1)
                }

                // Repetition Penalty
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Repetition Penalty")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", repetitionPenalty))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: $repetitionPenalty, in: 1.0...2.0, step: 0.05)
                }

                // Presence Penalty
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Presence Penalty")
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", presencePenalty))
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: $presencePenalty, in: 0.0...2.0, step: 0.05)
                }

                // Max New Tokens
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("Max Output Tokens")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        Spacer()
                        Text("\(maxNewTokens) tokens")
                            .font(.system(.caption, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.purple)
                    }

                    Slider(value: Binding(
                        get: { Float(maxNewTokens) },
                        set: { maxNewTokens = Int($0) }
                    ), in: 32...10000, step: 32)

                    // Quick Token Presets
                    HStack(spacing: 6) {
                        Text("Presets:")
                            .font(.caption2)
                            .foregroundColor(.secondary)
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
                }

                // JetSpec Causal Speculative Tree Acceleration
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.triangle.branch")
                                    .foregroundColor(.purple)
                                Text("JetSpec Causal Speculative Tree")
                                    .font(.subheadline)
                                    .fontWeight(.semibold)
                            }
                            Text("Accelerates generation via parallel tree drafting and dynamic MoE budget pruning on resident standard attention models. Gated DeltaNet recurrent models (such as Ornith 1.5) and disk-streamed models automatically use optimized direct execution for maximum speed and state accuracy.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Toggle("", isOn: $jetSpecEnabled)
                            .toggleStyle(.switch)
                    }

                    if jetSpecEnabled {
                        // Max Tree Depth
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Max Tree Depth")
                                    .font(.caption)
                                Spacer()
                                Text("\(jetSpecMaxDepth)")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.purple)
                            }
                            Slider(value: Binding(
                                get: { Float(jetSpecMaxDepth) },
                                set: { jetSpecMaxDepth = Int($0) }
                            ), in: 1...5, step: 1)
                        }

                        // Branching Factor
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Branching Factor")
                                    .font(.caption)
                                Spacer()
                                Text("\(jetSpecBranchingFactor)")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.purple)
                            }
                            Slider(value: Binding(
                                get: { Float(jetSpecBranchingFactor) },
                                set: { jetSpecBranchingFactor = Int($0) }
                            ), in: 1...4, step: 1)
                        }

                        // Max Unique MoE Expert Cap
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text("Max Expert Cap per Step")
                                    .font(.caption)
                                Spacer()
                                Text("\(jetSpecMaxExpertCap) experts")
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.purple)
                            }
                            Slider(value: Binding(
                                get: { Float(jetSpecMaxExpertCap) },
                                set: { jetSpecMaxExpertCap = Int($0) }
                            ), in: 2...16, step: 1)
                        }
                    }
                }
                .padding(12)
                .background(Color.purple.opacity(0.06))
                .cornerRadius(8)

                // System Prompt Editor & Conjunction Combination
                VStack(alignment: .leading, spacing: 8) {
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
            .padding(14)
            .background(Color.secondary.opacity(0.06))
            .cornerRadius(10)
        }
    }

    // MARK: - Tab 3: Memory & Working Set
    private var memorySettingsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Memory & Dynamic SSD Expert Paging")
                .font(.headline)

            VStack(alignment: .leading, spacing: 16) {
                // Execution Mode Picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("Memory Execution Mode")
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    Picker("Mode", selection: $memoryExecutionMode) {
                        ForEach(MemoryExecutionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                // Budget Mode Picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("RAM Working Set Budget")
                        .font(.subheadline)
                        .fontWeight(.semibold)

                    Picker("Budget", selection: $memoryBudgetMode) {
                        ForEach(MemoryBudgetMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                // KV-Cache Precision Picker
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("KV-Cache Precision & Compression")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Spacer()
                        Text(kvCachePrecision == .fp8 ? "75% VRAM Reduction" : (kvCachePrecision == .fp16 ? "50% VRAM Reduction" : "Original"))
                            .font(.caption2)
                            .foregroundColor(.indigo)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.indigo.opacity(0.1))
                            .cornerRadius(4)
                    }

                    Picker("KV Precision", selection: $kvCachePrecision) {
                        ForEach(KVCachePrecision.allCases) { prec in
                            Text(prec.rawValue).tag(prec)
                        }
                    }
                    .pickerStyle(.segmented)

                    Text("FP16 (Half) cuts KV-cache memory by 50% with ~2x memory bandwidth during autoregressive GQA decoding. FP8 cuts memory footprint by 75% for ultra-long context windows.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }

                // Speculative Prefetching & Layer Lookahead
                VStack(alignment: .leading, spacing: 8) {
                    Toggle(isOn: $speculativePrefetchEnabled) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Speculative MoE Expert & Layer Prefetching")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                            Text("Asynchronously warms next-layer backbone weights and predicted expert slices in background before GPU execution.")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    .toggleStyle(.switch)

                    if speculativePrefetchEnabled {
                        HStack {
                            Text("Prefetch Lookahead Depth:")
                                .font(.caption)
                                .fontWeight(.medium)
                            Spacer()
                            Picker("Lookahead", selection: $prefetchLookaheadDepth) {
                                Text("1 Layer").tag(1)
                                Text("2 Layers").tag(2)
                            }
                            .pickerStyle(.segmented)
                            .frame(maxWidth: 180)
                        }
                        .padding(.top, 2)
                    }
                }

                Divider()

                // Telemetry Metrics Grid
                HStack(spacing: 14) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 3) {
                            Text("WORKING SET RAM")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary)
                            Image(systemName: "info.circle")
                                .font(.system(size: 8))
                                .foregroundColor(.secondary)
                        }
                        Text(String(format: "%.2f GB", currentRssGB))
                            .font(.system(.subheadline, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.indigo)
                        Text(String(format: "Heap: %.2f GB", getProcessResidentMemoryGB()))
                            .font(.system(size: 8, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                    .help(String(format: "Working Set RAM: %.2f GB\nProcess Heap (Activity Monitor): %.2f GB\nUnified Memory Cache: %.2f GB\n\nApple Silicon places clean zero-copy model weights in Darwin's Unified Memory Buffer Cache, which Activity Monitor excludes from process footprint.", currentRssGB, getProcessResidentMemoryGB(), max(0, currentRssGB - getProcessResidentMemoryGB())))

                    VStack(alignment: .leading, spacing: 2) {
                        Text("RESIDENT EXPERTS")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text("\(residentExpertCount) / \(totalExpertCount)")
                            .font(.system(.subheadline, design: .monospaced))
                            .fontWeight(.bold)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("CACHE HIT RATE")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.1f%%", cacheHitRate))
                            .font(.system(.subheadline, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.green)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("PREFETCH EFFICIENCY")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.1f%%", prefetchEfficiency))
                            .font(.system(.subheadline, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.purple)
                    }
                }

                Divider()

                // Cache Actions
                HStack(spacing: 12) {
                    Button(action: onFlushCache) {
                        Label("Flush Expert Cache", systemImage: "trash")
                    }
                    .buttonStyle(.bordered)

                    Button(action: onPreFaultAll) {
                        Label("Pre-Fault All Weights into RAM", systemImage: "bolt.fill")
                    }
                    .buttonStyle(.bordered)
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.06))
            .cornerRadius(10)
        }
    }

    // MARK: - Tab 4: Agent & Tool Execution Suite
    private var agentSettingsSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Agent Mode & Tool Calling Configuration")
                .font(.headline)

            VStack(alignment: .leading, spacing: 14) {
                // Enable Agent Tools Toggle
                Toggle(isOn: $isAgentToolsGloballyEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Enable Agent Tools by Default")
                            .font(.system(size: 13, weight: .medium))
                        Text("Equips models with shell execution, file reading/writing, and semantic search via ChatML <tool_call> tags.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .toggleStyle(.switch)

                Divider()

                // Turbo Mode (Auto-Approve Mutations)
                Toggle(isOn: $isTurboModeEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text("Agent Turbo Mode (Auto-Approve)")
                                .font(.system(size: 13, weight: .medium))
                            Image(systemName: "bolt.fill")
                                .foregroundColor(.orange)
                                .font(.system(size: 11))
                        }
                        Text("When enabled, file writes, edits, and shell commands execute automatically without individual authorization prompts (similar to Antigravity). When disabled, actions require explicit approval.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .toggleStyle(.switch)

                Divider()

                // Grammar-Constrained Logit Masking
                Toggle(isOn: $isGrammarMaskingEnabled) {
                    VStack(alignment: .leading, spacing: 2) {
                        HStack(spacing: 6) {
                            Text("Grammar-Constrained Tool Sampling")
                                .font(.system(size: 13, weight: .medium))
                            Image(systemName: "checkmark.shield.fill")
                                .foregroundColor(.green)
                                .font(.system(size: 11))
                        }
                        Text("Dynamically sets disallowed token logits to -Float.infinity during tool calls, mathematically preventing syntax and schema errors.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .toggleStyle(.switch)

                Divider()

                // Working Directory Picker
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

                Divider()

                // Output Length Slider
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Tool Output Length")
                                .font(.system(size: 13, weight: .medium))
                            Text("Protects context window from large outputs via intelligent head/tail truncation.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text("\(maxToolOutputLength) chars")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: Binding(
                        get: { Double(maxToolOutputLength) },
                        set: { maxToolOutputLength = Int($0) }
                    ), in: 1000...16000, step: 500)
                }

                Divider()

                // Max Agent Iteration Steps
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Max Multi-Step Agent Iterations")
                                .font(.system(size: 13, weight: .medium))
                            Text("Maximum number of reasoning and tool execution turns per user prompt.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Spacer()
                        Text("\(maxAgentSteps) steps")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: Binding(
                        get: { Double(maxAgentSteps) },
                        set: { maxAgentSteps = Int($0) }
                    ), in: 1...30, step: 1)
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(10)

            // Web Search Engine Configuration
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "globe")
                        .foregroundColor(.blue)
                    Text("Web Search Engine")
                        .font(.subheadline)
                        .fontWeight(.semibold)
                    Spacer()
                    Text(braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "DuckDuckGo (Free & Built-in)" : "Brave Search API")
                        .font(.caption)
                        .fontWeight(.medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.green.opacity(0.12) : Color.orange.opacity(0.12))
                        .foregroundColor(braveApiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? .green : .orange)
                        .cornerRadius(6)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("Brave Search API Key (Optional)")
                        .font(.system(size: 12, weight: .medium))
                    SecureField("Paste Brave Search API token (e.g. BSA...)", text: $braveApiKey)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 11, design: .monospaced))

                    Text("DuckDuckGo HTML search is active by default with zero configuration or API keys needed. You can optionally paste a Brave Search API key for dedicated high-speed JSON queries.")
                        .font(.system(size: 10.5))
                        .foregroundColor(.secondary)
                }
            }
            .padding(14)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(10)

            // Semantic Codebase Indexing & Local RAG Configuration
            VStack(alignment: .leading, spacing: 10) {
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
            .padding(14)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(10)

            // Subagent & Multi-Agent Delegation Card
            VStack(alignment: .leading, spacing: 10) {
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
            .padding(14)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(10)

            // Deep Developer Tooling (Native Git & SourceKit-LSP) Card
            VStack(alignment: .leading, spacing: 10) {
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
            .padding(14)
            .background(Color.secondary.opacity(0.04))
            .cornerRadius(10)

            // Available Built-in Tools List
            VStack(alignment: .leading, spacing: 10) {
                Text("Installed Tool Suite (\(AgentHarness.shared.availableToolDefinitions.count) Tools)")
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
                    toolSummaryCard(name: "web_search", icon: "globe", desc: "Live web search via DuckDuckGo / Brave. Returns titles, URLs, and real-time snippets.")
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
            Text("Advanced Tensor Diagnostics & Inspector")
                .font(.headline)

            VStack(alignment: .leading, spacing: 12) {
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
            .padding(14)
            .background(Color.secondary.opacity(0.06))
            .cornerRadius(10)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
                            .foregroundColor(.purple)
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
