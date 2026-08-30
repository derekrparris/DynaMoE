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
    case about = "About"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .models: return "square.stack.3d.up.fill"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .agent: return "wrench.and.screwdriver.fill"
        case .advanced: return "waveform.path.ecg"
        case .about: return "info.circle.fill"
        }
    }
}

struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab: SettingsTab = .models
    @State private var isPromptSavedFeedback: Bool = false
    @State private var repackingModelId: String? = nil
    @State private var repackProgress: Double = 0.0
    @State private var repackStatus: String = ""
    @State private var repackError: String? = nil
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    @AppStorage("dynamoe_agent_tools_enabled") private var isAgentToolsGloballyEnabled: Bool = true
    @AppStorage("dynamoe_agent_working_directory") private var agentWorkingDirectory: String = ""
    @AppStorage("dynamoe_max_tool_output_length") private var maxToolOutputLength: Int = 4000
    @AppStorage("dynamoe_max_agent_steps") private var maxAgentSteps: Int = 15

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
    @Binding var maxNewTokens: Int
    @Binding var systemPrompt: String
    @Binding var targetLayerCount: Int

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
            HStack {
                Text("Settings & Diagnostics")
                    .font(.headline)
                    .fontWeight(.bold)

                Spacer()

                Button("Done") {
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
            if selectedTab == .about {
                AboutDynaMoEView()
            } else {
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
                        case .about:
                            EmptyView()
                        }
                    }
                    .padding(20)
                }
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
                        .padding(12)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(10)
                        .overlay(
                            RoundedRectangle(cornerRadius: 10)
                                .stroke(isCurrentActive ? Color.purple.opacity(0.3) : Color.primary.opacity(0.06), lineWidth: 1)
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
                        Text("PHYSICAL RSS")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.2f GB", currentRssGB))
                            .font(.system(.subheadline, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.indigo)
                    }

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

                // Working Directory Picker
                VStack(alignment: .leading, spacing: 6) {
                    Text("Agent Working Directory (CWD)")
                        .font(.system(size: 13, weight: .medium))
                    Text("Commands like zsh, find, and ripgrep run relative to this base directory.")
                        .font(.caption)
                        .foregroundColor(.secondary)

                    HStack(spacing: 8) {
                        Image(systemName: "folder.fill")
                            .foregroundColor(.secondary)
                        Text(agentWorkingDirectory.isEmpty ? FileManager.default.currentDirectoryPath : agentWorkingDirectory)
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.primary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(Color.secondary.opacity(0.06))
                            .cornerRadius(6)

                        Button("Browse...") {
                            let panel = NSOpenPanel()
                            panel.canChooseFiles = false
                            panel.canChooseDirectories = true
                            panel.allowsMultipleSelection = false
                            panel.canCreateDirectories = true
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

            // Available Built-in Tools List
            VStack(alignment: .leading, spacing: 10) {
                Text("Installed Tool Suite (\(AgentHarness.shared.availableToolDefinitions.count) Tools)")
                    .font(.subheadline)
                    .fontWeight(.semibold)

                LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                    toolSummaryCard(name: "shell_run", icon: "terminal.fill", desc: "Runs native /bin/zsh shell commands with timeout, stdout/stderr capture, and exit codes.")
                    toolSummaryCard(name: "file_read", icon: "doc.text.fill", desc: "Reads text files with optional 1-indexed line ranges (start_line, end_line).")
                    toolSummaryCard(name: "file_write", icon: "doc.badge.plus", desc: "Creates or overwrites files with automatic parent directory creation.")
                    toolSummaryCard(name: "file_edit", icon: "square.and.pencil", desc: "Performs precise anchor string search-and-replace edits.")
                    toolSummaryCard(name: "find_files", icon: "folder.badge.gearshape", desc: "Discovers files and directories using glob matching and max depth.")
                    toolSummaryCard(name: "grep_search", icon: "magnifyingglass", desc: "Fast regex and literal text pattern search across files using ripgrep or grep.")
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
