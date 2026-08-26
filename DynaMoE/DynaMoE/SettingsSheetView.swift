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
    case advanced = "Advanced Diagnostics"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .models: return "square.stack.3d.up.fill"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .advanced: return "waveform.path.ecg"
        }
    }
}

struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab: SettingsTab = .models
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared

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
    @Binding var topK: Int
    @Binding var repetitionPenalty: Float
    @Binding var maxNewTokens: Int
    @Binding var systemPrompt: String
    @Binding var targetLayerCount: Int

    // Memory & Working Set bindings
    @Binding var memoryExecutionMode: MemoryExecutionMode
    @Binding var memoryBudgetMode: MemoryBudgetMode
    var currentRssGB: Double
    var residentExpertCount: Int
    var totalExpertCount: Int
    var cacheHitRate: Double
    var lastPagingLatencyMs: Double
    var pagingStatusMessage: String?
    var onFlushCache: () -> Void
    var onPreFaultAll: () -> Void

    // Advanced Diagnostics bindings & callbacks
    var filteredTensors: [TensorMetadata]
    @Binding var searchText: String
    @Binding var selectedCategory: String
    var categoryFilters: [String]
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
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    switch selectedTab {
                    case .models:
                        modelsSettingsSection
                    case .generation:
                        generationSettingsSection
                    case .memory:
                        memorySettingsSection
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

                            // Action Buttons: Default toggle & Load button
                            HStack(spacing: 8) {
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
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Max Output Tokens")
                            .font(.subheadline)
                        Spacer()
                        Text("\(maxNewTokens)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.purple)
                    }
                    Slider(value: Binding(
                        get: { Float(maxNewTokens) },
                        set: { maxNewTokens = Int($0) }
                    ), in: 32...4096, step: 32)
                }

                // System Prompt Editor
                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text("System Prompt")
                            .font(.subheadline)
                            .fontWeight(.semibold)

                        Spacer()

                        let defaultPrompt = ModelConfig.resolveDefaultSystemPrompt(config: modelConfig, summary: summary)
                        let isNanbeige = defaultPrompt.contains("南北阁")

                        if isNanbeige {
                            Text("Nanbeige Preset")
                                .font(.system(size: 10, weight: .semibold))
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.purple.opacity(0.12))
                                .foregroundColor(.purple)
                                .cornerRadius(4)
                        }

                        Button("Reset to Model Default") {
                            systemPrompt = defaultPrompt
                        }
                        .buttonStyle(.plain)
                        .font(.caption2)
                        .foregroundColor(.purple)
                    }

                    TextEditor(text: $systemPrompt)
                        .font(.system(.caption, design: .monospaced))
                        .frame(height: 80)
                        .padding(6)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(8)
                        .overlay(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                        )

                    Text("The system prompt is dynamically set based on the active model architecture. You can customize or clear it above.")
                        .font(.caption2)
                        .foregroundColor(.secondary)
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

                Divider()

                // Telemetry Metrics Grid
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("PHYSICAL RSS")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.2f GB", currentRssGB))
                            .font(.system(.title3, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.indigo)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("RESIDENT EXPERTS")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text("\(residentExpertCount) / \(totalExpertCount)")
                            .font(.system(.title3, design: .monospaced))
                            .fontWeight(.bold)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("CACHE HIT RATE")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundColor(.secondary)
                        Text(String(format: "%.1f%%", cacheHitRate))
                            .font(.system(.title3, design: .monospaced))
                            .fontWeight(.bold)
                            .foregroundColor(.green)
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

    // MARK: - Tab 4: Advanced Diagnostics & Inspector
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
}
