//
//  SettingsSheetView.swift
//  DynaMoE
//

import SwiftUI
import Metal

enum SettingsTab: String, CaseIterable, Identifiable {
    case model = "Model"
    case generation = "Generation"
    case memory = "Memory & SSD"
    case advanced = "Advanced Diagnostics"

    var id: String { rawValue }

    var icon: String {
        switch self {
        case .model: return "cube.transparent"
        case .generation: return "slider.horizontal.3"
        case .memory: return "memorychip"
        case .advanced: return "waveform.path.ecg"
        }
    }
}

struct SettingsSheetView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab: SettingsTab = .model

    // Model & Tokenizer bindings
    var summary: ModelSummary?
    var modelConfig: ModelConfig? = nil
    var tokenizer: DynaMoeTokenizer?
    var metalStatus: String
    var detectedArchitecture: ModelArchitectureType
    var onSelectModel: () -> Void
    var onSelectTokenizer: () -> Void

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
                    case .model:
                        modelSettingsSection
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
        .frame(minWidth: 640, minHeight: 520)
        .background(Color(NSColor.windowBackgroundColor))
    }

    // MARK: - Tab 1: Model & Weights
    private var modelSettingsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Model & Weights Configuration")
                .font(.headline)

            VStack(alignment: .leading, spacing: 12) {
                // Model Weights Card
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Model Weights (Safetensors)")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        if let summary = summary {
                            Text("\(summary.layerCount) Layers • \(summary.tensors.count) Tensors • \(detectedArchitecture.shortName)")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        } else {
                            Text("No model weights loaded.")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }
                    Spacer()
                    Button("Select Model Folder / Index", action: onSelectModel)
                        .buttonStyle(.borderedProminent)
                }
                .padding(14)
                .background(Color.secondary.opacity(0.06))
                .cornerRadius(10)

                // Tokenizer Card
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Tokenizer (tokenizer.json)")
                            .font(.subheadline)
                            .fontWeight(.semibold)
                        Text(tokenizer != nil ? "Tokenizer Loaded & Ready ✅" : "No tokenizer loaded.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                    Button(tokenizer == nil ? "Load tokenizer.json" : "Replace Tokenizer", action: onSelectTokenizer)
                        .buttonStyle(.bordered)
                }
                .padding(14)
                .background(Color.secondary.opacity(0.06))
                .cornerRadius(10)

                // GPU & Metal Backend Status
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Metal GPU Acceleration")
                            .font(.subheadline)
                            .fontWeight(.semibold)
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
                .padding(14)
                .background(Color.secondary.opacity(0.06))
                .cornerRadius(10)
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
