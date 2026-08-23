import SwiftUI
import UniformTypeIdentifiers
import Metal

struct LayerTelemetry: Identifiable {
    var id: Int { layerIndex }
    let layerIndex: Int
    let durationMs: Double
    let topExperts: [Int]
    let l2Norm: Float
}

struct TokenPrediction: Identifiable {
    var id: Int { rank }
    let rank: Int
    let tokenId: UInt32
    let tokenString: String
    let logit: Float
    let probability: Float
}

extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    @State private var engine: DynaMoeEngine? = nil
    @State private var tokenizer: DynaMoeTokenizer? = nil
    @State private var summary: ModelSummary? = nil
    @State private var errorMessage: String? = nil
    @State private var metalStatus: String = "GPU Status: Waiting for weights..."
    @State private var searchText: String = ""
    @State private var selectedCategory: String = "All"
    
    // Multi-Shard Metal Buffers
    @State private var shardBuffers: [UInt32: MTLBuffer] = [:]
    
    // File Importers
    @State private var isWeightImporterPresented: Bool = false
    @State private var isTokenizerImporterPresented: Bool = false
    
    // Tokenizer Playground State
    @State private var promptInput: String = "Hello DynaMoE, routing tokens to experts..."
    @State private var tokenIDsOutput: String = "Load a tokenizer.json file to tokenize text"
    
    @State private var selectedTensorID: String? = nil
    @State private var gpuComputeOutput: String? = nil

    // Active Token Embedding & MoE Routing State
    @State private var activeH0Buffer: MTLBuffer? = nil
    @State private var activeTokenCount: Int = 0
    @State private var activeHiddenDim: Int = 2048
    
    @State private var selectedLayerForRouting: Int = 0
    @State private var routedExperts: [(id: Int, weight: Float)] = []
    @State private var sharedExpertWeight: Float? = nil
    @State private var routerStatusText: String? = nil

    // MoE Layer MLP Output State
    @State private var activeHmlpBuffer: MTLBuffer? = nil
    @State private var layerMlpStatusText: String? = nil
    @State private var layerMlpSampleOutput: String? = nil
    @State private var isExecutingMlp: Bool = false

    // Full Transformer Layer Execution State (h_l -> h_l+1)
    @State private var activeH1Buffer: MTLBuffer? = nil
    @State private var fullLayerStatusText: String? = nil
    @State private var fullLayerSampleOutput: String? = nil
    @State private var isExecutingFullLayer: Bool = false

    // Multi-Layer Backbone Execution State (h_0 -> h_N)
    @State private var targetLayerCount: Int = 40
    @State private var activeHFinalBuffer: MTLBuffer? = nil
    @State private var multiLayerStatusText: String? = nil
    @State private var multiLayerTelemetry: [LayerTelemetry] = []
    @State private var isExecutingMultiLayer: Bool = false

    // Final RMSNorm & LM Head Vocabulary Projection State
    @State private var activeLogitsBuffer: MTLBuffer? = nil
    @State private var topTokenPredictions: [TokenPrediction] = []
    @State private var lmHeadStatusText: String? = nil
    @State private var isExecutingLMHead: Bool = false

    let categoryFilters = ["All", "Self-Attention", "MoE Router", "Routed Expert", "Shared Expert", "Embedding", "LM Head"]

    var filteredTensors: [TensorMetadata] {
        guard let tensors = summary?.tensors else { return [] }
        return tensors.filter { tensor in
            let matchesSearch = searchText.isEmpty || tensor.name.localizedCaseInsensitiveContains(searchText)
            let matchesCategory = (selectedCategory == "All") || tensor.category.contains(selectedCategory)
            return matchesSearch && matchesCategory
        }
    }

    var selectedTensor: TensorMetadata? {
        summary?.tensors.first(where: { $0.name == selectedTensorID })
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("DynaMoE Engine & Tokenizer Inspector")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    if let summary = summary {
                        HStack(spacing: 12) {
                            Text(metalStatus)
                                .foregroundColor(metalStatus.contains("✅") ? .green : .secondary)
                            Text("•")
                            Text("\(summary.layerCount) Layers")
                                .fontWeight(.semibold)
                            if summary.maxExpertId > 0 {
                                Text("•")
                                Text("\(summary.maxExpertId) Routed Experts/Layer")
                                    .foregroundColor(.purple)
                                    .fontWeight(.semibold)
                            }
                        }
                        .font(.subheadline)
                    } else {
                        Text(metalStatus)
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
                }
                
                Spacer()
                
                HStack(spacing: 8) {
                    Button(tokenizer == nil ? "Load tokenizer.json" : "Tokenizer Loaded ✅") {
                        #if os(macOS)
                        selectTokenizerWithOpenPanel()
                        #else
                        isTokenizerImporterPresented = true
                        #endif
                    }
                    .buttonStyle(.bordered)
                    .fileImporter(
                        isPresented: $isTokenizerImporterPresented,
                        allowedContentTypes: [.json, .data],
                        allowsMultipleSelection: false
                    ) { result in
                        switch result {
                        case .success(let urls):
                            guard let url = urls.first else { return }
                            if url.startAccessingSecurityScopedResource() {
                                defer { url.stopAccessingSecurityScopedResource() }
                                loadTokenizer(filePath: url.path)
                            }
                        case .failure(let error):
                            errorMessage = error.localizedDescription
                        }
                    }
                    
                    Button("Select Model Folder / Index") {
                        #if os(macOS)
                        selectModelWithOpenPanel()
                        #else
                        isWeightImporterPresented = true
                        #endif
                    }
                    .buttonStyle(.borderedProminent)
                    .fileImporter(
                        isPresented: $isWeightImporterPresented,
                        allowedContentTypes: [.folder, .json, .data],
                        allowsMultipleSelection: true
                    ) { result in
                        switch result {
                        case .success(let urls):
                            guard let primaryUrl = urls.first else { return }
                            
                            // Start security access for all selected items/folders
                            for url in urls {
                                _ = url.startAccessingSecurityScopedResource()
                            }
                            
                            // If a directory was picked, locate model.safetensors.index.json inside it
                            var targetPath = primaryUrl.path
                            if primaryUrl.hasDirectoryPath {
                                let indexPath = primaryUrl.appendingPathComponent("model.safetensors.index.json").path
                                if FileManager.default.fileExists(atPath: indexPath) {
                                    targetPath = indexPath
                                }
                            }
                            
                            loadAndBridgeToMetal(filePath: targetPath)
                            
                        case .failure(let error):
                            errorMessage = error.localizedDescription
                        }
                    }
                }
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()

            // Tokenizer Playground Panel
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Tokenization Playground")
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(.secondary)
                    Spacer()
                }
                
                HStack {
                    TextField("Enter prompt to encode...", text: $promptInput)
                        .textFieldStyle(.roundedBorder)
                        .onChange(of: promptInput) { _, newValue in
                            runTokenization(text: newValue)
                        }
                    
                    Button("Encode") {
                        runTokenization(text: promptInput)
                    }
                    .disabled(tokenizer == nil)
                }
                
                Text(tokenIDsOutput)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundColor(tokenizer == nil ? .secondary : .purple)
                    .lineLimit(2)

                HStack(spacing: 12) {
                    if let tokenizer = tokenizer, let ids = try? tokenizer.encode(text: promptInput), !ids.isEmpty {
                        Button(action: { executeEmbeddingLookup(tokenIds: ids) }) {
                            Label("Generate Hidden State h_0", systemImage: "sparkles")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.blue)
                    }
                    
                    if activeH0Buffer != nil, let summary = summary, summary.layerCount > 0 {
                        HStack(spacing: 8) {
                            Picker("Layer", selection: $selectedLayerForRouting) {
                                ForEach(0..<Int(summary.layerCount), id: \.self) { l in
                                    Text("Layer \(l)").tag(l)
                                }
                            }
                            .frame(width: 120)
                            
                            Button(action: { executeMoERouter(layerIndex: selectedLayerForRouting, topK: 8) }) {
                                Label("Route Top-8", systemImage: "point.3.connected.trianglepath.dotted")
                            }
                            .buttonStyle(.bordered)
                            
                            Button(action: { executeMoELayerMLP(layerIndex: selectedLayerForRouting, topK: 8) }) {
                                Label(isExecutingMlp ? "Computing MLP..." : "MoE MLP", systemImage: "bolt.fill")
                            }
                            .buttonStyle(.bordered)
                            .tint(.purple)
                            .disabled(isExecutingMlp || isExecutingFullLayer)

                            Button(action: { executeFullLayerForward(layerIndex: selectedLayerForRouting) }) {
                                Label(isExecutingFullLayer ? "Computing Block..." : "Execute Full Block (h_l → h_l+1)", systemImage: "arrow.triangle.merge")
                            }
                            .buttonStyle(.bordered)
                            .tint(.indigo)
                            .disabled(isExecutingFullLayer || isExecutingMlp || isExecutingMultiLayer || isExecutingLMHead)

                            Button(action: { executeMultiLayerForward(numLayers: targetLayerCount) }) {
                                Label(isExecutingMultiLayer ? "Executing Backbone..." : "Execute All \(targetLayerCount) Layers", systemImage: "square.stack.3d.forward.dottedline.fill")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.teal)
                            .disabled(isExecutingMultiLayer || isExecutingFullLayer || isExecutingMlp || isExecutingLMHead)

                            Button(action: { executeLMHeadProjection() }) {
                                Label(isExecutingLMHead ? "Projecting Vocab..." : "Project LM Head (Next Token)", systemImage: "text.word.spacing")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.pink)
                            .disabled(isExecutingLMHead || isExecutingMultiLayer || isExecutingFullLayer || isExecutingMlp)
                        }
                    }
                }
                .padding(.top, 4)

                // Live MoE Routing Visualizer Card
                if !routedExperts.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Layer \(selectedLayerForRouting) Top-8 Routed Experts (of \(summary?.maxExpertId ?? 256))", systemImage: "cpu.fill")
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(.purple)
                            
                            Spacer()
                            
                            if let shared = sharedExpertWeight {
                                Text(String(format: "Shared Expert Gate: %.1f%%", shared * 100.0))
                                    .font(.caption)
                                    .fontWeight(.bold)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 3)
                                    .background(Color.blue.opacity(0.15))
                                    .foregroundColor(.blue)
                                    .cornerRadius(6)
                            }
                        }
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 4), spacing: 8) {
                            ForEach(Array(routedExperts.enumerated()), id: \.offset) { rank, expert in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("#\(rank + 1)")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text("Expert #\(expert.id)")
                                            .font(.system(.caption, design: .monospaced))
                                            .fontWeight(.bold)
                                            .foregroundColor(.purple)
                                        Spacer()
                                        Text(String(format: "%.1f%%", expert.weight * 100.0))
                                            .font(.system(.caption, design: .monospaced))
                                            .fontWeight(.semibold)
                                    }
                                    
                                    GeometryReader { geo in
                                        ZStack(alignment: .leading) {
                                            Capsule()
                                                .fill(Color.purple.opacity(0.15))
                                                .frame(height: 6)
                                            Capsule()
                                                .fill(LinearGradient(colors: [.purple, .blue], startPoint: .leading, endPoint: .trailing))
                                                .frame(width: max(4, geo.size.width * CGFloat(expert.weight)), height: 6)
                                        }
                                    }
                                    .frame(height: 6)
                                }
                                .padding(8)
                                .background(Color(NSColor.controlBackgroundColor))
                                .cornerRadius(8)
                            }
                        }
                        
                        if let status = routerStatusText {
                            Text(status)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(10)
                    .background(Color.purple.opacity(0.06))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.purple.opacity(0.2), lineWidth: 1)
                    )
                }

                // Live MoE Layer MLP Output Card
                if let mlpStatus = layerMlpStatusText {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Layer \(selectedLayerForRouting) MoE Forward Pass Output (h_mlp)", systemImage: "sparkles.rectangle.stack.fill")
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(.green)
                            Spacer()
                        }
                        
                        Text(mlpStatus)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.primary)
                        
                        if let sample = layerMlpSampleOutput {
                            Text(sample)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(10)
                    .background(Color.green.opacity(0.06))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.green.opacity(0.2), lineWidth: 1)
                    )
                }

                // Live Full Layer Transformer Block Output Card (h_l+1)
                if let fullStatus = fullLayerStatusText {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Layer \(selectedLayerForRouting) Full Forward Block Output (h_\(selectedLayerForRouting + 1))", systemImage: "arrow.triangle.merge")
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(.indigo)
                            Spacer()
                        }
                        
                        Text(fullStatus)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.primary)
                        
                        if let sample = fullLayerSampleOutput {
                            Text(sample)
                                .font(.system(.caption2, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    .padding(10)
                    .background(Color.indigo.opacity(0.06))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.indigo.opacity(0.2), lineWidth: 1)
                    )
                }

                // Live Multi-Layer Backbone Output Card (h_0 -> h_N)
                if let multiStatus = multiLayerStatusText {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Full Backbone Multi-Layer Forward Pass (h_0 → h_\(multiLayerTelemetry.count))", systemImage: "square.stack.3d.forward.dottedline.fill")
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(.teal)
                            Spacer()
                        }
                        
                        Text(multiStatus)
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.primary)
                        
                        // Per-layer telemetry horizontal feed
                        if !multiLayerTelemetry.isEmpty {
                            ScrollView(.horizontal, showsIndicators: false) {
                                HStack(spacing: 6) {
                                    ForEach(multiLayerTelemetry) { item in
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text("L\(item.layerIndex)")
                                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                                .foregroundColor(.teal)
                                            Text(String(format: "%.2fms", item.durationMs))
                                                .font(.system(size: 9, design: .monospaced))
                                                .foregroundColor(.secondary)
                                            Text("||h||:\(String(format: "%.1f", item.l2Norm))")
                                                .font(.system(size: 8, design: .monospaced))
                                                .foregroundColor(.primary)
                                            Text("[\(item.topExperts.prefix(3).map(String.init).joined(separator: ","))]")
                                                .font(.system(size: 8, design: .monospaced))
                                                .foregroundColor(.purple)
                                        }
                                        .padding(5)
                                        .background(Color(NSColor.controlBackgroundColor))
                                        .cornerRadius(6)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 6)
                                                .stroke(Color.teal.opacity(0.3), lineWidth: 1)
                                        )
                                    }
                                }
                            }
                        }
                    }
                    .padding(10)
                    .background(Color.teal.opacity(0.06))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.teal.opacity(0.2), lineWidth: 1)
                    )
                }

                // Live Predicted Token Candidates Card (LM Head Output)
                if !topTokenPredictions.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("LM Head Predicted Next Tokens (Top-\(topTokenPredictions.count) Candidates)", systemImage: "sparkles")
                                .font(.subheadline)
                                .fontWeight(.bold)
                                .foregroundColor(.pink)
                            Spacer()
                            if let status = lmHeadStatusText {
                                Text(status)
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }

                        LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 8), count: 5), spacing: 8) {
                            ForEach(topTokenPredictions) { pred in
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack {
                                        Text("#\(pred.rank)")
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                        Text("\"\(pred.tokenString.replacingOccurrences(of: " ", with: " "))\"")
                                            .font(.system(.subheadline, design: .monospaced))
                                            .fontWeight(.bold)
                                            .foregroundColor(.pink)
                                            .lineLimit(1)
                                        Spacer()
                                        Text(String(format: "%.1f%%", pred.probability * 100.0))
                                            .font(.system(.caption, design: .monospaced))
                                            .fontWeight(.semibold)
                                    }

                                    GeometryReader { geo in
                                        ZStack(alignment: .leading) {
                                            Capsule()
                                                .fill(Color.pink.opacity(0.15))
                                                .frame(height: 6)
                                            Capsule()
                                                .fill(LinearGradient(colors: [.pink, .orange], startPoint: .leading, endPoint: .trailing))
                                                .frame(width: max(4, geo.size.width * CGFloat(pred.probability)), height: 6)
                                        }
                                    }
                                    .frame(height: 6)

                                    HStack {
                                        Text("ID: \(pred.tokenId)")
                                            .font(.system(size: 9, design: .monospaced))
                                            .foregroundColor(.secondary)
                                        Spacer()
                                        Text(String(format: "logit: %.2f", pred.logit))
                                            .font(.system(size: 9, design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                }
                                .padding(8)
                                .background(Color(NSColor.controlBackgroundColor))
                                .cornerRadius(8)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 8)
                                        .stroke(Color.pink.opacity(0.2), lineWidth: 1)
                                )
                            }
                        }
                    }
                    .padding(10)
                    .background(Color.pink.opacity(0.06))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.pink.opacity(0.2), lineWidth: 1)
                    )
                }
            }
            .padding(10)
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()

            if let err = errorMessage {
                ContentUnavailableView("Error", systemImage: "exclamationmark.triangle", description: Text(err))
            } else if summary != nil {
                VStack(spacing: 0) {
                    // Search & Category Filter Chips
                    VStack(spacing: 8) {
                        HStack {
                            Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                            TextField("Filter tensors by name or layer...", text: $searchText).textFieldStyle(.plain)
                        }
                        .padding(8)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)

                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(categoryFilters, id: \.self) { cat in
                                    Button(action: { selectedCategory = cat }) {
                                        Text(cat)
                                            .font(.caption)
                                            .padding(.horizontal, 10)
                                            .padding(.vertical, 4)
                                            .background(selectedCategory == cat ? Color.accentColor : Color(NSColor.controlColor))
                                            .foregroundColor(selectedCategory == cat ? .white : .primary)
                                            .cornerRadius(12)
                                    }
                                    .buttonStyle(.plain)
                                }
                            }
                        }
                    }
                    .padding(10)

                    // Tensor Table
                    Table(filteredTensors, selection: $selectedTensorID) {
                        TableColumn("Tensor Name", value: \.name)
                        TableColumn("Category") { t in
                            Text(t.category)
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(t.category.contains("Expert") ? Color.purple.opacity(0.15) : Color.blue.opacity(0.15))
                                .foregroundColor(t.category.contains("Expert") ? .purple : .blue)
                                .cornerRadius(4)
                        }
                        TableColumn("Shape") { t in
                            Text(t.shapeDisplay).font(.system(.body, design: .monospaced))
                        }
                        TableColumn("Dtype") { t in
                            Text(t.dtype).font(.system(.body, design: .monospaced)).foregroundColor(.blue)
                        }
                        TableColumn("Shard") { t in
                            Text("Shard #\(t.shardIndex)").font(.system(.caption, design: .monospaced)).foregroundColor(.secondary)
                        }
                        TableColumn("Size") { t in
                            Text(String(format: "%.2f MB", t.sizeMb)).font(.system(.body, design: .monospaced))
                        }
                    }
                    
                    // Selected Tensor & GPU Execution Bar
                    if let tensor = selectedTensor {
                        Divider()
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Selected: \(tensor.name)")
                                    .font(.headline)
                                Text("Category: \(tensor.category) | Shard #\(tensor.shardIndex) | Offset: \(tensor.offsetStart) bytes")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Button("Run Metal Compute Kernel") {
                                executeGpuShader(on: tensor)
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(.purple)
                        }
                        .padding()
                        .background(Color(NSColor.controlBackgroundColor))
                        
                        if let output = gpuComputeOutput {
                            HStack {
                                Text(output)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.purple)
                                Spacer()
                            }
                            .padding([.horizontal, .bottom])
                            .background(Color(NSColor.controlBackgroundColor))
                        }
                    }
                }
            } else {
                ContentUnavailableView("No Weights Loaded", systemImage: "memorychip", description: Text("Select a .safetensors or index.json file to inspect MoE layer topology."))
            }
        }
        .frame(minWidth: 900, minHeight: 650)
    }
    
    // MARK: - Tokenizer Execution
    private func loadTokenizer(filePath: String) {
        do {
            let tok = try DynaMoeTokenizer(tokenizerPath: filePath)
            self.tokenizer = tok
            runTokenization(text: promptInput)
        } catch {
            tokenIDsOutput = "❌ Failed to load tokenizer.json: \(error.localizedDescription)"
        }
    }
    
    private func runTokenization(text: String) {
        guard let tokenizer = tokenizer else {
            tokenIDsOutput = "Load a tokenizer.json file to tokenize text"
            return
        }
        do {
            let ids = try tokenizer.encode(text: text)
            let decoded = try tokenizer.decode(ids: ids)
            tokenIDsOutput = "Tokens (\(ids.count)): \(ids) ➔ Decoded: \"\(decoded)\""
        } catch {
            tokenIDsOutput = "❌ Encoding Error: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Execute Token Embedding Lookup (h_0)
    private func executeEmbeddingLookup(tokenIds: [UInt32]) {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Error setting up Metal embedding pipeline."
            return
        }

        do {
            guard let embedWeight = summary.tensors.first(where: {
                ($0.name.contains("embed_tokens") || $0.name.contains("embed") || $0.name.contains("wte")) &&
                $0.name.contains("weight") && !$0.name.contains("scale")
            }) else {
                gpuComputeOutput = "❌ Couldn't locate embedding weight tensor."
                return
            }

            guard let rawBaseBuffer = shardBuffers[embedWeight.shardIndex] else {
                gpuComputeOutput = "❌ Shard #\(embedWeight.shardIndex) buffer not loaded."
                return
            }

            var weightOffset = embedWeight.offsetStart
            var hiddenDim: UInt32 = 2048 // Qwen 35B hidden dimension
            let tokenCount = tokenIds.count
            let totalVectorElements = tokenCount * Int(hiddenDim)
            
            guard let tokenBuffer = device.makeBuffer(bytes: tokenIds, length: tokenCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let h0OutputBuffer = device.makeBuffer(length: totalVectorElements * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let isBF16 = embedWeight.dtype.contains("BF16") || embedWeight.dtype.contains("BFLOAT16")
            
            if isBF16 {
                guard let kernelFunction = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16") else { return }
                let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                
                computeEncoder.setComputePipelineState(pipelineState)
                computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(h0OutputBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            } else {
                let embedScale = summary.tensors.first(where: {
                    $0.name.contains(embedWeight.name.replacingOccurrences(of: ".weight", with: "")) &&
                    ($0.name.contains("scale") || $0.name.contains("scales"))
                })
                var scaleOffset = embedScale?.offsetStart ?? weightOffset
                
                guard let kernelFunction = defaultLibrary.makeFunction(name: "lookup_embeddings_mxfp8") else { return }
                let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                
                computeEncoder.setComputePipelineState(pipelineState)
                computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(h0OutputBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
            }
            
            let gridSize = MTLSize(width: totalVectorElements, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(totalVectorElements, 256), height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let rawFloatPtr = h0OutputBuffer.contents().bindMemory(to: Float.self, capacity: totalVectorElements)
            var h0Sample: [String] = []
            for i in 0..<min(8, totalVectorElements) {
                h0Sample.append(String(format: "%.6f", rawFloatPtr[i]))
            }
            
            self.activeH0Buffer = h0OutputBuffer
            self.activeTokenCount = tokenCount
            self.activeHiddenDim = Int(hiddenDim)
            
            gpuComputeOutput = "🚀 Generated h_0 Vector (\(embedWeight.dtype)) [\(tokenCount) x \(hiddenDim)] from Shard #\(embedWeight.shardIndex)! First 8 dims of Token #0: [\(h0Sample.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Embedding Lookup Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute MoE Top-K Router
    private func executeMoERouter(layerIndex: Int, topK: Int = 8) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Please click 'Generate Hidden State h_0' first."
            return
        }

        do {
            // Locate the router gate weight tensor for the target layer
            guard let gateWeight = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) &&
                ($0.category == "MoE Router" || $0.name.hasSuffix("mlp.gate.weight") || $0.name.contains("mlp.gate")) &&
                !$0.name.contains("shared") &&
                !$0.name.contains("scale")
            }) else {
                gpuComputeOutput = "❌ Could not find mlp.gate.weight for Layer #\(layerIndex)."
                return
            }

            guard let rawGateBuffer = shardBuffers[gateWeight.shardIndex] else {
                gpuComputeOutput = "❌ Shard #\(gateWeight.shardIndex) containing Layer #\(layerIndex) gate is not loaded."
                return
            }

            let numExpertsVal: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : 256
            var topKVal: UInt32 = UInt32(topK)
            var hiddenDimVal: UInt32 = UInt32(activeHiddenDim)
            var gateOffset: UInt64 = gateWeight.offsetStart
            var numExperts: UInt32 = numExpertsVal

            guard let outIndicesBuffer = device.makeBuffer(length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let outWeightsBuffer = device.makeBuffer(length: topK * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            guard let kernelFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16") else {
                gpuComputeOutput = "❌ Failed to load moe_router_topk_bf16 kernel."
                return
            }
            let pipelineState = try device.makeComputePipelineState(function: kernelFunction)

            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(rawGateBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 1)
            computeEncoder.setBuffer(outIndicesBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(outWeightsBuffer, offset: 0, index: 3)
            computeEncoder.setBytes(&gateOffset, length: MemoryLayout<UInt64>.stride, index: 4)
            computeEncoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
            computeEncoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 6)
            computeEncoder.setBytes(&topKVal, length: MemoryLayout<UInt32>.stride, index: 7)

            let threadsPerThreadgroup = MTLSize(width: Int(numExpertsVal), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

            // Also check for shared expert gate
            var sharedOutBuffer: MTLBuffer? = nil
            if let sharedGateWeight = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("shared_expert_gate") && !$0.name.contains("scale")
            }), let sharedRawBuffer = shardBuffers[sharedGateWeight.shardIndex],
               let sharedKernel = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") {
                
                let sharedPipelineState = try device.makeComputePipelineState(function: sharedKernel)
                var sharedOffset: UInt64 = sharedGateWeight.offsetStart
                sharedOutBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
                
                if let sharedOutBuffer = sharedOutBuffer {
                    computeEncoder.setComputePipelineState(sharedPipelineState)
                    computeEncoder.setBuffer(sharedRawBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(sharedOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&sharedOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                    
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            let indicesPtr = outIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: topK)
            let weightsPtr = outWeightsBuffer.contents().bindMemory(to: Float.self, capacity: topK)

            var extractedExperts: [(id: Int, weight: Float)] = []
            var sumProb: Float = 0.0
            for i in 0..<topK {
                let expertId = Int(indicesPtr[i])
                let w = weightsPtr[i]
                extractedExperts.append((id: expertId, weight: w))
                sumProb += w
            }
            self.routedExperts = extractedExperts

            if let sharedOutBuffer = sharedOutBuffer {
                let sharedPtr = sharedOutBuffer.contents().bindMemory(to: Float.self, capacity: 1)
                self.sharedExpertWeight = sharedPtr[0]
            } else {
                self.sharedExpertWeight = nil
            }

            let topExpertSummary = extractedExperts.map { "#\($0.id) (\(String(format: "%.1f%%", $0.weight * 100)))" }.joined(separator: ", ")
            let status = "⚡ Layer #\(layerIndex) Gated on GPU in \(String(format: "%.3f", elapsedMs)) ms! Top-\(topK): [\(topExpertSummary)] (Sum: \(String(format: "%.4f", sumProb)))"
            self.routerStatusText = status
            self.gpuComputeOutput = status

        } catch {
            gpuComputeOutput = "❌ MoE Router Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute MoE Layer MLP (SwiGLU across Top-K + Shared Experts)
    private func executeMoELayerMLP(layerIndex: Int, topK: Int = 8) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingMlp = true
        defer { self.isExecutingMlp = false }

        do {
            // 1. Ensure Routing has been performed for this layer
            if routedExperts.isEmpty {
                executeMoERouter(layerIndex: layerIndex, topK: topK)
            }
            guard !routedExperts.isEmpty else {
                gpuComputeOutput = "❌ MoE routing failed for Layer #\(layerIndex)."
                return
            }

            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var intermediateDim: UInt32 = 512 // Default intermediate size for Qwen/Ornith 35B
            
            // Check if any expert tensor reveals intermediate dimension
            if let sampleGate = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
            }) {
                let dims = sampleGate.shapeDisplay
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                    .components(separatedBy: ",")
                    .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                if dims.count >= 2 {
                    intermediateDim = dims[0]
                    hiddenDim = dims[1]
                }
            }

            // 2. Allocate output accumulator h_mlp and intermediate buffer
            guard let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                gpuComputeOutput = "❌ Failed to allocate GPU command encoder/buffers."
                return
            }

            // 3. Clear h_mlp accumulator vector to 0.0
            if let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32") {
                let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
                let gridSize = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            }

            // Helper to lookup tensor by query
            let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) &&
                    t.name.contains(sub1) &&
                    t.name.contains(sub2) &&
                    (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                })
            }

            // Load pipeline states for MXFP8 and BF16 SwiGLU & Down Projections
            guard let mxfp8GateUpKernel = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up"),
                  let mxfp8DownKernel = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to locate SwiGLU / DownProj Metal kernels."
                return
            }

            let mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateUpKernel)
            let mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownKernel)
            let bf16GateUpPipeline = try device.makeComputePipelineState(function: bf16GateUpKernel)
            let bf16DownPipeline = try device.makeComputePipelineState(function: bf16DownKernel)

            var executedExpertCount = 0

            // 4. Dispatch each of the Top-K Routed Experts
            for expert in routedExperts {
                let expId = expert.id
                var p_k = expert.weight
                if p_k <= 0.00001 { continue }

                let expTag = "experts.\(expId)"
                guard let gateWeight = findTensor(expTag, "gate_proj", false),
                      let upWeight   = findTensor(expTag, "up_proj", false),
                      let downWeight = findTensor(expTag, "down_proj", false),
                      let gateRaw    = shardBuffers[gateWeight.shardIndex],
                      let upRaw      = shardBuffers[upWeight.shardIndex],
                      let downRaw    = shardBuffers[downWeight.shardIndex] else {
                    continue
                }

                let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isMXFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset

                    // Dispatch SwiGLU (Gate & Up Proj)
                    computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    // Dispatch Down Proj with Weighted Accumulation
                    computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 7)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)

                } else {
                    var gateWeightOffset = gateWeight.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var downWeightOffset = downWeight.offsetStart

                    // Dispatch BF16 SwiGLU
                    computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    // Dispatch BF16 Down Proj
                    computeEncoder.setComputePipelineState(bf16DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                }

                executedExpertCount += 1
            }

            // 5. Dispatch Shared Expert if present
            var sharedDispatched = false
            if var sharedWeight = sharedExpertWeight, sharedWeight > 0.0001 {
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isMXFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset

                        computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 7)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    } else {
                        var gateWeightOffset = gateWeight.offsetStart
                        var upWeightOffset   = upWeight.offsetStart
                        var downWeightOffset = downWeight.offsetStart

                        computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(bf16DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 6)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                    sharedDispatched = true
                }
            }

            // 6. Commit and Measure Execution Time
            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            // 7. Calculate L2 Norm & Sample Output
            let outPtr = hMlpBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var sumSquares: Float = 0.0
            var sampleValues: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = outPtr[i]
                sumSquares += v * v
                if i < 8 {
                    sampleValues.append(String(format: "%.5f", v))
                }
            }
            let l2Norm = sqrt(sumSquares)

            self.activeHmlpBuffer = hMlpBuffer
            let sharedDesc = sharedDispatched ? "+ 1 Shared Expert" : ""
            let status = "⚡ Layer #\(layerIndex) MoE MLP Computed in \(String(format: "%.3f", elapsedMs)) ms! (\(executedExpertCount) Routed Experts \(sharedDesc)) | L2 Norm: \(String(format: "%.4f", l2Norm))"
            self.layerMlpStatusText = status
            self.layerMlpSampleOutput = "First 8 dims of h_mlp: [\(sampleValues.joined(separator: ", "))]"
            self.gpuComputeOutput = "\(status)\n\(self.layerMlpSampleOutput!)"

        } catch {
            gpuComputeOutput = "❌ MoE MLP Execution Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute Full Layer Forward Pass (h_l -> h_l+1)
    private func executeFullLayerForward(layerIndex: Int) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingFullLayer = true
        defer { self.isExecutingFullLayer = false }

        do {
            // Ensure routing is calculated
            if routedExperts.isEmpty {
                executeMoERouter(layerIndex: layerIndex, topK: 8)
            }

            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var eps: Float = 1e-6

            let findTensorInLayer = { (query: String) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) && t.name.contains(query)
                })
            }

            guard let rmsKernel = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
                  let addKernel = defaultLibrary.makeFunction(name: "vector_add_f32"),
                  let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32"),
                  let mxfp8GateUpKernel = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up"),
                  let mxfp8DownKernel = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for full layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateUpKernel)
            let mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownKernel)
            let bf16GateUpPipeline = try device.makeComputePipelineState(function: bf16GateUpKernel)
            let bf16DownPipeline = try device.makeComputePipelineState(function: bf16DownKernel)

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                gpuComputeOutput = "❌ Failed to allocate GPU command encoder."
                return
            }

            let byteLength = Int(hiddenDim) * MemoryLayout<Float>.stride
            guard let xNorm1Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let attnOutBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMidBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm2Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMlpBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hNextBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate GPU layer buffers."
                return
            }

            // -------------------------------------------------------------
            // Step 1: Pre-Attention RMSNorm (h_l -> xNorm1)
            // -------------------------------------------------------------
            if let norm1Tensor = findTensorInLayer("input_layernorm"),
               let norm1Raw = shardBuffers[norm1Tensor.shardIndex] {
                var gammaOffset = norm1Tensor.offsetStart
                computeEncoder.setComputePipelineState(rmsPipeline)
                computeEncoder.setBuffer(h0Buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(norm1Raw, offset: 0, index: 1)
                computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
            }

            // -------------------------------------------------------------
            // Step 2: Attention Computation (xNorm1 -> attnOut)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(clearPipeline)
            computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            if let oProjTensor = findTensorInLayer("o_proj") ?? findTensorInLayer("out_proj"),
               let oProjRaw = shardBuffers[oProjTensor.shardIndex] {
                let isMXFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                var oProjOffset = oProjTensor.offsetStart
                var inAttnDim = hiddenDim

                if isMXFP8, let gemvKernel = defaultLibrary.makeFunction(name: "mxfp8_gemv") {
                    let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                    let oScaleTensor = summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(layerIndex) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                    })
                    var oScaleOffset = oScaleTensor?.offsetStart ?? oProjOffset

                    computeEncoder.setComputePipelineState(gemvPipeline)
                    computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), gemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                } else if let bf16GemvKernel = defaultLibrary.makeFunction(name: "bf16_gemv") {
                    let bf16GemvPipeline = try device.makeComputePipelineState(function: bf16GemvKernel)
                    computeEncoder.setComputePipelineState(bf16GemvPipeline)
                    computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), bf16GemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
            }

            // -------------------------------------------------------------
            // Step 3: Residual Connection 1 (h_mid = h_l + attnOut)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(addPipeline)
            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 0)
            computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            // -------------------------------------------------------------
            // Step 4: Post-Attention RMSNorm (h_mid -> xNorm2)
            // -------------------------------------------------------------
            if let norm2Tensor = findTensorInLayer("post_attention_layernorm"),
               let norm2Raw = shardBuffers[norm2Tensor.shardIndex] {
                var gammaOffset = norm2Tensor.offsetStart
                computeEncoder.setComputePipelineState(rmsPipeline)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(norm2Raw, offset: 0, index: 1)
                computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
            }

            // -------------------------------------------------------------
            // Step 5: MoE Router & SwiGLU Feed-Forward Compute (xNorm2 -> hMlp)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(clearPipeline)
            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            var intermediateDim: UInt32 = 512
            if let sampleGate = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
            }) {
                let dims = sampleGate.shapeDisplay
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                    .components(separatedBy: ",")
                    .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                if dims.count >= 2 {
                    intermediateDim = dims[0]
                }
            }

            guard let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                return
            }

            let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) &&
                    t.name.contains(sub1) &&
                    t.name.contains(sub2) &&
                    (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                })
            }

            for expert in routedExperts {
                let expId = expert.id
                var p_k = expert.weight
                if p_k <= 0.00001 { continue }

                let expTag = "experts.\(expId)"
                guard let gateWeight = findTensor(expTag, "gate_proj", false),
                      let upWeight   = findTensor(expTag, "up_proj", false),
                      let downWeight = findTensor(expTag, "down_proj", false),
                      let gateRaw    = shardBuffers[gateWeight.shardIndex],
                      let upRaw      = shardBuffers[upWeight.shardIndex],
                      let downRaw    = shardBuffers[downWeight.shardIndex] else {
                    continue
                }

                let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isMXFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset

                    computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 7)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                } else {
                    var gateWeightOffset = gateWeight.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var downWeightOffset = downWeight.offsetStart

                    computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    computeEncoder.setComputePipelineState(bf16DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                }
            }

            if var sharedWeight = sharedExpertWeight, sharedWeight > 0.0001 {
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isMXFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset

                        computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 7)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                }
            }

            // -------------------------------------------------------------
            // Step 6: Residual Connection 2 (h_next = h_mid + hMlp)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(addPipeline)
            computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(hNextBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            // -------------------------------------------------------------
            // Step 7: Commit & Measure Full Block Latency
            // -------------------------------------------------------------
            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            let outPtr = hNextBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var sumSquares: Float = 0.0
            var sampleValues: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = outPtr[i]
                sumSquares += v * v
                if i < 8 {
                    sampleValues.append(String(format: "%.5f", v))
                }
            }
            let l2Norm = sqrt(sumSquares)

            self.activeH1Buffer = hNextBuffer
            let status = "⚡ Layer #\(layerIndex) Full Block (RMSNorm + Attn + Res + MoE + Res) Executed in \(String(format: "%.3f", elapsedMs)) ms! | Output h_\(layerIndex + 1) L2 Norm: \(String(format: "%.4f", l2Norm))"
            self.fullLayerStatusText = status
            self.fullLayerSampleOutput = "First 8 dims of h_\(layerIndex + 1): [\(sampleValues.joined(separator: ", "))]"
            self.gpuComputeOutput = "\(status)\n\(self.fullLayerSampleOutput!)"

        } catch {
            gpuComputeOutput = "❌ Full Layer Forward Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute Sequential Multi-Layer Forward Pass (h_0 -> h_N)
    private func executeMultiLayerForward(numLayers: Int) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingMultiLayer = true
        defer { self.isExecutingMultiLayer = false }

        do {
            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var eps: Float = 1e-6

            guard let rmsKernel = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
                  let addKernel = defaultLibrary.makeFunction(name: "vector_add_f32"),
                  let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32"),
                  let mxfp8GateUpKernel = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up"),
                  let mxfp8DownKernel = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate"),
                  let routerKernel = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
                  let sharedGateKernel = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for multi-layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateUpKernel)
            let mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownKernel)
            let routerPipeline = try device.makeComputePipelineState(function: routerKernel)
            let sharedGatePipeline = try device.makeComputePipelineState(function: sharedGateKernel)

            let byteLength = Int(hiddenDim) * MemoryLayout<Float>.stride
            guard let hCurrBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hNextBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm1Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let attnOutBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMidBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm2Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMlpBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let routerScoresBuffer = device.makeBuffer(length: 256 * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate GPU multi-layer buffers."
                return
            }

            // Copy initial h0 into hCurrBuffer
            memcpy(hCurrBuffer.contents(), h0Buffer.contents(), byteLength)

            var telemetryList: [LayerTelemetry] = []
            let totalStartTime = CFAbsoluteTimeGetCurrent()
            let actualLayers = min(numLayers, Int(summary.layerCount))

            for l in 0..<actualLayers {
                let layerStartTime = CFAbsoluteTimeGetCurrent()

                let findTensorInLayer = { (query: String) -> TensorMetadata? in
                    return summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(l) && t.name.contains(query)
                    })
                }

                guard let commandBuffer = commandQueue.makeCommandBuffer(),
                      let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    continue
                }

                // ---------------------------------------------------------
                // Step 0: Dynamic Routing for Layer l
                // ---------------------------------------------------------
                var activeExperts: [(id: Int, weight: Float)] = []

                if let routerTensor = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("mlp.gate.weight")
                }), let routerRaw = shardBuffers[routerTensor.shardIndex] {
                    var routerOffset = routerTensor.offsetStart
                    var numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : 256

                    computeEncoder.setComputePipelineState(routerPipeline)
                    computeEncoder.setBuffer(routerRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(routerScoresBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&routerOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(numExperts), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numExperts), routerPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                if let sharedGateTensor = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("shared_expert_gate.weight")
                }), let sharedGateRaw = shardBuffers[sharedGateTensor.shardIndex] {
                    var gateOffset = sharedGateTensor.offsetStart
                    computeEncoder.setComputePipelineState(sharedGatePipeline)
                    computeEncoder.setBuffer(sharedGateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(sharedScoreBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gateOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 1: Pre-Attention RMSNorm (hCurr -> xNorm1)
                // ---------------------------------------------------------
                if let norm1Tensor = findTensorInLayer("input_layernorm"),
                   let norm1Raw = shardBuffers[norm1Tensor.shardIndex] {
                    var gammaOffset = norm1Tensor.offsetStart
                    computeEncoder.setComputePipelineState(rmsPipeline)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(norm1Raw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                    computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 2: Attention Computation (xNorm1 -> attnOut)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 0)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                if let oProjTensor = findTensorInLayer("o_proj") ?? findTensorInLayer("out_proj"),
                   let oProjRaw = shardBuffers[oProjTensor.shardIndex] {
                    let isMXFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                    var oProjOffset = oProjTensor.offsetStart
                    var inAttnDim = hiddenDim

                    if isMXFP8, let gemvKernel = defaultLibrary.makeFunction(name: "mxfp8_gemv") {
                        let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                        let oScaleTensor = summary.tensors.first(where: { t in
                            t.layerIndex == UInt32(l) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                        })
                        var oScaleOffset = oScaleTensor?.offsetStart ?? oProjOffset

                        computeEncoder.setComputePipelineState(gemvPipeline)
                        computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), gemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if let bf16GemvKernel = defaultLibrary.makeFunction(name: "bf16_gemv") {
                        let bf16GemvPipeline = try device.makeComputePipelineState(function: bf16GemvKernel)
                        computeEncoder.setComputePipelineState(bf16GemvPipeline)
                        computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), bf16GemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                // ---------------------------------------------------------
                // Step 3: Residual Connection 1 (hMid = hCurr + attnOut)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(addPipeline)
                computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                // ---------------------------------------------------------
                // Step 4: Post-Attention RMSNorm (hMid -> xNorm2)
                // ---------------------------------------------------------
                if let norm2Tensor = findTensorInLayer("post_attention_layernorm"),
                   let norm2Raw = shardBuffers[norm2Tensor.shardIndex] {
                    var gammaOffset = norm2Tensor.offsetStart
                    computeEncoder.setComputePipelineState(rmsPipeline)
                    computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(norm2Raw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                    computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 5: MoE Router & SwiGLU Feed-Forward Compute (xNorm2 -> hMlp)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                var intermediateDim: UInt32 = 512
                if let sampleGate = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
                }) {
                    let dims = sampleGate.shapeDisplay
                        .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                        .components(separatedBy: ",")
                        .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                    if dims.count >= 2 {
                        intermediateDim = dims[0]
                    }
                }

                guard let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                    continue
                }

                let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                    return summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(l) &&
                        t.name.contains(sub1) &&
                        t.name.contains(sub2) &&
                        (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                    })
                }

                let rawScores = routerScoresBuffer.contents().bindMemory(to: Float.self, capacity: 256)
                var scorePairs: [(id: Int, score: Float)] = []
                for idx in 0..<256 {
                    scorePairs.append((id: idx, score: rawScores[idx]))
                }
                scorePairs.sort(by: { $0.score > $1.score })
                let top8 = Array(scorePairs.prefix(8))
                var expSum: Float = 0.0
                for item in top8 {
                    expSum += Darwin.exp(item.score)
                }
                if expSum > 0.0 {
                    activeExperts = top8.map { (id: $0.id, weight: Darwin.exp($0.score) / expSum) }
                } else {
                    activeExperts = (0..<8).map { (id: $0, weight: 1.0 / 8.0) }
                }

                for expert in activeExperts {
                    let expId = expert.id
                    var p_k = expert.weight
                    if p_k <= 0.00001 { continue }

                    let expTag = "experts.\(expId)"
                    guard let gateWeight = findTensor(expTag, "gate_proj", false),
                          let upWeight   = findTensor(expTag, "up_proj", false),
                          let downWeight = findTensor(expTag, "down_proj", false),
                          let gateRaw    = shardBuffers[gateWeight.shardIndex],
                          let upRaw      = shardBuffers[upWeight.shardIndex],
                          let downRaw    = shardBuffers[downWeight.shardIndex] else {
                        continue
                    }

                    let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isMXFP8 {
                        let gateScale = findTensor(expTag, "gate_proj", true)
                        let upScale   = findTensor(expTag, "up_proj", true)
                        let downScale = findTensor(expTag, "down_proj", true)

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset

                        computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 7)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                }

                // Shared expert
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isMXFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isMXFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale?.offsetStart ?? gateWeightOffset
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale?.offsetStart ?? upWeightOffset
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale?.offsetStart ?? downWeightOffset
                        var sharedW: Float = 0.5

                        computeEncoder.setComputePipelineState(mxfp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 8)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 9)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), mxfp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(mxfp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&sharedW, length: MemoryLayout<Float>.stride, index: 7)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), mxfp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                }

                // ---------------------------------------------------------
                // Step 6: Residual Connection 2 (hNext = hMid + hMlp)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(addPipeline)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(hNextBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                // End encoding and commit layer
                computeEncoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()

                let layerElapsedMs = (CFAbsoluteTimeGetCurrent() - layerStartTime) * 1000.0

                // Read L2 norm
                let outPtr = hNextBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
                var sumSquares: Float = 0.0
                for i in 0..<Int(hiddenDim) {
                    let v = outPtr[i]
                    sumSquares += v * v
                }
                let l2Norm = sqrt(sumSquares)

                telemetryList.append(LayerTelemetry(
                    layerIndex: l,
                    durationMs: layerElapsedMs,
                    topExperts: activeExperts.map { $0.id },
                    l2Norm: l2Norm
                ))

                // Ping-pong copy for next layer
                memcpy(hCurrBuffer.contents(), hNextBuffer.contents(), byteLength)
            }

            let totalElapsedMs = (CFAbsoluteTimeGetCurrent() - totalStartTime) * 1000.0
            let avgMsPerLayer = totalElapsedMs / Double(max(1, actualLayers))
            let layersPerSec = 1000.0 / max(0.001, avgMsPerLayer)

            // Final L2 norm
            let finalPtr = hCurrBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var finalSumSq: Float = 0.0
            var finalSamples: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = finalPtr[i]
                finalSumSq += v * v
                if i < 8 {
                    finalSamples.append(String(format: "%.5f", v))
                }
            }
            let finalL2Norm = sqrt(finalSumSq)

            self.activeHFinalBuffer = hCurrBuffer
            self.multiLayerTelemetry = telemetryList

            let status = "🚀 Full \(actualLayers)-Layer Backbone Executed in \(String(format: "%.2f", totalElapsedMs)) ms! (Avg \(String(format: "%.3f", avgMsPerLayer)) ms/layer | \(String(format: "%.0f", layersPerSec)) layers/sec) | Final h_\(actualLayers) L2 Norm: \(String(format: "%.4f", finalL2Norm))"
            self.multiLayerStatusText = status
            self.gpuComputeOutput = "\(status)\nFirst 8 dims of h_\(actualLayers): [\(finalSamples.joined(separator: ", "))]"

        } catch {
            gpuComputeOutput = "❌ Multi-Layer Forward Error: \(error.localizedDescription)"
        }
    }

    private func executeLMHeadProjection() {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvFunction = defaultLibrary.makeFunction(name: "bf16_gemv") else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline or missing shaders."
            return
        }

        let inputBuffer: MTLBuffer
        let sourceDescription: String
        if let hFinal = activeHFinalBuffer {
            inputBuffer = hFinal
            sourceDescription = "h_40 (Backbone Output)"
        } else if let h1 = activeH1Buffer {
            inputBuffer = h1
            sourceDescription = "h_1 (Single Block Output)"
        } else if let h0 = activeH0Buffer {
            inputBuffer = h0
            sourceDescription = "h_0 (Initial Embedding)"
        } else {
            gpuComputeOutput = "⚠️ Please run 'Embed & Route' or 'Execute All 40 Layers' first to generate hidden activations."
            return
        }

        // Find Final RMSNorm Tensor
        guard let normTensor = summary.tensors.first(where: {
            $0.name == "model.language_model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers."))
        }), let normShardBuffer = shardBuffers[normTensor.shardIndex] else {
            gpuComputeOutput = "❌ Final RMSNorm weight (model.language_model.norm.weight) not found."
            return
        }

        // Find LM Head Weight Tensor
        guard let lmHeadTensor = summary.tensors.first(where: {
            $0.name == "lm_head.weight" ||
            $0.name == "language_model.lm_head.weight" ||
            $0.name == "model.lm_head.weight"
        }), let lmHeadShardBuffer = shardBuffers[lmHeadTensor.shardIndex] else {
            gpuComputeOutput = "❌ LM Head weight (lm_head.weight) not found."
            return
        }

        isExecutingLMHead = true
        topTokenPredictions = []
        lmHeadStatusText = "Projecting 248,320 vocabulary logits..."

        do {
            let rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)
            let gemvPipeline = try device.makeComputePipelineState(function: gemvFunction)

            var hiddenDim = UInt32(activeHiddenDim)
            var eps: Float = 1e-6
            var normOffset = normTensor.offsetStart

            // Parse or set vocab size
            var vocabSize: UInt32 = 248320
            let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
            let shapeParts = cleanShape.split(separator: ",")
            if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
                vocabSize = parsed
            }

            var lmHeadOffset = lmHeadTensor.offsetStart

            // Allocate buffers
            guard let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate xFinal or Logits buffers."
                isExecutingLMHead = false
                return
            }

            let startTime = CFAbsoluteTimeGetCurrent()

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                isExecutingLMHead = false
                return
            }

            // 1. Dispatch Final RMSNorm (input -> xFinal)
            computeEncoder.setComputePipelineState(rmsnormPipeline)
            computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(normShardBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(xFinalBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&normOffset, length: MemoryLayout<UInt64>.stride, index: 3)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)

            let normTgSize = min(1024, rmsnormPipeline.maxTotalThreadsPerThreadgroup)
            computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: normTgSize, height: 1, depth: 1))

            // 2. Dispatch LM Head GEMV (xFinal -> logits)
            computeEncoder.setComputePipelineState(gemvPipeline)
            computeEncoder.setBuffer(lmHeadShardBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(xFinalBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(logitsBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&lmHeadOffset, length: MemoryLayout<UInt64>.stride, index: 3)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            computeEncoder.setBytes(&vocabSize, length: MemoryLayout<UInt32>.stride, index: 5)

            let gemvTgSize = min(256, gemvPipeline.maxTotalThreadsPerThreadgroup)
            computeEncoder.dispatchThreads(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: gemvTgSize, height: 1, depth: 1))

            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let gpuElapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            // 3. Extract Top-10 Vocabulary Candidates
            let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
            var topCandidates: [(id: Int, logit: Float)] = []
            
            for v in 0..<Int(vocabSize) {
                let val = logitsPtr[v]
                if topCandidates.count < 10 {
                    topCandidates.append((id: v, logit: val))
                    if topCandidates.count == 10 {
                        topCandidates.sort(by: { $0.logit > $1.logit })
                    }
                } else if val > topCandidates.last!.logit {
                    topCandidates[9] = (id: v, logit: val)
                    topCandidates.sort(by: { $0.logit > $1.logit })
                }
            }

            // Softmax over top candidates
            let maxLogit = topCandidates.first?.logit ?? 0.0
            var expSum: Float = 0.0
            for c in topCandidates {
                expSum += Darwin.exp(c.logit - maxLogit)
            }

            var predictions: [TokenPrediction] = []
            for (idx, item) in topCandidates.enumerated() {
                let prob = expSum > 0.0 ? (Darwin.exp(item.logit - maxLogit) / expSum) : (1.0 / Float(topCandidates.count))
                var tokenStr = "<Token \(item.id)>"
                if let tok = self.tokenizer {
                    do {
                        let decoded = try tok.decode(ids: [UInt32(item.id)])
                        if !decoded.isEmpty {
                            tokenStr = decoded
                        }
                    } catch {}
                }
                predictions.append(TokenPrediction(
                    rank: idx + 1,
                    tokenId: UInt32(item.id),
                    tokenString: tokenStr,
                    logit: item.logit,
                    probability: prob
                ))
            }

            self.topTokenPredictions = predictions
            self.activeLogitsBuffer = logitsBuffer
            self.isExecutingLMHead = false

            let top1 = predictions.first
            let top1Str = top1?.tokenString.replacingOccurrences(of: "\n", with: "\\n") ?? "N/A"
            let top1Pct = String(format: "%.1f%%", (top1?.probability ?? 0) * 100.0)

            let status = "✨ LM Head Projected \(vocabSize) Vocab Logits in \(String(format: "%.2f", gpuElapsedMs)) ms from \(sourceDescription)! Top Candidate: \"\(top1Str)\" (\(top1Pct) prob)"
            self.lmHeadStatusText = status
            self.gpuComputeOutput = status

        } catch {
            isExecutingLMHead = false
            gpuComputeOutput = "❌ LM Head Pipeline Error: \(error.localizedDescription)"
        }
    }

    private func loadAndBridgeToMetal(filePath: String) {
        do {
            let loadedEngine = try DynaMoeEngine(filePath: filePath)
            let loadedSummary = try loadedEngine.getSummary()
            
            guard let device = MTLCreateSystemDefaultDevice() else {
                metalStatus = "❌ Failed to initialize Metal GPU."
                return
            }
            
            // Map every shard into Metal zero-copy space
            var buffers: [UInt32: MTLBuffer] = [:]
            var mappedGB: Double = 0.0
            
            for shard in loadedSummary.shards {
                let address = UInt(shard.baseAddress)
                guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else { continue }
                let length = Int(shard.length)
                
                if let buffer = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil) {
                    buffers[shard.index] = buffer
                    mappedGB += Double(length) / (1024.0 * 1024.0 * 1024.0)
                }
            }
            
            self.engine = loadedEngine
            self.summary = loadedSummary
            self.shardBuffers = buffers
            self.errorMessage = nil
            self.selectedTensorID = nil
            self.gpuComputeOutput = nil
            
            metalStatus = "✅ Zero-Copy Active! \(loadedSummary.shards.count) Shards Mapped (\(String(format: "%.2f", mappedGB)) GB)"
            
        } catch {
            self.errorMessage = "Core Engine Error: \(error.localizedDescription)"
            self.summary = nil
            self.engine = nil
            self.shardBuffers.removeAll()
        }
    }

    private func executeGpuShader(on tensor: TensorMetadata) {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_mxfp8_paired"),
              let rawBaseBuffer = shardBuffers[tensor.shardIndex] else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline or missing shard buffer."
            return
        }

        do {
            let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
            
            var weightOffset = tensor.offsetStart
            var scaleOffset = tensor.offsetStart
            
            let baseName = tensor.name.replacingOccurrences(of: ".weight", with: "").replacingOccurrences(of: ".scales", with: "")
            let weightTensor = summary.tensors.first(where: { $0.name == "\(baseName).weight" }) ?? tensor
            let scaleTensor  = summary.tensors.first(where: { $0.name == "\(baseName).scales" }) ?? tensor
            
            weightOffset = weightTensor.offsetStart
            scaleOffset  = scaleTensor.offsetStart

            let sampleCount = 8
            let outputByteLength = sampleCount * MemoryLayout<Float>.stride
            guard let outputBuffer = device.makeBuffer(length: outputByteLength, options: .storageModeShared) else { return }
            
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            
            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 2)
            computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 3)
            
            let gridSize = MTLSize(width: sampleCount, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(sampleCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let rawFloatPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
            var sampleValues: [String] = []
            for i in 0..<sampleCount {
                sampleValues.append(String(format: "%.6f", rawFloatPtr[i]))
            }
            
            gpuComputeOutput = "⚡ MXFP8 Dequantized! First 8 weights for '\(baseName)' (Shard #\(tensor.shardIndex)): [\(sampleValues.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Pipeline Error: \(error.localizedDescription)"
        }
    }

    #if os(macOS)
    private func selectTokenizerWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data]
        panel.message = "Select tokenizer.json"
        panel.prompt = "Load Tokenizer"
        
        if panel.runModal() == .OK, let url = panel.url {
            loadTokenizer(filePath: url.path)
        }
    }

    private func selectModelWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data, .folder]
        panel.message = "Select Model Snapshot Folder or model.safetensors.index.json"
        panel.prompt = "Load Model"
        
        if panel.runModal() == .OK, let url = panel.url {
            var targetPath = url.path
            if url.hasDirectoryPath {
                let indexPath = url.appendingPathComponent("model.safetensors.index.json").path
                if FileManager.default.fileExists(atPath: indexPath) {
                    targetPath = indexPath
                }
            }
            loadAndBridgeToMetal(filePath: targetPath)
        }
    }
    #endif
}
