import SwiftUI
import UniformTypeIdentifiers
import Metal
import Foundation

struct ExpertRoutingBadgeView: View {
    let rank: Int
    let expertId: Int
    let weight: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("#\(rank + 1)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("Expert #\(expertId)")
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.purple)
                Spacer()
                Text(String(format: "%.1f%%", weight * 100.0))
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
                        .frame(width: max(4, geo.size.width * CGFloat(weight)), height: 6)
                }
            }
            .frame(height: 6)
        }
        .padding(8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

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

struct CachedLayer {
    let layerIndex: UInt32
    let routerTensor: TensorMetadata?
    let sharedGateTensor: TensorMetadata?
    let norm1Tensor: TensorMetadata?
    let oProjTensor: TensorMetadata?
    let oScaleTensor: TensorMetadata?
    let norm2Tensor: TensorMetadata?
    let sharedGateWeight: TensorMetadata?
    let sharedUpWeight: TensorMetadata?
    let sharedDownWeight: TensorMetadata?
    let sharedGateScale: TensorMetadata?
    let sharedUpScale: TensorMetadata?
    let sharedDownScale: TensorMetadata?
    let expertGateWeights: [Int: TensorMetadata]
    let expertUpWeights: [Int: TensorMetadata]
    let expertDownWeights: [Int: TensorMetadata]
    let expertGateScales: [Int: TensorMetadata]
    let expertUpScales: [Int: TensorMetadata]
    let expertDownScales: [Int: TensorMetadata]
    let intermediateDim: UInt32
}

#if canImport(Darwin)
import Darwin

func getProcessResidentMemoryGB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / (1024.0 * 1024.0 * 1024.0)
    }
    return 0.0
}
#else
func getProcessResidentMemoryGB() -> Double {
    return 0.0
}
#endif

enum MemoryBudgetMode: String, CaseIterable, Identifiable {
    case lowMemory8GB = "8 GB (Low RAM)"
    case balanced16GB = "16 GB (Balanced)"
    case unrestricted = "Unrestricted (36GB+)"

    var id: String { rawValue }

    var maxResidentExperts: Int {
        switch self {
        case .lowMemory8GB: return 512       // Max 512 active resident experts at once (~768 MB)
        case .balanced16GB: return 2048      // Max 2048 active resident experts (~3.0 GB)
        case .unrestricted: return 10240     // All 40 layers * 256 experts resident
        }
    }

    var targetMaxRssGB: Double {
        switch self {
        case .lowMemory8GB: return 8.0
        case .balanced16GB: return 16.0
        case .unrestricted: return 36.6
        }
    }
}

struct ExpertSlice {
    let shardIndex: UInt32
    let offset: UInt64
    let length: UInt64
}

struct ExpertKey: Hashable {
    let layer: Int
    let expertId: Int
}

final class WorkingSetManager {
    static let shared = WorkingSetManager()

    private(set) var expertSlices: [ExpertKey: [ExpertSlice]] = [:]
    private(set) var denseSlices: [ExpertSlice] = []
    private(set) var residentExperts: Set<ExpertKey> = []
    private var lruList: [ExpertKey] = []
    private let prefetchQueue = DispatchQueue(label: "com.dynamoe.prefetch", qos: .userInitiated)

    private(set) var totalAccesses: Int = 0
    private(set) var cacheHits: Int = 0
    private(set) var cacheMisses: Int = 0
    private(set) var lastPagingLatencyMs: Double = 0.0

    var totalExpertKeysCount: Int {
        return expertSlices.count
    }

    var cacheHitRatePercent: Double {
        guard totalAccesses > 0 else { return 100.0 }
        return (Double(cacheHits) / Double(totalAccesses)) * 100.0
    }

    func initialize(summary: ModelSummary, shardBuffers: [UInt32: MTLBuffer], mode: MemoryBudgetMode) {
        expertSlices.removeAll()
        denseSlices.removeAll()
        residentExperts.removeAll()
        lruList.removeAll()
        totalAccesses = 0
        cacheHits = 0
        cacheMisses = 0
        lastPagingLatencyMs = 0.0

        for tensor in summary.tensors {
            let slice = ExpertSlice(
                shardIndex: tensor.shardIndex,
                offset: tensor.offsetStart,
                length: tensor.offsetEnd - tensor.offsetStart
            )

            if let l = tensor.layerIndex, let exp = tensor.expertId {
                let key = ExpertKey(layer: Int(l), expertId: Int(exp))
                expertSlices[key, default: []].append(slice)
            } else {
                denseSlices.append(slice)
            }
        }

        // Pin dense backbone immediately (Embeddings, Attn/QKV, RMSNorms, Routers, Shared Experts, LM Head)
        pinDenseBackbone(shardBuffers: shardBuffers)

        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        }
    }

    func pinDenseBackbone(shardBuffers: [UInt32: MTLBuffer]) {
        for slice in denseSlices {
            if let buf = shardBuffers[slice.shardIndex] {
                let ptr = buf.contents().advanced(by: Int(slice.offset))
                madvise(ptr, Int(slice.length), MADV_WILLNEED)
            }
        }
    }

    func preFaultAll(shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        for shard in summary.shards {
            if let buf = shardBuffers[shard.index] {
                madvise(buf.contents(), Int(shard.length), MADV_WILLNEED)
            }
        }
        for key in expertSlices.keys {
            residentExperts.insert(key)
        }
    }

    func setBudgetMode(mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        } else {
            let maxAllowed = mode.maxResidentExperts
            while residentExperts.count > maxAllowed, !lruList.isEmpty {
                let evictKey = lruList.removeFirst()
                residentExperts.remove(evictKey)
                if let slices = expertSlices[evictKey] {
                    for slice in slices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            madvise(ptr, Int(slice.length), MADV_DONTNEED)
                        }
                    }
                }
            }
        }
    }

    func prefetchLayerExperts(layer: Int, expertIds: [Int], shardBuffers: [UInt32: MTLBuffer]) {
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }
            for expId in expertIds {
                let key = ExpertKey(layer: layer, expertId: expId)
                if let slices = self.expertSlices[key] {
                    for slice in slices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            posix_madvise(ptr, Int(slice.length), POSIX_MADV_WILLNEED)
                        }
                    }
                }
            }
        }
    }

    func touchAndEvict(layer: Int, activeExpertIds: [Int], mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer]) {
        let t0 = CFAbsoluteTimeGetCurrent()
        var pageFaulted = false

        for expId in activeExpertIds {
            let key = ExpertKey(layer: layer, expertId: expId)
            totalAccesses += 1

            if residentExperts.contains(key) {
                cacheHits += 1
                if let idx = lruList.firstIndex(of: key) {
                    lruList.remove(at: idx)
                }
                lruList.append(key)
            } else {
                cacheMisses += 1
                pageFaulted = true
                residentExperts.insert(key)
                lruList.append(key)

                // Demand page-in from SSD
                if let slices = expertSlices[key] {
                    for slice in slices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            madvise(ptr, Int(slice.length), MADV_WILLNEED)
                        }
                    }
                }
            }
        }

        // If not unrestricted, enforce working set budget eviction
        if mode != .unrestricted {
            let maxAllowed = mode.maxResidentExperts
            while residentExperts.count > maxAllowed, !lruList.isEmpty {
                let evictKey = lruList.removeFirst()
                if evictKey.layer == layer && activeExpertIds.contains(evictKey.expertId) {
                    lruList.append(evictKey)
                    break
                }
                residentExperts.remove(evictKey)
                if let slices = expertSlices[evictKey] {
                    for slice in slices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            madvise(ptr, Int(slice.length), MADV_DONTNEED)
                        }
                    }
                }
            }
        }

        if pageFaulted {
            lastPagingLatencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        }
    }

    func flushAllExperts(shardBuffers: [UInt32: MTLBuffer]) {
        for key in residentExperts {
            if let slices = expertSlices[key] {
                for slice in slices {
                    if let buf = shardBuffers[slice.shardIndex] {
                        let ptr = buf.contents().advanced(by: Int(slice.offset))
                        madvise(ptr, Int(slice.length), MADV_DONTNEED)
                    }
                }
            }
        }
        residentExperts.removeAll()
        lruList.removeAll()
        cacheHits = 0
        cacheMisses = 0
        totalAccesses = 0
        lastPagingLatencyMs = 0.0
    }
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

    // Autoregressive Text Generation State
    @State private var temperature: Float = 0.7
    @State private var topP: Float = 0.9
    @State private var topK: Int = 50
    @State private var repetitionPenalty: Float = 1.1
    @State private var maxNewTokens: Int = 64
    @State private var isGeneratingText: Bool = false
    @State private var generatedStreamText: String = ""
    @State private var generationSpeedTokPerSec: Double = 0.0
    @State private var generationTotalTokens: Int = 0
    @State private var generationElapsedMs: Double = 0.0
    @State private var generationStatusText: String? = nil
    @State private var generationTask: Task<Void, Never>? = nil

    // Working Set & Dynamic SSD Expert Paging State
    @State private var memoryBudgetMode: MemoryBudgetMode = .balanced16GB
    @State private var currentRssGB: Double = 0.0
    @State private var residentExpertCount: Int = 0
    @State private var totalExpertCount: Int = 0
    @State private var cacheHitRate: Double = 100.0
    @State private var lastPagingLatencyMs: Double = 0.0
    @State private var pagingStatusMessage: String? = nil

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
                            .disabled(isExecutingLMHead || isExecutingMultiLayer || isExecutingFullLayer || isExecutingMlp || isGeneratingText)

                            Button(action: {
                                if isGeneratingText {
                                    stopAutoregressiveGeneration()
                                } else {
                                    startAutoregressiveGeneration()
                                }
                            }) {
                                Label(isGeneratingText ? "Stop Generation" : "Generate Text", systemImage: isGeneratingText ? "stop.fill" : "sparkles")
                            }
                            .buttonStyle(.borderedProminent)
                            .tint(isGeneratingText ? .red : .purple)
                            .disabled(summary == nil || tokenizer == nil || isExecutingLMHead || isExecutingMultiLayer || isExecutingFullLayer || isExecutingMlp)
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
                                ExpertRoutingBadgeView(rank: rank, expertId: expert.id, weight: expert.weight)
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

                // MARK: - Autoregressive Text Generation Playground Card
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("Interactive Text Generation (Autoregressive Engine)", systemImage: "sparkles")
                            .font(.headline)
                            .foregroundColor(.purple)

                        Spacer()

                        if isGeneratingText {
                            ProgressView()
                                .scaleEffect(0.7)
                            Text(generationStatusText ?? "Generating tokens...")
                                .font(.caption)
                                .foregroundColor(.purple)
                        } else if let status = generationStatusText {
                            Text(status)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Preset Prompts Chips
                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 6) {
                            Text("Presets:")
                                .font(.caption)
                                .foregroundColor(.secondary)

                            Button("🌟 Mixture of Experts") {
                                promptInput = "What is a Mixture of Experts (MoE) neural network and why is it efficient?"
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("⚛️ Quantum Computing") {
                                promptInput = "Explain the fundamental principles of quantum computing in simple terms:"
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("⚡ Apple Silicon") {
                                promptInput = "The architectural advantages of unified memory on Apple Silicon for LLMs are:"
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)

                            Button("💻 Swift Async") {
                                promptInput = "Write a Swift actor that manages thread-safe caching with async/await:"
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }

                    // Generation Hyperparameters Bar
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Temperature:")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.2f", temperature))
                                    .font(.system(.caption2, design: .monospaced))
                                    .fontWeight(.bold)
                            }
                            Slider(value: $temperature, in: 0.0...2.0, step: 0.05)
                                .frame(width: 120)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Top-P (Nucleus):")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.2f", topP))
                                    .font(.system(.caption2, design: .monospaced))
                                    .fontWeight(.bold)
                            }
                            Slider(value: $topP, in: 0.1...1.0, step: 0.05)
                                .frame(width: 120)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Rep. Penalty:")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text(String(format: "%.2f", repetitionPenalty))
                                    .font(.system(.caption2, design: .monospaced))
                                    .fontWeight(.bold)
                            }
                            Slider(value: $repetitionPenalty, in: 1.0...1.5, step: 0.05)
                                .frame(width: 120)
                        }

                        VStack(alignment: .leading, spacing: 2) {
                            HStack {
                                Text("Max Tokens:")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                                Text("\(maxNewTokens)")
                                    .font(.system(.caption2, design: .monospaced))
                                    .fontWeight(.bold)
                            }
                            Stepper("", value: $maxNewTokens, in: 1...512, step: 16)
                                .labelsHidden()
                        }

                        Spacer()

                        HStack(spacing: 8) {
                            if isGeneratingText {
                                Button(action: stopAutoregressiveGeneration) {
                                    Label("Stop", systemImage: "stop.fill")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.red)
                            } else {
                                Button(action: startAutoregressiveGeneration) {
                                    Label("Generate Text", systemImage: "sparkles")
                                }
                                .buttonStyle(.borderedProminent)
                                .tint(.purple)
                                .disabled(summary == nil || tokenizer == nil || promptInput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines).isEmpty)
                            }

                            if !generatedStreamText.isEmpty {
                                Button(action: {
                                    generatedStreamText = ""
                                    generationTotalTokens = 0
                                    generationStatusText = nil
                                }) {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.bordered)
                                .help("Clear Generated Text")
                            }
                        }
                    }
                    .padding(8)
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(8)

                    // Live Streaming Output Window
                    if !generatedStreamText.isEmpty || isGeneratingText {
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Label("Stream Output", systemImage: "text.bubble.fill")
                                    .font(.caption)
                                    .fontWeight(.semibold)
                                    .foregroundColor(.purple)

                                Spacer()

                                if generationTotalTokens > 0 {
                                    HStack(spacing: 10) {
                                        Text("⚡ \(String(format: "%.1f", generationSpeedTokPerSec)) tok/s")
                                            .font(.system(.caption2, design: .monospaced))
                                            .fontWeight(.bold)
                                            .foregroundColor(.green)

                                        Text("\(generationTotalTokens) tokens")
                                            .font(.system(.caption2, design: .monospaced))
                                            .foregroundColor(.secondary)

                                        Text("\(String(format: "%.0f", generationElapsedMs)) ms")
                                            .font(.system(.caption2, design: .monospaced))
                                            .foregroundColor(.secondary)
                                    }
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(Color.purple.opacity(0.1))
                                    .cornerRadius(4)
                                }
                            }

                            ScrollView {
                                VStack(alignment: .leading, spacing: 4) {
                                    HStack(alignment: .top, spacing: 0) {
                                        Text(generatedStreamText)
                                            .font(.system(.body, design: .default))
                                            .textSelection(.enabled)

                                        if isGeneratingText {
                                            Text("▊")
                                                .foregroundColor(.purple)
                                                .opacity(0.8)
                                        }
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                                .padding(10)
                            }
                            .frame(minHeight: 80, maxHeight: 220)
                            .background(Color(NSColor.textBackgroundColor))
                            .cornerRadius(6)
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(Color.purple.opacity(0.3), lineWidth: 1)
                            )
                        }
                        .padding(8)
                        .background(Color.purple.opacity(0.04))
                        .cornerRadius(8)
                    }
                }
                .padding(10)
                .background(Color.purple.opacity(0.06))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color.purple.opacity(0.2), lineWidth: 1)
                )

                // MARK: - Dynamic SSD Expert Paging & Working Set Controller Card
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("Dynamic SSD Expert Paging & Memory Controller", systemImage: "memorychip")
                            .font(.headline)
                            .foregroundColor(.indigo)

                        Spacer()

                        if let msg = pagingStatusMessage {
                            Text(msg)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    }

                    // Memory Budget Mode Selector & Real-Time RAM Gauge
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Working-Set Memory Budget:")
                                .font(.caption2)
                                .foregroundColor(.secondary)

                            Picker("Working-Set Budget", selection: $memoryBudgetMode) {
                                ForEach(MemoryBudgetMode.allCases) { mode in
                                    Text(mode.rawValue).tag(mode)
                                }
                            }
                            .pickerStyle(.segmented)
                            .frame(width: 340)
                            .onChange(of: memoryBudgetMode) { newMode in
                                applyMemoryBudget(newMode)
                            }
                        }

                        Spacer()

                        // Action Buttons
                        HStack(spacing: 8) {
                            Button(action: flushExpertCache) {
                                Label("Flush Cache", systemImage: "trash.circle")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Release all resident expert pages using madvise(MADV_DONTNEED) to minimize RAM to baseline")

                            Button(action: preFaultAllWeights) {
                                Label("Pre-Fault All", systemImage: "bolt.fill")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .help("Pre-fault all model shards into RAM for maximum raw throughput on high-RAM Macs")
                        }
                    }

                    // Real-Time Memory & Paging Telemetry Grid
                    HStack(spacing: 12) {
                        // Physical RSS
                        VStack(alignment: .leading, spacing: 2) {
                            Text("PHYSICAL RSS")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary)
                            HStack(alignment: .bottom, spacing: 4) {
                                Text(String(format: "%.2f", currentRssGB))
                                    .font(.system(.title3, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(currentRssGB > memoryBudgetMode.targetMaxRssGB ? .orange : .indigo)
                                Text("GB / \(String(format: "%.0f", memoryBudgetMode.targetMaxRssGB)) GB")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)

                        // Resident Experts
                        VStack(alignment: .leading, spacing: 2) {
                            Text("RESIDENT EXPERTS")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary)
                            HStack(alignment: .bottom, spacing: 4) {
                                Text("\(residentExpertCount)")
                                    .font(.system(.title3, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(.purple)
                                Text("/ \(totalExpertCount > 0 ? "\(totalExpertCount)" : "10,240")")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)

                        // Cache Hit Rate
                        VStack(alignment: .leading, spacing: 2) {
                            Text("PREFETCH HIT RATE")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary)
                            HStack(alignment: .bottom, spacing: 4) {
                                Text(String(format: "%.1f%%", cacheHitRate))
                                    .font(.system(.title3, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(cacheHitRate >= 80.0 ? .green : (cacheHitRate >= 50.0 ? .yellow : .orange))
                                Text("(\(WorkingSetManager.shared.totalAccesses) reqs)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)

                        // SSD Paging Latency
                        VStack(alignment: .leading, spacing: 2) {
                            Text("PAGING OVERHEAD")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundColor(.secondary)
                            HStack(alignment: .bottom, spacing: 4) {
                                Text(String(format: "%.2f", lastPagingLatencyMs))
                                    .font(.system(.title3, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(lastPagingLatencyMs < 2.0 ? .green : .blue)
                                Text("ms/layer")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(NSColor.controlBackgroundColor))
                        .cornerRadius(6)
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
        .onAppear {
            updatePagingStats()
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                updatePagingStats()
            }
        }
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
                let embedScaleRaw = (embedScale != nil) ? shardBuffers[embedScale!.shardIndex] : rawBaseBuffer
                var scaleOffset = embedScale?.offsetStart ?? 0
                
                if let kernelFunction = defaultLibrary.makeFunction(name: "lookup_embeddings_fp8") {
                    let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                    computeEncoder.setComputePipelineState(pipelineState)
                    computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0OutputBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(embedScaleRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                }
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

            // Load pipeline states for FP8 and BF16 SwiGLU & Down Projections
            guard let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to locate SwiGLU / DownProj Metal kernels."
                return
            }

            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
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

                let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                          let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                          let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                        continue
                    }

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale!.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale!.offsetStart
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale!.offsetStart

                    // Dispatch SwiGLU (Gate & Up Proj)
                    computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                    computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    // Dispatch Down Proj with Weighted Accumulation
                    computeEncoder.setComputePipelineState(fp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
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

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
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
                  let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for full layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
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
                let isFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                var oProjOffset = oProjTensor.offsetStart
                var inAttnDim = hiddenDim

                if isFP8, let gemvKernel = defaultLibrary.makeFunction(name: "fp8_gemv") {
                    let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                    let oScaleTensor = summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(layerIndex) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                    })
                    let oScaleRaw = (oScaleTensor != nil) ? shardBuffers[oScaleTensor!.shardIndex] : oProjRaw
                    var oScaleOffset = oScaleTensor?.offsetStart ?? 0

                    computeEncoder.setComputePipelineState(gemvPipeline)
                    computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(oScaleRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
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

                let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                          let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                          let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                        continue
                    }

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale!.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale!.offsetStart
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale!.offsetStart

                    computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                    computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    computeEncoder.setComputePipelineState(fp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
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

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
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
                  let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let routerKernel = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
                  let sharedGateKernel = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for multi-layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
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
                    let isFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                    var oProjOffset = oProjTensor.offsetStart
                    var inAttnDim = hiddenDim

                    if isFP8, let gemvKernel = defaultLibrary.makeFunction(name: "fp8_gemv") {
                        let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                        let oScaleTensor = summary.tensors.first(where: { t in
                            t.layerIndex == UInt32(l) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                        })
                        let oScaleRaw = (oScaleTensor != nil) ? shardBuffers[oScaleTensor!.shardIndex] : oProjRaw
                        var oScaleOffset = oScaleTensor?.offsetStart ?? 0

                        computeEncoder.setComputePipelineState(gemvPipeline)
                        computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(oScaleRaw, offset: 0, index: 3)
                        computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
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

                // Demand Paging & Working Set LRU Eviction for Active Experts
                let activeIds = activeExperts.map { $0.id }
                WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: memoryBudgetMode, shardBuffers: shardBuffers)

                // Async Lookahead Prefetching for layer l + 1
                if l + 1 < actualLayers {
                    WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: [0, 1, 2, 3, 4, 5, 6, 7], shardBuffers: shardBuffers)
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

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isFP8 {
                        let gateScale = findTensor(expTag, "gate_proj", true)
                        let upScale   = findTensor(expTag, "up_proj", true)
                        let downScale = findTensor(expTag, "down_proj", true)

                        guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                              let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                              let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                            continue
                        }

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale!.offsetStart
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale!.offsetStart
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale!.offsetStart

                        computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                        computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(fp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                        computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
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

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart
                            var sharedW: Float = 0.5

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedW, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
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
            updatePagingStats()

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

    // MARK: - Autoregressive Generation & Sampling Engine

    private func buildCachedLayers(summary: ModelSummary) -> [CachedLayer] {
        var tensorsByLayer: [UInt32: [TensorMetadata]] = [:]
        for t in summary.tensors {
            if let l = t.layerIndex {
                tensorsByLayer[l, default: []].append(t)
            }
        }

        var cached: [CachedLayer] = []
        let numLayers = min(targetLayerCount, Int(summary.layerCount > 0 ? summary.layerCount : 40))

        for l in 0..<numLayers {
            let layerTensors = tensorsByLayer[UInt32(l)] ?? []

            let router = layerTensors.first(where: { $0.name.contains("mlp.gate.weight") })
            let sharedGate = layerTensors.first(where: { $0.name.contains("shared_expert_gate.weight") })
            let norm1 = layerTensors.first(where: { $0.name.contains("input_layernorm") })
            let oProj = layerTensors.first(where: { ($0.name.contains("o_proj") || $0.name.contains("out_proj")) && !$0.name.contains("scale") })
            let oScale = layerTensors.first(where: { ($0.name.contains("o_proj") || $0.name.contains("out_proj")) && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let norm2 = layerTensors.first(where: { $0.name.contains("post_attention_layernorm") })

            let sharedGateW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("gate_proj") && !$0.name.contains("scale") })
            let sharedUpW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("up_proj") && !$0.name.contains("scale") })
            let sharedDownW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("down_proj") && !$0.name.contains("scale") })

            let sharedGateS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let sharedUpS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let sharedDownS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })

            var expGateW: [Int: TensorMetadata] = [:]
            var expUpW: [Int: TensorMetadata] = [:]
            var expDownW: [Int: TensorMetadata] = [:]
            var expGateS: [Int: TensorMetadata] = [:]
            var expUpS: [Int: TensorMetadata] = [:]
            var expDownS: [Int: TensorMetadata] = [:]

            for t in layerTensors {
                if let e = t.expertId {
                    let expId = Int(e)
                    let isScale = t.name.contains("scale") || t.name.contains("scales")
                    if t.name.contains("gate_proj") {
                        if isScale { expGateS[expId] = t } else { expGateW[expId] = t }
                    } else if t.name.contains("up_proj") {
                        if isScale { expUpS[expId] = t } else { expUpW[expId] = t }
                    } else if t.name.contains("down_proj") {
                        if isScale { expDownS[expId] = t } else { expDownW[expId] = t }
                    }
                }
            }

            var intermediateDim: UInt32 = 512
            if let sampleGate = sharedGateW ?? expGateW.values.first {
                let dims = sampleGate.shapeDisplay
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                    .components(separatedBy: ",")
                    .compactMap { UInt32($0.trimmingCharacters(in: CharacterSet.whitespaces)) }
                if dims.count >= 2 {
                    intermediateDim = dims[0]
                }
            }

            cached.append(CachedLayer(
                layerIndex: UInt32(l),
                routerTensor: router,
                sharedGateTensor: sharedGate,
                norm1Tensor: norm1,
                oProjTensor: oProj,
                oScaleTensor: oScale,
                norm2Tensor: norm2,
                sharedGateWeight: sharedGateW,
                sharedUpWeight: sharedUpW,
                sharedDownWeight: sharedDownW,
                sharedGateScale: sharedGateS,
                sharedUpScale: sharedUpS,
                sharedDownScale: sharedDownS,
                expertGateWeights: expGateW,
                expertUpWeights: expUpW,
                expertDownWeights: expDownW,
                expertGateScales: expGateS,
                expertUpScales: expUpS,
                expertDownScales: expDownS,
                intermediateDim: intermediateDim
            ))
        }
        return cached
    }

    private func sampleNextToken(
        logits: UnsafePointer<Float>,
        vocabSize: Int,
        contextTokens: [UInt32],
        temperature: Float,
        topP: Float,
        topK: Int,
        repetitionPenalty: Float
    ) -> UInt32 {
        let recentContextSet = Set(contextTokens.suffix(256))
        
        // High-performance greedy fast path (temperature <= 0.01)
        if temperature <= 0.01 {
            var bestIdx = 0
            var bestLogit: Float = -Float.greatestFiniteMagnitude
            for v in 0..<vocabSize {
                var logit = logits[v]
                if repetitionPenalty > 1.001 && recentContextSet.contains(UInt32(v)) {
                    logit = logit > 0 ? (logit / repetitionPenalty) : (logit * repetitionPenalty)
                }
                if logit > bestLogit {
                    bestLogit = logit
                    bestIdx = v
                }
            }
            return UInt32(bestIdx)
        }

        let effectiveTopK = max(1, min(topK, vocabSize))
        var candidates: [(id: Int, logit: Float)] = []
        candidates.reserveCapacity(effectiveTopK)

        for v in 0..<vocabSize {
            var logit = logits[v]
            if repetitionPenalty > 1.001 && recentContextSet.contains(UInt32(v)) {
                logit = logit > 0 ? (logit / repetitionPenalty) : (logit * repetitionPenalty)
            }

            if candidates.count < effectiveTopK {
                candidates.append((id: v, logit: logit))
                if candidates.count == effectiveTopK {
                    candidates.sort(by: { $0.logit > $1.logit })
                }
            } else if logit > candidates[effectiveTopK - 1].logit {
                var low = 0
                var high = effectiveTopK - 1
                while low < high {
                    let mid = (low + high) / 2
                    if candidates[mid].logit < logit {
                        high = mid
                    } else {
                        low = mid + 1
                    }
                }
                candidates.insert((id: v, logit: logit), at: low)
                candidates.removeLast()
            }
        }

        guard let first = candidates.first else { return 0 }

        let invTemp = 1.0 / max(temperature, 0.01)
        let maxLogit = first.logit
        var expSum: Float = 0.0
        var probs: [Float] = []
        probs.reserveCapacity(candidates.count)

        for c in candidates {
            let p = Darwin.exp((c.logit - maxLogit) * invTemp)
            probs.append(p)
            expSum += p
        }

        if expSum <= 0.0 {
            return UInt32(first.id)
        }

        for i in 0..<probs.count {
            probs[i] /= expSum
        }

        var cumulativeProb: Float = 0.0
        var cutoffIndex = candidates.count - 1
        for (i, p) in probs.enumerated() {
            cumulativeProb += p
            if cumulativeProb >= topP {
                cutoffIndex = i
                break
            }
        }

        let nucleusSum = probs[0...cutoffIndex].reduce(0, +)
        let randomVal = Float.random(in: 0..<1.0) * nucleusSum
        var runningSum: Float = 0.0
        for i in 0...cutoffIndex {
            runningSum += probs[i]
            if runningSum >= randomVal {
                return UInt32(candidates[i].id)
            }
        }

        return UInt32(candidates[0].id)
    }

    private func stopAutoregressiveGeneration() {
        isGeneratingText = false
        generationTask?.cancel()
        generationTask = nil
        generationStatusText = "⏹ Generation stopped by user."
    }

    private func startAutoregressiveGeneration() {
        guard let summary = summary,
              let tokenizer = tokenizer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            let err = "❌ Metal or Tokenizer not ready for text generation."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let prompt = promptInput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        guard !prompt.isEmpty else {
            let err = "⚠️ Please enter a prompt to generate text."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try tokenizer.encode(text: prompt)
        } catch {
            let err = "❌ Tokenizer failed to encode prompt: \(error.localizedDescription)"
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        guard !promptTokenIds.isEmpty else {
            let err = "⚠️ Prompt tokenization produced 0 tokens."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Find Embedding Weight
        guard let embedWeight = summary.tensors.first(where: {
            ($0.name.contains("embed_tokens") || $0.name.contains("embed") || $0.name.contains("wte")) &&
            $0.name.contains("weight") && !$0.name.contains("scale")
        }), let embedShardBuffer = shardBuffers[embedWeight.shardIndex] else {
            let err = "❌ Embedding weight tensor not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Find Final RMSNorm
        guard let normTensor = summary.tensors.first(where: {
            $0.name == "model.language_model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers."))
        }), let normShardBuffer = shardBuffers[normTensor.shardIndex] else {
            let err = "❌ Final RMSNorm weight not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Find LM Head Weight
        guard let lmHeadTensor = summary.tensors.first(where: {
            $0.name == "lm_head.weight" ||
            $0.name == "language_model.lm_head.weight" ||
            $0.name == "model.lm_head.weight"
        }), let lmHeadShardBuffer = shardBuffers[lmHeadTensor.shardIndex] else {
            let err = "❌ LM Head weight not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Compile Pipelines with correct kernel names
        guard let embedBF16Function = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16"),
              let routerFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvBF16Function = defaultLibrary.makeFunction(name: "bf16_gemv"),
              let addFunction = defaultLibrary.makeFunction(name: "vector_add_f32"),
              let clearFunction = defaultLibrary.makeFunction(name: "clear_vector_f32"),
              let fp8GateUpFunction = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
              let fp8DownFunction = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate") else {
            let err = "❌ Failed to load required Metal compute shaders."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let embedPipeline: MTLComputePipelineState
        let routerPipeline: MTLComputePipelineState
        let rmsnormPipeline: MTLComputePipelineState
        let gemvBF16Pipeline: MTLComputePipelineState
        let fp8GemvPipeline: MTLComputePipelineState?
        let addPipeline: MTLComputePipelineState
        let clearPipeline: MTLComputePipelineState
        let fp8GateUpPipeline: MTLComputePipelineState
        let fp8DownPipeline: MTLComputePipelineState
        let bf16GateUpPipeline: MTLComputePipelineState?
        let bf16DownPipeline: MTLComputePipelineState?

        do {
            embedPipeline = try device.makeComputePipelineState(function: embedBF16Function)
            routerPipeline = try device.makeComputePipelineState(function: routerFunction)
            rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)
            gemvBF16Pipeline = try device.makeComputePipelineState(function: gemvBF16Function)
            if let fp8GemvFunc = defaultLibrary.makeFunction(name: "fp8_gemv") {
                fp8GemvPipeline = try device.makeComputePipelineState(function: fp8GemvFunc)
            } else {
                fp8GemvPipeline = nil
            }
            addPipeline = try device.makeComputePipelineState(function: addFunction)
            clearPipeline = try device.makeComputePipelineState(function: clearFunction)
            fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpFunction)
            fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownFunction)
            if let bGateUp = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
               let bDown = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") {
                bf16GateUpPipeline = try device.makeComputePipelineState(function: bGateUp)
                bf16DownPipeline = try device.makeComputePipelineState(function: bDown)
            } else {
                bf16GateUpPipeline = nil
                bf16DownPipeline = nil
            }
        } catch {
            let err = "❌ Failed to create compute pipelines: \(error.localizedDescription)"
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Model Hyperparameters
        let hiddenDim: UInt32 = UInt32(activeHiddenDim > 0 ? activeHiddenDim : 2048)
        let numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : 256
        let embedOffset = embedWeight.offsetStart
        let normOffset = normTensor.offsetStart
        let lmHeadOffset = lmHeadTensor.offsetStart
        let eps: Float = 1e-6

        var vocabSize: UInt32 = 248320
        let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
        let shapeParts = cleanShape.split(separator: ",")
        if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
            vocabSize = parsed
        }

        let cachedLayers = buildCachedLayers(summary: summary)
        let actualLayers = cachedLayers.count
        guard actualLayers > 0 else {
            let err = "❌ No layer tensors found in model."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let maxInterDim = cachedLayers.map { $0.intermediateDim }.max() ?? 512

        // Allocate Shared Scratch Buffers (Reused across all generation steps)
        guard let singleTokenBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let hCurrBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hNextBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnOutBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMidBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm2Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let routerIndicesBuffer = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let routerWeightsBuffer = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuffer = device.makeBuffer(length: Int(maxInterDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            let err = "❌ Failed to allocate GPU scratch buffers."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let temp = self.temperature
        let topPVal = self.topP
        let topKVal = self.topK
        let repPen = self.repetitionPenalty
        let maxTokens = self.maxNewTokens
        let buffers = self.shardBuffers
        let budgetMode = self.memoryBudgetMode

        isGeneratingText = true
        generatedStreamText = ""
        generationTotalTokens = 0
        generationSpeedTokPerSec = 0.0
        generationElapsedMs = 0.0
        generationStatusText = "⚡ Initializing Autoregressive Generation..."

        generationTask = Task.detached(priority: .userInitiated) {
            var contextTokens = promptTokenIds
            let startTime = CFAbsoluteTimeGetCurrent()
            var tokensGenerated = 0

            for _ in 0..<maxTokens {
                if Task.isCancelled { break }

                let currentTokenId = contextTokens.last!

                // 1. Embed Current Token (lookup_embeddings_bf16)
                let singleTokenPtr = singleTokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
                singleTokenPtr[0] = currentTokenId

                guard let cmdBuffer = commandQueue.makeCommandBuffer(),
                      let enc = cmdBuffer.makeComputeCommandEncoder() else { break }

                enc.setComputePipelineState(embedPipeline)
                enc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                enc.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                enc.setBuffer(hCurrBuffer, offset: 0, index: 2)
                var wOffset = embedOffset
                var hDim = hiddenDim
                var tokCount: UInt32 = 1
                enc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                enc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                enc.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                enc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                enc.endEncoding()
                cmdBuffer.commit()
                cmdBuffer.waitUntilCompleted()

                // 2. Multi-Layer Transformer Backbone (0..<actualLayers)
                var currentH = hCurrBuffer
                var nextH = hNextBuffer

                for l in 0..<actualLayers {
                    if Task.isCancelled { break }
                    let layer = cachedLayers[l]

                    // Async Lookahead Prefetching for layer l + 1
                    if l + 1 < actualLayers {
                        WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: [0, 1, 2, 3, 4, 5, 6, 7], shardBuffers: buffers)
                    }

                    // Dynamic MoE Router Top-K Shader Dispatch
                    var activeExperts: [(id: Int, weight: Float)] = []
                    if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                        guard let lCmd = commandQueue.makeCommandBuffer(),
                              let lEnc = lCmd.makeComputeCommandEncoder() else { break }

                        var rOffset = routerTensor.offsetStart
                        var nExp = numExperts
                        var kVal: UInt32 = 8
                        lEnc.setComputePipelineState(routerPipeline)
                        lEnc.setBuffer(routerRaw, offset: 0, index: 0)
                        lEnc.setBuffer(currentH, offset: 0, index: 1)
                        lEnc.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                        lEnc.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                        lEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        lEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        lEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                        lEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        lEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                        lEnc.endEncoding()
                        lCmd.commit()
                        lCmd.waitUntilCompleted()

                        let indPtr = routerIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: 8)
                        let wPtr = routerWeightsBuffer.contents().bindMemory(to: Float.self, capacity: 8)
                        for i in 0..<8 {
                            activeExperts.append((id: Int(indPtr[i]), weight: wPtr[i]))
                        }
                    } else {
                        activeExperts = (0..<8).map { (id: $0, weight: 1.0 / 8.0) }
                    }

                    // Demand Paging & Working Set LRU Eviction for Active Experts
                    let activeIds = activeExperts.map { $0.id }
                    WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: budgetMode, shardBuffers: buffers)

                    guard let layerCmd = commandQueue.makeCommandBuffer(),
                          let layerEnc = layerCmd.makeComputeCommandEncoder() else { break }

                    // Step 1: Pre-Attention RMSNorm (currentH -> xNorm1)
                    if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                        var gammaOff = norm1.offsetStart
                        var epsVal = eps
                        layerEnc.setComputePipelineState(rmsnormPipeline)
                        layerEnc.setBuffer(currentH, offset: 0, index: 0)
                        layerEnc.setBuffer(norm1Raw, offset: 0, index: 1)
                        layerEnc.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                        layerEnc.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        layerEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    }

                    // Step 2: Attention Computation (xNorm1 -> attnOut)
                    layerEnc.setComputePipelineState(clearPipeline)
                    layerEnc.setBuffer(attnOutBuffer, offset: 0, index: 0)
                    layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    if let oProj = layer.oProjTensor, let oProjRaw = buffers[oProj.shardIndex] {
                        let isFP8 = !oProj.dtype.contains("BF16") && !oProj.dtype.contains("FLOAT")
                        var oOffset = oProj.offsetStart
                        var inAttnDim = hiddenDim

                        if isFP8, let fp8GemvPipe = fp8GemvPipeline {
                            let oScale = layer.oScaleTensor
                            let oScaleRaw = (oScale != nil) ? buffers[oScale!.shardIndex] : oProjRaw
                            var oScaleOff = oScale?.offsetStart ?? 0
                            layerEnc.setComputePipelineState(fp8GemvPipe)
                            layerEnc.setBuffer(oProjRaw, offset: 0, index: 0)
                            layerEnc.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                            layerEnc.setBuffer(attnOutBuffer, offset: 0, index: 2)
                            layerEnc.setBuffer(oScaleRaw, offset: 0, index: 3)
                            layerEnc.setBytes(&oOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc.setBytes(&oScaleOff, length: MemoryLayout<UInt64>.stride, index: 5)
                            layerEnc.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), fp8GemvPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else {
                            layerEnc.setComputePipelineState(gemvBF16Pipeline)
                            layerEnc.setBuffer(oProjRaw, offset: 0, index: 0)
                            layerEnc.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                            layerEnc.setBuffer(attnOutBuffer, offset: 0, index: 2)
                            layerEnc.setBytes(&oOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), gemvBF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                    }

                    // Step 3: Residual Connection 1 (hMid = currentH + attnOut)
                    layerEnc.setComputePipelineState(addPipeline)
                    layerEnc.setBuffer(currentH, offset: 0, index: 0)
                    layerEnc.setBuffer(attnOutBuffer, offset: 0, index: 1)
                    layerEnc.setBuffer(hMidBuffer, offset: 0, index: 2)
                    layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                    layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    // Step 4: Post-Attention RMSNorm (hMid -> xNorm2)
                    if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                        var gammaOff = norm2.offsetStart
                        var epsVal = eps
                        layerEnc.setComputePipelineState(rmsnormPipeline)
                        layerEnc.setBuffer(hMidBuffer, offset: 0, index: 0)
                        layerEnc.setBuffer(norm2Raw, offset: 0, index: 1)
                        layerEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        layerEnc.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        layerEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    }

                    // Step 5: MoE SwiGLU Feed-Forward (xNorm2 -> hMlp)
                    layerEnc.setComputePipelineState(clearPipeline)
                    layerEnc.setBuffer(hMlpBuffer, offset: 0, index: 0)
                    layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    var intermediateDim = layer.intermediateDim

                    // Active Top-8 Experts
                    for expert in activeExperts {
                        let expId = expert.id
                        var p_k = expert.weight
                        if p_k <= 0.00001 { continue }

                        if let gateW = layer.expertGateWeights[expId],
                           let upW = layer.expertUpWeights[expId],
                           let downW = layer.expertDownWeights[expId],
                           let gateRaw = buffers[gateW.shardIndex],
                           let upRaw = buffers[upW.shardIndex],
                           let downRaw = buffers[downW.shardIndex] {
                            let isFP8 = !gateW.dtype.contains("BF16") && !gateW.dtype.contains("FLOAT")
                            if isFP8 {
                                let gateS = layer.expertGateScales[expId]
                                let upS = layer.expertUpScales[expId]
                                let downS = layer.expertDownScales[expId]

                                guard let gateSRaw = (gateS != nil) ? buffers[gateS!.shardIndex] : nil,
                                      let upSRaw = (upS != nil) ? buffers[upS!.shardIndex] : nil,
                                      let downSRaw = (downS != nil) ? buffers[downS!.shardIndex] : nil else { continue }

                                var gWOff = gateW.offsetStart
                                var gSOff = gateS!.offsetStart
                                var uWOff = upW.offsetStart
                                var uSOff = upS!.offsetStart
                                var dWOff = downW.offsetStart
                                var dSOff = downS!.offsetStart

                                layerEnc.setComputePipelineState(fp8GateUpPipeline)
                                layerEnc.setBuffer(gateRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(upRaw, offset: 0, index: 1)
                                layerEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(gateSRaw, offset: 0, index: 4)
                                layerEnc.setBuffer(upSRaw, offset: 0, index: 5)
                                layerEnc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                layerEnc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                layerEnc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                                layerEnc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 10)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)
                                layerEnc.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc.setComputePipelineState(fp8DownPipeline)
                                layerEnc.setBuffer(downRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                layerEnc.setBuffer(downSRaw, offset: 0, index: 3)
                                layerEnc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                                layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if let bGateUpPipe = bf16GateUpPipeline, let bDownPipe = bf16DownPipeline {
                                var gWOff = gateW.offsetStart
                                var uWOff = upW.offsetStart
                                var dWOff = downW.offsetStart

                                layerEnc.setComputePipelineState(bGateUpPipe)
                                layerEnc.setBuffer(gateRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(upRaw, offset: 0, index: 1)
                                layerEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 3)
                                layerEnc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), bGateUpPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc.setComputePipelineState(bDownPipe)
                                layerEnc.setBuffer(downRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                layerEnc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), bDownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        }
                    }

                    // Shared Expert
                    if let gateW = layer.sharedGateWeight,
                       let upW = layer.sharedUpWeight,
                       let downW = layer.sharedDownWeight,
                       let gateRaw = buffers[gateW.shardIndex],
                       let upRaw = buffers[upW.shardIndex],
                       let downRaw = buffers[downW.shardIndex] {
                        let isFP8 = !gateW.dtype.contains("BF16") && !gateW.dtype.contains("FLOAT")
                        var sharedW: Float = 0.5
                        if isFP8 {
                            let gateS = layer.sharedGateScale
                            let upS = layer.sharedUpScale
                            let downS = layer.sharedDownScale

                            if let gateSRaw = (gateS != nil) ? buffers[gateS!.shardIndex] : nil,
                               let upSRaw = (upS != nil) ? buffers[upS!.shardIndex] : nil,
                               let downSRaw = (downS != nil) ? buffers[downS!.shardIndex] : nil {
                                var gWOff = gateW.offsetStart
                                var gSOff = gateS!.offsetStart
                                var uWOff = upW.offsetStart
                                var uSOff = upS!.offsetStart
                                var dWOff = downW.offsetStart
                                var dSOff = downS!.offsetStart

                                layerEnc.setComputePipelineState(fp8GateUpPipeline)
                                layerEnc.setBuffer(gateRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(upRaw, offset: 0, index: 1)
                                layerEnc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(gateSRaw, offset: 0, index: 4)
                                layerEnc.setBuffer(upSRaw, offset: 0, index: 5)
                                layerEnc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                layerEnc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                layerEnc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                                layerEnc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 10)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)
                                layerEnc.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc.setComputePipelineState(fp8DownPipeline)
                                layerEnc.setBuffer(downRaw, offset: 0, index: 0)
                                layerEnc.setBuffer(interBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                layerEnc.setBuffer(downSRaw, offset: 0, index: 3)
                                layerEnc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&sharedW, length: MemoryLayout<Float>.stride, index: 8)
                                layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        }
                    }

                    // Step 6: Residual Connection 2 (nextH = hMid + hMlp)
                    layerEnc.setComputePipelineState(addPipeline)
                    layerEnc.setBuffer(hMidBuffer, offset: 0, index: 0)
                    layerEnc.setBuffer(hMlpBuffer, offset: 0, index: 1)
                    layerEnc.setBuffer(nextH, offset: 0, index: 2)
                    layerEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                    layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    layerEnc.endEncoding()
                    layerCmd.commit()
                    layerCmd.waitUntilCompleted()

                    // Swap ping-pong buffers
                    let tempBuf = currentH
                    currentH = nextH
                    nextH = tempBuf
                }

                // 3. Final RMSNorm & LM Head Vocabulary Projection
                guard let finalCmd = commandQueue.makeCommandBuffer(),
                      let finalEnc = finalCmd.makeComputeCommandEncoder() else { break }

                // Final RMSNorm
                var nOff = normOffset
                var vSize = vocabSize
                var epsVal = eps
                finalEnc.setComputePipelineState(rmsnormPipeline)
                finalEnc.setBuffer(currentH, offset: 0, index: 0)
                finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 2)
                finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                finalEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                // LM Head GEMV
                var lmOff = lmHeadOffset
                finalEnc.setComputePipelineState(gemvBF16Pipeline)
                finalEnc.setBuffer(lmHeadShardBuffer, offset: 0, index: 0)
                finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 1)
                finalEnc.setBuffer(logitsBuffer, offset: 0, index: 2)
                finalEnc.setBytes(&lmOff, length: MemoryLayout<UInt64>.stride, index: 3)
                finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                finalEnc.setBytes(&vSize, length: MemoryLayout<UInt32>.stride, index: 5)
                finalEnc.dispatchThreads(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gemvBF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                finalEnc.endEncoding()
                finalCmd.commit()
                finalCmd.waitUntilCompleted()

                // 4. Sample Next Token
                let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                let nextToken = sampleNextToken(
                    logits: logitsPtr,
                    vocabSize: Int(vocabSize),
                    contextTokens: contextTokens,
                    temperature: temp,
                    topP: topPVal,
                    topK: topKVal,
                    repetitionPenalty: repPen
                )

                // 5. Check EOS
                if nextToken == 248044 || nextToken == 248046 {
                    break
                }

                // 6. Decode Token Chunk
                let decoded = (try? tokenizer.decode(ids: [nextToken])) ?? ""
                if decoded.contains("<|im_end|>") || decoded.contains("<|endoftext|>") {
                    break
                }

                contextTokens.append(nextToken)
                tokensGenerated += 1

                let elapsedSec = CFAbsoluteTimeGetCurrent() - startTime
                let tokPerSec = Double(tokensGenerated) / max(elapsedSec, 0.001)
                let elapsedMs = elapsedSec * 1000.0
                let currentRss = getProcessResidentMemoryGB()
                let resCount = WorkingSetManager.shared.residentExperts.count
                let totalExp = WorkingSetManager.shared.totalExpertKeysCount
                let hitRate = WorkingSetManager.shared.cacheHitRatePercent
                let pageLat = WorkingSetManager.shared.lastPagingLatencyMs

                await MainActor.run {
                    self.generatedStreamText += decoded
                    self.generationTotalTokens = tokensGenerated
                    self.generationElapsedMs = elapsedMs
                    self.generationSpeedTokPerSec = tokPerSec
                    self.generationStatusText = "⚡ Streaming: \(tokensGenerated) tokens | \(String(format: "%.1f", tokPerSec)) tok/s"
                    self.currentRssGB = currentRss
                    self.residentExpertCount = resCount
                    self.totalExpertCount = totalExp
                    self.cacheHitRate = hitRate
                    self.lastPagingLatencyMs = pageLat
                }
            }

            let finalElapsedSec = CFAbsoluteTimeGetCurrent() - startTime
            let finalTokPerSec = Double(tokensGenerated) / max(finalElapsedSec, 0.001)
            let finalElapsedMs = finalElapsedSec * 1000.0
            let finalRss = getProcessResidentMemoryGB()
            let finalResCount = WorkingSetManager.shared.residentExperts.count
            let finalHitRate = WorkingSetManager.shared.cacheHitRatePercent
            let finalPageLat = WorkingSetManager.shared.lastPagingLatencyMs

            await MainActor.run {
                self.isGeneratingText = false
                self.generationTask = nil
                self.generationStatusText = "✨ Generated \(tokensGenerated) tokens in \(String(format: "%.2f", finalElapsedMs)) ms (\(String(format: "%.1f", finalTokPerSec)) tok/s)"
                self.currentRssGB = finalRss
                self.residentExpertCount = finalResCount
                self.cacheHitRate = finalHitRate
                self.lastPagingLatencyMs = finalPageLat
            }
        }
    }

    private func updatePagingStats() {
        self.currentRssGB = getProcessResidentMemoryGB()
        self.residentExpertCount = WorkingSetManager.shared.residentExperts.count
        self.totalExpertCount = WorkingSetManager.shared.totalExpertKeysCount
        self.cacheHitRate = WorkingSetManager.shared.cacheHitRatePercent
        self.lastPagingLatencyMs = WorkingSetManager.shared.lastPagingLatencyMs
    }

    private func applyMemoryBudget(_ mode: MemoryBudgetMode) {
        guard let summary = summary else { return }
        WorkingSetManager.shared.setBudgetMode(mode: mode, shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "⚡ Budget updated: \(mode.rawValue)"
    }

    private func flushExpertCache() {
        WorkingSetManager.shared.flushAllExperts(shardBuffers: shardBuffers)
        updatePagingStats()
        pagingStatusMessage = "🧹 Expert cache flushed (MADV_DONTNEED)"
    }

    private func preFaultAllWeights() {
        guard let summary = summary else { return }
        WorkingSetManager.shared.preFaultAll(shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "🚀 All weights pre-faulted into RAM"
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
            
            WorkingSetManager.shared.initialize(summary: loadedSummary, shardBuffers: buffers, mode: self.memoryBudgetMode)
            updatePagingStats()
            
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
              let rawBaseBuffer = shardBuffers[tensor.shardIndex] else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline or missing shard buffer."
            return
        }

        do {
            let isBF16 = tensor.dtype.contains("BF16") || tensor.dtype.contains("FLOAT")
            let sampleCount = 8
            let outputByteLength = sampleCount * MemoryLayout<Float>.stride
            guard let outputBuffer = device.makeBuffer(length: outputByteLength, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let baseName = tensor.name.replacingOccurrences(of: ".weight", with: "").replacingOccurrences(of: ".scales", with: "").replacingOccurrences(of: ".weight_scale", with: "")

            if isBF16 {
                guard let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_bf16_preview") else { return }
                let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                var weightOffset = tensor.offsetStart
                computeEncoder.setComputePipelineState(pipelineState)
                computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
                computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 2)
                computeEncoder.dispatchThreads(MTLSize(width: sampleCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sampleCount, height: 1, depth: 1))
            } else {
                let weightTensor = summary.tensors.first(where: { $0.name == "\(baseName).weight" }) ?? tensor
                let scaleTensor = summary.tensors.first(where: { $0.name == "\(baseName).weight_scale" || $0.name == "\(baseName).scales" || $0.name == "\(baseName).scale" })
                let scaleShardRaw = (scaleTensor != nil) ? shardBuffers[scaleTensor!.shardIndex] : rawBaseBuffer

                var weightOffset = weightTensor.offsetStart
                var scaleOffset = scaleTensor?.offsetStart ?? 0

                if let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_fp8_row_scaled") {
                    let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                    computeEncoder.setComputePipelineState(pipelineState)
                    computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(scaleShardRaw, offset: 0, index: 2)
                    computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.dispatchThreads(MTLSize(width: sampleCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sampleCount, height: 1, depth: 1))
                }
            }

            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let rawFloatPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
            var sampleValues: [String] = []
            for i in 0..<sampleCount {
                sampleValues.append(String(format: "%.6f", rawFloatPtr[i]))
            }

            gpuComputeOutput = "⚡ Dequantized! First 8 values for '\(tensor.name)' (Shard #\(tensor.shardIndex)): [\(sampleValues.joined(separator: ", "))]"

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
