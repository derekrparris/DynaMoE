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

final class KVCacheManager {
    static let shared = KVCacheManager()
    
    var kCacheBuffer: MTLBuffer?
    var vCacheBuffer: MTLBuffer?
    var linearStateBuffer: MTLBuffer?
    var convStateBuffer: MTLBuffer?
    var allocatedSeqLen: Int = 2048
    
    func reset(device: MTLDevice, config: ModelConfig? = nil, actualLayers: Int = 40, maxSeqLen: Int = 2048) {
        self.allocatedSeqLen = maxSeqLen
        
        let numLoops = max(1, config?.effectiveNumLoops ?? 1)
        let numKvHeads = config?.effectiveNumKeyValueHeads ?? 2
        let headDim = config?.effectiveHeadDim ?? 256
        let totalSlots = actualLayers * numLoops
        let kvStride = numKvHeads * headDim
        let kvBytes = max(totalSlots, 40) * maxSeqLen * max(kvStride, 512) * MemoryLayout<Float>.stride
        
        self.kCacheBuffer = device.makeBuffer(length: kvBytes, options: .storageModeShared)
        self.vCacheBuffer = device.makeBuffer(length: kvBytes, options: .storageModeShared)
        
        if let kBuf = kCacheBuffer { memset(kBuf.contents(), 0, kvBytes) }
        if let vBuf = vCacheBuffer { memset(vBuf.contents(), 0, kvBytes) }
        
        let linLayers = actualLayers
        let linStateBytes = max(linLayers, 40) * 32 * 128 * 128 * MemoryLayout<Float>.stride
        self.linearStateBuffer = device.makeBuffer(length: linStateBytes, options: .storageModeShared)
        if let sBuf = linearStateBuffer { memset(sBuf.contents(), 0, linStateBytes) }

        let convBytes = max(linLayers, 40) * 8192 * 4 * MemoryLayout<Float>.stride
        self.convStateBuffer = device.makeBuffer(length: convBytes, options: .storageModeShared)
        if let cBuf = convStateBuffer { memset(cBuf.contents(), 0, convBytes) }
    }
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
    case lowMemory8GB = "8 GB"
    case balanced16GB = "16 GB"
    case unrestricted = "Unrestricted"

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
    @State private var maxNewTokens: Int = 256
    @State private var isGeneratingText: Bool = false
    @State private var generatedStreamText: String = ""
    @State private var thinkingText: String = ""
    @State private var responseText: String = ""
    @State private var isThinking: Bool = false
    @State private var isThinkingExpanded: Bool = true
    @State private var generationSpeedTokPerSec: Double = 0.0
    @State private var generationTotalTokens: Int = 0
    @State private var generationElapsedMs: Double = 0.0
    @State private var generationStatusText: String? = nil
    @State private var generationTask: Task<Void, Never>? = nil

    // Working Set & Dynamic SSD Expert Paging State
    @State private var memoryExecutionMode: MemoryExecutionMode = .autoDetect
    @State private var memoryBudgetMode: MemoryBudgetMode = .balanced16GB
    @State private var modelConfig: ModelConfig? = nil
    @State private var detectedArchitecture: ModelArchitectureType = .hybridSsmMoe
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
                        HStack(spacing: 10) {
                            HStack(spacing: 5) {
                                Image(systemName: detectedArchitecture.icon)
                                    .foregroundColor(.purple)
                                Text(detectedArchitecture.shortName)
                                    .fontWeight(.bold)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color.purple.opacity(0.12))
                            .cornerRadius(6)

                            Text("•")
                                .foregroundColor(.secondary)
                            Text("\(summary.layerCount) Layers")
                                .fontWeight(.semibold)
                            if summary.maxExpertId > 0 {
                                Text("•")
                                    .foregroundColor(.secondary)
                                Text("\(summary.maxExpertId) Experts/Layer")
                                    .foregroundColor(.purple)
                                    .fontWeight(.semibold)
                            }
                            Text("•")
                                .foregroundColor(.secondary)
                            Text(metalStatus)
                                .foregroundColor(metalStatus.contains("✅") ? .green : .secondary)
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
                        reasoningAndResponseSection
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
                memoryControllerCard
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
    
    // MARK: - Modular UI Subcomponents
    private var shouldShowBudgetPicker: Bool {
        if memoryExecutionMode == .dynamicStreaming { return true }
        if memoryExecutionMode == .autoDetect {
            let footprint = self.summary?.sizeGb ?? 18.0
            return memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: footprint) == .dynamicStreaming
        }
        return false
    }

    @ViewBuilder
    private var reasoningAndResponseSection: some View {
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
                VStack(alignment: .leading, spacing: 8) {
                    // Collapsible Reasoning Process Block
                    if !thinkingText.isEmpty {
                        DisclosureGroup(
                            isExpanded: $isThinkingExpanded,
                            content: {
                                Text(thinkingText)
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundColor(.secondary)
                                    .textSelection(.enabled)
                                    .padding(8)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Color.secondary.opacity(0.08))
                                    .cornerRadius(6)
                            },
                            label: {
                                HStack(spacing: 6) {
                                    Image(systemName: isThinking ? "brain.head.profile" : "brain")
                                        .foregroundColor(.purple)
                                    Text(isThinking ? "Reasoning in progress..." : "Thought Process")
                                        .font(.caption)
                                        .fontWeight(.medium)
                                        .foregroundColor(.purple)
                                    if isThinking {
                                        ProgressView()
                                            .scaleEffect(0.5)
                                    }
                                }
                            }
                        )
                    }

                    // Main Text Response
                    if !responseText.isEmpty || (!isThinking && !generatedStreamText.isEmpty) || isGeneratingText {
                        HStack(alignment: .top, spacing: 0) {
                            let displayText = !responseText.isEmpty ? responseText : (thinkingText.isEmpty ? generatedStreamText : "")
                            Text(displayText)
                                .font(.system(.body, design: .default))
                                .textSelection(.enabled)

                            if isGeneratingText && !isThinking {
                                Text("▊")
                                    .foregroundColor(.purple)
                                    .opacity(0.8)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(10)
            }
            .frame(minHeight: 100, maxHeight: 300)
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

    @ViewBuilder
    private var memoryControllerCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header: Title, Status Message & Action Buttons
            HStack(spacing: 12) {
                Label("Dynamic SSD Expert Paging & Memory Controller", systemImage: "memorychip")
                    .font(.headline)
                    .foregroundColor(.indigo)

                Spacer()

                if let msg = pagingStatusMessage {
                    Text(msg)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                HStack(spacing: 8) {
                    Button(action: flushExpertCache) {
                        Label("Flush Cache", systemImage: "trash")
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

            // Controls Section: Execution Mode & Budget (if applicable)
            HStack(alignment: .top, spacing: 20) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Execution Mode")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundColor(.secondary)

                    Picker("Memory Mode", selection: $memoryExecutionMode) {
                        ForEach(MemoryExecutionMode.allCases) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(width: 320)
                    .onChange(of: memoryExecutionMode) { newMode in
                        applyMemoryExecutionMode(newMode)
                    }
                }

                if shouldShowBudgetPicker {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Working-Set Budget")
                            .font(.caption)
                            .fontWeight(.semibold)
                            .foregroundColor(.secondary)

                        Picker("Budget", selection: $memoryBudgetMode) {
                            ForEach(MemoryBudgetMode.allCases) { mode in
                                Text(mode.rawValue).tag(mode)
                            }
                        }
                        .pickerStyle(.segmented)
                        .frame(width: 240)
                        .onChange(of: memoryBudgetMode) { newMode in
                            applyMemoryBudget(newMode)
                        }
                    }
                }

                Spacer()
            }

            // Mode Description Callout
            HStack(alignment: .center, spacing: 6) {
                Image(systemName: memoryExecutionMode.icon)
                    .font(.caption)
                    .foregroundColor(.indigo)
                Text(memoryExecutionMode.description)
                    .font(.caption2)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.indigo.opacity(0.05))
            .cornerRadius(4)

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
                !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
                ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
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
        return InferenceEngine.shared.buildCachedLayers(summary: summary, config: modelConfig, targetLayerCount: targetLayerCount)
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

        let isNanbeige = (modelConfig?.modelType?.lowercased().contains("nanbeige") == true) ||
                         (modelConfig?.architectures?.contains(where: { $0.lowercased().contains("nanbeige") }) == true) ||
                         (summary.layerCount == 22 && summary.maxExpertId == 0) ||
                         (summary.tensors.contains(where: { $0.name.contains("dense_gate_up_proj") })) ||
                         (summary.tensors.contains(where: { $0.name.hasPrefix("model.layers.0.mlp.gate_proj") }) && summary.layerCount == 22)

        let formattedPrompt: String
        if prompt.contains("<|im_start|>") {
            if isNanbeige && !prompt.contains("<|im_start|>system") {
                formattedPrompt = "<|im_start|>system\n你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。<|im_end|>\n" + prompt
            } else {
                formattedPrompt = prompt
            }
        } else if isNanbeige {
            formattedPrompt = "<|im_start|>system\n你是南北阁，一款由BOSS直聘自主研发并训练的专业大语言模型。<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
        } else {
            formattedPrompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n<think>\n"
        }

        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try tokenizer.encode(text: formattedPrompt)
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

        // Find Embedding Weight & Affine Scales/Biases
        guard let embedWeight = summary.tensors.first(where: {
            !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }), let embedShardBuffer = shardBuffers[embedWeight.shardIndex] else {
            let err = "❌ Embedding weight tensor not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let embedScale = summary.tensors.first(where: {
            !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            ($0.name.contains("scale") || $0.name.contains("scales"))
        })
        let embedBias = summary.tensors.first(where: {
            !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            ($0.name.contains("bias") || $0.name.contains("biases"))
        })

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

        // Find LM Head Weight & Affine Scales/Biases
        guard let lmHeadTensor = summary.tensors.first(where: {
            ($0.name == "lm_head.weight" ||
             $0.name == "language_model.lm_head.weight" ||
             $0.name == "model.lm_head.weight" ||
             $0.name == "lm_head") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }), let lmHeadShardBuffer = shardBuffers[lmHeadTensor.shardIndex] else {
            let err = "❌ LM Head weight not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let lmHeadScale = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("scale") || $0.name.contains("scales"))
        })
        let lmHeadBias = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("bias") || $0.name.contains("biases"))
        })

        // Compile Pipelines with correct kernel names
        guard let embedBF16Function = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16"),
              let routerFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvBF16Function = defaultLibrary.makeFunction(name: "bf16_gemv"),
              let addFunction = defaultLibrary.makeFunction(name: "vector_add_f32"),
              let clearFunction = defaultLibrary.makeFunction(name: "clear_vector_f32") else {
            let err = "❌ Failed to load core Metal compute shaders."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let embedPipeline: MTLComputePipelineState
        let embedQ4Pipeline: MTLComputePipelineState?
        let embedMXFP8Pipeline: MTLComputePipelineState?
        let routerPipeline: MTLComputePipelineState
        let routerQ4Pipeline: MTLComputePipelineState?
        let routerQ8Pipeline: MTLComputePipelineState?
        let rmsnormPipeline: MTLComputePipelineState
        let rmsnormF16Pipeline: MTLComputePipelineState?
        let gemvBF16Pipeline: MTLComputePipelineState
        let bf16GemvSimdPipeline: MTLComputePipelineState?
        let fp8GemvPipeline: MTLComputePipelineState?
        let mxfp8GemvPipeline: MTLComputePipelineState?
        let q4GemvPipeline: MTLComputePipelineState?
        let q8GemvPipeline: MTLComputePipelineState?
        let addPipeline: MTLComputePipelineState
        let clearPipeline: MTLComputePipelineState
        let fp8GateUpPipeline: MTLComputePipelineState?
        let fp8DownPipeline: MTLComputePipelineState?
        let fp8GateUpSimdPipeline: MTLComputePipelineState?
        let fp8DownSimdPipeline: MTLComputePipelineState?
        let mxfp8GateUpPipeline: MTLComputePipelineState?
        let mxfp8DownPipeline: MTLComputePipelineState?
        let bf16GateUpPipeline: MTLComputePipelineState?
        let bf16DownPipeline: MTLComputePipelineState?
        let bf16GateUpSimdPipeline: MTLComputePipelineState?
        let bf16DownSimdPipeline: MTLComputePipelineState?
        let q4GateUpPipeline: MTLComputePipelineState?
        let q4DownPipeline: MTLComputePipelineState?
        let headRmsnormPipeline: MTLComputePipelineState?
        let headRmsnormF16Pipeline: MTLComputePipelineState?
        let ropePipeline: MTLComputePipelineState?
        let storeKvCachePipeline: MTLComputePipelineState?
        let gqaDecodePipeline: MTLComputePipelineState?
        let gqaStandardPipeline: MTLComputePipelineState?
        let causalConv1dPipeline: MTLComputePipelineState?
        let l2NormQkPipeline: MTLComputePipelineState?
        let linearAttnStepPipeline: MTLComputePipelineState?

        do {
            embedPipeline = try device.makeComputePipelineState(function: embedBF16Function)
            routerPipeline = try device.makeComputePipelineState(function: routerFunction)
            rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)

            if let rmsF16Func = defaultLibrary.makeFunction(name: "rmsnorm_f16") {
                rmsnormF16Pipeline = try device.makeComputePipelineState(function: rmsF16Func)
            } else { rmsnormF16Pipeline = nil }
            gemvBF16Pipeline = try device.makeComputePipelineState(function: gemvBF16Function)
            addPipeline = try device.makeComputePipelineState(function: addFunction)
            clearPipeline = try device.makeComputePipelineState(function: clearFunction)

            if let rQ4Func = defaultLibrary.makeFunction(name: "moe_router_topk_q4") {
                routerQ4Pipeline = try device.makeComputePipelineState(function: rQ4Func)
            } else { routerQ4Pipeline = nil }

            if let rQ8Func = defaultLibrary.makeFunction(name: "moe_router_topk_q8") {
                routerQ8Pipeline = try device.makeComputePipelineState(function: rQ8Func)
            } else { routerQ8Pipeline = nil }

            if let embedQ4Func = defaultLibrary.makeFunction(name: "lookup_embeddings_q4") {
                embedQ4Pipeline = try device.makeComputePipelineState(function: embedQ4Func)
            } else { embedQ4Pipeline = nil }

            if let embedMXFP8Func = defaultLibrary.makeFunction(name: "lookup_embeddings_mxfp8") {
                embedMXFP8Pipeline = try device.makeComputePipelineState(function: embedMXFP8Func)
            } else { embedMXFP8Pipeline = nil }

            if let bGemvSimd = defaultLibrary.makeFunction(name: "bf16_gemv_simd") {
                bf16GemvSimdPipeline = try device.makeComputePipelineState(function: bGemvSimd)
            } else { bf16GemvSimdPipeline = nil }

            if let fp8GemvFunc = defaultLibrary.makeFunction(name: "fp8_gemv") {
                fp8GemvPipeline = try device.makeComputePipelineState(function: fp8GemvFunc)
            } else { fp8GemvPipeline = nil }

            if let mxfp8GemvFunc = defaultLibrary.makeFunction(name: "mxfp8_gemv") {
                mxfp8GemvPipeline = try device.makeComputePipelineState(function: mxfp8GemvFunc)
            } else { mxfp8GemvPipeline = nil }

            if let q4GemvFunc = defaultLibrary.makeFunction(name: "q4_gemv") {
                q4GemvPipeline = try device.makeComputePipelineState(function: q4GemvFunc)
            } else { q4GemvPipeline = nil }

            if let q8GemvFunc = defaultLibrary.makeFunction(name: "q8_gemv") {
                q8GemvPipeline = try device.makeComputePipelineState(function: q8GemvFunc)
            } else { q8GemvPipeline = nil }

            if let fp8GateUpFunc = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up") {
                fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpFunc)
            } else { fp8GateUpPipeline = nil }

            if let fp8DownFunc = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate") {
                fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownFunc)
            } else { fp8DownPipeline = nil }

            if let fp8GateUpSimd = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up_simd") {
                fp8GateUpSimdPipeline = try device.makeComputePipelineState(function: fp8GateUpSimd)
            } else { fp8GateUpSimdPipeline = nil }

            if let fp8DownSimd = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate_simd") {
                fp8DownSimdPipeline = try device.makeComputePipelineState(function: fp8DownSimd)
            } else { fp8DownSimdPipeline = nil }

            if let mxfp8GateFunc = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up") {
                mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateFunc)
            } else { mxfp8GateUpPipeline = nil }

            if let mxfp8DownFunc = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate") {
                mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownFunc)
            } else { mxfp8DownPipeline = nil }

            if let bGateUp = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up") {
                bf16GateUpPipeline = try device.makeComputePipelineState(function: bGateUp)
            } else { bf16GateUpPipeline = nil }

            if let bDown = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") {
                bf16DownPipeline = try device.makeComputePipelineState(function: bDown)
            } else { bf16DownPipeline = nil }

            if let bGateUpSimd = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up_simd") {
                bf16GateUpSimdPipeline = try device.makeComputePipelineState(function: bGateUpSimd)
            } else { bf16GateUpSimdPipeline = nil }

            if let bDownSimd = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate_simd") {
                bf16DownSimdPipeline = try device.makeComputePipelineState(function: bDownSimd)
            } else { bf16DownSimdPipeline = nil }

            if let q4GateUpFunc = defaultLibrary.makeFunction(name: "q4_swiglu_gate_up") {
                q4GateUpPipeline = try device.makeComputePipelineState(function: q4GateUpFunc)
            } else { q4GateUpPipeline = nil }

            if let q4DownFunc = defaultLibrary.makeFunction(name: "q4_down_proj_accumulate") {
                q4DownPipeline = try device.makeComputePipelineState(function: q4DownFunc)
            } else { q4DownPipeline = nil }

            if let hNormFunc = defaultLibrary.makeFunction(name: "per_head_rmsnorm_bf16") {
                headRmsnormPipeline = try device.makeComputePipelineState(function: hNormFunc)
            } else { headRmsnormPipeline = nil }

            if let hNormF16Func = defaultLibrary.makeFunction(name: "per_head_rmsnorm_f16") {
                headRmsnormF16Pipeline = try device.makeComputePipelineState(function: hNormF16Func)
            } else { headRmsnormF16Pipeline = nil }

            if let ropeFunc = defaultLibrary.makeFunction(name: "apply_rope_qwen") {
                ropePipeline = try device.makeComputePipelineState(function: ropeFunc)
            } else { ropePipeline = nil }

            if let storeKvFunc = defaultLibrary.makeFunction(name: "store_kv_cache") {
                storeKvCachePipeline = try device.makeComputePipelineState(function: storeKvFunc)
            } else { storeKvCachePipeline = nil }

            if let gqaFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused") {
                gqaDecodePipeline = try device.makeComputePipelineState(function: gqaFunc)
            } else { gqaDecodePipeline = nil }

            if let gqaStdFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard") {
                gqaStandardPipeline = try device.makeComputePipelineState(function: gqaStdFunc)
            } else { gqaStandardPipeline = nil }

            if let convFunc = defaultLibrary.makeFunction(name: "causal_conv1d_silu") {
                causalConv1dPipeline = try device.makeComputePipelineState(function: convFunc)
            } else { causalConv1dPipeline = nil }

            if let l2Func = defaultLibrary.makeFunction(name: "l2_norm_qk") {
                l2NormQkPipeline = try device.makeComputePipelineState(function: l2Func)
            } else { l2NormQkPipeline = nil }

            if let linStepFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_step") {
                linearAttnStepPipeline = try device.makeComputePipelineState(function: linStepFunc)
            } else { linearAttnStepPipeline = nil }
        } catch {
            let err = "❌ Failed to create compute pipelines: \(error.localizedDescription)"
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Model Hyperparameters & Dynamic Sizing
        let hiddenDim: UInt32 = UInt32(modelConfig?.effectiveHiddenSize ?? (activeHiddenDim > 0 ? activeHiddenDim : 2048))
        let numHeads: UInt32 = UInt32(modelConfig?.effectiveNumAttentionHeads ?? 16)
        let numKvHeads: UInt32 = UInt32(modelConfig?.effectiveNumKeyValueHeads ?? 2)
        let headDim: UInt32 = UInt32(modelConfig?.effectiveHeadDim ?? 128)
        let rotaryDim: UInt32 = UInt32(modelConfig?.effectiveRotaryDim ?? 128)
        let thetaVal: Float = modelConfig?.effectiveRopeTheta ?? 10000000.0
        let totalLoops: Int = max(1, modelConfig?.effectiveNumLoops ?? 1)
        let eosTokenId: UInt32 = UInt32(modelConfig?.effectiveEosTokenId ?? 248044)

        let cachedLayers = buildCachedLayers(summary: summary)
        let actualLayers = cachedLayers.count
        guard actualLayers > 0 else {
            let err = "❌ No layer tensors found in model."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let arch = modelConfig?.resolveArchitectureType(summary: summary) ?? (summary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)
        let isHybridArch = (arch == .hybridSsmMoe)

        let qOutDim: UInt32 = isHybridArch ? (numHeads * headDim * 2) : (numHeads * headDim)
        let kvOutDim: UInt32 = numKvHeads * headDim
        let attnCtxDim: UInt32 = numHeads * headDim
        let kvStride: UInt32 = numKvHeads * headDim

        let numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : UInt32(modelConfig?.effectiveNumExperts ?? 0)
        let embedOffset = embedWeight.offsetStart
        let normOffset = normTensor.offsetStart
        let lmHeadOffset = lmHeadTensor.offsetStart
        let eps: Float = modelConfig?.effectiveRmsNormEps ?? 1e-6

        var vocabSize: UInt32 = 248320
        let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
        let shapeParts = cleanShape.split(separator: ",")
        if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
            vocabSize = parsed
        } else if let cfgVocab = modelConfig?.effectiveVocabSize {
            vocabSize = UInt32(cfgVocab)
        }

        let maxInterDim = cachedLayers.map { $0.intermediateDim }.max() ?? 512

        // Allocate Shared Scratch Buffers (Reused across all generation steps)
        guard let singleTokenBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let hCurrBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hNextBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let qGateBuffer = device.makeBuffer(length: max(Int(qOutDim), 8192) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let kVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 512) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let vVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 512) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zGateBuffer = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aVectorBuffer = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bVectorBuffer = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnCtxBuffer = device.makeBuffer(length: max(Int(attnCtxDim), 4096) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnOutBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMidBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm2Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let routerIndicesBuffer = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let routerWeightsBuffer = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuffer = device.makeBuffer(length: max(Int(maxInterDim), 512) * MemoryLayout<Float>.stride, options: .storageModeShared),
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

        KVCacheManager.shared.reset(device: device, config: modelConfig, actualLayers: actualLayers, maxSeqLen: 2048)

        isGeneratingText = true
        generatedStreamText = ""
        thinkingText = ""
        responseText = ""
        isThinking = false
        isThinkingExpanded = true
        generationTotalTokens = 0
        generationSpeedTokPerSec = 0.0
        generationElapsedMs = 0.0
        generationStatusText = "⚡ Initializing Autoregressive Generation..."

        generationTask = Task.detached(priority: .userInitiated) {
            var contextTokens = promptTokenIds
            let startTime = CFAbsoluteTimeGetCurrent()
            var tokensGenerated = 0
            var currentStep: UInt32 = 0

            // Helper for Linear/GEMV Projections (handles Q4, FP8, and BF16 SIMD)
            func dispatchLinear(
                enc: MTLComputeCommandEncoder,
                weight: TensorMetadata?,
                scale: TensorMetadata?,
                bias: TensorMetadata?,
                inBuf: MTLBuffer,
                outBuf: MTLBuffer,
                inDim: UInt32,
                outDim: UInt32,
                groupSize: UInt32 = 64
            ) {
                guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
                var wOff = w.offsetStart
                var inD = inDim
                var outD = outDim
                var grp = groupSize

                let hasScale = (scale != nil)
                let hasBias = (bias != nil)
                let isMXFP8 = hasScale && (scale!.dtype.contains("U8") || scale!.dtype.contains("UINT8") || (!hasBias && (w.dtype.contains("U32") || w.dtype.contains("U8") || w.dtype.contains("FP8")) && (scale!.offsetEnd - scale!.offsetStart) >= UInt64((inDim / 32) * outDim)))
                let isQuantizedAffine = (hasBias || w.dtype.contains("Q4")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQuantizedAffine && !w.dtype.contains("BF16") && !w.dtype.contains("F16") && !w.dtype.contains("FLOAT")

                if isMXFP8, let mxfp8Pipe = mxfp8GemvPipeline, let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil {
                    var sOff = scale!.offsetStart
                    enc.setComputePipelineState(mxfp8Pipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBuffer(sRaw, offset: 0, index: 3)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                    enc.dispatchThreads(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                } else if isQuantizedAffine {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil
                    let bRaw = (bias != nil) ? buffers[bias!.shardIndex] : nil
                    guard let sBuffer = sRaw, let bBuffer = bRaw else { return }
                    var sOff = scale!.offsetStart
                    var bOff = bias!.offsetStart

                    let is8Bit = (w.offsetEnd - w.offsetStart) >= UInt64(outDim) * UInt64(inDim)
                    if is8Bit, let q8Pipe = q8GemvPipeline {
                        enc.setComputePipelineState(q8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: 0, index: 3)
                        enc.setBuffer(outBuf, offset: 0, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let q4Pipe = q4GemvPipeline {
                        enc.setComputePipelineState(q4Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: 0, index: 3)
                        enc.setBuffer(outBuf, offset: 0, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    }
                } else if isFP8, let fp8Pipe = fp8GemvPipeline {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : wRaw
                    var sOff = scale?.offsetStart ?? 0
                    enc.setComputePipelineState(fp8Pipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBuffer(sRaw, offset: 0, index: 3)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                    enc.dispatchThreads(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), fp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                } else if let bSimdPipe = bf16GemvSimdPipeline {
                    enc.setComputePipelineState(bSimdPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(gemvBF16Pipeline)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreads(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), gemvBF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
            }

            // Helper for MoE Expert MLP (SwiGLU Gate/Up + Down Proj Accumulate)
            func dispatchExpertMlp(
                enc: MTLComputeCommandEncoder,
                gateW: TensorMetadata,
                gateS: TensorMetadata?,
                gateB: TensorMetadata?,
                upW: TensorMetadata,
                upS: TensorMetadata?,
                upB: TensorMetadata?,
                downW: TensorMetadata,
                downS: TensorMetadata?,
                downB: TensorMetadata?,
                inBuf: MTLBuffer,
                interBuf: MTLBuffer,
                accumBuf: MTLBuffer,
                inDim: UInt32,
                interDim: UInt32,
                routingWeight: Float,
                groupSize: UInt32 = 64
            ) {
                guard let gRaw = buffers[gateW.shardIndex],
                      let uRaw = buffers[upW.shardIndex],
                      let dRaw = buffers[downW.shardIndex] else { return }

                var gWOff = gateW.offsetStart
                var uWOff = upW.offsetStart
                var dWOff = downW.offsetStart
                var hDimVal = inDim
                var interDimVal = interDim
                var p_k = routingWeight
                var grp = groupSize

                let hasGateScale = (gateS != nil)
                let hasGateBias = (gateB != nil)
                let isMXFP8 = hasGateScale && (gateS!.dtype.contains("U8") || gateS!.dtype.contains("UINT8") || (!hasGateBias && (gateW.dtype.contains("U32") || gateW.dtype.contains("U8") || gateW.dtype.contains("FP8")) && (gateS!.offsetEnd - gateS!.offsetStart) >= UInt64((inDim / 32) * interDim)))
                let isQ4 = (hasGateBias || gateW.dtype.contains("Q4")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQ4 && !gateW.dtype.contains("BF16") && !gateW.dtype.contains("F16") && !gateW.dtype.contains("FLOAT")

                if isMXFP8, let mxfp8GatePipe = mxfp8GateUpPipeline, let mxfp8DownPipe = mxfp8DownPipeline {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    enc.setComputePipelineState(mxfp8GatePipe)
                    enc.setBuffer(gRaw, offset: 0, index: 0)
                    enc.setBuffer(uRaw, offset: 0, index: 1)
                    enc.setBuffer(inBuf, offset: 0, index: 2)
                    enc.setBuffer(interBuf, offset: 0, index: 3)
                    enc.setBuffer(gSRaw, offset: 0, index: 4)
                    enc.setBuffer(uSRaw, offset: 0, index: 5)
                    enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                    enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                    enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                    enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                    enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                    enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                    enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    enc.setComputePipelineState(mxfp8DownPipe)
                    enc.setBuffer(dRaw, offset: 0, index: 0)
                    enc.setBuffer(interBuf, offset: 0, index: 1)
                    enc.setBuffer(accumBuf, offset: 0, index: 2)
                    enc.setBuffer(dSRaw, offset: 0, index: 3)
                    enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                    enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                    enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                    enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                    enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                } else if isQ4, let q4SwigluPipe = q4GateUpPipeline, let q4DownPipe = q4DownPipeline {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let gB = gateB, let gBRaw = buffers[gB.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let uB = upB, let uBRaw = buffers[uB.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex],
                          let dB = downB, let dBRaw = buffers[dB.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var gBOff = gB.offsetStart
                    var uSOff = uS.offsetStart
                    var uBOff = uB.offsetStart
                    var dSOff = dS.offsetStart
                    var dBOff = dB.offsetStart

                    // Step 1: Q4 SwiGLU Gate & Up Proj
                    enc.setComputePipelineState(q4SwigluPipe)
                    enc.setBuffer(gRaw, offset: 0, index: 0)
                    enc.setBuffer(gSRaw, offset: 0, index: 1)
                    enc.setBuffer(gBRaw, offset: 0, index: 2)
                    enc.setBuffer(uRaw, offset: 0, index: 3)
                    enc.setBuffer(uSRaw, offset: 0, index: 4)
                    enc.setBuffer(uBRaw, offset: 0, index: 5)
                    enc.setBuffer(inBuf, offset: 0, index: 6)
                    enc.setBuffer(interBuf, offset: 0, index: 7)
                    enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                    enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                    enc.setBytes(&gBOff, length: MemoryLayout<UInt64>.stride, index: 10)
                    enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 11)
                    enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 12)
                    enc.setBytes(&uBOff, length: MemoryLayout<UInt64>.stride, index: 13)
                    enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 14)
                    enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 15)
                    enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 16)
                    enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    enc.memoryBarrier(scope: .buffers)

                    // Step 2: Q4 Down Proj Accumulate
                    enc.setComputePipelineState(q4DownPipe)
                    enc.setBuffer(dRaw, offset: 0, index: 0)
                    enc.setBuffer(dSRaw, offset: 0, index: 1)
                    enc.setBuffer(dBRaw, offset: 0, index: 2)
                    enc.setBuffer(interBuf, offset: 0, index: 3)
                    enc.setBuffer(accumBuf, offset: 0, index: 4)
                    enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                    enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 6)
                    enc.setBytes(&dBOff, length: MemoryLayout<UInt64>.stride, index: 7)
                    enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 8)
                    enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                    enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                    enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 11)
                    enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    enc.memoryBarrier(scope: .buffers)
                } else if isFP8 {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    if let fp8GateSimd = fp8GateUpSimdPipeline, let fp8DownSimd = fp8DownSimdPipeline {
                        enc.setComputePipelineState(fp8GateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: 0, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                        enc.setComputePipelineState(fp8DownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: 0, index: 1)
                        enc.setBuffer(accumBuf, offset: 0, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let fp8GatePipe = fp8GateUpPipeline, let fp8DownPipe = fp8DownPipeline {
                        enc.setComputePipelineState(fp8GatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: 0, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDim), fp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                        enc.setComputePipelineState(fp8DownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: 0, index: 1)
                        enc.setBuffer(accumBuf, offset: 0, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(inDim), fp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                } else {
                    // BF16
                    if let bGateSimd = bf16GateUpSimdPipeline, let bDownSimd = bf16DownSimdPipeline {
                        enc.setComputePipelineState(bGateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: 0, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

                        enc.setComputePipelineState(bDownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: 0, index: 1)
                        enc.setBuffer(accumBuf, offset: 0, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let bGatePipe = bf16GateUpPipeline, let bDownPipe = bf16DownPipeline {
                        enc.setComputePipelineState(bGatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: 0, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDim), bGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                        enc.setComputePipelineState(bDownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: 0, index: 1)
                        enc.setBuffer(accumBuf, offset: 0, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(inDim), bDownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }
            }

            // Helper for Single Token Forward Pass
            func runTokenForward(tokenId: UInt32, step: UInt32, computeLogits: Bool) -> Bool {
                let singleTokenPtr = singleTokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
                singleTokenPtr[0] = tokenId
                var hDim = hiddenDim
                let isStandardGqa = !isHybridArch

                // 2. Multi-Layer Transformer Backbone (0..<actualLayers, looped totalLoops times)
                var currentH = hCurrBuffer
                var nextH = hNextBuffer

                guard var activeCmd = commandQueue.makeCommandBuffer() else { return false }

                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]

                        if l + 1 < actualLayers {
                            WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: [0, 1, 2, 3, 4, 5, 6, 7], shardBuffers: buffers)
                        }

                        // --- Phase A: Attention & Routing Sub-Block ---
                        guard let layerEnc1 = activeCmd.makeComputeCommandEncoder() else { return false }

                        // Step 0: Embed Token on Loop 0, Layer 0
                        if loopIdx == 0 && l == 0 {
                            let hasEmbedScale = (embedScale != nil)
                            let hasEmbedBias = (embedBias != nil)
                            let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                            let isEmbedQ4 = (hasEmbedBias || embedWeight.dtype.contains("Q4")) && !isEmbedMXFP8

                            if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                               let embedScaleRaw = buffers[embedScale!.shardIndex] {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var hDimVal = hiddenDim
                                layerEnc1.setComputePipelineState(embedMXFP8Pipe)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 3)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if isEmbedQ4, let embedQ4Pipe = embedQ4Pipeline,
                               let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                               let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var bOffset = embedBias!.offsetStart
                                var tok = tokenId
                                var grpSize: UInt32 = 64
                                layerEnc1.setComputePipelineState(embedQ4Pipe)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 3)
                                layerEnc1.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc1.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else {
                                var wOffset = embedOffset
                                var tokCount: UInt32 = 1
                                layerEnc1.setComputePipelineState(embedPipeline)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        }

                        // Step 1: Pre-Attention RMSNorm (currentH -> xNorm1)
                        if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                            var gammaOff = norm1.offsetStart
                            var epsVal = eps
                            let isNorm1F16 = (norm1.dtype.contains("F16") || norm1.dtype.contains("HALF") || norm1.dtype.contains("FLOAT16")) && !norm1.dtype.contains("BF16") && !norm1.dtype.contains("BFLOAT")
                            let norm1Pipe = (isNorm1F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            layerEnc1.setComputePipelineState(norm1Pipe)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm1Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                        }

                        // Step 2: Attention Computation
                        if layer.attentionType == .fullAttention {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            let currentQDim = isHybridArch ? (numHeads * headDim * 2) : qOutDim
                            let currentKvDim = kvStride

                            dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: currentQDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)

                            // Q-Norm (if present)
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex] {
                                let isQNormF16 = (qNorm.dtype.contains("F16") || qNorm.dtype.contains("HALF") || qNorm.dtype.contains("FLOAT16")) && !qNorm.dtype.contains("BF16") && !qNorm.dtype.contains("BFLOAT")
                                let qHeadNormPipe = (isQNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                if let headNormPipe = qHeadNormPipe {
                                    var qNormOff = qNorm.offsetStart
                                    var nQ = numHeads
                                    var hD = headDim
                                    var hStride: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), headNormPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                            }

                            // K-Norm (if present)
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex] {
                                let isKNormF16 = (kNorm.dtype.contains("F16") || kNorm.dtype.contains("HALF") || kNorm.dtype.contains("FLOAT16")) && !kNorm.dtype.contains("BF16") && !kNorm.dtype.contains("BFLOAT")
                                let kHeadNormPipe = (isKNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                if let headNormPipe = kHeadNormPipe {
                                    var kNormOff = kNorm.offsetStart
                                    var nK = numKvHeads
                                    var hD = headDim
                                    var hStride = headDim
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), headNormPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                            }

                            // RoPE on Q and K
                            if let ropePipe = ropePipeline {
                                var pos = step
                                var nQ = numHeads
                                var nK = numKvHeads
                                var hD = headDim
                                var rD = rotaryDim
                                var qStr: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                var kStr = headDim
                                var theta = thetaVal

                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }

                            // Store KV-Cache and Fused GQA Decode
                            if let storePipe = storeKvCachePipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let layerByteOffset = slot * 2048 * Int(kvStride) * MemoryLayout<Float>.stride
                                var pos = step
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc1.setComputePipelineState(storePipe)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                if isStandardGqa, let gqaStdPipe = gqaStandardPipeline {
                                    var seqLen = step + 1
                                    var nQ = numHeads
                                    layerEnc1.setComputePipelineState(gqaStdPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                } else if let gqaPipe = gqaDecodePipeline {
                                    var seqLen = step + 1
                                    var nQ = numHeads
                                    layerEnc1.setComputePipelineState(gqaPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                    layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                    layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                            }

                            // Attention Output Projection (o_proj)
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: attnCtxDim, outDim: hiddenDim)
                        } else {
                            // Linear Attention (GatedDeltaNet Recurrent State)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: 8192)

                            if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                               let convPipe = causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                let convStateByteOffset = layer.linAttnIndex * 8192 * 4 * MemoryLayout<Float>.stride
                                var cOff = conv1d.offsetStart
                                var numChannels: UInt32 = 8192
                                layerEnc1.setComputePipelineState(convPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(convRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(convState, offset: convStateByteOffset, index: 2)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 3)
                                layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&numChannels, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: 8192, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }

                            if let l2Pipe = l2NormQkPipeline {
                                var numHeads: UInt32 = 16
                                var headDim: UInt32 = 128
                                layerEnc1.setComputePipelineState(l2Pipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: 16, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buffer, outBuf: zGateBuffer, inDim: hiddenDim, outDim: 4096)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buffer, outBuf: aVectorBuffer, inDim: hiddenDim, outDim: 32)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: 32)

                            if let linPipe = linearAttnStepPipeline,
                               let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                let linIdx = layer.linAttnIndex
                                let stateByteOffset = linIdx * (32 * 128 * 128) * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var linNormOff = linNorm.offsetStart
                                var numValHeads: UInt32 = 32
                                var numKeyHeads: UInt32 = 16
                                var headDim: UInt32 = 128
                                var epsVal = eps
                                
                                layerEnc1.setComputePipelineState(linPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(aVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(aLogRaw, offset: 0, index: 4)
                                layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                layerEnc1.setBuffer(linNormRaw, offset: 0, index: 6)
                                layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 8)
                                layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                layerEnc1.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                layerEnc1.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 12)
                                layerEnc1.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 14)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 15)
                                layerEnc1.dispatchThreads(MTLSize(width: 32, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: 4096, outDim: hiddenDim)
                        }

                        // Step 3: Residual Connection 1 (hMid = currentH + attnOut)
                        layerEnc1.setComputePipelineState(addPipeline)
                        layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                        layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                        layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 2)
                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                        layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                        // Step 4: Post-Attention RMSNorm (hMid -> xNorm2)
                        if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                            var gammaOff = norm2.offsetStart
                            var epsVal = eps
                            let isNorm2F16 = (norm2.dtype.contains("F16") || norm2.dtype.contains("HALF") || norm2.dtype.contains("FLOAT16")) && !norm2.dtype.contains("BF16") && !norm2.dtype.contains("BFLOAT")
                            let norm2Pipe = (isNorm2F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            layerEnc1.setComputePipelineState(norm2Pipe)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                        }

                        // Step 5: Dense Feed-Forward OR MoE
                        let intermediateDim = layer.intermediateDim

                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            // Dense SwiGLU Feed-Forward (Unified within layerEnc1)
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight {
                                let gateS = layer.denseGateScale
                                let gateB = layer.denseGateBias
                                let upS = layer.denseUpScale
                                let upB = layer.denseUpBias
                                let downS = layer.denseDownScale
                                let downB = layer.denseDownBias

                                dispatchExpertMlp(
                                    enc: layerEnc1,
                                    gateW: gateW,
                                    gateS: gateS,
                                    gateB: gateB,
                                    upW: upW,
                                    upS: upS,
                                    upB: upB,
                                    downW: downW,
                                    downS: downS,
                                    downB: downB,
                                    inBuf: xNorm2Buffer,
                                    interBuf: interBuffer,
                                    accumBuf: hMlpBuffer,
                                    inDim: hiddenDim,
                                    interDim: intermediateDim,
                                    routingWeight: 1.0
                                )
                            }

                            // Step 6: Residual Connection 2 (nextH = hMid + hMlp)
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(nextH, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            layerEnc1.endEncoding()
                        } else {
                            // MoE Model with dynamic routing
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = 8
                                var grp: UInt32 = 64

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if routerTensor.shapeDisplay.contains("512"), let rQ8 = routerQ8Pipeline {
                                        layerEnc1.setComputePipelineState(rQ8)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    } else if let rQ4 = routerQ4Pipeline {
                                        layerEnc1.setComputePipelineState(rQ4)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                } else {
                                    layerEnc1.setComputePipelineState(routerPipeline)
                                    layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                    layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                    layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                }
                            }

                            layerEnc1.endEncoding()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()

                            var activeExperts: [(id: Int, weight: Float)] = []
                            if layer.routerTensor != nil {
                                let indPtr = routerIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: 8)
                                let wPtr = routerWeightsBuffer.contents().bindMemory(to: Float.self, capacity: 8)
                                for i in 0..<8 {
                                    activeExperts.append((id: Int(indPtr[i]), weight: wPtr[i]))
                                }
                            } else {
                                activeExperts = (0..<8).map { (id: $0, weight: 1.0 / 8.0) }
                            }

                            var sharedW: Float = 1.0
                            if let sg = layer.sharedGateTensor, let sgRaw = buffers[sg.shardIndex] {
                                let xPtr = xNorm2Buffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
                                var sum: Float = 0.0
                                if let sgScale = layer.sharedGateTensorScale, let sgBias = layer.sharedGateTensorBias,
                                   let sgScaleRaw = buffers[sgScale.shardIndex], let sgBiasRaw = buffers[sgBias.shardIndex] {
                                    let sRaw = sgScaleRaw.contents().advanced(by: Int(sgScale.offsetStart))
                                    let bRaw = sgBiasRaw.contents().advanced(by: Int(sgBias.offsetStart))
                                    let wRaw = sgRaw.contents().advanced(by: Int(sg.offsetStart))
                                    let groupSize = 64
                                    let numGroups = Int(hiddenDim) / groupSize
                                    let is8Bit = (sg.offsetEnd - sg.offsetStart) >= UInt64(hiddenDim)
                                    if is8Bit {
                                        for g in 0..<numGroups {
                                            let s0 = UInt16(sRaw.load(fromByteOffset: g * 2, as: UInt8.self))
                                            let s1 = UInt16(sRaw.load(fromByteOffset: g * 2 + 1, as: UInt8.self))
                                            let sU16 = s0 | (s1 << 8)
                                            let b0 = UInt16(bRaw.load(fromByteOffset: g * 2, as: UInt8.self))
                                            let b1 = UInt16(bRaw.load(fromByteOffset: g * 2 + 1, as: UInt8.self))
                                            let bU16 = b0 | (b1 << 8)
                                            let s = Float(bitPattern: UInt32(sU16) << 16)
                                            let b = Float(bitPattern: UInt32(bU16) << 16)
                                            let colStart = g * groupSize
                                            var groupSum: Float = 0.0
                                            var groupXSum: Float = 0.0
                                            for c in 0..<groupSize {
                                                let col = colStart + c
                                                let x = xPtr[col]
                                                let w8 = Float(wRaw.load(fromByteOffset: col, as: UInt8.self))
                                                groupSum += w8 * x
                                                groupXSum += x
                                            }
                                            sum += s * groupSum + b * groupXSum
                                        }
                                    } else {
                                        for g in 0..<numGroups {
                                            let s0 = UInt16(sRaw.load(fromByteOffset: g * 2, as: UInt8.self))
                                            let s1 = UInt16(sRaw.load(fromByteOffset: g * 2 + 1, as: UInt8.self))
                                            let sU16 = s0 | (s1 << 8)
                                            let b0 = UInt16(bRaw.load(fromByteOffset: g * 2, as: UInt8.self))
                                            let b1 = UInt16(bRaw.load(fromByteOffset: g * 2 + 1, as: UInt8.self))
                                            let bU16 = b0 | (b1 << 8)
                                            let s = Float(bitPattern: UInt32(sU16) << 16)
                                            let b = Float(bitPattern: UInt32(bU16) << 16)
                                            let u32Start = (g * groupSize) / 8
                                            var groupSum: Float = 0.0
                                            var groupXSum: Float = 0.0
                                            for u in 0..<(groupSize / 8) {
                                                let byteIdx = (u32Start + u) * 4
                                                let u0 = UInt32(wRaw.load(fromByteOffset: byteIdx, as: UInt8.self))
                                                let u1 = UInt32(wRaw.load(fromByteOffset: byteIdx + 1, as: UInt8.self))
                                                let u2 = UInt32(wRaw.load(fromByteOffset: byteIdx + 2, as: UInt8.self))
                                                let u3 = UInt32(wRaw.load(fromByteOffset: byteIdx + 3, as: UInt8.self))
                                                let u32 = u0 | (u1 << 8) | (u2 << 16) | (u3 << 24)
                                                let baseCol = (u32Start + u) * 8
                                                for nib in 0..<8 {
                                                    let w4 = Float((u32 >> (nib * 4)) & 0x0F)
                                                    let x = xPtr[baseCol + nib]
                                                    groupSum += w4 * x
                                                    groupXSum += x
                                                }
                                            }
                                            sum += s * groupSum + b * groupXSum
                                        }
                                    }
                                } else {
                                    let sgRawPtr = sgRaw.contents().advanced(by: Int(sg.offsetStart))
                                    for d in 0..<Int(hiddenDim) {
                                        let w0 = UInt16(sgRawPtr.load(fromByteOffset: d * 2, as: UInt8.self))
                                        let w1 = UInt16(sgRawPtr.load(fromByteOffset: d * 2 + 1, as: UInt8.self))
                                        let u = UInt32(w0 | (w1 << 8)) << 16
                                        let w = Float(bitPattern: u)
                                        sum += w * xPtr[d]
                                    }
                                }
                                sharedW = 1.0 / (1.0 + exp(-sum))
                            }

                            let activeIds = activeExperts.map { $0.id }
                            WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: budgetMode, shardBuffers: buffers)

                            guard let moeCmd = commandQueue.makeCommandBuffer(),
                                  let layerEnc2 = moeCmd.makeComputeCommandEncoder() else { return false }

                            layerEnc2.setComputePipelineState(clearPipeline)
                            layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            for expert in activeExperts {
                                let expId = expert.id
                                let p_k = expert.weight
                                if p_k <= 0.00001 { continue }

                                if let gateW = layer.expertGateWeights[expId],
                                   let upW = layer.expertUpWeights[expId],
                                   let downW = layer.expertDownWeights[expId] {
                                    let gateS = layer.expertGateScales[expId]
                                    let gateB = layer.expertGateBiases[expId]
                                    let upS = layer.expertUpScales[expId]
                                    let upB = layer.expertUpBiases[expId]
                                    let downS = layer.expertDownScales[expId]
                                    let downB = layer.expertDownBiases[expId]

                                    dispatchExpertMlp(
                                        enc: layerEnc2,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: p_k
                                    )
                                }
                            }

                            if let gateW = layer.sharedGateWeight,
                               let upW = layer.sharedUpWeight,
                               let downW = layer.sharedDownWeight {
                                let gateS = layer.sharedGateScale
                                let gateB = layer.sharedGateBias
                                let upS = layer.sharedUpScale
                                let upB = layer.sharedUpBias
                                let downS = layer.sharedDownScale
                                let downB = layer.sharedDownBias

                                dispatchExpertMlp(
                                    enc: layerEnc2,
                                    gateW: gateW,
                                    gateS: gateS,
                                    gateB: gateB,
                                    upW: upW,
                                    upS: upS,
                                    upB: upB,
                                    downW: downW,
                                    downS: downS,
                                    downB: downB,
                                    inBuf: xNorm2Buffer,
                                    interBuf: interBuffer,
                                    accumBuf: hMlpBuffer,
                                    inDim: hiddenDim,
                                    interDim: intermediateDim,
                                    routingWeight: sharedW
                                )
                            }

                            layerEnc2.setComputePipelineState(addPipeline)
                            layerEnc2.setBuffer(hMidBuffer, offset: 0, index: 0)
                            layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                            layerEnc2.setBuffer(nextH, offset: 0, index: 2)
                            layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            layerEnc2.endEncoding()
                            moeCmd.commit()
                            moeCmd.waitUntilCompleted()

                            guard let nextCmd = commandQueue.makeCommandBuffer() else { return false }
                            activeCmd = nextCmd
                        }

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }

                    // Apply intermediate loop final norm if multiple loops exist
                    if loopIdx + 1 < totalLoops {
                        guard let loopNormEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                        var nOff = normOffset
                        var epsVal = eps
                        let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                        let finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                        loopNormEnc.setComputePipelineState(finalNormPipe)
                        loopNormEnc.setBuffer(currentH, offset: 0, index: 0)
                        loopNormEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                        loopNormEnc.setBuffer(nextH, offset: 0, index: 2)
                        loopNormEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        loopNormEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        loopNormEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        loopNormEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        loopNormEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                        loopNormEnc.endEncoding()

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }
                }

                guard computeLogits else {
                    activeCmd.commit()
                    activeCmd.waitUntilCompleted()
                    return true
                }

                // 3. Final RMSNorm & LM Head
                guard let finalEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                var nOff = normOffset
                var epsVal = eps
                let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                let finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                finalEnc.setComputePipelineState(finalNormPipe)
                finalEnc.setBuffer(currentH, offset: 0, index: 0)
                finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 2)
                finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                finalEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                dispatchLinear(enc: finalEnc, weight: lmHeadTensor, scale: lmHeadScale, bias: lmHeadBias, inBuf: xFinalBuffer, outBuf: logitsBuffer, inDim: hiddenDim, outDim: vocabSize)

                finalEnc.endEncoding()
                activeCmd.commit()
                activeCmd.waitUntilCompleted()

                return true
            }

            var generatedTokenIds: [UInt32] = []

            // Ingest prompt tokens into KV-cache and recurrent states
            for pIdx in 0..<(promptTokenIds.count - 1) {
                if Task.isCancelled { break }
                let pTok = promptTokenIds[pIdx]
                let ok = runTokenForward(tokenId: pTok, step: currentStep, computeLogits: false)
                if !ok { break }
                currentStep += 1
            }

            // Autoregressive generation loop
            for _ in 0..<maxTokens {
                if Task.isCancelled { break }
                let currentTokenId = contextTokens.last!

                let ok = runTokenForward(tokenId: currentTokenId, step: currentStep, computeLogits: true)
                if !ok { break }
                currentStep += 1

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
                if nextToken == eosTokenId || nextToken == 248044 || nextToken == 248046 || nextToken == 248045 || nextToken == 166101 || nextToken == 166102 {
                    break
                }

                generatedTokenIds.append(nextToken)
                contextTokens.append(nextToken)
                tokensGenerated += 1

                // 6. Decode Cumulative Tokens
                let fullDecoded = (try? tokenizer.decode(ids: generatedTokenIds)) ?? ""
                if fullDecoded.contains("<|im_end|>") || fullDecoded.contains("<|endoftext|>") || fullDecoded.contains("<|im_start|>") {
                    break
                }

                let elapsedSec = CFAbsoluteTimeGetCurrent() - startTime
                let tokPerSec = Double(tokensGenerated) / max(elapsedSec, 0.001)
                let elapsedMs = elapsedSec * 1000.0
                let currentRss = getProcessResidentMemoryGB()
                let resCount = WorkingSetManager.shared.residentExperts.count
                let totalExp = WorkingSetManager.shared.totalExpertKeysCount
                let hitRate = WorkingSetManager.shared.cacheHitRatePercent
                let pageLat = WorkingSetManager.shared.lastPagingLatencyMs

                var updatedRaw = fullDecoded
                var thinkPart = ""
                var respPart = ""
                var activeThink = false

                if updatedRaw.contains("</think>") {
                    let parts = updatedRaw.components(separatedBy: "</think>")
                    thinkPart = parts[0].replacingOccurrences(of: "<think>", with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                    respPart = parts.dropFirst().joined(separator: "</think>").trimmingCharacters(in: .whitespacesAndNewlines)
                    activeThink = false
                } else if updatedRaw.contains("<think>") || formattedPrompt.contains("<think>") {
                    thinkPart = updatedRaw.replacingOccurrences(of: "<think>", with: "").trimmingCharacters(in: .whitespaces)
                    respPart = ""
                    activeThink = true
                } else {
                    respPart = updatedRaw
                    activeThink = false
                }

                await MainActor.run {
                    self.generatedStreamText = updatedRaw
                    self.thinkingText = thinkPart
                    self.responseText = respPart
                    self.isThinking = activeThink
                    self.generationTotalTokens = tokensGenerated
                    self.generationElapsedMs = elapsedMs
                    self.generationSpeedTokPerSec = tokPerSec
                    self.generationStatusText = activeThink ? "🧠 Reasoning: \(tokensGenerated) tokens | \(String(format: "%.1f", tokPerSec)) tok/s" : "⚡ Streaming: \(tokensGenerated) tokens | \(String(format: "%.1f", tokPerSec)) tok/s"
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

    private func applyMemoryExecutionMode(_ mode: MemoryExecutionMode) {
        guard let summary = summary else { return }
        let mappedGB = summary.sizeGb
        let eff = mode.resolveEffectiveMode(modelFootprintGB: mappedGB)
        if eff == .residentRAM {
            WorkingSetManager.shared.preFaultAll(shardBuffers: shardBuffers, summary: summary)
            pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
        } else {
            WorkingSetManager.shared.initialize(summary: summary, shardBuffers: shardBuffers, mode: memoryBudgetMode)
            pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
        }
        updatePagingStats()
    }

    private func loadAndBridgeToMetal(filePath: String) {
        do {
            let loadedEngine = try DynaMoeEngine(filePath: filePath)
            let loadedSummary = try loadedEngine.getSummary()
            
            guard let device = MTLCreateSystemDefaultDevice() else {
                metalStatus = "❌ Failed to initialize Metal GPU."
                return
            }

            // Attempt to load and parse HuggingFace config.json if present
            let fileUrl = URL(fileURLWithPath: filePath)
            let dirUrl = fileUrl.hasDirectoryPath ? fileUrl : fileUrl.deletingLastPathComponent()
            if let cfg = ModelConfig.load(from: dirUrl) {
                self.modelConfig = cfg
                self.detectedArchitecture = cfg.resolveArchitectureType(summary: loadedSummary)
            } else {
                self.modelConfig = nil
                self.detectedArchitecture = loadedSummary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer
            }

            // Auto-load tokenizer.json from model directory if present
            let tokUrl = dirUrl.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: tokUrl.path) {
                loadTokenizer(filePath: tokUrl.path)
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

            let effMode = self.memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: mappedGB)
            if effMode == .residentRAM {
                WorkingSetManager.shared.preFaultAll(shardBuffers: buffers, summary: loadedSummary)
                self.pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
            } else {
                WorkingSetManager.shared.initialize(summary: loadedSummary, shardBuffers: buffers, mode: self.memoryBudgetMode)
                self.pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
            }
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
        panel.message = "Select Model Snapshot Folder, model_weights.bin, or model.safetensors.index.json"
        panel.prompt = "Load Model"
        
        if panel.runModal() == .OK, let url = panel.url {
            var targetPath = url.path
            if url.hasDirectoryPath {
                let indexPath = url.appendingPathComponent("model.safetensors.index.json").path
                let flashMoeJson = url.appendingPathComponent("model_weights.json").path
                let flashMoeBin = url.appendingPathComponent("model_weights.bin").path
                
                if FileManager.default.fileExists(atPath: indexPath) {
                    targetPath = indexPath
                } else if FileManager.default.fileExists(atPath: flashMoeJson) {
                    targetPath = flashMoeJson
                } else if FileManager.default.fileExists(atPath: flashMoeBin) {
                    targetPath = flashMoeBin
                }
            }
            loadAndBridgeToMetal(filePath: targetPath)
        }
    }
    #endif
}
