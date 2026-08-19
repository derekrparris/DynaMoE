import SwiftUI
import UniformTypeIdentifiers
import Metal

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

                if let tokenizer = tokenizer, let ids = try? tokenizer.encode(text: promptInput), !ids.isEmpty {
                    Button("Generate Initial Hidden State h_0") {
                        executeEmbeddingLookup(tokenIds: ids)
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.blue)
                    .padding(.top, 4)
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
            for i in 0..<8 {
                h0Sample.append(String(format: "%.6f", rawFloatPtr[i]))
            }
            
            gpuComputeOutput = "🚀 Generated h_0 Vector (\(embedWeight.dtype)) [\(tokenCount) x \(hiddenDim)] from Shard #\(embedWeight.shardIndex)! First 8 dims of Token #0: [\(h0Sample.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Embedding Lookup Error: \(error.localizedDescription)"
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
