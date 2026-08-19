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
                        isTokenizerImporterPresented = true
                    }
                    .buttonStyle(.bordered)
                    
                    Button("Select .safetensors") {
                        isWeightImporterPresented = true
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()

            // Tokenizer Playground
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
            }
            .padding(10)
            .background(Color(NSColor.controlBackgroundColor))
            
            Divider()

            if let err = errorMessage {
                ContentUnavailableView("Error", systemImage: "exclamationmark.triangle", description: Text(err))
            } else if summary != nil {
                VStack(spacing: 0) {
                    // Search & Topology Filter Chips
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
                                Text("Category: \(tensor.category) | Offset: \(tensor.offsetStart) bytes")
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
                ContentUnavailableView("No Weights Loaded", systemImage: "memorychip", description: Text("Select a .safetensors file to inspect MoE layer topology."))
            }
        }
        .frame(minWidth: 900, minHeight: 650)
        // File Importers
        .fileImporter(isPresented: $isWeightImporterPresented, allowedContentTypes: [.data], allowsMultipleSelection: false) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                if url.startAccessingSecurityScopedResource() {
                    defer { url.stopAccessingSecurityScopedResource() }
                    loadAndBridgeToMetal(filePath: url.path)
                } else {
                    errorMessage = "macOS Sandbox denied file access."
                }
            case .failure(let error):
                errorMessage = error.localizedDescription
            }
        }
        .fileImporter(isPresented: $isTokenizerImporterPresented, allowedContentTypes: [.json, .data], allowsMultipleSelection: false) { result in
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
    
    private func loadAndBridgeToMetal(filePath: String) {
        do {
            let loadedEngine = try DynaMoeEngine(filePath: filePath)
            self.engine = loadedEngine
            self.summary = try loadedEngine.getSummary()
            self.errorMessage = nil
            self.selectedTensorID = nil
            self.gpuComputeOutput = nil
            
            guard let device = MTLCreateSystemDefaultDevice() else {
                metalStatus = "❌ Failed to initialize Metal GPU."
                return
            }
            
            let address = UInt(loadedEngine.bufferBaseAddress())
            guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else {
                metalStatus = "❌ Invalid memory pointer received from Rust."
                return
            }
            let length = Int(loadedEngine.bufferLength())
            
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: length,
                options: .storageModeShared,
                deallocator: nil
            ) else {
                metalStatus = "❌ Metal rejected buffer."
                return
            }
            
            let mbSize = Double(buffer.length) / (1024.0 * 1024.0)
            metalStatus = "✅ Zero-Copy Active! GPU mapped \(String(format: "%.2f", mbSize)) MB"
            
        } catch {
            self.errorMessage = "Core Engine Error: \(error.localizedDescription)"
            self.summary = nil
            self.engine = nil
        }
    }

    private func executeGpuShader(on tensor: TensorMetadata) {
        guard let engine = engine,
              let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_mxfp8_paired") else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline."
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

            let address = UInt(engine.bufferBaseAddress())
            guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else { return }
            let length = Int(engine.bufferLength())
            
            guard let rawBaseBuffer = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil) else {
                gpuComputeOutput = "❌ Failed to instantiate Metal base buffer."
                return
            }
            
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
            
            gpuComputeOutput = "⚡ MXFP8 Dequantized! First 8 weights for '\(baseName)': [\(sampleValues.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Pipeline Error: \(error.localizedDescription)"
        }
    }
}
