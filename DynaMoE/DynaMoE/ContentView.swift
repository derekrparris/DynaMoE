import SwiftUI
import UniformTypeIdentifiers
import Metal

extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    @State private var engine: DynaMoeEngine? = nil
    @State private var summary: ModelSummary? = nil
    @State private var errorMessage: String? = nil
    @State private var metalStatus: String = "GPU Status: Waiting for weights..."
    @State private var searchText: String = ""
    @State private var isImporterPresented: Bool = false
    
    // Selection and GPU Compute State
    @State private var selectedTensorID: String? = nil
    @State private var gpuComputeOutput: String? = nil

    var filteredTensors: [TensorMetadata] {
        guard let tensors = summary?.tensors else { return [] }
        return searchText.isEmpty ? tensors : tensors.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
    }

    var selectedTensor: TensorMetadata? {
        summary?.tensors.first(where: { $0.name == selectedTensorID })
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("DynaMoE Tensor Inspector")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text(metalStatus)
                        .font(.subheadline)
                        .foregroundColor(metalStatus.contains("✅") ? .green : .secondary)
                }
                
                Spacer()
                
                Button("Select .safetensors") {
                    isImporterPresented = true
                }
                .buttonStyle(.borderedProminent)
            }
            .padding()
            .background(Color(NSColor.windowBackgroundColor))
            
            Divider()

            if let err = errorMessage {
                ContentUnavailableView("Error", systemImage: "exclamationmark.triangle", description: Text(err))
            } else if summary != nil {
                VStack(spacing: 0) {
                    // Search Bar
                    HStack {
                        Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                        TextField("Filter tensors...", text: $searchText).textFieldStyle(.plain)
                    }
                    .padding(8)
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(6)
                    .padding(10)

                    // Interactive Tensor Table
                    Table(filteredTensors, selection: $selectedTensorID) {
                        TableColumn("Tensor Name", value: \.name)
                        TableColumn("Shape") { t in
                            Text(t.shapeDisplay).font(.system(.body, design: .monospaced))
                        }
                        TableColumn("Dtype") { t in
                            Text(t.dtype).font(.system(.body, design: .monospaced)).foregroundColor(.blue)
                        }
                        TableColumn("Size") { t in
                            Text(String(format: "%.2f MB", t.sizeMb)).font(.system(.body, design: .monospaced))
                        }
                        TableColumn("Byte Offset Range") { t in
                            Text("\(t.offsetStart) ..< \(t.offsetEnd)")
                                .font(.system(.caption, design: .monospaced))
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    // GPU Inspector Action Bar
                    if let tensor = selectedTensor {
                        Divider()
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Selected: \(tensor.name)")
                                    .font(.headline)
                                Text("Shape: \(tensor.shapeDisplay) | Offset: \(tensor.offsetStart) bytes")
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
                ContentUnavailableView("No Weights Loaded", systemImage: "memorychip", description: Text("Select a .safetensors file to test Metal Zero-Copy Handoff."))
            }
        }
        .frame(minWidth: 850, minHeight: 550)
        .fileImporter(isPresented: $isImporterPresented, allowedContentTypes: [.data], allowsMultipleSelection: false) { result in
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

    // MARK: - Execute MSL Shader on Selected Tensor
    private func executeGpuShader(on tensor: TensorMetadata) {
        guard let engine = engine,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_mxfp8_weights") else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline."
            return
        }

        do {
            let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
            
            // 1. Wrap mapped pointer for Metal
            let address = UInt(engine.bufferBaseAddress())
            guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else { return }
            let length = Int(engine.bufferLength())
            
            guard let rawBaseBuffer = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil) else {
                gpuComputeOutput = "❌ Failed to instantiate Metal base buffer."
                return
            }
            
            // 2. Create GPU Output Buffer for 8 sample weights (FP32)
            let sampleCount = 8
            let outputByteLength = sampleCount * MemoryLayout<Float>.stride
            guard let outputBuffer = device.makeBuffer(length: outputByteLength, options: .storageModeShared) else { return }
            
            // 3. Prepare Command Buffer
            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
            
            var byteOffset = tensor.offsetStart
            
            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
            computeEncoder.setBytes(&byteOffset, length: MemoryLayout<UInt64>.stride, index: 2)
            
            // Dispatch 8 threads to read sample weights
            let gridSize = MTLSize(width: sampleCount, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(sampleCount, pipelineState.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            // 4. Read back GPU Output Buffer
            let rawFloatPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
            var sampleValues: [String] = []
            for i in 0..<sampleCount {
                sampleValues.append(String(format: "%.4f", rawFloatPtr[i]))
            }
            
            gpuComputeOutput = "⚡ GPU Execution Complete! First 8 weights read from NVMe: [\(sampleValues.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Pipeline Error: \(error.localizedDescription)"
        }
    }
}
