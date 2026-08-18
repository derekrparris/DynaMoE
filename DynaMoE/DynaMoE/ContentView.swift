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

    var filteredTensors: [TensorMetadata] {
        guard let tensors = summary?.tensors else { return [] }
        return searchText.isEmpty ? tensors : tensors.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
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
                    HStack {
                        Image(systemName: "magnifyingglass").foregroundColor(.secondary)
                        TextField("Filter tensors...", text: $searchText).textFieldStyle(.plain)
                    }
                    .padding(8)
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(6)
                    .padding(10)

                    Table(filteredTensors) {
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
                }
            } else {
                ContentUnavailableView("No Weights Loaded", systemImage: "memorychip", description: Text("Select a .safetensors file to test Metal Zero-Copy Handoff."))
            }
        }
        .frame(minWidth: 800, minHeight: 500)
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
}
