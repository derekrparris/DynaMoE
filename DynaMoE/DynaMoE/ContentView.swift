//
//  ContentView.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.

import SwiftUI
import UniformTypeIdentifiers
import Metal

extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    // Hold the active Rust engine in memory so the file stays mapped
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
                    .padding(8).background(Color(NSColor.controlBackgroundColor)).cornerRadius(6).padding(10)

                    Table(filteredTensors) {
                        TableColumn("Tensor Name", value: \.name)
                        TableColumn("Shape") { t in Text(t.shapeDisplay).font(.system(.body, design: .monospaced)) }
                        TableColumn("Dtype") { t in Text(t.dtype).font(.system(.body, design: .monospaced)).foregroundColor(.blue) }
                        TableColumn("Size") { t in Text(String(format: "%.2f MB", t.sizeMb)).font(.system(.body, design: .monospaced)) }
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
    
    // MARK: - Metal Zero-Copy Logic
    private func loadAndBridgeToMetal(filePath: String) {
        do {
            // 1. Initialize the stateful Rust engine (mmaps the file)
            let loadedEngine = try DynaMoeEngine(filePath: filePath)
            self.engine = loadedEngine
            self.summary = try loadedEngine.getSummary()
            self.errorMessage = nil
            
            // 2. Get the default Apple Silicon GPU
            guard let device = MTLCreateSystemDefaultDevice() else {
                metalStatus = "❌ Failed to initialize Metal GPU."
                return
            }
            
            // 3. Convert Rust's u64 address into a Swift memory pointer
            let address = UInt(loadedEngine.bufferBaseAddress())
            guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else {
                metalStatus = "❌ Invalid memory pointer received from Rust."
                return
            }
            let length = Int(loadedEngine.bufferLength())
            
            // 4. THE HANDOFF: Tell Metal to wrap the SSD pointer directly without copying
            guard let buffer = device.makeBuffer(
                bytesNoCopy: pointer,
                length: length,
                options: .storageModeShared,
                deallocator: { _, _ in
                    // Empty closure: Rust owns the lifecycle, so Metal shouldn't try to free it.
                }
            ) else {
                metalStatus = "❌ Metal rejected the buffer (Page Alignment Error?)"
                return
            }
            
            let mbSize = Double(buffer.length) / (1024.0 * 1024.0)
            metalStatus = "✅ Zero-Copy Active! GPU is directly mapping \(String(format: "%.2f", mbSize)) MB"
            
        } catch {
            self.errorMessage = "Core Engine Error: \(error.localizedDescription)"
            self.summary = nil
            self.engine = nil
        }
    }
}
