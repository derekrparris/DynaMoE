//
//  ContentView.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.
//

import SwiftUI
import UniformTypeIdentifiers

// Make UniFFI record identifiable for SwiftUI Table usage
extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    @State private var summary: ModelSummary? = nil
    @State private var errorMessage: String? = nil
    @State private var searchText: String = ""
    @State private var isImporterPresented: Bool = false

    var filteredTensors: [TensorMetadata] {
        guard let tensors = summary?.tensors else { return [] }
        if searchText.isEmpty {
            return tensors
        } else {
            return tensors.filter { $0.name.localizedCaseInsensitiveContains(searchText) }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            // Header Bar
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("DynaMoE Tensor Inspector")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    if let summary = summary {
                        Text("\(summary.tensorCount) matrices | \(String(format: "%.2f", summary.sizeGb)) GB (mmap mapped)")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    } else {
                        Text("No model loaded")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                    }
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
                ContentUnavailableView("Error Loading Model", systemImage: "exclamationmark.triangle", description: Text(err))
            } else if let summary = summary {
                // Filter bar & Table View
                VStack(spacing: 0) {
                    HStack {
                        Image(systemName: "magnifyingglass")
                            .foregroundColor(.secondary)
                        TextField("Filter tensors (e.g. 'attn', 'expert', 'mlp')...", text: $searchText)
                            .textFieldStyle(.plain)
                    }
                    .padding(8)
                    .background(Color(NSColor.controlBackgroundColor))
                    .cornerRadius(6)
                    .padding(10)

                    Table(filteredTensors) {
                        TableColumn("Tensor Name", value: \.name)
                        TableColumn("Shape") { tensor in
                            Text(tensor.shapeDisplay)
                                .font(.system(.body, design: .monospaced))
                        }
                        TableColumn("Dtype") { tensor in
                            Text(tensor.dtype)
                                .font(.system(.body, design: .monospaced))
                                .foregroundColor(.blue)
                        }
                        TableColumn("Size") { tensor in
                            Text(String(format: "%.2f MB", tensor.sizeMb))
                                .font(.system(.body, design: .monospaced))
                        }
                    }
                }
            } else {
                ContentUnavailableView("No Weights Loaded", systemImage: "doc.badge.gearshape", description: Text("Select a .safetensors file to inspect its memory-mapped tensor layout."))
            }
        }
        .frame(minWidth: 800, minHeight: 500)
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.data],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                if url.startAccessingSecurityScopedResource() {
                    defer { url.stopAccessingSecurityScopedResource() }
                    let res = inspectModelWeights(filePath: url.path)
                    if let err = res.error {
                        self.errorMessage = err
                        self.summary = nil
                    } else {
                        self.errorMessage = nil
                        self.summary = res
                    }
                } else {
                    self.errorMessage = "macOS Sandbox denied file access."
                }
            case .failure(let error):
                self.errorMessage = error.localizedDescription
            }
        }
    }
}

#Preview {
    ContentView()
}
