//
//  ContentView.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @State private var engineOutput = "Waiting for model weights..."
    @State private var isImporterPresented = false

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "memorychip")
                .imageScale(.large)
                .font(.system(size: 40))
                .foregroundStyle(.tint)
            
            Text("DynaMoE SSD Streamer")
                .font(.title2)
                .fontWeight(.bold)
            
            Text(engineOutput)
                .font(.system(.body, design: .monospaced))
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .padding()
                .frame(minWidth: 400, minHeight: 100)
                .background(Color(NSColor.controlBackgroundColor))
                .cornerRadius(8)
            
            Button("Select .safetensors File") {
                isImporterPresented = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        }
        .padding(40)
        // Native macOS File Picker
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: [.data], // Accepts binary data files
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                
                // Security scoped resource access for macOS App Sandbox
                if url.startAccessingSecurityScopedResource() {
                    defer { url.stopAccessingSecurityScopedResource() }
                    
                    // Pass the file path down into our Rust mmap engine!
                    engineOutput = inspectModelWeights(filePath: url.path)
                } else {
                    engineOutput = "Error: macOS Sandbox denied access to the file."
                }
                
            case .failure(let error):
                engineOutput = "Failed to select file: \(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    ContentView()
}
