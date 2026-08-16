//
//  ContentView.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/16/26.
//

import SwiftUI

struct ContentView: View {
    @State private var rustMessage = "Waiting for Rust..."

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "cpu")
                .imageScale(.large)
                .foregroundStyle(.tint)
            
            Text("DynaMoE Engine")
                .font(.headline)
            
            Text(rustMessage)
                .foregroundColor(.secondary)
            
            Button("Ping Rust") {
                // Here we call the Rust function exactly like native Swift!
                rustMessage = helloFromDynamoe(name: "macOS")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }
}

#Preview {
    ContentView()
}
