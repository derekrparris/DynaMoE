//
//  ExpertRepacker.swift
//  DynaMoE
//
//  High-Performance Safetensors -> Packed Expert Binary Blob Repacker
//  Extracts non-expert weights into model_weights.bin and packs expert weights
//  contiguously per-layer (packed_experts/layer_XX.bin) with exact bit-level identity.
//

import Foundation

public struct FlashMoELayout: Codable {
    public struct Component: Codable {
        public let name: String
        public let offset: UInt64
        public let size: UInt64
        public let dtype: String
        public let shape: [Int]
    }

    public let expert_size: UInt64
    public let num_layers: Int
    public let num_experts: Int
    public let components: [Component]
}

public struct FlashMoEWeightsManifest: Codable {
    public struct TensorEntry: Codable {
        public let offset: UInt64
        public let size: UInt64
        public let shape: [Int]
        public let dtype: String
    }

    public let model: String
    public let num_tensors: Int
    public let tensors: [String: TensorEntry]
}

public enum ExpertRepackerError: LocalizedError {
    case invalidSourceDirectory
    case indexNotFound
    case failedToOpenFile(String)
    case writeFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidSourceDirectory: return "Source directory does not exist or is invalid."
        case .indexNotFound: return "Neither model.safetensors.index.json nor safetensors shards were found."
        case .failedToOpenFile(let path): return "Failed to open file at path: \(path)"
        case .writeFailed(let msg): return "Repack write failed: \(msg)"
        }
    }
}

public final class ExpertRepacker {
    public static let shared = ExpertRepacker()

    private init() {}

    /// Checks if a directory is already in FlashMoE pre-packed format
    public static func isPackedFormat(dir: URL) -> Bool {
        let weightsBin = dir.appendingPathComponent("model_weights.bin")
        let packedDir = dir.appendingPathComponent("packed_experts")
        let layoutJson = packedDir.appendingPathComponent("layout.json")

        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: weightsBin.path),
              FileManager.default.fileExists(atPath: packedDir.path, isDirectory: &isDir),
              isDir.boolValue,
              FileManager.default.fileExists(atPath: layoutJson.path) else {
            return false
        }
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: weightsBin.path),
              let size = attrs[.size] as? UInt64, size > 1024 * 1024 else {
            return false
        }
        guard let lData = try? Data(contentsOf: layoutJson),
              let layout = try? JSONDecoder().decode(FlashMoELayout.self, from: lData),
              layout.expert_size > 0,
              !layout.components.isEmpty else {
            return false
        }
        return true
    }

    /// Fast repacking of a Safetensors HuggingFace snapshot into FlashMoE packed binary format
    public func repackSafetensors(
        sourceDir: URL,
        outputDir: URL,
        progress: @escaping (Double, String) -> Void
    ) throws {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: outputDir.path) {
            try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)
        }
        let packedExpertsDir = outputDir.appendingPathComponent("packed_experts")
        if !fileManager.fileExists(atPath: packedExpertsDir.path) {
            try fileManager.createDirectory(at: packedExpertsDir, withIntermediateDirectories: true)
        }

        // Copy auxiliary config and tokenizer files
        let auxFiles = ["config.json", "tokenizer.json", "tokenizer.model", "generation_config.json", "added_tokens.json", "special_tokens_map.json", "vocab.bin", "tokenizer.bin"]
        for aux in auxFiles {
            let srcAux = sourceDir.appendingPathComponent(aux)
            let dstAux = outputDir.appendingPathComponent(aux)
            if fileManager.fileExists(atPath: srcAux.path) && !fileManager.fileExists(atPath: dstAux.path) {
                try? fileManager.copyItem(at: srcAux, to: dstAux)
            }
        }

        progress(0.05, "Parsing Safetensors metadata...")
        let engine = try DynaMoeEngine(filePath: sourceDir.path)
        let summary = try engine.getSummary()

        let numLayers = Int(summary.layerCount > 0 ? summary.layerCount : 40)
        let numExperts = Int(summary.maxExpertId > 0 ? summary.maxExpertId : 256)

        // Map shard handles for zero-copy reading
        var shardData: [UInt32: (ptr: UnsafeRawPointer, length: Int)] = [:]
        for shard in summary.shards {
            let addr = UInt(shard.baseAddress)
            if let ptr = UnsafeRawPointer(bitPattern: addr) {
                shardData[shard.index] = (ptr: ptr, length: Int(shard.length))
            }
        }

        // 1. Separate Non-Expert Tensors -> model_weights.bin
        progress(0.10, "Extracting dense backbone weights...")
        let nonExpertTensors = summary.tensors.filter { tensor in
            guard tensor.expertId == nil else { return false }
            let name = tensor.name.lowercased()
            if name.contains("ple.") || name.contains("ngram_embedding") { return false }
            if name.starts(with: "mtp.") || name.contains(".mtp.") { return false }
            if name.starts(with: "visual.") || name.starts(with: "model.visual.") { return false }
            return true
        }
        var nonExpertManifest: [String: FlashMoEWeightsManifest.TensorEntry] = [:]

        let weightsBinUrl = outputDir.appendingPathComponent("model_weights.bin")
        guard let weightsBinHandle = fopen(weightsBinUrl.path, "wb") else {
            throw ExpertRepackerError.writeFailed("Cannot open model_weights.bin for writing")
        }
        defer { fclose(weightsBinHandle) }

        var currentOffset: UInt64 = 0
        for tensor in nonExpertTensors {
            guard let sInfo = shardData[tensor.shardIndex] else { continue }
            let tensorSize = tensor.offsetEnd - tensor.offsetStart
            let srcPtr = sInfo.ptr.advanced(by: Int(tensor.offsetStart))

            let written = fwrite(srcPtr, 1, Int(tensorSize), weightsBinHandle)
            guard written == Int(tensorSize) else {
                throw ExpertRepackerError.writeFailed("Short write in model_weights.bin for \(tensor.name)")
            }

            nonExpertManifest[tensor.name] = FlashMoEWeightsManifest.TensorEntry(
                offset: currentOffset,
                size: tensorSize,
                shape: [],
                dtype: tensor.dtype
            )
            currentOffset += tensorSize
        }

        let weightsManifest = FlashMoEWeightsManifest(
            model: sourceDir.path,
            num_tensors: nonExpertManifest.count,
            tensors: nonExpertManifest
        )
        let manifestData = try JSONEncoder().encode(weightsManifest)
        try manifestData.write(to: outputDir.appendingPathComponent("model_weights.json"))

        // 2. Repack Per-Layer Experts -> packed_experts/layer_XX.bin
        progress(0.20, "Repacking \(numLayers) layers of expert weights...")

        // Dynamically discover component suffixes from the first available routed expert
        let allExpertTensors = summary.tensors.filter { $0.expertId != nil && $0.layerIndex != nil }
        guard let firstLayer = allExpertTensors.compactMap({ $0.layerIndex }).min(),
              let sampleExpertTensor = allExpertTensors.first(where: { $0.layerIndex == firstLayer && $0.expertId == 0 }) ?? allExpertTensors.first(where: { $0.layerIndex == firstLayer }),
              let sampleExpId = sampleExpertTensor.expertId else {
            throw ExpertRepackerError.writeFailed("No MoE expert tensors found in model summary.")
        }

        let sampleExpertTensors = summary.tensors.filter { $0.layerIndex == firstLayer && $0.expertId == sampleExpId }
        var componentSuffixes: [String] = []

        let projPrefixes = ["gate_proj", "up_proj", "down_proj"]
        for prefix in projPrefixes {
            let matching = sampleExpertTensors.filter { $0.name.contains(prefix) }
            // Sort: weight first, then scales/scale/weight_scale, then biases/bias
            let sorted = matching.sorted { t1, t2 in
                func priority(_ name: String) -> Int {
                    if name.hasSuffix(".weight") { return 0 }
                    if name.contains("scale") { return 1 }
                    if name.contains("bias") { return 2 }
                    return 3
                }
                return priority(t1.name) < priority(t2.name)
            }
            for t in sorted {
                if let range = t.name.range(of: prefix) {
                    let suffix = String(t.name[range.lowerBound...])
                    if !componentSuffixes.contains(suffix) {
                        componentSuffixes.append(suffix)
                    }
                }
            }
        }

        if componentSuffixes.isEmpty {
            componentSuffixes = [
                "gate_proj.weight", "gate_proj.weight_scale", "gate_proj.scales", "gate_proj.biases",
                "up_proj.weight", "up_proj.weight_scale", "up_proj.scales", "up_proj.biases",
                "down_proj.weight", "down_proj.weight_scale", "down_proj.scales", "down_proj.biases"
            ]
        }

        var expertTensorMap: [String: TensorMetadata] = [:]
        expertTensorMap.reserveCapacity(numLayers * numExperts * max(componentSuffixes.count, 1))
        for t in summary.tensors {
            if let l = t.layerIndex, let e = t.expertId {
                for suffix in componentSuffixes {
                    if t.name.hasSuffix(suffix) {
                        expertTensorMap["\(l)_\(e)_\(suffix)"] = t
                        break
                    }
                }
            }
        }

        var layoutComponents: [FlashMoELayout.Component] = []
        var singleExpertSize: UInt64 = 0

        // Calculate offsets and single expert size using sample layer and expert
        var componentOffset: UInt64 = 0
        for suffix in componentSuffixes {
            if let t = expertTensorMap["\(firstLayer)_\(sampleExpId)_\(suffix)"] {
                let size = t.offsetEnd - t.offsetStart
                layoutComponents.append(FlashMoELayout.Component(
                    name: suffix,
                    offset: componentOffset,
                    size: size,
                    dtype: t.dtype,
                    shape: []
                ))
                componentOffset += size
            }
        }
        singleExpertSize = componentOffset

        guard singleExpertSize > 0, !layoutComponents.isEmpty else {
            throw ExpertRepackerError.writeFailed("Failed to resolve expert layout components (singleExpertSize=0).")
        }

        // Repack each layer
        for l in 0..<numLayers {
            let layerBinUrl = packedExpertsDir.appendingPathComponent(String(format: "layer_%02d.bin", l))

            // Check if this layer has any routed experts
            let layerHasExperts = (0..<numExperts).contains { expertTensorMap["\(l)_\($0)_\(layoutComponents[0].name)"] != nil }
            guard layerHasExperts else {
                // If a stale or zero-length binary exists from an earlier failed run, remove it
                try? FileManager.default.removeItem(at: layerBinUrl)
                print("ℹ️ [ExpertRepacker] Skipping layer \(l) (no routed experts detected, dense MLP).")
                continue
            }

            guard let layerHandle = fopen(layerBinUrl.path, "wb") else {
                throw ExpertRepackerError.writeFailed("Cannot open layer_\(l).bin for writing")
            }

            for e in 0..<numExperts {
                for comp in layoutComponents {
                    let key = "\(l)_\(e)_\(comp.name)"
                    if let t = expertTensorMap[key], let sInfo = shardData[t.shardIndex] {
                        let srcPtr = sInfo.ptr.advanced(by: Int(t.offsetStart))
                        let size = Int(t.offsetEnd - t.offsetStart)
                        _ = fwrite(srcPtr, 1, size, layerHandle)
                    }
                }
            }
            fclose(layerHandle)

            let p = 0.20 + (Double(l + 1) / Double(numLayers)) * 0.75
            progress(p, "Repacked layer \(l + 1)/\(numLayers)...")
        }

        // 3. Write layout.json
        let layout = FlashMoELayout(
            expert_size: singleExpertSize,
            num_layers: numLayers,
            num_experts: numExperts,
            components: layoutComponents
        )
        let layoutData = try JSONEncoder().encode(layout)
        try layoutData.write(to: packedExpertsDir.appendingPathComponent("layout.json"))

        progress(1.0, "Repacking complete!")
    }
}
