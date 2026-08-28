//
//  LocalModelManager.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/26/26.
//

import Foundation
import SwiftUI
import Combine

public struct DiscoveredModel: Identifiable, Hashable, Codable, Equatable {
    public var id: String // Unique identifier (e.g. repoId or snapshot path)
    public var displayName: String // e.g. "Ornith 1.5 35B A3B FP8"
    public var author: String // e.g. "ornith-ai"
    public var repoId: String // e.g. "ornith-ai/Ornith-1.5-35B-A3B-FP8"
    public var snapshotPath: String // Directory containing model snapshot
    public var weightsEntryPath: String // Path to index.json, safetensors, or snapshot folder
    public var sizeBytes: Int64
    public var formattedSize: String
    public var architectureName: String
    public var isMoE: Bool
    public var expertCount: Int?
    public var hasTokenizer: Bool
    public var lastModified: Date
    public var quantization: String?
    public var rawModelType: String?
    public var supportsThinking: Bool

    public init(
        id: String,
        displayName: String,
        author: String,
        repoId: String,
        snapshotPath: String,
        weightsEntryPath: String,
        sizeBytes: Int64,
        formattedSize: String,
        architectureName: String,
        isMoE: Bool,
        expertCount: Int? = nil,
        hasTokenizer: Bool = true,
        lastModified: Date = Date(),
        quantization: String? = nil,
        rawModelType: String? = nil,
        supportsThinking: Bool = false
    ) {
        self.id = id
        self.displayName = displayName
        self.author = author
        self.repoId = repoId
        self.snapshotPath = snapshotPath
        self.weightsEntryPath = weightsEntryPath
        self.sizeBytes = sizeBytes
        self.formattedSize = formattedSize
        self.architectureName = architectureName
        self.isMoE = isMoE
        self.expertCount = expertCount
        self.hasTokenizer = hasTokenizer
        self.lastModified = lastModified
        self.quantization = quantization
        self.rawModelType = rawModelType
        self.supportsThinking = supportsThinking
    }
}

public class LocalModelManager: ObservableObject {
    public static let shared = LocalModelManager()

    private let defaultModelKey = "DynaMoE_DefaultModelId"
    private let lastUsedModelKey = "DynaMoE_LastUsedModelId"
    private let customPathsKey = "DynaMoE_CustomScanPaths"

    @Published public var discoveredModels: [DiscoveredModel] = []
    @Published public var isScanning: Bool = false
    @Published public var lastScanDate: Date? = nil
    @Published public var defaultModelId: String? {
        didSet {
            UserDefaults.standard.set(defaultModelId, forKey: defaultModelKey)
        }
    }
    @Published public var lastUsedModelId: String? {
        didSet {
            UserDefaults.standard.set(lastUsedModelId, forKey: lastUsedModelKey)
        }
    }
    @Published public var customScanPaths: [String] = [] {
        didSet {
            UserDefaults.standard.set(customScanPaths, forKey: customPathsKey)
        }
    }

    public init() {
        self.defaultModelId = UserDefaults.standard.string(forKey: defaultModelKey)
        self.lastUsedModelId = UserDefaults.standard.string(forKey: lastUsedModelKey)
        self.customScanPaths = UserDefaults.standard.stringArray(forKey: customPathsKey) ?? []
        scanLocalModels()
    }

    // MARK: - Local HuggingFace & Directory Scanning

    public func scanLocalModels() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            DispatchQueue.main.async {
                self.isScanning = true
            }

            var scanned: [DiscoveredModel] = []

            // 1. Resolve Hugging Face hub paths
            var searchRoots: [URL] = []
            
            // Environment overrides
            if let hfHome = ProcessInfo.processInfo.environment["HF_HOME"] {
                let url = URL(fileURLWithPath: hfHome).appendingPathComponent("hub")
                searchRoots.append(url)
            } else if let hfHubCache = ProcessInfo.processInfo.environment["HF_HUB_CACHE"] {
                searchRoots.append(URL(fileURLWithPath: hfHubCache))
            }

            // Standard default: ~/.cache/huggingface/hub
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            let standardHFCache = homeDir.appendingPathComponent(".cache/huggingface/hub")
            if !searchRoots.contains(where: { $0.path == standardHFCache.path }) {
                searchRoots.append(standardHFCache)
            }

            // User custom paths
            for path in self.customScanPaths {
                let customUrl = URL(fileURLWithPath: path)
                if !searchRoots.contains(where: { $0.path == customUrl.path }) {
                    searchRoots.append(customUrl)
                }
            }

            // 2. Scan each root
            for root in searchRoots {
                guard FileManager.default.fileExists(atPath: root.path) else { continue }
                scanned.append(contentsOf: self.scanDirectory(rootUrl: root))
            }

            // Deduplicate by repoId or snapshotPath
            var uniqueMap: [String: DiscoveredModel] = [:]
            for model in scanned {
                uniqueMap[model.id] = model
            }
            let sortedModels = Array(uniqueMap.values).sorted {
                $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending
            }

            DispatchQueue.main.async {
                self.discoveredModels = sortedModels
                self.isScanning = false
                self.lastScanDate = Date()
            }
        }
    }

    private func scanDirectory(rootUrl: URL) -> [DiscoveredModel] {
        var results: [DiscoveredModel] = []
        let fileManager = FileManager.default

        guard let contents = try? fileManager.contentsOfDirectory(
            at: rootUrl,
            includingPropertiesForKeys: [.isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        for itemUrl in contents {
            let name = itemUrl.lastPathComponent

            // Check if it's a HuggingFace hub repo directory: models--<author>--<model_name>
            if name.hasPrefix("models--") {
                if let model = parseHFHubRepo(repoDirUrl: itemUrl) {
                    results.append(model)
                }
            } else {
                // Check if the directory itself is a direct model snapshot or folder with weights
                if let model = inspectModelDirectory(dirUrl: itemUrl, customRepoId: nil) {
                    results.append(model)
                }
            }
        }

        return results
    }

    private func parseHFHubRepo(repoDirUrl: URL) -> DiscoveredModel? {
        let repoFolderName = repoDirUrl.lastPathComponent
        // Strip "models--"
        let cleanName = String(repoFolderName.dropFirst("models--".count))
        let parts = cleanName.components(separatedBy: "--")
        let author = parts.count > 1 ? parts[0] : "HuggingFace"
        let modelShortName = parts.count > 1 ? parts.dropFirst().joined(separator: "-") : cleanName
        let repoId = "\(author)/\(modelShortName)"

        // Look for snapshots directory
        let snapshotsUrl = repoDirUrl.appendingPathComponent("snapshots")
        guard FileManager.default.fileExists(atPath: snapshotsUrl.path),
              let snapshotFolders = try? FileManager.default.contentsOfDirectory(at: snapshotsUrl, includingPropertiesForKeys: [URLResourceKey.isDirectoryKey, URLResourceKey.contentModificationDateKey], options: .skipsHiddenFiles) else {
            // Check direct folder
            return inspectModelDirectory(dirUrl: repoDirUrl, customRepoId: repoId)
        }

        let validSnapshotFolders = snapshotFolders.filter { url in
            let isDir = (try? url.resourceValues(forKeys: [URLResourceKey.isDirectoryKey]).isDirectory) ?? false
            return isDir && !url.lastPathComponent.hasPrefix(".")
        }

        guard !validSnapshotFolders.isEmpty else { return nil }

        // Sort by most recently modified
        let sortedSnapshots = validSnapshotFolders.sorted {
            let d1 = (try? $0.resourceValues(forKeys: [URLResourceKey.contentModificationDateKey]).contentModificationDate) ?? Date.distantPast
            let d2 = (try? $1.resourceValues(forKeys: [URLResourceKey.contentModificationDateKey]).contentModificationDate) ?? Date.distantPast
            return d1 > d2
        }

        for snapshot in sortedSnapshots {
            if let model = inspectModelDirectory(dirUrl: snapshot, customRepoId: repoId) {
                return model
            }
        }
        return nil
    }

    private func inspectModelDirectory(dirUrl: URL, customRepoId: String?) -> DiscoveredModel? {
        let fileManager = FileManager.default
        let dirPath = dirUrl.path

        // Check if directory contains valid model weights
        let indexPath = dirUrl.appendingPathComponent("model.safetensors.index.json").path
        let singleSafetensors = dirUrl.appendingPathComponent("model.safetensors").path
        let flashMoeJson = dirUrl.appendingPathComponent("model_weights.json").path
        let flashMoeBin = dirUrl.appendingPathComponent("model_weights.bin").path

        var weightsEntryPath: String? = nil

        if fileManager.fileExists(atPath: flashMoeJson) {
            weightsEntryPath = flashMoeJson
        } else if fileManager.fileExists(atPath: flashMoeBin) {
            weightsEntryPath = flashMoeBin
        } else if fileManager.fileExists(atPath: indexPath) {
            weightsEntryPath = indexPath
        } else if fileManager.fileExists(atPath: singleSafetensors) {
            weightsEntryPath = singleSafetensors
        } else {
            // Check if there are any .safetensors files in directory
            if let files = try? fileManager.contentsOfDirectory(atPath: dirPath) {
                if let firstSafe = files.first(where: { $0.hasSuffix(".safetensors") }) {
                    weightsEntryPath = dirUrl.appendingPathComponent(firstSafe).path
                }
            }
        }

        guard let validWeightsPath = weightsEntryPath else {
            return nil
        }

        // Parse config.json if present
        let configUrl = dirUrl.appendingPathComponent("config.json")
        var architectureName = "Transformer"
        var isMoE = false
        var expertCount: Int? = nil
        var quantization: String? = nil
        var rawModelType: String? = nil

        if fileManager.fileExists(atPath: configUrl.path),
           let data = try? Data(contentsOf: configUrl),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {

            rawModelType = json["model_type"] as? String
            let archs = json["architectures"] as? [String] ?? []
            let archString = archs.first ?? rawModelType ?? ""

            if let numExperts = json["num_experts"] as? Int ?? json["n_routed_experts"] as? Int ?? json["num_local_experts"] as? Int {
                isMoE = true
                expertCount = numExperts
            }

            if archString.localizedCaseInsensitiveContains("moe") || (rawModelType?.localizedCaseInsensitiveContains("moe") ?? false) {
                isMoE = true
            }

            if archString.localizedCaseInsensitiveContains("qwen3_5") || archString.localizedCaseInsensitiveContains("ornith") || (rawModelType?.localizedCaseInsensitiveContains("qwen3_5") ?? false) || (rawModelType?.localizedCaseInsensitiveContains("ornith") ?? false) {
                architectureName = isMoE ? "Hybrid SSM-MoE" : "Hybrid SSM-Dense"
            } else if archString.localizedCaseInsensitiveContains("nanbeige") {
                architectureName = isMoE ? "Dense/MoE Transformer" : "Nanbeige Transformer"
            } else if archString.localizedCaseInsensitiveContains("deepseek") {
                architectureName = "DeepSeek MoE (MLA)"
            } else if isMoE {
                architectureName = "Mixture of Experts"
            } else {
                architectureName = "Dense Transformer"
            }

            if let quantConfig = json["quantization_config"] as? [String: Any],
               let quantMethod = quantConfig["quant_method"] as? String {
                quantization = quantMethod.uppercased()
            } else if let quantConfig = json["quantization_config"] as? [String: Any],
                      let bits = quantConfig["bits"] as? Int {
                let mode = quantConfig["mode"] as? String ?? "Affine"
                quantization = "MLX \(bits)-Bit (\(mode.capitalized))"
            } else if let quant = json["quantization"] as? [String: Any],
                      let bits = quant["bits"] as? Int {
                let mode = quant["mode"] as? String ?? "Affine"
                quantization = "MLX \(bits)-Bit (\(mode.capitalized))"
            } else if let dtype = json["torch_dtype"] as? String {
                quantization = dtype
            }
        }

        let tokPath = dirUrl.appendingPathComponent("tokenizer.json").path
        let hasTokenizer = fileManager.fileExists(atPath: tokPath)

        // Compute total size of snapshot files
        // Compute total size of snapshot files
        let (totalBytes, modDate) = calculateDirectorySizeAndModDate(dirUrl: dirUrl)

        // Generate clean display name and repo identifier
        let repoId = customRepoId ?? dirUrl.lastPathComponent
        let displayName = formatDisplayName(repoId: repoId)
        let author = repoId.contains("/") ? repoId.components(separatedBy: "/")[0] : "Local"

        let formattedSize = ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file)

        // Check if model supports thinking / reasoning (pre-computed on background scanning thread)
        var supportsThinking = false
        let nameLower = displayName.lowercased()
        let repoLower = repoId.lowercased()
        let typeLower = (rawModelType ?? "").lowercased()
        let archLower = architectureName.lowercased()
        let pathLower = dirPath.lowercased()

        if nameLower.contains("nanbeige") || repoLower.contains("nanbeige") || typeLower.contains("nanbeige") || archLower.contains("nanbeige") || pathLower.contains("nanbeige") ||
           nameLower.contains("ornith") || repoLower.contains("ornith") || typeLower.contains("ornith") || archLower.contains("ornith") || pathLower.contains("ornith") ||
           nameLower.contains("qwen") || repoLower.contains("qwen") || typeLower.contains("qwen") || archLower.contains("qwen") || pathLower.contains("qwen") ||
           nameLower.contains("deepseek") || repoLower.contains("deepseek") || typeLower.contains("deepseek") || archLower.contains("deepseek") || pathLower.contains("deepseek") ||
           nameLower.contains("r1") || repoLower.contains("r1") || nameLower.contains("reason") || repoLower.contains("reason") ||
           nameLower.contains("qwq") || repoLower.contains("qwq") || typeLower.contains("qwq") ||
           nameLower.contains("glm") || repoLower.contains("glm") || typeLower.contains("glm") ||
           nameLower.contains("nemotron") || repoLower.contains("nemotron") || typeLower.contains("nemotron") ||
           nameLower.contains("think") || repoLower.contains("think") {
            supportsThinking = true
        } else {
            // Check only lightweight config templates (never huge tokenizer.json)
            for fName in ["tokenizer_config.json", "chat_template.jinja"] {
                let fUrl = dirUrl.appendingPathComponent(fName)
                if fileManager.fileExists(atPath: fUrl.path), let content = try? String(contentsOf: fUrl, encoding: .utf8) {
                    if content.contains("<think>") || content.contains("enable_thinking") || content.contains("<|thought|>") || content.contains("reasoning_content") {
                        supportsThinking = true
                        break
                    }
                }
            }
        }

        return DiscoveredModel(
            id: repoId,
            displayName: displayName,
            author: author,
            repoId: repoId,
            snapshotPath: dirPath,
            weightsEntryPath: validWeightsPath,
            sizeBytes: totalBytes,
            formattedSize: formattedSize,
            architectureName: architectureName,
            isMoE: isMoE,
            expertCount: expertCount,
            hasTokenizer: hasTokenizer,
            lastModified: modDate,
            quantization: quantization,
            rawModelType: rawModelType,
            supportsThinking: supportsThinking
        )
    }

    private func formatDisplayName(repoId: String) -> String {
        let nameOnly = repoId.contains("/") ? repoId.components(separatedBy: "/").dropFirst().joined(separator: "/") : repoId
        // Replace dashes and underscores with spaces for clean readability
        var formatted = nameOnly.replacingOccurrences(of: "-", with: " ").replacingOccurrences(of: "_", with: " ")
        // Remove duplicate spaces
        formatted = formatted.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }.joined(separator: " ")
        return formatted
    }

    private func calculateDirectorySizeAndModDate(dirUrl: URL) -> (Int64, Date) {
        let fileManager = FileManager.default
        var totalSize: Int64 = 0
        var latestDate: Date = Date.distantPast

        guard let enumerator = fileManager.enumerator(
            at: dirUrl,
            includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey, .isSymbolicLinkKey],
            options: [.skipsHiddenFiles]
        ) else {
            return (0, Date())
        }

        for case let fileUrl as URL in enumerator {
            guard let values = try? fileUrl.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey, .isSymbolicLinkKey]) else { continue }
            
            if let date = values.contentModificationDate, date > latestDate {
                latestDate = date
            }

            // If it's a symlink (as HuggingFace blobs are), resolve destination size
            if values.isSymbolicLink == true {
                if let destinationPath = try? fileManager.destinationOfSymbolicLink(atPath: fileUrl.path) {
                    let fullDestUrl = fileUrl.deletingLastPathComponent().appendingPathComponent(destinationPath)
                    if let destValues = try? fullDestUrl.resourceValues(forKeys: [.fileSizeKey]),
                       let size = destValues.fileSize {
                        totalSize += Int64(size)
                    }
                }
            } else if let size = values.fileSize {
                totalSize += Int64(size)
            }
        }

        if latestDate == Date.distantPast {
            latestDate = Date()
        }

        return (totalSize, latestDate)
    }

    // MARK: - Model Preferences

    public func setDefaultModel(id: String?) {
        self.defaultModelId = id
    }

    public func setLastUsedModel(id: String) {
        self.lastUsedModelId = id
    }

    public func getModel(byId id: String) -> DiscoveredModel? {
        return discoveredModels.first(where: { $0.id == id || $0.repoId == id || $0.snapshotPath == id || $0.weightsEntryPath == id })
    }

    public func getDefaultOrFirstModel() -> DiscoveredModel? {
        if let defaultId = defaultModelId, let model = getModel(byId: defaultId) {
            return model
        }
        if let lastId = lastUsedModelId, let model = getModel(byId: lastId) {
            return model
        }
        return discoveredModels.first
    }
}
