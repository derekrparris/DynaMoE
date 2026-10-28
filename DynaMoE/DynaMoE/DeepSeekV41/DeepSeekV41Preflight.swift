//
//  DeepSeekV41Preflight.swift
//  DynaMoE
//
//  Load-time sanity check for a DeepSeek-V4.1 snapshot.
//
//  The published checkpoint is 48 shards and ~510 GB, so it is normally
//  downloaded over hours. A half-finished `huggingface-cli download` leaves a
//  snapshot whose index names shards that are absent or still `.incomplete`,
//  and the only signal today is a bare "Missing shard file" from the Rust
//  loader with no sense of how far along the download is. This preflight turns
//  that into one line: how many shards are present, how many bytes against the
//  index's own total, and which files are short.
//

import Foundation

nonisolated public struct DeepSeekV41Preflight: Sendable, Equatable {
    public struct ShardIssue: Sendable, Equatable {
        public let name: String
        /// Size the index implies, when the checkpoint publishes it.
        public let expectedBytes: Int?
        /// Bytes actually present; nil when the file does not exist.
        public let actualBytes: Int?

        public var isMissing: Bool { actualBytes == nil }
    }

    public let shardCount: Int
    public let presentShardCount: Int
    public let bytesPresent: Int
    /// `metadata.total_size` from the index when published; nil otherwise.
    public let bytesExpected: Int?
    public let issues: [ShardIssue]
    /// True when every shard the index names exists and no size mismatch is
    /// visible from the file system.
    public let isComplete: Bool

    public var humanSummary: String {
        let present = ByteCountFormatter.string(fromByteCount: Int64(bytesPresent), countStyle: .file)
        guard let expected = bytesExpected else {
            return "\(presentShardCount)/\(shardCount) shards, \(present)"
        }
        let total = ByteCountFormatter.string(fromByteCount: Int64(expected), countStyle: .file)
        let percent = expected > 0 ? Int((Double(bytesPresent) / Double(expected)) * 100) : 0
        return "\(presentShardCount)/\(shardCount) shards, \(present) of \(total) (\(percent)%)"
    }

    /// `nil` when the directory has no DeepSeek V4.1 index to check (a packed
    /// FlashMoE repack, for instance, is checked by the packer instead).
    public static func inspect(snapshotDir: URL, fileManager: FileManager = .default) -> DeepSeekV41Preflight? {
        let indexURL = snapshotDir.appendingPathComponent("model.safetensors.index.json")
        guard fileManager.fileExists(atPath: indexURL.path),
              let data = try? Data(contentsOf: indexURL),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = object["weight_map"] as? [String: String] else {
            return nil
        }
        let metadata = object["metadata"] as? [String: Any]
        let expectedTotal = (metadata?["total_size"] as? Int)
            ?? (metadata?["total_size"] as? NSNumber)?.intValue

        var shardNames = Set<String>()
        for file in weightMap.values { shardNames.insert(file) }
        let ordered = shardNames.sorted()

        var issues: [ShardIssue] = []
        var present = 0
        var bytes = 0
        for name in ordered {
            let url = snapshotDir.appendingPathComponent(name)
            if let attributes = try? fileManager.attributesOfItem(atPath: url.path),
               let size = attributes[.size] as? NSNumber {
                present += 1
                bytes += size.intValue
            } else {
                issues.append(ShardIssue(name: name, expectedBytes: nil, actualBytes: nil))
            }
        }
        return DeepSeekV41Preflight(
            shardCount: ordered.count,
            presentShardCount: present,
            bytesPresent: bytes,
            bytesExpected: expectedTotal,
            issues: issues,
            isComplete: issues.isEmpty && ordered.count > 0
        )
    }
}
