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
//  "Short" is decided against each shard's own safetensors header, not against
//  the index: a truncated shard is present and has a plausible size, so only the
//  offsets the header declares can tell a finished file from a half-written one.
//  A shard is therefore reported when its size disagrees with the header it
//  carries, and the index's `metadata.total_size` is only used for the summary.
//

import Foundation

nonisolated public struct DeepSeekV41Preflight: Sendable, Equatable {
    public struct ShardIssue: Sendable, Equatable {
        public let name: String
        /// Size the shard's own safetensors header implies (8-byte length prefix
        /// plus header plus the largest declared data offset).
        public let expectedBytes: Int?
        /// Bytes actually present; nil when the file does not exist.
        public let actualBytes: Int?
        /// Whether the safetensors header could be read. A present file whose
        /// header is unreadable or zero-length is exactly the half-written
        /// download this preflight exists to catch, so it is an issue, not a
        /// reason to skip the shard.
        public let headerReadable: Bool

        public init(
            name: String,
            expectedBytes: Int?,
            actualBytes: Int?,
            headerReadable: Bool = true
        ) {
            self.name = name
            self.expectedBytes = expectedBytes
            self.actualBytes = actualBytes
            self.headerReadable = headerReadable
        }

        public var isMissing: Bool { actualBytes == nil }

        public var humanDescription: String {
            guard let actualBytes else { return "\(name): missing" }
            guard headerReadable else { return "\(name): header unreadable, \(actualBytes) bytes" }
            guard let expectedBytes else { return "\(name): \(actualBytes) bytes" }
            let state = actualBytes < expectedBytes ? "short" : "larger than its header"
            return "\(name): \(state), \(actualBytes) of \(expectedBytes) bytes"
        }
    }

    public let shardCount: Int
    public let presentShardCount: Int
    public let bytesPresent: Int
    /// `metadata.total_size` from the index when published; nil otherwise.
    public let bytesExpected: Int?
    public let issues: [ShardIssue]
    /// True when every shard the index names exists and matches the size its own
    /// header declares.
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
        // Int64: the published checkpoint's total is ~510 GB, which an Int32 read
        // would truncate into a summary that reads as a fraction of the download.
        let expectedTotal = (metadata?["total_size"] as? NSNumber)?.int64Value
            ?? (metadata?["total_size"] as? Int).map(Int64.init)

        var shardNames = Set<String>()
        for file in weightMap.values { shardNames.insert(file) }
        let ordered = shardNames.sorted()

        var issues: [ShardIssue] = []
        var present = 0
        var bytes: Int64 = 0
        for name in ordered {
            let url = snapshotDir.appendingPathComponent(name)
            guard let attributes = try? fileManager.attributesOfItem(atPath: url.path),
                  let size = attributes[.size] as? NSNumber else {
                issues.append(ShardIssue(name: name, expectedBytes: nil, actualBytes: nil))
                continue
            }
            let actual = size.int64Value
            present += 1
            bytes += actual
            guard let expected = expectedShardBytes(at: url) else {
                // Present but unreadable/zero-length header: report it as an issue
                // instead of pretending the shard is complete.
                issues.append(
                    ShardIssue(
                        name: name,
                        expectedBytes: nil,
                        actualBytes: Int(clamping: actual),
                        headerReadable: false
                    )
                )
                continue
            }
            if Int64(expected) != actual {
                issues.append(
                    ShardIssue(name: name, expectedBytes: expected, actualBytes: Int(clamping: actual))
                )
            }
        }
        return DeepSeekV41Preflight(
            shardCount: ordered.count,
            presentShardCount: present,
            bytesPresent: Int(clamping: bytes),
            bytesExpected: expectedTotal.map { Int(clamping: $0) },
            issues: issues,
            isComplete: issues.isEmpty && ordered.count > 0
        )
    }

    /// Bytes a shard file must have according to the safetensors header it starts
    /// with: the 8-byte little-endian header length, the JSON header, then the
    /// largest `data_offsets` end any tensor declares.
    ///
    /// Returns nil when the header cannot be read or parsed, which is itself
    /// reported elsewhere; a nil here must not veto a present shard.
    static func expectedShardBytes(at url: URL) -> Int? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let prefix = try? handle.read(upToCount: 8), prefix.count == 8 else { return nil }
        let headerLength = prefix.withUnsafeBytes { raw -> UInt64 in
            UInt64(littleEndian: raw.loadUnaligned(as: UInt64.self))
        }
        // A header is a few MB at most; anything past this is not a safetensors
        // file and must not be turned into an allocation.
        guard headerLength > 0, headerLength <= 64 * 1_024 * 1_024 else { return nil }
        guard let headerData = try? handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let header = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            return nil
        }
        var dataEnd: Int64 = 0
        for (key, value) in header where key != "__metadata__" {
            guard let tensor = value as? [String: Any],
                  let offsets = tensor["data_offsets"] as? [NSNumber],
                  offsets.count == 2 else { continue }
            // Int64: a ~10 GB shard has data offsets past what a 32-bit read holds.
            dataEnd = max(dataEnd, offsets[1].int64Value)
        }
        guard dataEnd > 0 else { return nil }
        return Int(clamping: 8 + Int64(headerLength) + dataEnd)
    }
}
