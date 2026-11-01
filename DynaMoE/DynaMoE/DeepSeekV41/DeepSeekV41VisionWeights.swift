//
//  DeepSeekV41VisionWeights.swift
//  DynaMoE
//
//  The vision tower's weights, straight out of the checkpoint's shards.
//
//  The tower is the one part of this checkpoint whose forward runs on the host: it
//  is 970 MB of BF16, and one image costs the tower's 32 blocks once, where the
//  510 GB backbone costs a full pass per position. Its weights are still read
//  through `pread` and decoded on demand rather than pinned, because one encode
//  touches each tensor exactly once: a pinned copy would be 1.9 GB of `Float` to
//  save reads that happen a single time.
//
//  Resolution is by tensor NAME through the checkpoint's own `index.json` and
//  safetensors headers, not through the segmented `ModelSummary`. The summary is
//  what the GPU binds, and a tensor no kernel reads has no reason to survive
//  segmentation; the tower would then lose a block's weights with no error anywhere,
//  which reads as a tower that runs and answers wrongly. A header is authoritative
//  and costs one read per shard.
//
//  Every failure is recorded rather than swallowed: a dtype this decoder does not
//  know and a short read both surface through `problems`, because the alternative is
//  the tower reporting a missing tensor when the bytes are there and wrong.
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

nonisolated public enum DeepSeekV41VisionWeightError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(String)
    case unreadableHeader(String)
    case shortRead(name: String, offset: Int, wanted: Int, got: Int)
    case unsupportedDtype(name: String, dtype: String)

    public var description: String {
        switch self {
        case .cannotOpen(let detail):
            return "cannot open a vision shard: \(detail)"
        case .unreadableHeader(let detail):
            return "cannot read a vision shard header: \(detail)"
        case let .shortRead(name, offset, wanted, got):
            return "vision tensor '\(name)' read at \(offset) returned \(got) of \(wanted) bytes"
        case let .unsupportedDtype(name, dtype):
            return "vision tensor '\(name)' has dtype '\(dtype)', which the host decoder does not read"
        }
    }
}

/// A role's tensor inside one shard file.
nonisolated private struct VisionTensorLocation {
    let name: String
    let path: String
    /// Absolute byte offset in the file, header included.
    let offset: Int
    let byteLength: Int
    /// Published shape: a matrix's rows and columns, a vector's count and 1.
    let rows: Int
    let columns: Int
    let dtype: String
}

/// The tower's weights as a `DeepSeekV41WeightSource`, decoded per lookup.
///
/// Not thread-safe by design and documented rather than locked: one encode reads
/// each tensor once, from one thread, and the cache-free decode is what keeps a
/// 970 MB tower from needing 1.9 GB of resident `Float`.
nonisolated public final class DeepSeekV41VisionWeights: DeepSeekV41WeightSource, @unchecked Sendable {

    /// Everything the decoder refused, in the order it was refused. A missing role
    /// is not listed here: the tower names it itself through `missingTensor`.
    public private(set) var problems: [String] = []

    private var locations: [String: VisionTensorLocation] = [:]
    /// One descriptor per shard a vision tensor lives in. The published checkpoint
    /// keeps the whole tower in two of them.
    private var descriptors: [String: Int32] = [:]
    /// The text width the tower's rows are projected to, when the aligner says so.
    public let textHiddenSize: Int?

    /// Whether a snapshot publishes the tower at all, decided from the shard index
    /// (or the shard headers when there is no index) without building a catalog.
    ///
    /// This backs the scan-time `supportsVision` flag, so it has to be cheap AND
    /// honest: a checkpoint whose config advertises a tower but whose vision shards
    /// were never downloaded must not put an attachment button in the composer.
    /// The sample names cover the three families the tower reads — a block, the
    /// aligner, and the marker embeddings, which live in a DIFFERENT shard from the
    /// blocks and are the part a partial download loses first.
    public static func directoryHasTower(_ snapshotDir: URL) -> Bool {
        let required = [
            "vision.blocks.0.attn.wqkv.weight",
            "vision.blocks.0.mlp.w2.weight",
            "aligner.w2.weight",
            "image_start",
            "image_end",
            "image_newline"
        ]
        let weightMap = DeepSeekV41EngramTableLocator.indexWeightMap(in: snapshotDir)
        if !weightMap.isEmpty {
            return required.allSatisfy { !(weightMap[$0] ?? "").isEmpty }
        }
        let shards = DeepSeekV41EngramTableLocator.shardFiles(in: snapshotDir)
        guard !shards.isEmpty else { return false }
        var names: Set<String> = []
        for shard in shards {
            names.formUnion(
                DeepSeekV41EngramTableLocator.headerNames(
                    in: snapshotDir.appendingPathComponent(shard)
                )
            )
        }
        return required.allSatisfy { names.contains($0) }
    }

    public init?(snapshotDir: URL?, catalog: DeepSeekV41WeightCatalog) {
        guard let snapshotDir else { return nil }
        // Every vision-scope slot the catalog classified, minus the scale
        // companions (the tower has none, and a `*scale*` name would be read as a
        // vector and quietly decoded as garbage).
        var wanted: [(role: DeepSeekV41TensorRole, layer: Int?, name: String)] = []
        for slot in catalog.slots where slot.scope == .vision {
            guard !slot.isScale, !slot.ref.name.isEmpty else { continue }
            wanted.append((slot.role, slot.layer, slot.ref.name))
        }
        guard !wanted.isEmpty else { return nil }

        let weightMap = DeepSeekV41EngramTableLocator.indexWeightMap(in: snapshotDir)
        var byShard: [String: [(role: DeepSeekV41TensorRole, layer: Int?, name: String)]] = [:]
        for entry in wanted {
            let shard: String
            if let listed = weightMap[entry.name], !listed.isEmpty {
                shard = listed
            } else if let located = DeepSeekV41EngramTableLocator.shardFilename(
                forTensorNamed: entry.name, in: snapshotDir
            ) {
                shard = located
            } else {
                problems.append("vision tensor '\(entry.name)' is not in any shard")
                continue
            }
            byShard[shard, default: []].append(entry)
        }

        for (shard, entries) in byShard {
            let url = snapshotDir.appendingPathComponent(shard)
            guard let header = Self.readHeader(url) else {
                problems.append(DeepSeekV41VisionWeightError.unreadableHeader(shard).description)
                continue
            }
            let dataStart = 8 + header.length
            for entry in entries {
                guard let object = header.objects[entry.name] else {
                    problems.append("vision tensor '\(entry.name)' is not in \(shard)'s header")
                    continue
                }
                guard let dtype = object["dtype"] as? String,
                      let shape = object["shape"] as? [Int],
                      let ranges = object["data_offsets"] as? [Int],
                      ranges.count == 2, ranges[0] >= 0, ranges[1] >= ranges[0] else {
                    problems.append("vision tensor '\(entry.name)' has an unreadable header entry")
                    continue
                }
                let rows: Int
                let columns: Int
                switch shape.count {
                case 1:
                    rows = shape[0]
                    columns = 1
                case 2:
                    rows = shape[0]
                    columns = shape[1]
                default:
                    problems.append(
                        "vision tensor '\(entry.name)' is rank \(shape.count), which the tower never reads"
                    )
                    continue
                }
                guard Self.bytesPerValue(dtype) != nil else {
                    problems.append(
                        DeepSeekV41VisionWeightError.unsupportedDtype(
                            name: entry.name, dtype: dtype
                        ).description
                    )
                    continue
                }
                locations[Self.key(entry.role, layer: entry.layer)] = VisionTensorLocation(
                    name: entry.name,
                    path: url.path,
                    offset: dataStart + ranges[0],
                    byteLength: ranges[1] - ranges[0],
                    rows: rows,
                    columns: columns,
                    dtype: dtype
                )
            }
        }
        textHiddenSize = locations[Self.key(.visionAlignerW2, layer: nil)].map(\.rows)
        guard !locations.isEmpty else { return nil }
    }

    deinit {
        for descriptor in descriptors.values { close(descriptor) }
    }

    // MARK: Reading

    public func matrix(
        _ role: DeepSeekV41TensorRole,
        layer: Int?,
        expert: Int?
    ) -> DeepSeekV41Matrix? {
        guard let location = locations[Self.key(role, layer: layer)],
              let values = read(location),
              values.count >= location.rows * location.columns else { return nil }
        return DeepSeekV41Matrix(
            rows: location.rows,
            columns: location.columns,
            values: values
        )
    }

    public func vector(_ role: DeepSeekV41TensorRole, layer: Int?) -> [Float]? {
        guard let location = locations[Self.key(role, layer: layer)] else { return nil }
        return read(location)
    }

    /// Whether the catalog published this role at all, so a test can tell a
    /// missing tensor from a decoder that refused the bytes.
    public func publishes(_ role: DeepSeekV41TensorRole, layer: Int? = nil) -> Bool {
        locations[Self.key(role, layer: layer)] != nil
    }

    /// The tensor name behind a role, for diagnostics.
    public func tensorName(_ role: DeepSeekV41TensorRole, layer: Int? = nil) -> String? {
        locations[Self.key(role, layer: layer)]?.name
    }

    public var tensorCount: Int { locations.count }

    private func read(_ location: VisionTensorLocation) -> [Float]? {
        guard let width = Self.bytesPerValue(location.dtype) else {
            problems.append(
                DeepSeekV41VisionWeightError.unsupportedDtype(
                    name: location.name, dtype: location.dtype
                ).description
            )
            return nil
        }
        let count = location.rows * location.columns
        let wanted = count * width
        guard wanted > 0, location.byteLength >= wanted else {
            problems.append(
                "vision tensor '\(location.name)' holds \(location.byteLength) bytes for "
                    + "\(count) values of \(width) bytes"
            )
            return nil
        }
        var bytes = [UInt8](repeating: 0, count: wanted)
        guard let descriptor = descriptor(for: location.path) else {
            problems.append(
                DeepSeekV41VisionWeightError.cannotOpen(location.path).description
            )
            return nil
        }
        var total = 0
        let read = bytes.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) -> Bool in
            guard let base = buffer.baseAddress else { return false }
            while total < wanted {
                let got = pread(
                    descriptor,
                    base.advanced(by: total),
                    wanted - total,
                    off_t(location.offset + total)
                )
                if got < 0 {
                    if errno == EINTR { continue }
                    return false
                }
                if got == 0 { return false }
                total += got
            }
            return true
        }
        guard read else {
            problems.append(
                DeepSeekV41VisionWeightError.shortRead(
                    name: location.name, offset: location.offset, wanted: wanted, got: total
                ).description
            )
            return nil
        }
        return bytes.withUnsafeBytes { Self.decode($0, dtype: location.dtype, count: count) }
    }

    private func descriptor(for path: String) -> Int32? {
        if let existing = descriptors[path] { return existing }
        let opened = open(path, O_RDONLY | O_CLOEXEC)
        guard opened >= 0 else { return nil }
        descriptors[path] = opened
        return opened
    }

    // MARK: Format

    private static func key(_ role: DeepSeekV41TensorRole, layer: Int?) -> String {
        "\(role.rawValue)|\(layer.map(String.init) ?? "-")"
    }

    /// Byte width of one value, or nil for a format the decoder does not read.
    /// The tower is published BF16; FP32 is accepted because a converted
    /// checkpoint is byte-identical in meaning and costs one branch.
    static func bytesPerValue(_ dtype: String) -> Int? {
        let upper = dtype.uppercased()
        if upper.contains("BF16") || upper.contains("FLOAT16") || upper == "F16" {
            return 2
        }
        if upper.contains("F32") || upper.contains("FLOAT32") {
            return 4
        }
        return nil
    }

    /// BF16 and FP32 to host floats. Both are read as integer bits and reproduced
    /// exactly: a bfloat16 is the high half of the float it came from, so widening
    /// it is a shift and never a conversion.
    ///
    /// The widening runs over typed destinations rather than one byte-addressable
    /// load per value, because the published tower is 485 million values and the
    /// fixed cost of an encode is one full widen of all of them.
    static func decode(
        _ bytes: UnsafeRawBufferPointer,
        dtype: String,
        count: Int
    ) -> [Float] {
        let width = bytesPerValue(dtype) ?? 2
        var values = [Float](repeating: 0, count: count)
        guard count > 0, bytes.count >= count * width, let base = bytes.baseAddress else {
            return values
        }
        values.withUnsafeMutableBufferPointer { destination in
            guard let floats = destination.baseAddress else { return }
            floats.withMemoryRebound(to: UInt32.self, capacity: count) { bits in
                if width == 2 {
                    for index in 0..<count {
                        let half = base.loadUnaligned(fromByteOffset: index * 2, as: UInt16.self)
                        bits[index] = UInt32(half) << 16
                    }
                } else {
                    for index in 0..<count {
                        bits[index] = base.loadUnaligned(fromByteOffset: index * 4, as: UInt32.self)
                    }
                }
            }
        }
        return values
    }

    // MARK: Safetensors headers

    struct Header {
        let length: Int
        let objects: [String: [String: Any]]
    }

    /// The JSON header of a safetensors file, whose own length prefix is the first
    /// eight bytes. Never reads tensor data.
    static func readHeader(_ url: URL) -> Header? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else {
            return nil
        }
        let length = Int(lengthData.withUnsafeBytes { $0.load(as: UInt64.self) })
        // Far past any real header; the published checkpoint's are a few hundred
        // kilobytes because they name every tensor in the shard.
        guard length > 0, length < 64 << 20 else { return nil }
        var header = Data()
        while header.count < length {
            guard let piece = try? handle.read(upToCount: length - header.count),
                  !piece.isEmpty else { return nil }
            header.append(piece)
        }
        guard let object = try? JSONSerialization.jsonObject(with: header),
              let root = object as? [String: Any] else { return nil }
        var objects: [String: [String: Any]] = [:]
        for (name, value) in root where name != "__metadata__" {
            if let entry = value as? [String: Any] { objects[name] = entry }
        }
        return Header(length: length, objects: objects)
    }
}
