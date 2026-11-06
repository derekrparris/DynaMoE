//
//  DenseRepacker.swift
//  DynaMoE
//
//  Quantizes a DENSE checkpoint's weights into a contiguous blob the engine can
//  mmap, so a model with no routed experts stops reading BF16 for every token.
//
//  This is NOT `ExpertRepacker`: that one is a byte copier that needs
//  `(layer, expert)` tensors and throws when it finds none. A dense model has no
//  expert axis, so there is nothing to repack around, and the per-token cost is
//  the whole weight set (66 GB for Nanbeige 4.3, i.e. ~7 tok/s at 500 GB/s
//  whatever the chip does). Quantizing is the only lever that moves it.
//
//  Format, read off the kernels rather than invented:
//
//      dense_weights.bin        contiguous logical tensors, in manifest order
//      dense_weights.json       { quantization, group_size, tensors: [...] }
//
//  A quantized tensor is THREE manifest rows, because the Swift side resolves a
//  projection's companions by name substring and its affine path requires both:
//
//      <name>            packed weights, dtype "Q4" | "Q8"
//      <name>.scale      per-group BF16 scales
//      <name>.bias       per-group BF16 biases (the group MINIMUM, not zero: the
//                        kernels compute `sum += scale * dot(q, x) + bias * sum(x)`)
//
//  Layout inside a row is MLX affine, group size 64 (`q4_gemv`/`q8_gemv` index
//  `scale[row * numGroups + col / 64]`):
//
//      Q4  row = inDim/8 little-endian uint32 words, LOW nibble = LOWER column
//      Q8  row = inDim bytes
//
//  `group_size` is 64 and cannot be anything else: it is hardcoded in both
//  kernels and no config field feeds it.
//

import Foundation

public struct DenseRepackManifest: Codable, Sendable {
    public struct TensorEntry: Codable, Sendable {
        public let name: String
        public let dtype: String
        public let shape: [Int]
        public let offset: UInt64
        public let size: UInt64
    }

    public let quantization: String
    /// Optional, and absent reads as "unstated", because the Rust loader
    /// (`try_load_dense_repack`) treats an absent `group_size` the same way and
    /// the two sides decide "is this a repack" between them: a rule only one of
    /// them applies is a directory the app half-believes.
    public let groupSize: UInt64?
    public let tensors: [TensorEntry]
    /// Length of the blob these offsets were written against.
    ///
    /// Bounds alone cannot say the manifest and the blob are the SAME pair: a
    /// Q4 blob and a Q8 blob of the same tensors both contain every Q4 offset,
    /// so a stale manifest over a freshly written blob loads at the wrong
    /// stride and reports the wrong quantization. Optional because manifests
    /// written before this field existed are still readable, and their offsets
    /// are validated exactly as before.
    public let blobSize: UInt64?

    enum CodingKeys: String, CodingKey {
        case quantization
        case groupSize = "group_size"
        case blobSize = "blob_size"
        case tensors
    }

    public init(
        quantization: String,
        groupSize: UInt64?,
        tensors: [TensorEntry],
        blobSize: UInt64? = nil
    ) {
        self.quantization = quantization
        self.groupSize = groupSize
        self.tensors = tensors
        self.blobSize = blobSize
    }
}

public enum DenseRepackerError: LocalizedError {
    case nothingToQuantize
    case notARepackableCheckpoint
    case writeFailed(String)
    case unsupportedTensor(String)
    case alreadyQuantized(String)
    case unreachableTensor(String)

    public var errorDescription: String? {
        switch self {
        case .nothingToQuantize:
            return "No 2-D BF16 tensors wide enough to quantize were found (inDim must be a multiple of 64)."
        case .notARepackableCheckpoint:
            return "This checkpoint cannot take a dense repack: it has routed experts (use FlashMoE Repack instead) or no usable vocabulary."
        case .writeFailed(let detail):
            return "Dense repack write failed: \(detail)"
        case .unsupportedTensor(let name):
            return "Dense repack: \(name) is not BF16, so this checkpoint is already quantized."
        case .alreadyQuantized(let detail):
            return "This checkpoint is already quantized, so packing it again would quantize quantized bytes. \(detail)"
        case .unreachableTensor(let name):
            return "Dense repack: \(name) lives in a shard the engine could not map, so it cannot be copied. Nothing was written."
        }
    }
}

public final class DenseRepacker {
    public static let shared = DenseRepacker()

    /// The one group size the Metal kernels implement.
    public static let groupSize = 64

    /// Every tensor starts on a 16-byte boundary in the blob.
    ///
    /// The packed-weight kernels read their rows as `uchar4` (4-byte) vector
    /// loads and the input as `float4` (16-byte) ones, and Apple GPUs require a
    /// vector load to be naturally aligned: a row that begins mid-word reads
    /// garbage rather than the checkpoint's bytes. A row's start is
    /// `offset + row * inDim`, and `isQuantizable` already requires
    /// `inDim % groupSize == 0`, so aligning each entry's `offset` is what makes
    /// every row aligned (see `ComputeShaders.metal`'s note at the top of
    /// `q8_gemv`). Aligning to 16 rather than 4 covers the `float4` reads too.
    public static let tensorAlignment: UInt64 = 16

    /// Names never quantized: the token embedding is read a row at a time, and the
    /// output head keeps full precision so the logits the sampler reads are the
    /// checkpoint's own. `.scale`/`.bias` are the companions of an ALREADY packed
    /// tensor: a 64-wide companion row would pass `isQuantizable`, and quantizing
    /// it would leave the packed weight with scales of scales.
    static let excludedNameFragments = [
        "embed_tokens", "lm_head", "visual.", "mtp.", "ple.", "ngram_embedding",
        ".scale", ".bias", ".scales",
    ]

    private init() {}

    public enum Quantization: String, CaseIterable, Sendable {
        case q8 = "q8"
        case q4 = "q4"

        public var bits: UInt32 { self == .q4 ? 4 : 8 }
        public var levels: Float { self == .q4 ? 15 : 255 }
        public var displayName: String { self == .q4 ? "Q4 (dense repack)" : "Q8 (dense repack)" }
        public var wireDtype: String { self == .q4 ? "Q4" : "Q8" }
    }

    /// The quantization a repacked directory already carries, from the manifest.
    ///
    /// Deliberately requires the manifest to be COMPLETE (`isRepacked`): an
    /// interrupted repack leaves a manifest describing bytes that are not there,
    /// and answering "Q4 (dense repack)" for it both mislabels the model row and
    /// hides the Quantize action that is the only way to repair it.
    public static func quantizedFormat(dir: URL) -> Quantization? {
        guard isRepacked(dir: dir) else { return nil }
        guard let manifest = manifest(in: dir),
              let quant = Quantization(rawValue: manifest.quantization) else {
            return nil
        }
        return quant
    }

    /// Why a manifest does not describe the blob beside it, or nil when it does.
    ///
    /// The SAME rule exists in the Rust loader (`try_load_dense_repack`,
    /// core/src/lib.rs) and the two must not drift: Swift uses it to decide
    /// whether the model row may say "Q8 (dense repack)" (which also hides the
    /// Quantize action that repairs the directory), and Rust uses it to decide
    /// whether the blob may be loaded at all. A disagreement means one half of
    /// the app believes a repack the other half refuses.
    ///
    /// Every check is here rather than at the call sites so the answers cannot
    /// differ:
    ///
    /// - the blob exists and its length matches the manifest's `blob_size`
    ///   (a Q4 blob and a Q8 blob of the same tensors contain every one of each
    ///   other's offsets, so bounds alone cannot tell a stale manifest from a
    ///   fresh one);
    /// - every entry lies inside the blob, without wrapping;
    /// - every PACKED entry starts on `tensorAlignment`, because the kernels
    ///   read its rows as vector loads, and its row width divides into the
    ///   kernels' 16-byte chunks;
    /// - every packed weight has BOTH companions, because the quantized path
    ///   needs the scale and the bias.
    static func manifestProblem(
        _ manifest: DenseRepackManifest,
        blobBytes: UInt64
    ) -> String? {
        guard !manifest.tensors.isEmpty else { return "the manifest lists no tensors" }
        // The kernels hardcode 64, and an absent value reads as "unstated" (the
        // Rust loader accepts `group_size` 0 the same way).
        if let declared = manifest.groupSize, declared != 0, declared != UInt64(groupSize) {
            return "group_size \(declared) is not \(groupSize)"
        }
        if !manifest.quantization.isEmpty,
           Quantization(rawValue: manifest.quantization) == nil,
           manifest.quantization != "bf16" {
            return "unknown quantization \(manifest.quantization)"
        }
        if let recorded = manifest.blobSize, recorded != blobBytes {
            return "it describes a \(recorded) byte blob but \(blobBytes) bytes are on disk"
        }
        let names = Set(manifest.tensors.map(\.name))
        for entry in manifest.tensors {
            let (end, overflowed) = entry.offset.addingReportingOverflow(entry.size)
            guard !overflowed else { return "\(entry.name) has an overflowing byte range" }
            guard end <= blobBytes else { return "\(entry.name) runs past the end of the blob" }
            guard isPackedWeightDtype(entry.dtype) else { continue }
            guard entry.offset % tensorAlignment == 0 else {
                return "\(entry.name) starts at \(entry.offset), which is not \(tensorAlignment)-byte aligned"
            }
            guard entry.shape.count == 2, entry.shape[1] > 0, entry.shape[1] % 16 == 0 else {
                return "\(entry.name) has a row width the kernels cannot read as 16-byte chunks"
            }
            guard names.contains(entry.name + ".scale"), names.contains(entry.name + ".bias") else {
                return "\(entry.name) is packed \(entry.dtype) with no scale/bias companion"
            }
        }
        return nil
    }

    /// Whether a repack is present AND reads as complete. See
    /// `manifestProblem(_:blobBytes:)` for what "complete" means and why it is
    /// shared with the Rust loader.
    public static func isRepacked(dir: URL) -> Bool {
        guard let manifest = manifest(in: dir) else { return false }
        let blobUrl = dir.appendingPathComponent("dense_weights.bin")
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: blobUrl.path),
              let blobBytes = attributes[.size] as? UInt64 else {
            return false
        }
        return manifestProblem(manifest, blobBytes: blobBytes) == nil
    }

    /// The decoded manifest, or nil when the file is absent or unreadable.
    static func manifest(in dir: URL) -> DenseRepackManifest? {
        let manifestUrl = dir.appendingPathComponent("dense_weights.json")
        guard let data = try? Data(contentsOf: manifestUrl),
              let manifest = try? JSONDecoder().decode(DenseRepackManifest.self, from: data) else {
            return nil
        }
        return manifest
    }

    /// The file names a parked repack keeps, so parking is reversible by hand and
    /// a re-quantize can find what it replaced.
    static let parkedBlobName = "dense_weights.bin.unusable"
    static let parkedManifestName = "dense_weights.json.unusable"

    /// Renames the repack out of the way without deleting it.
    ///
    /// Used both by the GPU-limit path below and by a RE-QUANTIZE, which needs the
    /// checkpoint's own safetensors back: the engine prefers the repack, so while
    /// the files are named `dense_weights.*` a new repack would be handed the
    /// packed rows it is trying to replace.
    @discardableResult
    private static func parkRepackFiles(dir: URL, fileManager: FileManager = .default) -> Bool {
        let blobUrl = dir.appendingPathComponent("dense_weights.bin")
        let manifestUrl = dir.appendingPathComponent("dense_weights.json")
        let parkedBlob = dir.appendingPathComponent(parkedBlobName)
        let parkedManifest = dir.appendingPathComponent(parkedManifestName)
        // Each file separately: an interrupted repack can leave a manifest with no
        // blob (or the reverse), and parking must clear what is there rather than
        // abandon the pair because one half is missing.
        var parkedAny = false
        try? fileManager.removeItem(at: parkedBlob)
        try? fileManager.removeItem(at: parkedManifest)
        if fileManager.fileExists(atPath: blobUrl.path) {
            do {
                try fileManager.moveItem(at: blobUrl, to: parkedBlob)
                parkedAny = true
            } catch {
                print("⚠️ [Dense] could not set the repack blob aside: \(error)")
                return false
            }
        }
        if fileManager.fileExists(atPath: manifestUrl.path) {
            do {
                try fileManager.moveItem(at: manifestUrl, to: parkedManifest)
                parkedAny = true
            } catch {
                print("⚠️ [Dense] could not set the repack manifest aside: \(error)")
                return false
            }
        }
        return parkedAny
    }

    /// Moves a dense repack aside so a new one can be written from the
    /// checkpoint's own shards (the Q8 → Q4 path). Answers false when there is
    /// nothing to park.
    @discardableResult
    public static func setAsideRepackForRequantize(dir: URL) -> Bool {
        parkRepackFiles(dir: dir)
    }

    /// Moves a dense repack aside when this GPU cannot map it, so the loader falls
    /// back to the checkpoint's own safetensors shards.
    ///
    /// A repack is ONE contiguous blob. `segmentShardsForMetal` splits a blob that
    /// is merely large, but a single packed tensor still has to fit one
    /// `MTLBuffer`, and a blob that cannot be mapped leaves its weights
    /// unreachable: the load succeeds, then the first dispatch that touches one of
    /// them throws "weight tensor not found", which names neither the repack nor
    /// the GPU limit. The manifest names every span, so this needs no mapping to
    /// answer.
    @discardableResult
    public static func setAsideRepackIfUnusable(dir: URL, maxBufferLength: UInt64) -> Bool {
        guard maxBufferLength > 0 else { return false }
        let fileManager = FileManager.default
        let blobUrl = dir.appendingPathComponent("dense_weights.bin")
        guard let manifest = manifest(in: dir),
              let attributes = try? fileManager.attributesOfItem(atPath: blobUrl.path),
              let blobBytes = attributes[.size] as? UInt64 else {
            return false
        }
        // 16 KB is the largest Apple Silicon page, so this can only overestimate a
        // span, which is the safe direction for a refusal. A range that does not
        // fit the blob is refused as unusable too: at least one span of a manifest
        // the reader would reject cannot be mapped.
        let page: UInt64 = 16 * 1024
        func alignedSpan(_ entry: DenseRepackManifest.TensorEntry) -> UInt64 {
            let (rawEnd, overflowed) = entry.offset.addingReportingOverflow(entry.size)
            guard !overflowed, rawEnd <= blobBytes else { return UInt64.max }
            let start = entry.offset & ~(page - 1)
            let clipped = min((rawEnd + page - 1) & ~(page - 1), blobBytes)
            return clipped > start ? clipped - start : 0
        }
        guard manifest.tensors.contains(where: { alignedSpan($0) > maxBufferLength }) else {
            return false
        }
        return parkRepackFiles(dir: dir, fileManager: fileManager)
    }

    /// Whether this directory holds a dense BF16 checkpoint that would benefit.
    /// A repacked directory, an MoE checkpoint and a checkpoint whose weights are
    /// already quantized all answer false.
    public static func canRepack(dir: URL) -> Bool {
        guard !isRepacked(dir: dir) else { return false }
        guard let config = ModelConfig.load(from: dir) else { return false }
        return !config.isMoE && config.effectiveVocabSize > 0
    }

    /// Quantizes every eligible tensor and rewrites the checkpoint's weights in
    /// place (`model.safetensors*` stays on disk; the engine prefers the repack
    /// and falls back to the safetensors index if it is ever incomplete).
    public func repackSafetensors(
        sourceDir: URL,
        outputDir: URL,
        quantization: Quantization,
        progress: @escaping (Double, String) -> Void
    ) throws {
        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: outputDir.path) {
            try fileManager.createDirectory(at: outputDir, withIntermediateDirectories: true)
        }
        // `tokenizer_config.json` is what carries the CHAT TEMPLATE the family's
        // prompt renderer is verified against, so a repack into a different
        // directory would otherwise be a checkpoint that lost the thing that says
        // how a turn is shaped.
        let auxFiles = [
            "config.json", "tokenizer_config.json", "tokenizer.json", "tokenizer.model",
            "generation_config.json", "added_tokens.json", "special_tokens_map.json",
            "vocab.bin", "tokenizer.bin"
        ]
        for aux in auxFiles {
            let source = sourceDir.appendingPathComponent(aux)
            let destination = outputDir.appendingPathComponent(aux)
            guard fileManager.fileExists(atPath: source.path),
                  !fileManager.fileExists(atPath: destination.path) else { continue }
            // By BYTES, not by `copyItem`: a hub snapshot's files are relative
            // symlinks into `../../blobs/...`, so copying the link itself leaves a
            // repack in another directory pointing at nothing and the model will
            // not load at all.
            guard let contents = try? Data(contentsOf: source) else { continue }
            try? contents.write(to: destination)
        }

        progress(0.02, "Reading the checkpoint index…")
        let engine = try DynaMoeEngine(filePath: sourceDir.path)
        let summary = try engine.getSummary()

        // The engine PREFERS a dense repack over the safetensors shards, so
        // re-running the quantize on a directory that already holds one hands
        // this function the repack's own Q4/Q8 rows. Copying those verbatim
        // while the manifest is labelled with the REQUESTED format publishes a
        // blob the kernels would read at the wrong stride, and quantizing them
        // again would quantize quantized bytes. Refuse instead: the repair is to
        // move the repack aside, which re-enables the Quantize action.
        if let packed = summary.tensors.first(where: { Self.isPackedWeightDtype($0.dtype) }) {
            throw DenseRepackerError.alreadyQuantized(
                "\(packed.name) is already \(packed.dtype), so re-packing would quantize quantized bytes. Move dense_weights.bin and dense_weights.json aside to re-quantize from the checkpoint's own shards."
            )
        }

        var shardPointers: [UInt32: UnsafeRawPointer] = [:]
        for shard in summary.shards {
            if let pointer = UnsafeRawPointer(bitPattern: UInt(shard.baseAddress)) {
                shardPointers[shard.index] = pointer
            }
        }

        let blobUrl = outputDir.appendingPathComponent("dense_weights.bin")
        // Written beside the target and renamed only once it is whole. Truncating
        // `dense_weights.bin` in place meant an interrupted re-repack (Q4 over an
        // existing Q8, say) left the OLD, longer manifest validating against the
        // NEW, shorter partial blob, i.e. offsets that point into nothing. The
        // manifest is committed last, and atomically, for the same reason.
        let stagingBlobUrl = outputDir.appendingPathComponent("dense_weights.bin.tmp")
        try? fileManager.removeItem(at: stagingBlobUrl)
        guard let blob = fopen(stagingBlobUrl.path, "wb") else {
            throw DenseRepackerError.writeFailed("cannot open dense_weights.bin for writing")
        }
        var blobHandleOpen = true
        defer {
            if blobHandleOpen {
                fclose(blob)
                try? fileManager.removeItem(at: stagingBlobUrl)
            }
        }

        var entries: [DenseRepackManifest.TensorEntry] = []
        var offset: UInt64 = 0
        var quantizedCount = 0

        // Sorted so two runs of the same checkpoint produce the same blob.
        let tensors = summary.tensors.sorted { $0.name < $1.name }
        let quantizable = tensors.filter { Self.isQuantizable($0) }
        guard !quantizable.isEmpty else {
            throw DenseRepackerError.nothingToQuantize
        }
        let quantizableBytes = quantizable.reduce(UInt64(0)) { $0 + ($1.offsetEnd - $1.offsetStart) }
        var quantizedSoFar: UInt64 = 0

        for tensor in tensors {
            // A tensor the engine could not map cannot be copied. Dropping it from
            // the manifest used to publish a model with a missing projection,
            // which loads and silently contributes only its residual, so the
            // whole repack fails instead and names the tensor.
            guard let base = shardPointers[tensor.shardIndex] else {
                throw DenseRepackerError.unreachableTensor(tensor.name)
            }
            let source = base.advanced(by: Int(tensor.offsetStart))
            let shape = ExpertRepacker.parseShapeDisplay(tensor.shapeDisplay)
            // Every entry starts on `tensorAlignment`, which is what keeps the
            // packed kernels' vector loads naturally aligned (see that constant).
            try padToAlignment(&offset, handle: blob)

            guard Self.isQuantizable(tensor), shape.count == 2 else {
                // Copied verbatim: norms, the embedding, the head, and anything
                // already narrow enough that grouping would not divide.
                let bytes = Int(tensor.offsetEnd - tensor.offsetStart)
                guard fwrite(source, 1, bytes, blob) == bytes else {
                    throw DenseRepackerError.writeFailed("short write copying \(tensor.name)")
                }
                entries.append(.init(
                    name: tensor.name,
                    dtype: tensor.dtype,
                    shape: shape,
                    offset: offset,
                    size: UInt64(bytes)
                ))
                offset += UInt64(bytes)
                continue
            }
            guard tensor.dtype.contains("BF16") else {
                throw DenseRepackerError.unsupportedTensor(tensor.name)
            }

            let outDim = shape[0]
            let inDim = shape[1]
            let groups = inDim / Self.groupSize
            let weightBytes = outDim * (quantization == .q4 ? inDim / 2 : inDim)
            let companionBytes = outDim * groups * MemoryLayout<UInt16>.stride
            var scales = [UInt16](repeating: 0, count: outDim * groups)
            var biases = [UInt16](repeating: 0, count: outDim * groups)
            let packed = Self.quantizeMatrix(
                source: source.assumingMemoryBound(to: UInt16.self),
                outDim: outDim,
                inDim: inDim,
                quantization: quantization,
                scales: &scales,
                biases: &biases
            )

            let weightOffset = offset
            try write(packed, to: blob, name: tensor.name)
            offset += UInt64(weightBytes)
            entries.append(.init(
                name: tensor.name,
                dtype: quantization.wireDtype,
                shape: shape,
                offset: weightOffset,
                size: UInt64(weightBytes)
            ))

            let scaleOffset = offset
            try write(bf16Bytes(scales), to: blob, name: tensor.name + ".scale")
            offset += UInt64(companionBytes)
            entries.append(.init(
                name: tensor.name + ".scale",
                dtype: "BF16",
                shape: [outDim, groups],
                offset: scaleOffset,
                size: UInt64(companionBytes)
            ))

            let biasOffset = offset
            try write(bf16Bytes(biases), to: blob, name: tensor.name + ".bias")
            offset += UInt64(companionBytes)
            entries.append(.init(
                name: tensor.name + ".bias",
                dtype: "BF16",
                shape: [outDim, groups],
                offset: biasOffset,
                size: UInt64(companionBytes)
            ))

            quantizedCount += 1
            quantizedSoFar += tensor.offsetEnd - tensor.offsetStart
            let fraction = Double(quantizedSoFar) / Double(max(quantizableBytes, 1))
            progress(
                0.05 + fraction * 0.9,
                String(
                    format: "Quantized %d tensors — %@ (%@) · %@ of %@",
                    quantizedCount,
                    tensor.name,
                    quantization.rawValue,
                    ByteCountFormatter.string(fromByteCount: Int64(offset), countStyle: .file),
                    ByteCountFormatter.string(fromByteCount: Int64(summary.sizeGb * 1_073_741_824), countStyle: .file)
                )
            )
        }

        // Flush and fsync BEFORE the rename: a rename that is durable while the
        // bytes behind it are not would publish a blob the next boot reads short.
        guard fflush(blob) == 0, fsync(fileno(blob)) == 0 else {
            throw DenseRepackerError.writeFailed("cannot flush dense_weights.bin to disk")
        }
        guard fclose(blob) == 0 else {
            throw DenseRepackerError.writeFailed("cannot close dense_weights.bin")
        }
        blobHandleOpen = false
        guard rename(stagingBlobUrl.path, blobUrl.path) == 0 else {
            try? fileManager.removeItem(at: stagingBlobUrl)
            throw DenseRepackerError.writeFailed("cannot move dense_weights.bin into place")
        }

        let manifest = DenseRepackManifest(
            quantization: quantization.rawValue,
            groupSize: UInt64(Self.groupSize),
            tensors: entries,
            blobSize: offset
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        // `.atomic` is the manifest half of the same rule: a half-written manifest
        // is the file every offset is validated against.
        try encoder.encode(manifest).write(
            to: outputDir.appendingPathComponent("dense_weights.json"),
            options: .atomic
        )
        progress(1.0, "Dense repack complete — \(entries.count) manifest rows, \(quantizedCount) tensors quantized.")
    }

    /// The packing itself, separated from the file walk so it can be pinned
    /// without a checkpoint: `scales`/`biases` are filled in place and the packed
    /// rows are returned, exactly as `q4_gemv`/`q8_gemv` address them.
    ///
    /// One pass over the matrix, group at a time: a group is 64 values, so its
    /// minimum and maximum are still in cache when the same 64 values are packed.
    /// The earlier shape computed every scale first and then re-read the tensor,
    /// which doubled the work on a 66 GB checkpoint for no benefit.
    static func quantizeMatrix(
        source: UnsafePointer<UInt16>,
        outDim: Int,
        inDim: Int,
        quantization: Quantization,
        scales: inout [UInt16],
        biases: inout [UInt16]
    ) -> [UInt8] {
        let groups = inDim / groupSize
        precondition(scales.count == outDim * groups && biases.count == outDim * groups)
        let rowBytes = quantization == .q4 ? inDim / 2 : inDim
        var packed = [UInt8](repeating: 0, count: outDim * rowBytes)

        // Rows are independent and disjoint in the destination, so the work splits
        // cleanly. A dense checkpoint is ~38 billion values here; single-threaded
        // Swift carries that past half an hour in an unoptimized build.
        packed.withUnsafeMutableBufferPointer { packedBuffer in
            scales.withUnsafeMutableBufferPointer { scaleBuffer in
                biases.withUnsafeMutableBufferPointer { biasBuffer in
                    let packedBase = packedBuffer.baseAddress!
                    let scaleBase = scaleBuffer.baseAddress!
                    let biasBase = biasBuffer.baseAddress!
                    DispatchQueue.concurrentPerform(iterations: outDim) { row in
                        let rowBase = row * inDim
                        let rowPacked = row * rowBytes
                        for group in 0..<groups {
                            var lowest = Float.greatestFiniteMagnitude
                            var highest = -Float.greatestFiniteMagnitude
                            let groupBase = rowBase + group * groupSize
                            for column in 0..<groupSize {
                                let value = bf16ToFloat(source[groupBase + column])
                                if value < lowest { lowest = value }
                                if value > highest { highest = value }
                            }
                            var scale = (highest - lowest) / quantization.levels
                            if !(scale > 0) { scale = 1 }
                            // The QUANTIZED companions are what the kernel reads
                            // back, so the encode uses them too: a blob encoded
                            // against the unrounded scale is not what the device
                            // will decode.
                            let storedScale = floatToBF16(scale)
                            let storedBias = floatToBF16(lowest)
                            scaleBase[row * groups + group] = storedScale
                            biasBase[row * groups + group] = storedBias
                            let bias = bf16ToFloat(storedBias)
                            let descending = 1 / bf16ToFloat(storedScale)

                            if quantization == .q4 {
                                var word: UInt32 = 0
                                for column in 0..<groupSize {
                                    let value = bf16ToFloat(source[groupBase + column])
                                    let quantized = UInt32(clamp(round((value - bias) * descending), quantization.levels))
                                    // Nibble position WITHIN the word: the word holds
                                    // eight columns, so the shift wraps every 8
                                    // (`column * 4` overflows a UInt32 from column 8
                                    // on and silently drops those columns).
                                    word |= (quantized & 0xF) << UInt32((column % 8) * 4)
                                    if column % 8 == 7 {
                                        let index = rowPacked + group * (groupSize / 2) + (column / 8) * 4
                                        packedBase[index] = UInt8(word & 0xFF)
                                        packedBase[index + 1] = UInt8((word >> 8) & 0xFF)
                                        packedBase[index + 2] = UInt8((word >> 16) & 0xFF)
                                        packedBase[index + 3] = UInt8((word >> 24) & 0xFF)
                                        word = 0
                                    }
                                }
                            } else {
                                for column in 0..<groupSize {
                                    let value = bf16ToFloat(source[groupBase + column])
                                    packedBase[rowPacked + group * groupSize + column] = UInt8(
                                        clamp(round((value - bias) * descending), quantization.levels)
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
        return packed
    }

    // MARK: - Rules and converters

    /// Whether a tensor is worth and able to be quantized: 2-D, BF16, a row width
    /// that divides into both 64-value groups and 8-value Q4 words, and not one of
    /// the names the engine reads specially.
    static func isQuantizable(_ tensor: TensorMetadata) -> Bool {
        guard tensor.expertId == nil else { return false }
        let lowered = tensor.name.lowercased()
        guard !excludedNameFragments.contains(where: { lowered.contains($0) }) else { return false }
        guard tensor.dtype.contains("BF16") else { return false }
        let shape = ExpertRepacker.parseShapeDisplay(tensor.shapeDisplay)
        guard shape.count == 2, shape[0] > 0, shape[1] > 0 else { return false }
        let width = shape[1]
        return width % groupSize == 0 && width % 8 == 0
    }

    static func bf16ToFloat(_ bits: UInt16) -> Float {
        Float(bitPattern: UInt32(bits) << 16)
    }

    /// The dtypes a dense repack writes for a PACKED weight, as opposed to a
    /// tensor it copied verbatim.
    public static func isPackedWeightDtype(_ dtype: String) -> Bool {
        dtype.contains("Q4") || dtype.contains("Q8")
    }

    /// Scale/bias companions of a dense repack, keyed by the tensor they annotate.
    ///
    /// A packed tensor's companions are SEPARATE manifest rows (`<name>.scale`,
    /// `<name>.bias`), but several of the engine's projection call sites pass
    /// `scale: nil, bias: nil` — the MLA and KDA projections and the SSM `in_proj`
    /// trio — because a plain safetensors checkpoint has no companions to pass.
    /// Those sites would otherwise be handed a packed weight with nothing to
    /// dequantize it and silently skip the dispatch, so the companions are resolved
    /// by name once per turn instead of depending on every call site remembering.
    public struct Companions: Sendable {
        public let scale: TensorMetadata?
        public let bias: TensorMetadata?
    }

    public static func companionIndex(_ tensors: [TensorMetadata]) -> [String: Companions] {
        var scales: [String: TensorMetadata] = [:]
        var biases: [String: TensorMetadata] = [:]
        for tensor in tensors {
            if tensor.name.hasSuffix(".scale") {
                scales[String(tensor.name.dropLast(".scale".count))] = tensor
            } else if tensor.name.hasSuffix(".bias") {
                biases[String(tensor.name.dropLast(".bias".count))] = tensor
            }
        }
        var index: [String: Companions] = [:]
        index.reserveCapacity(scales.count + biases.count)
        for key in Set(scales.keys).union(biases.keys) {
            index[key] = Companions(scale: scales[key], bias: biases[key])
        }
        return index
    }

    /// Round-to-nearest-even, which is what `MLX` and numpy both do; the encoder
    /// below is float32 → bfloat16 with ties handled by adding the round bit.
    static func floatToBF16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        if bits & 0x7FFF_FFFF > 0x7F80_0000 { return UInt16(truncatingIfNeeded: bits >> 16) | 0x0040 }
        let rounding = 0x7FFF + ((bits >> 16) & 1)
        return UInt16(truncatingIfNeeded: (bits &+ UInt32(rounding)) >> 16)
    }

    static func clamp(_ value: Float, _ levels: Float) -> Float {
        if !(value > 0) { return 0 }
        if value > levels { return levels }
        return value
    }

    private func write(_ bytes: [UInt8], to handle: UnsafeMutablePointer<FILE>, name: String) throws {
        guard !bytes.isEmpty else { return }
        let written = bytes.withUnsafeBufferPointer { fwrite($0.baseAddress, 1, $0.count, handle) }
        guard written == bytes.count else {
            throw DenseRepackerError.writeFailed("short write for \(name)")
        }
    }

    /// Writes zero bytes up to the next `tensorAlignment` boundary and advances
    /// `offset` by them, so every manifest offset is also the entry's byte
    /// position in the file.
    private func padToAlignment(_ offset: inout UInt64, handle: UnsafeMutablePointer<FILE>) throws {
        let remainder = offset % Self.tensorAlignment
        guard remainder != 0 else { return }
        let padding = Int(Self.tensorAlignment - remainder)
        try write([UInt8](repeating: 0, count: padding), to: handle, name: "alignment padding")
        offset += UInt64(padding)
    }

    private func bf16Bytes(_ values: [UInt16]) -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: values.count * 2)
        for (index, value) in values.enumerated() {
            bytes[index * 2] = UInt8(value & 0xFF)
            bytes[index * 2 + 1] = UInt8(value >> 8)
        }
        return bytes
    }
}
