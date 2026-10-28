//
//  DeepSeekV41MemoryPlan.swift
//  DynaMoE
//
//  Where a DeepSeek-V4.1 turn's bytes go.
//
//  This model is unusual in that the KV cache is NOT the constraint: the two
//  compressed caches are 288 bytes per token per owner, so a million-token
//  context is under 600 MB. What costs memory is everything else:
//
//    * the dense backbone (attention, shared experts, compressors, indexers,
//      norms, head) which must stay resident because it is read every token,
//    * 384 routed experts per layer, of which 6 are used per token, so an arena
//      of a few dozen resident experts plus SSD streaming covers the rest,
//    * the Engram tables, ~100 GB each, which are NEVER resident: only a
//      double-buffered staging block of a few dozen rows is.
//
//  Getting this arithmetic wrong is how a 510 GB checkpoint appears to "fit" in
//  128 GB and then pages itself to death, so it is computed once, from the
//  config and the tensor catalog, before anything is allocated.
//

import Foundation

nonisolated public enum DeepSeekV41DtypeWidth {
    /// Bytes per stored value, including the scale payload:
    ///   E4M3 with one E8M0 per 32  -> 1 + 1/32,
    ///   E2M1 (MXFP4) packed two per byte with one E8M0 per 32 -> 0.5 + 1/32,
    ///   plain 8-bit integers -> 1,
    ///   BF16 / F16 -> 2, F32 -> 4, F64 -> 8.
    public static func bytesPerValue(dtype: String) -> Double {
        let upper = dtype.uppercased()
        if upper.contains("E2M1") || upper.contains("MXFP4") || upper == "FP4" || upper.hasPrefix("F4") {
            return 0.5 + 1.0 / 32.0
        }
        if upper.contains("F8_E4M3") || upper.contains("F8_E5M2") || upper == "FP8" {
            return 1.0 + 1.0 / 32.0
        }
        if upper.contains("F8_E8M0") {
            return 1.0
        }
        if upper == "I8" || upper.contains("U8") || upper.contains("INT8") || upper == "BOOL" {
            return 1.0
        }
        if upper == "I64" || upper == "U64" || upper.contains("F64") {
            return 8.0
        }
        if upper.contains("BF16") || upper.contains("F16") || upper.contains("FLOAT16") {
            return 2.0
        }
        if upper.contains("F32") || upper.contains("FLOAT32") {
            return 4.0
        }
        return 2.0
    }

    public static func byteCount(shape: [Int], dtype: String) -> Int {
        let values = shape.reduce(1) { $0 * max(1, $1) }
        return Int((Double(values) * bytesPerValue(dtype: dtype)).rounded(.up))
    }
}

nonisolated public struct DeepSeekV41MemoryPlan: Sendable, Equatable {
    public let contextLength: Int
    public let useFP4MainCache: Bool
    /// Compressed K/V + indexer state for every cache-owning layer.
    public let compressedCacheBytes: Int
    /// Window rings: one per layer, each `window_size` records.
    public let windowRingBytes: Int
    /// Indexer keys/weights for the layers that own an indexer.
    public let indexerBytes: Int
    /// Candidate block scores (per token, per candidate source layer).
    public let candidateBytes: Int
    /// One resident expert of every routed matrix, at MXFP4 width.
    public let bytesPerExpert: Int
    /// Dense, non-expert, non-streamed weights, from the tensor catalog.
    public let residentWeightBytes: Int
    /// Double-buffered Engram staging: one block per Engram layer.
    public let engramStagingBytes: Int

    /// Bytes the caches need at this context length.
    public var cacheBytes: Int { compressedCacheBytes + windowRingBytes + indexerBytes + candidateBytes }

    /// Bytes needed with `expertSlots` resident (layer, expert) pairs, in the one
    /// global arena, plus the Engram staging block. `expertSlots` is a total, not
    /// a per-layer count: the arena is a single LRU pool.
    public func requiredBytes(expertSlots: Int) -> Int {
        let slotCount = max(1, expertSlots)
        return cacheBytes + residentWeightBytes + slotCount * bytesPerExpert + engramStagingBytes
    }

    /// One-line load-log summary.
    public func describe(expertSlots: Int) -> String {
        func gb(_ bytes: Int) -> String {
            String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
        }
        return [
            "dense \(gb(residentWeightBytes))",
            "experts \(expertSlots)×\(gb(bytesPerExpert)) = \(gb(expertSlots * bytesPerExpert))",
            "kv \(gb(cacheBytes)) @\(contextLength) tok",
            "engram staging \(gb(engramStagingBytes))",
            "total \(gb(requiredBytes(expertSlots: expertSlots)))"
        ].joined(separator: ", ")
    }

    /// Builds the plan. `residentWeightBytes` comes from the catalog so a
    /// checkpoint that ships extra dense tensors is still counted exactly.
    public static func make(
        config: DeepSeekV41Config,
        plan: [DeepSeekV41LayerPlan],
        catalog: DeepSeekV41WeightCatalog?,
        contextLength: Int,
        useFP4MainCache: Bool = true,
        engramStagingRows: Int = 64,
        recordBytesOverride: Int? = nil
    ) -> DeepSeekV41MemoryPlan {
        // `recordBytesOverride` lets the runtime price the cache it actually
        // allocated (FP16, while the FP8 kernels are unverified) instead of the
        // design width, so the reported total never flatters the allocation.
        let recordBytes = recordBytesOverride
            ?? (useFP4MainCache ? config.fp4CacheRecordBytes : config.fp8CacheRecordBytes)
        let owners = plan.filter(\.ownsCompressedKV).count
        let compressed = owners * recordBytes * max(0, contextLength)

        // Window rings are per layer and bounded by window_size, so they do not
        // grow with the context. Priced at `recordBytes`, the same width the
        // compressed cache uses, so an FP16 runtime is not billed at the FP8
        // design width.
        let ringRecords = max(0, config.windowSize)
        let windowRings = plan.count * ringRecords * recordBytes

        // The indexer keeps one key per COMPRESSED row. Its width comes from
        // the checkpoint's own `attn.indexer.wk` shape when the catalog has it
        // (the probe prints it), because widening this by `index_n_heads` would
        // turn a ~250 MB cache into ~8 GB. The candidate pass keeps one score
        // per candidate block per token.
        let defaultKeyWidth = config.indexHeadDim
        var indexerBytes = 0
        for layer in plan where layer.ownsCompressedIndex {
            let rows = max(0, contextLength) / max(1, layer.compressRatio)
            let keyWidth = catalog?.tensor(.indexerWk, layer: layer.index)?.shape.first
                ?? defaultKeyWidth
            indexerBytes += rows * keyWidth * MemoryLayout<UInt16>.stride
        }
        let blockSize = max(1, config.candidateBlockSize)
        let candidateBytes = config.candidateSourceLayerID >= 0
            ? (max(0, contextLength) / blockSize) * MemoryLayout<Float>.stride
            : 0

        let expertValues = 3 * config.hiddenSize * max(1, config.moeIntermediateSize)
        // The routed experts are MXFP4, named explicitly so a genuinely 8-bit
        // resident tensor is not priced at the packed FP4 rate.
        let bytesPerExpert = Int(
            (Double(expertValues) * DeepSeekV41DtypeWidth.bytesPerValue(dtype: "MXFP4")).rounded(.up)
        )

        let engramRowBytes = config.engramTables.reduce(0) { $0 + $1.rowBytes }
        let staging = max(1, engramStagingRows) * engramRowBytes * 2  // double buffered

        var resident = 0
        if let catalog {
            for slot in catalog.slots where !slot.isScale {
                if slot.role.isStreamed { continue }
                if slot.expert != nil { continue }
                resident += DeepSeekV41DtypeWidth.byteCount(shape: slot.ref.shape, dtype: slot.ref.dtype)
            }
        }

        return DeepSeekV41MemoryPlan(
            contextLength: contextLength,
            useFP4MainCache: useFP4MainCache,
            compressedCacheBytes: compressed,
            windowRingBytes: windowRings,
            indexerBytes: indexerBytes,
            candidateBytes: candidateBytes,
            bytesPerExpert: bytesPerExpert,
            residentWeightBytes: resident,
            engramStagingBytes: staging
        )
    }
}
