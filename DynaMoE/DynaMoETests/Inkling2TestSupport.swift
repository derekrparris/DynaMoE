//
//  Inkling2TestSupport.swift
//  DynaMoE
//
//  Helpers the Inkling 2 suites share.
//

import Metal
@testable import DynaMoE

/// The summary the Metal side of a test must use: shards split to fit the GPU.
///
/// The CATALOG has to be built from the same summary the buffers were, because
/// a tensor's shard index and byte offsets are rewritten by segmentation. A
/// catalog from the raw summary addresses the unsplit file, so a tensor past
/// `maxBufferLength` resolves into the wrong segment (or past it).
func inkling2MetalReadySummary(_ summary: ModelSummary) throws -> ModelSummary {
    guard let device = MTLCreateSystemDefaultDevice() else { return summary }
    return try InferenceEngine.segmentShardsForMetal(
        summary, maxBufferLength: UInt64(device.maxBufferLength)
    )
}

/// The checkpoint's shards as Metal buffers, split to fit the GPU.
///
/// A single-file checkpoint is ONE shard of the whole model (18 GB here), far
/// past `maxBufferLength` (8.8 GiB on a 16 GB M1 Pro), and
/// `makeBuffer(bytesNoCopy:)` returns nil for it -- so every tensor would read
/// as "its shard is not mapped". The app's own loader segments first
/// (`InferenceEngine.segmentShardsForMetal`), which rewrites each tensor's
/// shard index and offsets to its segment; a test that maps the raw shard is
/// exercising a mapping the app never makes.
func inkling2MetalShardBuffers(_ summary: ModelSummary, device: MTLDevice) throws -> [UInt32: MTLBuffer] {
    // Idempotent: a caller that already segmented passes a no-op, and one that
    // did not is still safe.
    let segmented = try InferenceEngine.segmentShardsForMetal(
        summary, maxBufferLength: UInt64(device.maxBufferLength)
    )
    var buffers: [UInt32: MTLBuffer] = [:]
    for shard in segmented.shards {
        let length = Int(shard.length)
        guard length > 0, length <= Int(device.maxBufferLength),
              let pointer = UnsafeMutableRawPointer(bitPattern: UInt(shard.baseAddress)) else { continue }
        if let buffer = device.makeBuffer(
            bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil
        ) { buffers[shard.index] = buffer }
    }
    return buffers
}
