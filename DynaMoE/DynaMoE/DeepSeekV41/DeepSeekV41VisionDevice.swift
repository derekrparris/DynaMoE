//
//  DeepSeekV41VisionDevice.swift
//  DynaMoE
//
//  Phase: DeepSeek-V4.1-Flash port -- the vision tower's device attention.
//
//  The tower's quadratic term is its bidirectional attention: 8,649 patch tokens is
//  74.8 million (query, key) pairs per head per block, and the host path keeps a
//  `tokens * tokens` score block per head while it computes them. That block is what
//  makes the largest admissible image cost 41.71 s on the app's own -O build, 76% of
//  it in attention plus softmax ("DSV4.1 #37").
//
//  This file is the device half of that stage and only that stage: the rotation and
//  head split that feed it (`dsv41_vision_rope_split_f32`), and the online-softmax
//  attention itself (`dsv41_vision_attention_f32`). The rest of the tower -- patch
//  embed, the block projections, pixel-unshuffle, the aligner -- still runs on the
//  host, which is why this is a stage runner rather than a second `encode`. Keeping
//  it separate is what lets the numerical rule be checked against the host oracle
//  before any of the surrounding pipeline moves.
//

import Foundation
import Metal

nonisolated public enum DeepSeekV41VisionDeviceError: Error, Equatable, CustomStringConvertible {
    case unavailable(String)
    case shapeMismatch(String)

    public var description: String {
        switch self {
        case .unavailable(let detail):
            return "vision device path unavailable: \(detail)"
        case .shapeMismatch(let detail):
            return "vision device shape mismatch: \(detail)"
        }
    }
}

/// The device form of the tower's attention stage.
///
/// `split` turns one `3 * hidden` row per token into three `[heads, tokens, headDim]`
/// planes, rotating q and k with the tower's split-half rule; `attention` runs the
/// quadratic stage over those planes. They are separate calls because the host path
/// rotates per token and attends per head, and a measurement of the quadratic term
/// has to be a measurement of the second half alone.
nonisolated public final class DeepSeekV41VisionAttentionDevice {

    /// One SIMD group per query token, four to a threadgroup.
    public static let threadgroupWidth = 128
    /// A lane owns two output dimensions, so this is the widest head the kernel takes.
    public static let maximumHeadDimension = 64

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let ropeSplitPipeline: MTLComputePipelineState
    private let attentionPipeline: MTLComputePipelineState

    public init(device: MTLDevice? = nil, library: MTLLibrary? = nil) throws {
        guard let device = device ?? MTLCreateSystemDefaultDevice() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no Metal device")
        }
        guard let library = library ?? device.makeDefaultLibrary() else {
            throw DeepSeekV41VisionDeviceError.unavailable("the default Metal library is missing")
        }
        guard let queue = device.makeCommandQueue() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no Metal command queue")
        }
        func make(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw DeepSeekV41VisionDeviceError.unavailable("kernel \(name) is missing")
            }
            return try device.makeComputePipelineState(function: function)
        }
        self.device = device
        self.queue = queue
        self.ropeSplitPipeline = try make("dsv41_vision_rope_split_f32")
        self.attentionPipeline = try make("dsv41_vision_attention_f32")
    }

    /// q, k and v for every head, `[heads, tokens, headDim]`, ready for `attention`.
    nonisolated public final class SplitPlanes {
        public let queries: MTLBuffer
        public let keys: MTLBuffer
        public let values: MTLBuffer
        public let tokens: Int
        public let heads: Int
        public let headDim: Int

        init(
            queries: MTLBuffer,
            keys: MTLBuffer,
            values: MTLBuffer,
            tokens: Int,
            heads: Int,
            headDim: Int
        ) {
            self.queries = queries
            self.keys = keys
            self.values = values
            self.tokens = tokens
            self.heads = heads
            self.headDim = headDim
        }
    }

    /// Rotates q and k and splits the interleaved row into head-major planes.
    ///
    /// `cosine` and `sine` carry one row of `ropeDim` angles per token, exactly as
    /// `DeepSeekV41VisionForward.rotaryTables` emits them. An empty pair of tables
    /// means "no rotation", which is the host path's own rule for a missing table.
    public func splitAndRotate(
        qkv: [[Float]],
        heads: Int,
        headDim: Int,
        ropeDim: Int,
        cosine: [[Float]],
        sine: [[Float]]
    ) throws -> SplitPlanes {
        let tokens = qkv.count
        try validate(
            tokens: tokens, heads: heads, headDim: headDim, ropeDim: ropeDim,
            cosineCount: cosine.count, sineCount: sine.count
        )
        let hidden = heads * headDim
        let rowWidth = 3 * hidden
        guard qkv.allSatisfy({ $0.count == rowWidth }) else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "every qkv row must be \(rowWidth) wide"
            )
        }
        let flat = qkv.flatMap { $0 }
        let rotated = !cosine.isEmpty && !sine.isEmpty && ropeDim > 0
        let angles = rotated ? tokens * ropeDim : 1
        let flatCos: [Float] = rotated ? cosine.flatMap { $0.prefix(ropeDim) } : [0]
        let flatSin: [Float] = rotated ? sine.flatMap { $0.prefix(ropeDim) } : [0]
        guard flatCos.count == angles, flatSin.count == angles else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "angle rows must be \(ropeDim) wide for \(tokens) tokens"
            )
        }
        let planeCount = heads * tokens * headDim

        guard let qkvBuffer = makeFloatBuffer(flat),
              let cosineBuffer = makeFloatBuffer(flatCos),
              let sineBuffer = makeFloatBuffer(flatSin),
              let queryBuffer = device.makeBuffer(
                  length: max(planeCount, 1) * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let keyBuffer = device.makeBuffer(
                  length: max(planeCount, 1) * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ),
              let valueBuffer = device.makeBuffer(
                  length: max(planeCount, 1) * MemoryLayout<Float>.stride,
                  options: .storageModeShared
              ) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate the vision planes")
        }

        var tokenCount = UInt32(tokens)
        var headCount = UInt32(heads)
        var dimension = UInt32(headDim)
        var rotationWidth = UInt32(rotated ? ropeDim : 0)
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }
        encoder.setComputePipelineState(ropeSplitPipeline)
        encoder.setBuffer(qkvBuffer, offset: 0, index: 0)
        encoder.setBuffer(cosineBuffer, offset: 0, index: 1)
        encoder.setBuffer(sineBuffer, offset: 0, index: 2)
        encoder.setBuffer(queryBuffer, offset: 0, index: 3)
        encoder.setBuffer(keyBuffer, offset: 0, index: 4)
        encoder.setBuffer(valueBuffer, offset: 0, index: 5)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&rotationWidth, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: max(headDim / 2, 1), height: tokens * heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("rope split failed: \(error)")
        }

        return SplitPlanes(
            queries: queryBuffer,
            keys: keyBuffer,
            values: valueBuffer,
            tokens: tokens,
            heads: heads,
            headDim: headDim
        )
    }

    /// The tower's attention over prepared planes, one row per token of
    /// `heads * headDim` values.
    public func attention(_ planes: SplitPlanes) throws -> [[Float]] {
        let hidden = planes.heads * planes.headDim
        let outputCount = max(planes.tokens * hidden, 1)
        guard let outputBuffer = device.makeBuffer(
            length: outputCount * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate the attention output")
        }
        _ = try runAttention(planes, into: outputBuffer)
        let output = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        return (0..<planes.tokens).map { token in
            Array(UnsafeBufferPointer(
                start: output + token * hidden, count: hidden
            ))
        }
    }

    /// Runs one attention dispatch into a caller-owned output buffer and returns its
    /// GPU time, without the readback.
    ///
    /// The readback is not part of the quadratic stage: at the largest image the
    /// output is `tokens * hidden` floats, and materialising it as `[[Float]]` costs
    /// more than the dispatch it describes. A measurement of this stage times this
    /// call; the rows are read separately when a test needs to compare them.
    @discardableResult
    public func runAttention(_ planes: SplitPlanes, into outputBuffer: MTLBuffer) throws -> Double {
        let tokens = planes.tokens
        let heads = planes.heads
        let headDim = planes.headDim
        guard outputBuffer.length >= tokens * heads * headDim * MemoryLayout<Float>.stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the output buffer is smaller than \(tokens) x \(heads) x \(headDim)"
            )
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }
        var tokenCount = UInt32(tokens)
        var headCount = UInt32(heads)
        var dimension = UInt32(headDim)
        encoder.setComputePipelineState(attentionPipeline)
        encoder.setBuffer(planes.queries, offset: 0, index: 0)
        encoder.setBuffer(planes.keys, offset: 0, index: 1)
        encoder.setBuffer(planes.values, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        let queriesPerGroup = Self.threadgroupWidth / 32
        let queryBlocks = (tokens + queriesPerGroup - 1) / queriesPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: heads * max(queryBlocks, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("attention failed: \(error)")
        }
        return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    /// Rotation, split and attention in one call, the shape the host oracle's
    /// `attention(qkv:)` takes.
    public func attention(
        qkv: [[Float]],
        heads: Int,
        headDim: Int,
        ropeDim: Int,
        cosine: [[Float]],
        sine: [[Float]]
    ) throws -> [[Float]] {
        let planes = try splitAndRotate(
            qkv: qkv,
            heads: heads,
            headDim: headDim,
            ropeDim: ropeDim,
            cosine: cosine,
            sine: sine
        )
        return try attention(planes)
    }

    private func validate(
        tokens: Int,
        heads: Int,
        headDim: Int,
        ropeDim: Int,
        cosineCount: Int,
        sineCount: Int
    ) throws {
        guard tokens > 0, heads > 0 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch("no tokens or heads")
        }
        guard headDim > 0, headDim <= Self.maximumHeadDimension, headDim % 2 == 0 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "head dimension \(headDim) must be even and at most \(Self.maximumHeadDimension)"
            )
        }
        guard ropeDim >= 0, ropeDim <= headDim / 2 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "rope dimension \(ropeDim) exceeds half a head of \(headDim)"
            )
        }
        if ropeDim > 0 {
            guard cosineCount == tokens, sineCount == tokens else {
                throw DeepSeekV41VisionDeviceError.shapeMismatch(
                    "the rotation needs one angle row per token"
                )
            }
        }
    }

    private func makeFloatBuffer(_ values: [Float]) -> MTLBuffer? {
        let length = max(values.count, 1) * MemoryLayout<Float>.stride
        return values.withUnsafeBytes { bytes in
            guard let base = bytes.baseAddress else {
                return device.makeBuffer(length: length, options: .storageModeShared)
            }
            return device.makeBuffer(bytes: base, length: max(bytes.count, 4), options: .storageModeShared)
        }
    }
}
