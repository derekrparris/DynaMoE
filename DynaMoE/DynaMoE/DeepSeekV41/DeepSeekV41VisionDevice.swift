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
//  attention itself (`dsv41_vision_attention_f32`). It stays a stage runner even
//  though the shipped tower now runs on the device: `DeepSeekV41VisionEncodeDevice`
//  composes this with the projections and the elementwise stages into one pass, and
//  the runner is the unit that can be checked against the host oracle on its own
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
    /// The number of keys one staged tile holds (`DSV41_VISION_STAGED_TILE`).
    public static let stagedTile = 32
    /// The widest dispatch the kernels accept, in SIMD groups (`DSV41_VISION_QUERIES_PER_GROUP`).
    public static let maximumQueriesPerGroup = 16

    /// Which of the two attention kernels a dispatch names.
    ///
    /// `direct` walks each key row in place: a lane owns one key of a 32-key tile and
    /// reads that key's whole row, so a load instruction spreads over 32 cache lines.
    /// `stagedKeyTile` copies the tile into threadgroup memory with a coalesced read
    /// first, which the threadgroup's other SIMD groups then share. They compute the
    /// same rule; only the read pattern differs, which is what the measurement A/Bs.
    ///
    /// `pairedFold` is the one that ships: a lane owns two adjacent dimensions of the
    /// value fold, so its pair is one aligned `float2`, and the tile's additions split
    /// across four accumulator pairs instead of one 32-long dependent chain. It measures
    /// 1.39x at the smallest admissible image and 1.03x at the largest ("DSV4.1 #44"),
    /// which is where `direct` stays as the ablation's baseline. `stagedKeyTile` lost
    /// 18% and is kept only as the measured negative.
    public enum AttentionForm: String, Sendable {
        case direct
        case stagedKeyTile
        case pairedFold
    }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let ropeSplitPipeline: MTLComputePipelineState
    private let attentionPipeline: MTLComputePipelineState
    private let stagedAttentionPipeline: MTLComputePipelineState
    private let pairedFoldPipeline: MTLComputePipelineState

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
        self.stagedAttentionPipeline = try make("dsv41_vision_attention_staged_f32")
        self.pairedFoldPipeline = try make("dsv41_vision_attention_pairfold_f32")
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

    /// Empty planes for a caller that reuses one allocation across every block.
    ///
    /// A device encode runs 32 blocks over the same token grid, and allocating the
    /// three `heads * tokens * headDim` planes per block would churn a hundred
    /// megabytes at the larger image sizes. The planes are pure scratch: the split
    /// overwrites all of them before attention reads them.
    public static func allocatePlanes(
        device: MTLDevice,
        tokens: Int,
        heads: Int,
        headDim: Int
    ) throws -> SplitPlanes {
        let count = max(tokens * heads * headDim, 1) * MemoryLayout<Float>.stride
        guard tokens > 0, heads > 0, headDim > 0 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch("empty vision planes")
        }
        guard let queries = device.makeBuffer(length: count, options: .storageModeShared),
              let keys = device.makeBuffer(length: count, options: .storageModeShared),
              let values = device.makeBuffer(length: count, options: .storageModeShared) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate the vision planes")
        }
        return SplitPlanes(
            queries: queries, keys: keys, values: values,
            tokens: tokens, heads: heads, headDim: headDim
        )
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
        let planes = try Self.allocatePlanes(
            device: device, tokens: tokens, heads: heads, headDim: headDim
        )
        let flat = qkv.flatMap { $0 }
        guard let qkvBuffer = makeFloatBuffer(flat) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate the qkv buffer")
        }
        let rotated = !cosine.isEmpty && !sine.isEmpty && ropeDim > 0
        let angles = rotated ? tokens * ropeDim : 1
        let flatCos: [Float] = rotated ? cosine.flatMap { $0.prefix(ropeDim) } : [0]
        let flatSin: [Float] = rotated ? sine.flatMap { $0.prefix(ropeDim) } : [0]
        guard flatCos.count == angles, flatSin.count == angles else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "angle rows must be \(ropeDim) wide for \(tokens) tokens"
            )
        }
        guard let cosineBuffer = makeFloatBuffer(flatCos),
              let sineBuffer = makeFloatBuffer(flatSin) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate the angle buffers")
        }
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }
        try dispatchSplitAndRotate(
            on: encoder,
            qkv: qkvBuffer, cosine: cosineBuffer, sine: sineBuffer,
            planes: planes, ropeDim: rotated ? ropeDim : 0
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("rope split failed: \(error)")
        }
        return planes
    }

    /// The rotation-and-split dispatch alone, for a caller that owns the command
    /// buffer and the planes.
    public func dispatchSplitAndRotate(
        on encoder: MTLComputeCommandEncoder,
        qkv: MTLBuffer,
        cosine: MTLBuffer,
        sine: MTLBuffer,
        planes: SplitPlanes,
        ropeDim: Int
    ) throws {
        let tokens = planes.tokens
        let heads = planes.heads
        let headDim = planes.headDim
        guard tokens > 0, heads > 0, headDim > 0, headDim % 2 == 0,
              headDim <= Self.maximumHeadDimension else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the split needs positive, even, at most \(Self.maximumHeadDimension)-wide heads"
            )
        }
        guard ropeDim >= 0, ropeDim <= headDim / 2 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "rope dimension \(ropeDim) exceeds half a head of \(headDim)"
            )
        }
        let stride = MemoryLayout<Float>.stride
        guard qkv.length >= tokens * 3 * heads * headDim * stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the qkv buffer is smaller than \(tokens) x \(3 * heads * headDim)"
            )
        }
        let planeBytes = tokens * heads * headDim * stride
        guard planes.queries.length >= planeBytes, planes.keys.length >= planeBytes,
              planes.values.length >= planeBytes else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "a vision plane is smaller than \(tokens) x \(heads) x \(headDim)"
            )
        }
        if ropeDim > 0 {
            let angleBytes = tokens * ropeDim * stride
            guard cosine.length >= angleBytes, sine.length >= angleBytes else {
                throw DeepSeekV41VisionDeviceError.shapeMismatch(
                    "the rotation needs one \(ropeDim)-wide angle row per token"
                )
            }
        }
        var tokenCount = UInt32(tokens)
        var headCount = UInt32(heads)
        var dimension = UInt32(headDim)
        var rotationWidth = UInt32(ropeDim)
        encoder.setComputePipelineState(ropeSplitPipeline)
        encoder.setBuffer(qkv, offset: 0, index: 0)
        encoder.setBuffer(cosine, offset: 0, index: 1)
        encoder.setBuffer(sine, offset: 0, index: 2)
        encoder.setBuffer(planes.queries, offset: 0, index: 3)
        encoder.setBuffer(planes.keys, offset: 0, index: 4)
        encoder.setBuffer(planes.values, offset: 0, index: 5)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.stride, index: 7)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 8)
        encoder.setBytes(&rotationWidth, length: MemoryLayout<UInt32>.stride, index: 9)
        encoder.dispatchThreads(
            MTLSize(width: max(headDim / 2, 1), height: tokens * heads, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
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
        return try attention(planes, into: outputBuffer)
    }

    /// The same readback over a caller-owned buffer and a named form, for a caller that
    /// is comparing two forms on one set of planes.
    public func attention(
        _ planes: SplitPlanes,
        into outputBuffer: MTLBuffer,
        form: AttentionForm = .pairedFold
    ) throws -> [[Float]] {
        _ = try runAttention(planes, into: outputBuffer, form: form)
        let hidden = planes.heads * planes.headDim
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
    public func runAttention(
        _ planes: SplitPlanes,
        into outputBuffer: MTLBuffer,
        form: AttentionForm = .pairedFold,
        threadgroupWidth: Int = DeepSeekV41VisionAttentionDevice.threadgroupWidth
    ) throws -> Double {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }
        do {
            try dispatchAttention(
                on: encoder, planes: planes, into: outputBuffer,
                form: form, threadgroupWidth: threadgroupWidth
            )
        } catch {
            // A refused dispatch must still close its encoder: Metal asserts when one is
            // released without `endEncoding`, and that assertion is a Debug-build abort
            // rather than an error a caller can handle.
            encoder.endEncoding()
            throw error
        }
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("attention failed: \(error)")
        }
        return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    /// The attention dispatch alone, for a caller that owns the command buffer.
    ///
    /// `form` and `threadgroupWidth` are the two shape knobs the measurement sweeps;
    /// the shipped tower takes both defaults.
    public func dispatchAttention(
        on encoder: MTLComputeCommandEncoder,
        planes: SplitPlanes,
        into outputBuffer: MTLBuffer,
        form: AttentionForm = .pairedFold,
        threadgroupWidth: Int = DeepSeekV41VisionAttentionDevice.threadgroupWidth
    ) throws {
        let tokens = planes.tokens
        let heads = planes.heads
        let headDim = planes.headDim
        guard outputBuffer.length >= tokens * heads * headDim * MemoryLayout<Float>.stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the output buffer is smaller than \(tokens) x \(heads) x \(headDim)"
            )
        }
        guard threadgroupWidth % 32 == 0,
              threadgroupWidth / 32 >= 1,
              threadgroupWidth / 32 <= Self.maximumQueriesPerGroup else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "a threadgroup of \(threadgroupWidth) is not 1...\(Self.maximumQueriesPerGroup) SIMD groups"
            )
        }
        let keyStride = headDim + 1
        var tileStride = UInt32(keyStride)
        var tokenCount = UInt32(tokens)
        var headCount = UInt32(heads)
        var dimension = UInt32(headDim)
        switch form {
        case .direct:
            encoder.setComputePipelineState(attentionPipeline)
        case .pairedFold:
            guard headDim % 2 == 0 else {
                throw DeepSeekV41VisionDeviceError.shapeMismatch(
                    "the paired fold owns two adjacent dimensions, so head dimension \(headDim) must be even"
                )
            }
            encoder.setComputePipelineState(pairedFoldPipeline)
        case .stagedKeyTile:
            let tileBytes = Self.stagedTile * keyStride * MemoryLayout<Float>.stride
            guard tileBytes <= device.maxThreadgroupMemoryLength else {
                throw DeepSeekV41VisionDeviceError.shapeMismatch(
                    "a \(Self.stagedTile) x \(keyStride) key tile needs \(tileBytes) bytes of threadgroup memory"
                )
            }
            encoder.setComputePipelineState(stagedAttentionPipeline)
            encoder.setThreadgroupMemoryLength(tileBytes, index: 0)
            encoder.setBytes(&tileStride, length: MemoryLayout<UInt32>.stride, index: 7)
        }
        encoder.setBuffer(planes.queries, offset: 0, index: 0)
        encoder.setBuffer(planes.keys, offset: 0, index: 1)
        encoder.setBuffer(planes.values, offset: 0, index: 2)
        encoder.setBuffer(outputBuffer, offset: 0, index: 3)
        encoder.setBytes(&tokenCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&headCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&dimension, length: MemoryLayout<UInt32>.stride, index: 6)
        let queriesPerGroup = threadgroupWidth / 32
        let queryBlocks = (tokens + queriesPerGroup - 1) / queriesPerGroup
        encoder.dispatchThreadgroups(
            MTLSize(width: heads * max(queryBlocks, 1), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1)
        )
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
