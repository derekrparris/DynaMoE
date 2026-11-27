//
//  Inkling2MetalForward.swift
//  DynaMoE
//
//  The Inkling 2 Mini device graph: the same 48-visit schedule the host
//  reference runs (`Inkling2Forward.swift`), expressed as Metal dispatches over
//  the checkpoint's own zero-copy shard buffers.
//
//  Built STAGE BY STAGE against the host graph, not ported wholesale: the host
//  graph is what matches the checkpoint's Transformers reference, so each
//  device stage is compared with the host's value for the same stage before the
//  next one is written. A whole-graph port that is 2% off somewhere is a week of
//  bisecting; a stage that is 2% off the moment it is written names itself.
//
//  This file starts with the weight binding and the attention half of one
//  visit, because that half is where a mistake is most likely (relative bias,
//  sink, grouped KV heads, two different head geometries) and it reads the
//  cache the position writes.
//
//  Layout rule, inherited from the DeepSeek executor: every "constant" is a
//  small buffer filled by the host, never a shader literal, so one pipeline
//  serves every layer and shape.
//

import Accelerate
import Foundation
import Metal

// MARK: - Weight binding

/// Resolves a catalog role to the buffer and byte offset a kernel binds.
///
/// The shards are the app's existing zero-copy mappings, one `MTLBuffer` per
/// shard index, so a weight is `(buffer, offset)` and never a copy. A role that
/// is missing, or whose shard is not mapped, is an error that names it -- the
/// host weight source fails closed the same way.
nonisolated public final class Inkling2DeviceWeights {
    /// One expert's slice of a fused bank, as a kernel addresses it: the bank's
    /// buffer and base offset, plus this expert's own byte offset, row strides
    /// and outer scale.
    ///
    /// A tuple because every dispatch site binds the fields positionally; the
    /// alias exists so the mixture can carry eight of them through a resolution
    /// pass before encoding anything.
    public typealias ExpertBank = (
        buffer: MTLBuffer, baseOffset: UInt64, expertOffset: UInt64,
        rowStride: Int, scaleBuffer: MTLBuffer, scaleBaseOffset: UInt64,
        scaleExpertOffset: UInt64, scaleRowStride: Int,
        layout: Inkling2ExpertQuantLayout, amax: Float, denominator: Float,
        fileWeightOffset: UInt64, fileScaleOffset: UInt64
    )

    public let config: Inkling2Config
    public let catalog: Inkling2WeightCatalog
    private let shardBuffers: [UInt32: MTLBuffer]
    /// Where each shard's BUFFER starts inside its file.
    ///
    /// A buffer's offsets are relative to the buffer, and after segmentation
    /// that is not the file: a reader that opens the file itself (the expert
    /// arena's `pread`) needs the origin added back, or it reads a plausible
    /// range of the wrong bytes. Empty means "buffers are whole shards".
    private let fileOrigins: [UInt32: UInt64]
    /// Dequantized FP32 matrices, keyed by role and layer.
    ///
    /// The small vectors this graph needs as VALUES (the relative bank, the
    /// convolution taps, the clip scales) are read from the shards on the host,
    /// and a visit asks for the same ones on every position: without this, one
    /// turn re-decoded them 48 times per token. Guarded because the executor
    /// runs on a turn's own task while another turn can be preparing.
    private let cacheLock = NSLock()
    private var matrixCache: [String: [Float]] = [:]

    public init(
        config: Inkling2Config, catalog: Inkling2WeightCatalog, shardBuffers: [UInt32: MTLBuffer],
        fileOrigins: [UInt32: UInt64] = [:]
    ) {
        self.config = config
        self.catalog = catalog
        self.shardBuffers = shardBuffers
        self.fileOrigins = fileOrigins
    }

    /// One tensor's buffer and byte offset, for a kernel's `[[buffer(n)]]` plus
    /// its `weightOffset` constant.
    public func tensor(
        _ role: Inkling2TensorRole, layer: Int? = nil
    ) throws -> (buffer: MTLBuffer, offset: UInt64) {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight(
                "\(role.rawValue)\(layer.map { " layer \($0)" } ?? "")"
            )
        }
        guard let shard = ref.shardIndex, let buffer = shardBuffers[shard] else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): its shard is not mapped")
        }
        guard let offset = ref.byteOffset else {
            throw Inkling2ForwardError.missingWeight("\(ref.name): no byte offset")
        }
        return (buffer, offset)
    }

    /// A role's stored shape, straight from the catalog.
    public func shape(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> [Int] {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight("\(role.rawValue)")
        }
        return ref.shape
    }

    /// A weight as the host graph sees it: dequantized FP32. Used only for the
    /// small vectors the kernels need as VALUES (the relative bank, clip
    /// scales), never for a stage the port exists to move -- those read the
    /// stored bytes directly.
    public func hostMatrix(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> [Float] {
        let key = "\(role.rawValue)|\(layer.map(String.init) ?? "-")"
        cacheLock.lock()
        if let cached = matrixCache[key] {
            cacheLock.unlock()
            return cached
        }
        cacheLock.unlock()
        let matrix = try Inkling2SnapshotWeights(
            config: config, catalog: catalog, shardBuffers: shardBuffers
        ).matrix(role, layer: layer)
        cacheLock.lock()
        matrixCache[key] = matrix
        cacheLock.unlock()
        return matrix
    }

    public func dtype(_ role: Inkling2TensorRole, layer: Int? = nil) throws -> String {
        guard let ref = catalog.tensor(role, layer: layer) else {
            throw Inkling2ForwardError.missingWeight("\(role.rawValue)")
        }
        return ref.dtype
    }

    /// Whether a shard's position inside its FILE is known.
    ///
    /// A reader that opens the file itself -- the expert arena's `pread` -- must
    /// have it: segmentation rewrites a tensor's offset to be segment-relative,
    /// so an empty origin set means "these buffers are whole shards", which a
    /// segmented model is not. A caller that segmented has to pass
    /// `fileOrigins`; an arena installed without one would fill its slots with
    /// plausible wrong bytes and report hits over them.
    public func knowsFileOrigin(of shard: UInt32) -> Bool {
        fileOrigins[shard] != nil
    }

    /// One expert's slice of a FUSED bank, resolved.
    ///
    /// The catalog holds a bank whole (`[experts][rows][columns]`), so expert
    /// `e` starts at `e * rows * columns * width`; a tensor offset inside a
    /// shard is not 256-byte aligned, so every one of these travels as a
    /// `constant` and the kernel adds it. The layout is inferred from the
    /// published dtypes exactly as the host source infers it -- a bank read as
    /// the wrong layout dequantizes to plausible numbers, not to a crash.
    public func expertBank(
        _ role: Inkling2TensorRole, layer: Int, expert: Int
    ) throws -> ExpertBank {
        let scaleRole: Inkling2TensorRole = role == .expertW13 ? .expertW13Scale : .expertW2Scale
        let ref = catalog.tensor(role, layer: layer)
        let scaleRef = catalog.tensor(scaleRole, layer: layer)
        guard let bank = ref, let scale = scaleRef, bank.shape.count == 3, scale.shape.count == 3,
              expert >= 0, expert < bank.shape[0] else {
            throw Inkling2ForwardError.missingWeight("\(role.rawValue) layer \(layer) expert \(expert)")
        }
        guard let shard = bank.shardIndex, let buffer = shardBuffers[shard],
              let baseOffset = bank.byteOffset else {
            throw Inkling2ForwardError.missingWeight("\(bank.name): its shard is not mapped")
        }
        guard let scaleShard = scale.shardIndex, let scaleBuffer = shardBuffers[scaleShard],
              let scaleBaseOffset = scale.byteOffset else {
            throw Inkling2ForwardError.missingWeight("\(scale.name): its shard is not mapped")
        }
        let rows = bank.shape[1]
        let columns = bank.shape[2]
        let width = Inkling2SnapshotWeights.elementWidth(bank.dtype)
        let scaleRows = scale.shape[1]
        let scaleColumns = scale.shape[2]
        let scaleWidth = Inkling2SnapshotWeights.elementWidth(scale.dtype)

        let w13 = try catalog.tensor(.expertW13, layer: layer)
        let w13Scale = try catalog.tensor(.expertW13Scale, layer: layer)
        let w2 = try catalog.tensor(.expertW2, layer: layer)
        let w2Scale = try catalog.tensor(.expertW2Scale, layer: layer)
        guard let w13, let w13Scale, let w2, let w2Scale else {
            throw Inkling2ForwardError.missingWeight("a bank or its scale is missing on layer \(layer)")
        }
        let layout: Inkling2ExpertQuantLayout
        switch Inkling2ExpertQuantLayout.infer(
            w13: w13, w13Scale: w13Scale, w2: w2, w2Scale: w2Scale, config: config
        ) {
        case .success(let inferred): layout = inferred
        case .failure(let error): throw Inkling2ForwardError.unsupported(error.description)
        }

        var amax: Float = 1
        var denominator: Float = 1
        if layout == .nvfp4 {
            let amaxRole: Inkling2TensorRole = role == .expertW13 ? .expertW13Amax : .expertW2Amax
            let denominatorRole: Inkling2TensorRole =
                role == .expertW13 ? .expertW13OuterScaleDenominator : .expertW2OuterScaleDenominator
            amax = try hostMatrix(amaxRole, layer: layer)[expert]
            denominator = try hostMatrix(denominatorRole, layer: layer)[0]
        }
        let expertByteOffset = UInt64(expert) * UInt64(rows) * UInt64(columns) * UInt64(width)
        let expertScaleOffset = UInt64(expert) * UInt64(scaleRows) * UInt64(scaleColumns) * UInt64(scaleWidth)
        // The same two ranges as FILE offsets, which is what a `pread` reader
        // (the expert arena) needs: the buffer's offsets are relative to a
        // segment once the shard was segmented.
        let fileWeight = (fileOrigins[shard] ?? 0) + baseOffset + expertByteOffset
        let fileScale = (fileOrigins[scaleShard] ?? 0) + scaleBaseOffset + expertScaleOffset
        return (
            buffer, baseOffset, expertByteOffset,
            columns,
            scaleBuffer, scaleBaseOffset,
            expertScaleOffset,
            scaleColumns, layout, amax, denominator, fileWeight, fileScale
        )
    }
}

// MARK: - Workspace

/// How a visit's attention rows are split across threadgroups.
///
/// The single-threadgroup kernel gives one visit `numAttentionHeads`
/// threadgroups of 32 lanes and walks every row serially, so its cost is one
/// reduction and three barriers per row. Splitting the rows measured 13.5x at
/// 1,024 rows, 15.2x at 4,096 and 35.2x at 8,192, with a one-split control at
/// 1.00x -- the split, not the rewrite, is the win
/// (`Inkling2DeviceKernelTests.testWhatSplittingALongVisitRowsCostsAndWhetherItAgrees`).
///
/// The chunk target is 64 rows because 64 was the best chunk at every length
/// measured (1,024 rows: 16 splits at 13.53x; 4,096: 64 splits at 15.17x;
/// 8,192: 128 splits at 35.16x), and a finer chunk starts losing to the
/// per-split partial traffic (8,192 rows: 128 splits 1.202 ms, chunk 32 would
/// be 256 splits). `maximumSplits` bounds the partials buffer: past
/// `maximumSplits * targetChunk` rows the chunk grows instead, which keeps the
/// threadgroup staging request under `maximumChunk` at every length the
/// checkpoint declares.
///
/// Every plan gives every split at least one row, which is what lets the kernel
/// write its slot unconditionally.
nonisolated public enum Inkling2AttentionSplit {
    public static let maximumSplits = 256
    public static let targetChunk = 64
    /// The device's 32,768 B of threadgroup memory, less the 32 reduction slots.
    public static let maximumChunk = 8_160

    public static func plan(rowCount: Int) -> (splits: Int, chunk: Int) {
        let rows = max(1, rowCount)
        let wanted = (rows + targetChunk - 1) / targetChunk
        let splits = min(maximumSplits, max(1, wanted))
        let chunk = (rows + splits - 1) / splits
        precondition(chunk <= maximumChunk, "a chunk of \(chunk) rows cannot be staged")
        return (splits, chunk)
    }
}

/// The device buffers a visit's stages share.
///
/// Sized for ONE position and one visit: the staging vectors are
/// `hiddenSize`-wide (the widest a single stage produces), and the KV rows live
/// in the sequence's own per-visit buffers rather than here. A batched form
/// would replicate the staging by slot, which is why the vector strides are
/// explicit parameters on the kernels rather than assumed.
nonisolated public final class Inkling2DeviceWorkspace {
    public let hidden: MTLBuffer
    public let attentionInput: MTLBuffer
    public let query: MTLBuffer
    public let key: MTLBuffer
    public let value: MTLBuffer
    public let gate: MTLBuffer
    public let attentionOutput: MTLBuffer
    /// The convolved K/V, kept apart from the projections that feed them: the
    /// visit's convolution history advances by the PRE-convolution value, so
    /// the projection must survive the convolution to be pushed.
    public let convolvedKey: MTLBuffer
    public let convolvedValue: MTLBuffer
    /// The MLP half's staging: router scores, the top-k selection, one expert's
    /// gate/up rows and its down projection, and the running mixture.
    public let routerScores: MTLBuffer
    public let routeIndices: MTLBuffer
    public let routeWeights: MTLBuffer
    public let expertGateUp: MTLBuffer
    public let expertIntermediate: MTLBuffer
    public let expertDown: MTLBuffer
    public let residualAccumulator: MTLBuffer
    public let projectionOutput: MTLBuffer
    /// Scratch for the reduction kernels' warp sums.
    public let reductionScratch: MTLBuffer
    /// One visit's key and value rows, staged from the sequence state before an
    /// attention dispatch. The state itself stays host-side (it is what makes a
    /// resume provable), so a device attention reads a copy; the rows are
    /// `kvWidth` wide and the buffer holds the checkpoint's maximum context.
    ///
    /// How the rows are held matches `stagedStorage`: FP32 rows for the
    /// reference path, one signed byte plus one FP16 scale per (row, KV head)
    /// for `.int8`. The INT8 staging is a quarter of the bytes, and it is also
    /// a straight byte copy -- the cache's own rows are already in the kernel's
    /// form, so the host never dequantizes to move them.
    public let stagedKeys: MTLBuffer
    public let stagedValues: MTLBuffer
    public let stagedKeyScales: MTLBuffer
    public let stagedValueScales: MTLBuffer
    public let stagedCapacity: Int
    /// The storage the staging buffers were laid out for. A visit whose own
    /// cache holds a different form cannot be staged into them (see `stage`):
    /// reading one-byte rows as floats is a wrong number, not an error the GPU
    /// would report.
    public let stagedStorage: Inkling2KVStorage
    /// The attention stage's remaining inputs, all small host-computed vectors
    /// or weights staged once per visit: the three projection inputs, the
    /// post-clip relative states, the sink logits and the relative bank.
    public let queryInput: MTLBuffer
    public let keyInput: MTLBuffer
    public let valueInput: MTLBuffer
    public let relativeStates: MTLBuffer
    public let sinkLogits: MTLBuffer
    public let bank: MTLBuffer
    /// The attention's per-row scores, for a parity check of the stage inside.
    public let scoreDump: MTLBuffer
    /// One split's local max, denominator and unnormalized value fold, for every
    /// (split, head) pair the widest visit can dispatch. Laid out
    /// `[split][head][headDim + 2]` and sized for `maximumSplits` on the wider
    /// of the two head geometries.
    public let attentionPartials: MTLBuffer
    /// The unembedding's output: one FP32 logit per vocab row, so it is the one
    /// workspace buffer sized by the vocabulary rather than by the width.
    public let headLogits: MTLBuffer

    /// `stagedPositions` bounds the KV staging, i.e. the longest sequence one
    /// attention dispatch can read. The default is the checkpoint's declared
    /// maximum, but a turn is capped by its KV budget, so the engine passes the
    /// cap it provisioned for -- the declared maximum at these widths is
    /// gigabytes of staging for rows that can never exist.
    ///
    /// `storage` is the form the KV staging buffers are laid out for, and the
    /// sequence this workspace serves must hold its rows the same way.
    public init(
        device: MTLDevice,
        config: Inkling2Config,
        stagedPositions: Int? = nil,
        storage: Inkling2KVStorage = .float32
    ) throws {
        let hiddenBytes = config.hiddenSize * MemoryLayout<Float>.stride
        let attentionWidth = config.numAttentionHeads * config.headDim
        func make(_ length: Int, _ label: String) throws -> MTLBuffer {
            guard let buffer = device.makeBuffer(length: max(length, 16), options: .storageModeShared) else {
                throw Inkling2ForwardError.unsupported("could not allocate the workspace's \(label)")
            }
            return buffer
        }
        hidden = try make(hiddenBytes, "hidden staging")
        attentionInput = try make(hiddenBytes, "attention input")
        query = try make(attentionWidth * MemoryLayout<Float>.stride, "query")
        key = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "key")
        value = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "value")
        gate = try make(attentionWidth * MemoryLayout<Float>.stride, "output gate")
        attentionOutput = try make(attentionWidth * MemoryLayout<Float>.stride, "attention output")
        convolvedKey = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "convolved key")
        convolvedValue = try make(config.numKeyValueHeads * config.headDim * MemoryLayout<Float>.stride, "convolved value")
        routerScores = try make(config.numRoutedExperts * MemoryLayout<Float>.stride, "router scores")
        routeIndices = try make(config.numExpertsPerToken * MemoryLayout<UInt32>.stride, "route indices")
        routeWeights = try make(config.numExpertsPerToken * MemoryLayout<Float>.stride, "route weights")
        expertGateUp = try make(2 * config.moeIntermediateSize * MemoryLayout<Float>.stride, "expert gate/up")
        expertIntermediate = try make(config.moeIntermediateSize * MemoryLayout<Float>.stride, "expert intermediate")
        expertDown = try make(hiddenBytes, "expert down")
        residualAccumulator = try make(hiddenBytes, "residual accumulator")
        projectionOutput = try make(hiddenBytes, "projection output")
        reductionScratch = try make(32 * MemoryLayout<Float>.stride, "reduction scratch")
        // The attention reads at most the window plus the compressed rows; the
        // checkpoint's declared maximum is the safe bound and a few dozen MB at
        // FP32, which is the same order as the host's own rows.
        let kvWidth = config.numKeyValueHeads * config.headDim
        stagedStorage = storage
        stagedCapacity = max(1, stagedPositions ?? config.maxPositionEmbeddings)
        let stagedBytes = stagedCapacity * kvWidth * storage.bytesPerElement
        stagedKeys = try make(stagedBytes, "staged keys")
        stagedValues = try make(stagedBytes, "staged values")
        // A scale is per (row, KV head), and the two geometries may disagree
        // about how many KV heads a visit has, so the block is sized by the
        // wider one. In FP32 storage this is a placeholder byte range: the
        // kernel is never handed it.
        let scaleHeads = max(config.numKeyValueHeads, config.swaNumKeyValueHeads)
        let scaleBytes = stagedCapacity * scaleHeads * max(1, storage.scaleBytesPerHead)
        stagedKeyScales = try make(scaleBytes, "staged key scales")
        stagedValueScales = try make(scaleBytes, "staged value scales")
        queryInput = try make(hiddenBytes, "query input")
        keyInput = try make(hiddenBytes, "key input")
        valueInput = try make(hiddenBytes, "value input")
        relativeStates = try make(
            config.numAttentionHeads * config.dRel * MemoryLayout<Float>.stride, "relative states"
        )
        sinkLogits = try make(config.numAttentionHeads * MemoryLayout<Float>.stride, "sink logits")
        bank = try make(
            config.dRel * max(config.relExtent, config.slidingWindowSize) * MemoryLayout<Float>.stride,
            "relative bank"
        )
        scoreDump = try make(config.maxPositionEmbeddings * MemoryLayout<Float>.stride, "score dump")
        let partialHeads = max(config.numAttentionHeads, config.swaNumAttentionHeads)
        let partialHeadDim = max(config.headDim, config.swaHeadDim)
        attentionPartials = try make(
            Inkling2AttentionSplit.maximumSplits * partialHeads * (partialHeadDim + 2)
                * MemoryLayout<Float>.stride,
            "attention split partials"
        )
        headLogits = try make(
            config.unpaddedVocabSize * MemoryLayout<Float>.stride, "head logits"
        )
    }

    /// Copies one visit's retained key and value rows into the staging buffers.
    /// Returns the row count; the caller passes it to the attention kernel with
    /// the visit's `windowStart`.
    ///
    /// One `vDSP_mmov` per row instead of a scalar element loop: the copy is
    /// row-major and contiguous on both sides, so it is the same bytes in the
    /// same order (bit-identical, it is a copy) and measured 4.4-5.1x faster at
    /// the published row width -- 0.25 ms -> 0.06 ms for a full 1,024-row
    /// window, and this stage grows with the context.
    ///
    /// In `.int8` storage there is nothing to decode: the cache's rows are
    /// already one signed byte per element plus one FP16 scale per KV head, so
    /// the staging is a memcpy of both. Decoding on the host instead would cost
    /// a multiply per element per position AND the 4x staging buffer that the
    /// quantized cache exists to avoid.
    public func stage(_ state: Inkling2VisitState) throws -> Int {
        guard state.storage == stagedStorage else {
            throw Inkling2ForwardError.unsupported(
                "this workspace stages \(stagedStorage.rawValue) KV rows but the visit holds "
                    + "\(state.storage.rawValue): build the workspace for the same storage"
            )
        }
        let rows = state.rowCount
        let width = state.rowWidth
        guard rows > 0, width > 0, rows <= stagedCapacity else { return 0 }
        switch stagedStorage {
        case .float32:
            // Row-major copy: the host keeps [[Float]], the kernel wants one block.
            let keyRows = state.keys
            let valueRows = state.values
            let keyPointer = stagedKeys.contents().bindMemory(to: Float.self, capacity: rows * width)
            let valuePointer = stagedValues.contents().bindMemory(to: Float.self, capacity: rows * width)
            for row in 0..<rows {
                let keyRow = keyRows[row]
                let valueRow = valueRows[row]
                let count = min(width, keyRow.count)
                keyRow.withUnsafeBufferPointer { source in
                    vDSP_mmov(
                        source.baseAddress!, keyPointer + row * width,
                        vDSP_Length(count), 1, vDSP_Length(count), vDSP_Length(count)
                    )
                }
                valueRow.withUnsafeBufferPointer { source in
                    vDSP_mmov(
                        source.baseAddress!, valuePointer + row * width,
                        vDSP_Length(min(count, valueRow.count)), 1,
                        vDSP_Length(min(count, valueRow.count)), vDSP_Length(min(count, valueRow.count))
                    )
                }
            }
        case .int8:
            // The kernel indexes a scale by (row, KV head), so the cache's own
            // per-head count has to be this visit's -- a sequence built with a
            // different count would misread a row that is otherwise the right
            // bytes.
            let heads = state.heads
            guard heads > 0 else {
                throw Inkling2ForwardError.unsupported("an INT8 visit holds a row with no KV heads")
            }
            let keyPointer = stagedKeys.contents().bindMemory(to: Int8.self, capacity: rows * width)
            let valuePointer = stagedValues.contents().bindMemory(to: Int8.self, capacity: rows * width)
            let keyScalePointer = stagedKeyScales.contents().bindMemory(
                to: UInt16.self, capacity: rows * heads
            )
            let valueScalePointer = stagedValueScales.contents().bindMemory(
                to: UInt16.self, capacity: rows * heads
            )
            for row in 0..<rows {
                guard let key = state.quantizedKey(row), let value = state.quantizedValue(row) else {
                    throw Inkling2ForwardError.unsupported(
                        "row \(row) of this INT8 visit has no stored bytes"
                    )
                }
                let bytes = min(width, key.stored.count)
                key.stored.withUnsafeBytes { source in
                    memcpy(keyPointer + row * width, source.baseAddress!, bytes)
                }
                value.stored.withUnsafeBytes { source in
                    memcpy(valuePointer + row * width, source.baseAddress!, min(bytes, value.stored.count))
                }
                key.scales.withUnsafeBytes { source in
                    memcpy(
                        keyScalePointer + row * heads, source.baseAddress!,
                        min(heads, key.scales.count) * MemoryLayout<UInt16>.stride
                    )
                }
                value.scales.withUnsafeBytes { source in
                    memcpy(
                        valueScalePointer + row * heads, source.baseAddress!,
                        min(heads, value.scales.count) * MemoryLayout<UInt16>.stride
                    )
                }
            }
        }
        return rows
    }
}

// MARK: - Scalar arguments

/// Keeps every `constant` argument buffer alive for as long as a command buffer
/// may read it.
///
/// A dispatch reads its arguments when the GPU RUNS it, not when the host
/// encodes it, so a scalar freed at the end of the helper that made it is a
/// use-after-free -- and the GPU reports that as an address fault, not as a
/// wrong number. Per-helper arrays were tried and every new helper forgot one;
/// one arena owned by the caller of the whole sequence cannot.
nonisolated public final class Inkling2ScalarArena {
    private let device: MTLDevice
    private var buffers: [MTLBuffer] = []

    public init(device: MTLDevice) { self.device = device }

    /// A copy of `value` as a buffer, retained until the arena dies.
    public func make<T>(_ value: T) -> MTLBuffer? {
        var copy = value
        guard let buffer = device.makeBuffer(
            bytes: &copy, length: MemoryLayout<T>.stride, options: .storageModeShared
        ) else { return nil }
        buffers.append(buffer)
        return buffer
    }

    /// A staged copy of a host array (weights, history, anything a kernel reads
    /// that is not already in a workspace buffer), retained until the arena is
    /// cleared. The same lifetime rule as `make`: a buffer the GPU reads must
    /// outlive the commit.
    public func make<Element>(staging values: [Element]) -> MTLBuffer? {
        // A zero-length `makeBuffer(bytes:)` is not allowed, and an empty
        // history reads no row, so one zeroed element is a safe stand-in. It is
        // allocated zeroed rather than `unsafeBitCast(0, to: Element.self)`,
        // which traps for any element wider than an Int (a Float's zero is
        // four bytes and an Int's is eight).
        let buffer: MTLBuffer?
        if values.isEmpty {
            buffer = device.makeBuffer(
                length: MemoryLayout<Element>.stride, options: .storageModeShared
            )
        } else {
            buffer = values.withUnsafeBytes { raw in
                device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
            }
        }
        if let buffer { buffers.append(buffer) }
        return buffer
    }

    /// Drops the retained buffers: call once the command buffer has completed.
    public func clear() { buffers.removeAll() }
}

// MARK: - The device forward

/// One visit's device path. It owns the pipelines and the workspace; the turn
/// engine owns the sequence state and the weights.
nonisolated public final class Inkling2MetalForward {
    public let config: Inkling2Config
    public let weights: Inkling2DeviceWeights
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    /// The host graph, used ONLY for the pieces this stage deliberately leaves
    /// on the host (two small vectors' worth of arithmetic) and as the oracle a
    /// parity test compares against. It reads the same weights.
    public let host: Inkling2CPUForward

    /// Touches every byte of a weight range on the HOST before a dispatch reads
    /// it, so the GPU meets resident pages instead of faulting them in.
    ///
    /// Off by default and a measurement seam only: this family has no expert
    /// arena, its banks are the shard `mmap`, and `Benchmarks/PERF_FINDINGS.md`
    /// "DSV4.1 #19" measured a fault path at 0.065-0.2 GB/s once it interleaves
    /// with GPU work. Whether that is what the mixture's in-situ rate is missing
    /// is a question this switch answers, not an assumption.
    public var prefaultsExpertBanks = false
    /// The bytes `prefaultsExpertBanks` read, accumulated so the loads cannot
    /// be eliminated as dead. Zero in every production path.
    public private(set) var touchedChecksum: UInt8 = 0

    /// Whether `runVisit` bills its stages into `visitProfile`.
    ///
    /// Off by default and off in every production path: the seam that turns it
    /// on is `dynamoe_inkling2_profile`, read once per turn. A nil profile at
    /// every call site is what keeps the cost of the instrument out of the
    /// shipping graph rather than a promise that it is cheap.
    public var recordsVisitTiming = false

    /// The per-stage table `runVisit` fills while `recordsVisitTiming` is on.
    /// Its owner calls `reset()` at the start of a turn and `describe()` at the
    /// end; a turn that is cancelled still leaves what it measured.
    public let visitProfile = Inkling2VisitProfile()

    /// The expert arena the mixture reads from, when one is installed.
    ///
    /// Nil in every production path -- this family's banks are the shard mmap
    /// until a residency policy is measured to pay. When it is set, the mixture
    /// asks it for each routed expert and projects that expert's SLOT; a miss
    /// preads the expert's four ranges into a slot before the dispatch encodes,
    /// so the mmap is never bound while an arena holds the bank's geometry (it
    /// is the binding of two whole segments that reads zeros here). The arena's
    /// own counters then reach the profile through `recordExperts`, so a
    /// cold/warm comparison is read from the same line as the stage table rather
    /// than a second instrument.
    ///
    /// One slot is a byte copy at FIXED strides, so an arena serves one packed
    /// LAYOUT. The published checkpoint mixes two -- NVFP4 for the first twenty
    /// layers and MXFP8 for the last four -- and `installExpertArena` therefore
    /// may be called twice; a bank goes to the arena whose geometry holds it.
    /// A capacity under the visit's own routing width (`numExpertsPerToken`)
    /// would alias two experts onto one slot, which is why
    /// `Inkling2ArenaCapacity` floors it there.
    public private(set) var expertArenas: [Inkling2InstalledExpertArena] = []

    /// The first installed arena. Callers that install one (every test) keep
    /// reading this; the mixture reads `expertArenas`.
    public var expertArena: Inkling2ExpertArena? { expertArenas.first?.arena }

    /// Installs a bounded expert arena and the reader that fills it.
    ///
    /// `capacity` is in experts, and the caller owns the budget decision: one
    /// slot is `geometry.slotBytes`, so `capacity * slotBytes` is the RAM the
    /// arena takes from everything else (the KV cache included). The slot
    /// buffer is allocated here at `capacity * slotBytes`, and a capacity the
    /// device refuses is thrown rather than silently reduced -- an arena that
    /// is smaller than asked for would change the hit rate under measurement.
    ///
    /// Installing the same geometry twice replaces the first arena of that
    /// layout (a re-install is a resize, not a second cache); a different
    /// geometry adds one beside it, because the two cannot share slots.
    public func installExpertArena(
        capacity: Int,
        geometry: Inkling2ExpertSlotGeometry,
        reader: Inkling2ArenaShardReader
    ) throws {
        let bytes = capacity * geometry.slotBytes
        guard bytes > 0, bytes <= device.maxBufferLength,
              let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
            throw Inkling2ForwardError.shapeMismatch(
                "an expert arena of \(capacity) slot(s) needs \(bytes) bytes, which this device will not allocate"
            )
        }
        let arena = Inkling2ExpertArena(capacity: capacity, geometry: geometry)
        arena.attachSlotBuffer(buffer)
        expertArenas.removeAll { $0.geometry == geometry }
        expertArenas.append(Inkling2InstalledExpertArena(
            geometry: geometry, arena: arena, reader: reader
        ))
    }

    /// Removes every arena, so the mixture goes back to the shard mmap.
    public func removeExpertArena() {
        for installed in expertArenas { installed.arena.evictAll() }
        expertArenas.removeAll()
    }

    private let pipelines: [String: MTLComputePipelineState]

    /// Every pipeline this graph dispatches, by kernel name. A missing one is
    /// a load-time failure that names it, not a crash at the first dispatch.
    public static let kernelNames: [String] = [
        "inkling_rmsnorm_f32",
        "inkling_rmsnorm_no_gain_f32",
        "inkling_softcapped_swiglu_f32",
        "inkling_softcapped_swiglu_interleaved_f32",
        "inkling_sigmoid_inplace_f32",
        "inkling_weighted_accumulate_f32",
        "inkling_short_conv_f32",
        "inkling_dynamic_conv_f32",
        "inkling_message_gate_f32",
        "inkling_l2_clip_scale_f32",
        "inkling_l2_clip_f32",
        "inkling_gate_multiply_f32",
        "inkling_block_read_f32",
        "inkling_attention_f32",
        "inkling_attention_int8_f32",
        "inkling_attention_split_f32",
        "inkling_attention_split_int8_f32",
        "inkling_attention_split_combine_f32",
        "inkling_expert_row_f32",
        "inkling_router_topk_f32",
        "bf16_gemv_rows8",
        "bf16_gemv_rows",
        "bf16_gemv",
        "vector_add_f32"
    ]

    public init(
        device: MTLDevice,
        library: MTLLibrary,
        queue: MTLCommandQueue,
        weights: Inkling2DeviceWeights,
        host: Inkling2CPUForward
    ) throws {
        self.config = weights.config
        self.weights = weights
        self.device = device
        self.queue = queue
        self.host = host
        var built: [String: MTLComputePipelineState] = [:]
        for name in Self.kernelNames {
            guard let function = library.makeFunction(name: name) else {
                throw Inkling2ForwardError.unsupported("the shader library has no kernel '\(name)'")
            }
            built[name] = try device.makeComputePipelineState(function: function)
        }
        self.pipelines = built
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        guard let pipeline = pipelines[name] else {
            throw Inkling2ForwardError.unsupported("no pipeline for '\(name)'")
        }
        return pipeline
    }

    /// The block's Q/K/V projections in ONE submission: position `p`'s row is
    /// slot `p` of `input`, and each projection streams its weight once for the
    /// whole block.
    ///
    /// This is the first stage that actually batches, and it is deliberately
    /// only the projections: everything after them reads the visit state's
    /// history per position, so the block driver still runs those one position
    /// at a time. It takes EIGHT slots (the kernel reads eight rows
    /// unconditionally), and a longer input is the caller's to split.
    ///
    /// `q`, `k` and `v` are one block-sized buffer each, `slots` rows wide; a
    /// caller that wants a short block pads its input and reads back only what
    /// it wrote.
    public func encodeBlockQKV(
        layer: Int,
        queryInput: MTLBuffer,
        keyInput: MTLBuffer,
        valueInput: MTLBuffer,
        query: MTLBuffer,
        key: MTLBuffer,
        value: MTLBuffer,
        slots: Int
    ) throws {
        let isFull = !config.isLocalLayer(layer)
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let hidden = config.hiddenSize
        guard slots == 8 else {
            throw Inkling2ForwardError.unsupported(
                "the batched projection runs exactly eight slots, got \(slots)"
            )
        }
        // The same three segments the single-position front binds, in one
        // command buffer: priced before the encoder exists, so a refusal cannot
        // leave one unclosed (Metal asserts on release, which is a process
        // abort rather than an error a caller can read).
        try guardDenseBinding(
            "batched attention front", roles: [.attnWqDu, .attnWkDv, .attnWvDv], layer: layer
        )
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Inkling2ForwardError.unsupported("could not make a command buffer")
        }
        let arena = Inkling2ScalarArena(device: device)
        defer { arena.clear() }
        var encoded = false
        defer { if !encoded { encoder.endEncoding() } }
        try gemm(
            encoder: encoder, role: .attnWqDu, layer: layer,
            input: queryInput, inputOffset: 0, output: query, outputOffset: 0,
            rows: heads * headDim, columns: hidden, positions: slots, arena: arena
        )
        try gemm(
            encoder: encoder, role: .attnWkDv, layer: layer,
            input: keyInput, inputOffset: 0, output: key, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden, positions: slots, arena: arena
        )
        try gemm(
            encoder: encoder, role: .attnWvDv, layer: layer,
            input: valueInput, inputOffset: 0, output: value, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden, positions: slots, arena: arena
        )
        encoder.endEncoding()
        encoded = true
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
    }

    // MARK: Small constants

    /// A `constant` argument, as the buffer the shader binds.
    ///
    /// The caller keeps the returned buffers alive until the command buffer is
    /// committed: a dispatch's arguments are read when the GPU runs it, not
    /// when it is encoded, so a temporary freed at the end of the encoding
    /// scope would be read after it was gone.
    private func scalar<T>(_ value: T) -> MTLBuffer? {
        var copy = value
        return device.makeBuffer(
            bytes: &copy, length: MemoryLayout<T>.stride, options: .storageModeShared
        )
    }

    /// `y = W x` for a BF16 weight, dispatched through the wide-grid block form.
    ///
    /// The host's `linear` is FP32 over a dequantized matrix; the device reads
    /// the checkpoint's BF16 bytes, which is the same number widened, so the
    /// only difference is the accumulation order.
    func gemm(
        encoder: MTLComputeCommandEncoder,
        role: Inkling2TensorRole,
        layer: Int?,
        input: MTLBuffer,
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        rows: Int,
        columns: Int,
        positions: Int = 1,
        arena: Inkling2ScalarArena
    ) throws {
        let (weightBuffer, weightOffset) = try weights.tensor(role, layer: layer)
        try gemv(
            encoder: encoder, weight: weightBuffer, weightOffset: weightOffset,
            input: input, inputOffset: inputOffset, output: output, outputOffset: outputOffset,
            rows: rows, columns: columns, positions: positions, arena: arena
        )
    }

    /// The shared body of `gemm`, for a weight already resolved (a fused bank's
    /// expert slice has no single role).
    ///
    /// `positions` is a BLOCK: eight or more positions whose rows are already
    /// contiguous in `input`/`output` take `bf16_gemv_rows8`, which streams the
    /// weight row ONCE for all of them. That is the batched-prefill win -- the
    /// walk re-reads the weight per position (measured 4.9-8.3x on this
    /// checkpoint's own projection shapes). A shorter block falls back to the
    /// one-thread-per-row form with its token axis, because the eight-position
    /// kernel reads eight rows unconditionally.
    func gemv(
        encoder: MTLComputeCommandEncoder,
        weight: MTLBuffer, weightOffset: UInt64,
        input: MTLBuffer, inputOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        rows: Int, columns: Int, positions: Int = 1,
        arena: Inkling2ScalarArena
    ) throws {
        let kernel = positions >= 8 ? "bf16_gemv_rows8" : "bf16_gemv"
        // The eight-position form reads and writes exactly eight rows; a taller
        // block would silently run only its first eight, so it is refused.
        if positions > 8 {
            throw Inkling2ForwardError.unsupported(
                "a batched projection is one to eight positions, got \(positions)"
            )
        }
        let offset = arena.make(UInt64(weightOffset))
        let inDim = arena.make(UInt32(columns))
        let outDim = arena.make(UInt32(rows))
        let zero: UInt32 = 0
        let inputStride = arena.make(zero)
        let outputStride = arena.make(zero)
        encoder.setComputePipelineState(try pipeline(kernel))
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBuffer(offset, offset: 0, index: 3)
        encoder.setBuffer(inDim, offset: 0, index: 4)
        encoder.setBuffer(outDim, offset: 0, index: 5)
        encoder.setBuffer(inputStride, offset: 0, index: 6)
        encoder.setBuffer(outputStride, offset: 0, index: 7)
        if kernel == "bf16_gemv_rows8" {
            encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
        } else {
            encoder.dispatchThreads(
                MTLSize(width: rows, height: positions, depth: 1),
                threadsPerThreadgroup: MTLSize(width: min(rows, 128), height: 1, depth: 1)
            )
        }
    }

    // MARK: - The output head

    /// Which kernel the unembedding dispatches.
    ///
    /// The head is the one stage where the two shipped forms differ by roughly
    /// an order of magnitude in COALESCING rather than in arithmetic: 200,058
    /// rows of 2,304, read once per token. `rowThread` gives a thread a row, so
    /// a warp's 32 lanes touch 32 rows at the same column and every 16-byte load
    /// lands in its own 128-byte sector; `coalesced` gives a threadgroup a row
    /// and strides the columns, so a warp reads one contiguous window. They are
    /// both kept so the choice stays a measurement (see
    /// `Inkling2DecodeMeasurementTests`) rather than a copied threshold.
    public enum HeadKernelForm: String, Sendable, CaseIterable {
        case coalesced
        case rowThread
    }

    /// The form the graph ships, chosen by the rotated A/B in
    /// `Inkling2PublishedHeadTests`: the coalesced form reads the 921 MB table
    /// at 358 GB/s against the one-thread form's 209, and the very first pass --
    /// on an idle GPU -- read the opposite. See the plan's Phase 6 slice-1
    /// record for the numbers and the trap.
    public static let shippedHeadForm: HeadKernelForm = .coalesced

    /// The unembedding on the device: `logits = lmHead * hidden`, the raw
    /// pre-softcap values for the first `unpaddedVocabSize` rows.
    ///
    /// One commit and one wait: the caller samples from the result, so there is
    /// nothing to overlap it with. The 922 MB read is the cost and the GPU's
    /// request rate is the point.
    ///
    /// It stages the hidden row in `workspace.queryInput`, which is the right
    /// width and is not live here: the head runs after the last visit, and every
    /// visit re-stages that buffer before it reads it.
    public func headLogits(
        hidden: [Float],
        workspace: Inkling2DeviceWorkspace,
        form: HeadKernelForm = Inkling2MetalForward.shippedHeadForm
    ) throws -> [Float] {
        let rows = config.unpaddedVocabSize
        let columns = config.hiddenSize
        guard hidden.count == columns else {
            throw Inkling2ForwardError.shapeMismatch(
                "the head wants a \(columns)-wide hidden row, got \(hidden.count)"
            )
        }
        try guardDenseBinding("head", roles: [.lmHead], layer: nil)
        let (weightBuffer, weightOffset) = try weights.tensor(.lmHead, layer: nil)
        let input = try staging(workspace.queryInput, hidden)
        let arena = Inkling2ScalarArena(device: device)
        var encoded = false
        let (commandBuffer, encoder) = try commandComponents()
        defer {
            if !encoded { encoder.endEncoding() }
            arena.clear()
        }
        let offset = arena.make(UInt64(weightOffset))
        let inDim = arena.make(UInt32(columns))
        let outDim = arena.make(UInt32(rows))
        let zero: UInt32 = 0
        let inputStride = arena.make(zero)
        let outputStride = arena.make(zero)
        encoder.setBuffer(weightBuffer, offset: 0, index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(workspace.headLogits, offset: 0, index: 2)
        encoder.setBuffer(offset, offset: 0, index: 3)
        encoder.setBuffer(inDim, offset: 0, index: 4)
        encoder.setBuffer(outDim, offset: 0, index: 5)
        encoder.setBuffer(inputStride, offset: 0, index: 6)
        encoder.setBuffer(outputStride, offset: 0, index: 7)
        switch form {
        case .coalesced:
            let lanes = 128
            encoder.setComputePipelineState(try pipeline("bf16_gemv_rows"))
            // One float per 32 lanes, sized for the widest group a caller can
            // ask for rather than for this one.
            encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
            encoder.dispatchThreadgroups(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: lanes, height: 1, depth: 1)
            )
        case .rowThread:
            encoder.setComputePipelineState(try pipeline("bf16_gemv"))
            encoder.dispatchThreads(
                MTLSize(width: rows, height: 1, depth: 1),
                threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
            )
        }
        encoder.endEncoding()
        encoded = true
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        return readBack(workspace.headLogits, count: rows)
    }

    /// The block read: `softmax(q . rmsnorm(row) / sqrt(width)) . rmsnorm(row)`
    /// over the stack plus the partial, one threadgroup per query.
    func blockRead(
        encoder: MTLComputeCommandEncoder,
        stack: MTLBuffer,
        stackRows: Int,
        partial: MTLBuffer,
        partialOffset: Int,
        queries: MTLBuffer,
        queryCount: Int,
        output: MTLBuffer,
        outputOffset: Int,
        arena: Inkling2ScalarArena
    ) throws {
        let width = config.hiddenSize
        let stackRowsValue = arena.make(UInt32(stackRows))
        let widthValue = arena.make(UInt32(width))
        encoder.setComputePipelineState(try pipeline("inkling_block_read_f32"))
        encoder.setBuffer(stack, offset: 0, index: 0)
        encoder.setBuffer(partial, offset: partialOffset, index: 1)
        encoder.setBuffer(queries, offset: 0, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        encoder.setBuffer(stackRowsValue, offset: 0, index: 4)
        encoder.setBuffer(widthValue, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(
            (32 + 64 + 64) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: queryCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// The l2 clip with a learned scale, one threadgroup per head.
    func l2Clip(
        encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        vectorOffset: Int,
        scale: MTLBuffer,
        scaleOffset: Int,
        headDim: Int,
        heads: Int,
        threshold: Float,
        tau: Float,
        arena: Inkling2ScalarArena
    ) throws {
        let headDimValue = arena.make(UInt32(headDim))
        let scaleOffsetValue = arena.make(UInt64(scaleOffset))
        let thresholdValue = arena.make(threshold)
        let tauValue = arena.make(tau)
        encoder.setComputePipelineState(try pipeline("inkling_l2_clip_scale_f32"))
        encoder.setBuffer(vector, offset: vectorOffset, index: 0)
        // The scale's buffer at offset ZERO plus its byte offset as a constant:
        // a tensor's offset inside a shard is not buffer-aligned.
        encoder.setBuffer(scale, offset: 0, index: 1)
        encoder.setBuffer(scaleOffsetValue, offset: 0, index: 2)
        encoder.setBuffer(headDimValue, offset: 0, index: 3)
        encoder.setBuffer(thresholdValue, offset: 0, index: 4)
        encoder.setBuffer(tauValue, offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// The plain per-head L2 clip (no learned scale), which is the key path.
    func l2ClipPlain(
        encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        headDim: Int,
        heads: Int,
        threshold: Float,
        arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_l2_clip_f32"))
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(arena.make(UInt32(headDim)), offset: 0, index: 1)
        encoder.setBuffer(arena.make(threshold), offset: 0, index: 2)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    /// One head's unweighted norm in place, which the value projection needs.
    func headNorm(
        encoder: MTLComputeCommandEncoder,
        vector: MTLBuffer,
        headDim: Int,
        heads: Int,
        arena: Inkling2ScalarArena
    ) throws {
        let headDimValue = arena.make(UInt32(headDim))
        let epsValue = arena.make(config.rmsNormEps)
        encoder.setComputePipelineState(try pipeline("inkling_rmsnorm_no_gain_f32"))
        // In place: the kernel reads a row fully into its reduction before its
        // write loop touches it, and each threadgroup owns one head's row, so
        // no group reads another's.
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(vector, offset: 0, index: 1)
        encoder.setBuffer(headDimValue, offset: 0, index: 2)
        encoder.setBuffer(epsValue, offset: 0, index: 3)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        // One threadgroup per head; the kernel normalizes rows of `dim`, and
        // the vector is heads * headDim, so the grid is the head count.
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    // MARK: Staging and host-side pieces

    /// Copies a host vector into a workspace buffer.
    func staging(_ buffer: MTLBuffer, _ values: [Float]) throws -> MTLBuffer {
        guard values.count * MemoryLayout<Float>.stride <= buffer.length else {
            throw Inkling2ForwardError.shapeMismatch(
                "staging \(values.count) values into a \(buffer.length)-byte buffer"
            )
        }
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: values.count)
        for index in values.indices { pointer[index] = values[index] }
        return buffer
    }

    func readBack(_ buffer: MTLBuffer, count: Int) -> [Float] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count
        ))
    }

    /// A weight as the host sees it: the same dequantized matrix the host graph
    /// multiplies by. Used only for the small vectors the kernel needs as
    /// values (the relative bank), never for a stage the port is meant to move.
    func hostMatrix(_ role: Inkling2TensorRole, layer: Int?) throws -> [Float] {
        try weights.hostMatrix(role, layer: layer)
    }

    private func commandComponents() throws -> (MTLCommandBuffer, MTLComputeCommandEncoder) {
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw Inkling2ForwardError.unsupported("could not make a command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw Inkling2ForwardError.unsupported("could not make a compute encoder")
        }
        return (commandBuffer, encoder)
    }

    /// The relative states: `wr @ input`, per head scaled by the learned scale,
    /// L2-clipped, and (on a full layer) log-scaled. Host arithmetic, because it
    /// is two small vectors and the kernel consumes only the result.
    func relativeStates(layer: Int, queryInput: [Float], position: Int, isFull: Bool) throws -> [Float] {
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let wr = try hostMatrix(.attnWrDu, layer: layer)
        let rScale = try hostMatrix(.attnRelL2ClipScale, layer: layer)
        var relative = try host.linear(wr, queryInput, rows: heads * config.dRel)
        let tau = isFull
            ? Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
            : 1
        for head in 0..<heads {
            var slice = Array(relative[(head * config.dRel)..<((head + 1) * config.dRel)])
            for dimension in 0..<config.dRel { slice[dimension] *= rScale[head * config.dRel + dimension] }
            slice = Inkling2Math.l2Clip(slice, threshold: config.relL2ClipThreshold)
            if isFull { slice = slice.map { $0 * tau } }
            for dimension in 0..<config.dRel { relative[head * config.dRel + dimension] = slice[dimension] }
        }
        return relative
    }

    /// The sink logits: `sinkProj @ input`, one per head.
    func sinkLogits(layer: Int, queryInput: [Float]) throws -> [Float] {
        let heads = !config.isLocalLayer(layer) ? config.numAttentionHeads : config.swaNumAttentionHeads
        let projection = try hostMatrix(.attnSinkProj, layer: layer)
        return try host.linear(projection, queryInput, rows: heads)
    }

    /// The short convolution on device, static or dynamic, for one of a visit's
    /// four convolution states.
    ///
    /// It stages what the dispatch reads and does NOT touch the host state:
    /// the state has to advance by the value the GPU actually produced, and
    /// that value does not exist until the command buffer completes. A first
    /// cut pushed the input here, which left the history one commit stale --
    /// exact for the first rows, wrong from the point the convolution reads
    /// them. The caller pushes after the wait.
    func convolve(
        encoder: MTLComputeCommandEncoder,
        state: Inkling2VisitState,
        index: Int,
        values: MTLBuffer,
        valuesCount: Int,
        dynamicInput: [Float],
        layer: Int,
        isFull: Bool,
        headDim: Int,
        output: MTLBuffer,
        arena: Inkling2ScalarArena
    ) throws {
        let kernelSize = config.convKernelSize
        let history = state.history(index)
        // A full-attention layer's K/V convolutions are DYNAMIC: static taps
        // plus per-head taps projected from the projection's input, normalized
        // by `sqrt(2 * kernel + 1)`. The post-attention and post-MLP streams are
        // always the static form. One weight per stream: K and V have their
        // own, and the last two are DIFFERENT tensors (`attn_sconv` and
        // `mlp_sconv`) even though both are hidden-wide -- folding them onto
        // one made the post-MLP convolution read the attention's taps.
        let isDynamic = isFull && (index == 0 || index == 1)
        let tapsRole: Inkling2TensorRole
        switch index {
        case 0: tapsRole = isDynamic ? .attnKSconvStatic : .attnKSconvWeight
        case 1: tapsRole = isDynamic ? .attnVSconvStatic : .attnVSconvWeight
        case 2: tapsRole = .attnSconv
        default: tapsRole = .mlpSconv
        }
        let taps = try hostMatrix(tapsRole, layer: layer)
        // The dynamic taps come from a projection of the SAME input the K/V
        // projection read, one `kernel`-wide vector per KV head.
        var dynamicTaps: [Float] = []
        if isDynamic {
            let projection = try hostMatrix(
                index == 0 ? .attnKSconvProj : .attnVSconvProj, layer: layer
            )
            let projected = try host.linear(projection, dynamicInput, rows: (valuesCount / headDim) * kernelSize)
            dynamicTaps = projected
        }
        // Staged through the ARENA, which owns every buffer a dispatch reads
        // until after the wait: a local buffer is freed when this function
        // returns, and the GPU may not have run the dispatch yet -- a
        // use-after-free it reports as an address fault.
        guard let tapsBuffer = arena.make(staging: taps) else {
            throw Inkling2ForwardError.unsupported("could not stage the convolution taps")
        }
        let historyRows = history.count
        var flatHistory = [Float](repeating: 0, count: historyRows * valuesCount)
        for row in 0..<historyRows {
            for channel in 0..<valuesCount { flatHistory[row * valuesCount + channel] = history[row][channel] }
        }
        guard let historyBuffer = arena.make(staging: flatHistory) else {
            throw Inkling2ForwardError.unsupported("could not stage the convolution history")
        }
        let channels = arena.make(UInt32(valuesCount))
        let tapCount = arena.make(UInt32(kernelSize))
        let rows = arena.make(UInt32(historyRows))
        if isDynamic {
            guard let dynamicBuffer = arena.make(staging: dynamicTaps) else {
                throw Inkling2ForwardError.unsupported("could not stage the dynamic convolution taps")
            }
            encoder.setComputePipelineState(try pipeline("inkling_dynamic_conv_f32"))
            encoder.setBuffer(values, offset: 0, index: 0)
            encoder.setBuffer(historyBuffer, offset: 0, index: 1)
            encoder.setBuffer(tapsBuffer, offset: 0, index: 2)
            encoder.setBuffer(dynamicBuffer, offset: 0, index: 3)
            encoder.setBuffer(output, offset: 0, index: 4)
            encoder.setBuffer(channels, offset: 0, index: 5)
            encoder.setBuffer(tapCount, offset: 0, index: 6)
            encoder.setBuffer(rows, offset: 0, index: 7)
            encoder.setBuffer(arena.make(UInt32(headDim)), offset: 0, index: 8)
        } else {
            encoder.setComputePipelineState(try pipeline("inkling_short_conv_f32"))
            encoder.setBuffer(values, offset: 0, index: 0)
            encoder.setBuffer(historyBuffer, offset: 0, index: 1)
            encoder.setBuffer(tapsBuffer, offset: 0, index: 2)
            encoder.setBuffer(output, offset: 0, index: 3)
            encoder.setBuffer(channels, offset: 0, index: 4)
            encoder.setBuffer(tapCount, offset: 0, index: 5)
            encoder.setBuffer(rows, offset: 0, index: 6)
        }
        encoder.dispatchThreads(
            MTLSize(width: valuesCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    /// `output *= sigmoid(gate)`, elementwise.
    func gateMultiply(
        encoder: MTLComputeCommandEncoder, output: MTLBuffer, gate: MTLBuffer,
        count: Int, arena: Inkling2ScalarArena
    ) throws {
        let countValue = arena.make(UInt32(count))
        encoder.setComputePipelineState(try pipeline("inkling_gate_multiply_f32"))
        encoder.setBuffer(output, offset: 0, index: 0)
        encoder.setBuffer(gate, offset: 0, index: 1)
        encoder.setBuffer(countValue, offset: 0, index: 2)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    // MARK: The attention half

    /// The attention half of one visit, stage for stage the host's
    /// `Inkling2CPUForward.attention`, returning the same vector the host
    /// returns (after the gate and the grouped output projection).
    ///
    /// Host arrays in, host array out: the sequence state is host-side (it is
    /// what makes a resume provable), so this stages what the kernels read and
    /// copies the result back. The staging is a few hundred KB against tens of
    /// GFLOP of work at these widths, so it is not what the port is for -- the
    /// port is the arithmetic.
    public func attentionHalf(
        layer: Int,
        state: Inkling2VisitState,
        queryInput: [Float],
        keyInput: [Float],
        valueInput: [Float],
        position: Int,
        workspace: Inkling2DeviceWorkspace,
        dumpsAttentionScores: Bool = false,
        // Folds the post-attention convolution into this half's last buffer, so
        // the visit pays one commit for both. Opt-in and off by default: the
        // caller then owns the wait, the history push and the read-back, which
        // is what `runVisit` does.
        foldsPostAttentionConvolution: Bool = false,
        profile: Inkling2VisitProfile? = nil
    ) throws -> [Float] {
        let stageStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let isFull = !config.isLocalLayer(layer)
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let hidden = config.hiddenSize

        // Price both command buffers' bindings before anything is staged: a
        // binding past the device's ceiling returns zeros with no error at all.
        try guardDenseBinding(
            "attention front", roles: [.attnWqDu, .attnWkDv, .attnWvDv], layer: layer
        )
        try guardDenseBinding(
            "attention tail", roles: [.attnGate, .attnWoUd], layer: layer
        )

        // Inputs staged once; every stage reads them.
        let queryInputBuffer = try staging(workspace.queryInput, queryInput)
        let keyInputBuffer = try staging(workspace.keyInput, keyInput)
        let valueInputBuffer = try staging(workspace.valueInput, valueInput)

        let encodeStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let (commandBuffer, encoder) = try commandComponents()
        // Every dispatch's scalar argument lives in the arena until the command
        // buffer has completed: the GPU reads them when it runs the work.
        let arena = Inkling2ScalarArena(device: device)
        defer { arena.clear() }
        // Every early throw below must still close the encoder: Metal asserts on
        // release, and that assertion is a process abort rather than an error a
        // caller can read (the same rule the vision kernels recorded).
        var encoded = false
        defer { if !encoded { encoder.endEncoding() } }

        // Q: project, then scale + L2 clip (+ log scale on a full layer).
        try gemm(
            encoder: encoder, role: .attnWqDu, layer: layer,
            input: queryInputBuffer, inputOffset: 0,
            output: workspace.query, outputOffset: 0,
            rows: heads * headDim, columns: hidden,
            arena: arena
        )
        let qScale = try weights.tensor(.attnQL2ClipScale, layer: layer)
        let tau = isFull
            ? Inkling2Math.logScale(position: position, alpha: config.logScalingAlpha, floor: config.logScalingNFloor)
            : 1
        try l2Clip(
            encoder: encoder, vector: workspace.query, vectorOffset: 0,
            scale: qScale.buffer, scaleOffset: Int(qScale.offset),
            headDim: headDim, heads: heads, threshold: config.qkL2ClipThreshold, tau: tau,
            arena: arena
        )

        // K: project into `key`, convolve into `convolvedKey`, L2 clip there.
        // The two buffers are separate because the visit's history advances by
        // the PRE-convolution value, which the convolution would overwrite.
        try gemm(
            encoder: encoder, role: .attnWkDv, layer: layer,
            input: keyInputBuffer, inputOffset: 0,
            output: workspace.key, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden,
            arena: arena
        )
        try convolve(
            encoder: encoder, state: state, index: 0,
            values: workspace.key, valuesCount: kvHeads * headDim, dynamicInput: keyInput,
            layer: layer, isFull: isFull, headDim: headDim, output: workspace.convolvedKey, arena: arena
        )
        // The keys are clipped PLAINLY: the reference multiplies only the
        // queries by `q_l2_clip.scale`, so passing that scale here made K 0.47
        // off while Q was exact.
        try l2ClipPlain(
            encoder: encoder, vector: workspace.convolvedKey,
            headDim: headDim, heads: kvHeads, threshold: config.qkL2ClipThreshold,
            arena: arena
        )

        // V: project into `value`, convolve into `convolvedValue`, norm there.
        try gemm(
            encoder: encoder, role: .attnWvDv, layer: layer,
            input: valueInputBuffer, inputOffset: 0,
            output: workspace.value, outputOffset: 0,
            rows: kvHeads * headDim, columns: hidden,
            arena: arena
        )
        try convolve(
            encoder: encoder, state: state, index: 1,
            values: workspace.value, valuesCount: kvHeads * headDim, dynamicInput: valueInput,
            layer: layer, isFull: isFull, headDim: headDim, output: workspace.convolvedValue, arena: arena
        )
        try headNorm(encoder: encoder, vector: workspace.convolvedValue, headDim: headDim, heads: kvHeads, arena: arena)

        encoder.endEncoding()
        encoded = true
        let frontCommit = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        let frontWaitBegan = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        commandBuffer.waitUntilCompleted()
        let frontWaitEnded = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        if let error = commandBuffer.error { throw error }
        if let profile {
            var billed = Inkling2StageTiming()
            billed.add(
                Inkling2BufferClock(
                    encodeStart: encodeStart, commit: frontCommit, buffer: commandBuffer
                ),
                drainNanoseconds: frontWaitEnded - frontWaitBegan
            )
            profile.record(
                stage: .attentionFront,
                wallNanoseconds: frontWaitEnded - stageStart,
                buffers: billed
            )
        }

        // The host state advances NOW, from what the GPU actually produced:
        // the convolution histories take the projections, and the position's own
        // K/V joins the cache before the attention is encoded -- the host's
        // `attention` appends at exactly this point, and a row staged before the
        // write would be the previous position's.
        state.pushConvolution(0, value: readBack(workspace.key, count: kvHeads * headDim))
        state.pushConvolution(1, value: readBack(workspace.value, count: kvHeads * headDim))
        _ = state.append(
            key: readBack(workspace.convolvedKey, count: kvHeads * headDim),
            value: readBack(workspace.convolvedValue, count: kvHeads * headDim)
        )

        // The relative states and the sink logits are small host-side
        // projections; the kernel only needs their results, and computing them
        // on the host keeps this stage's device part to what it is for.
        let relative = try relativeStates(layer: layer, queryInput: queryInput, position: position, isFull: isFull)
        let sink = try sinkLogits(layer: layer, queryInput: queryInput)
        let relativeBuffer = try staging(workspace.relativeStates, relative)
        let sinkBuffer = try staging(workspace.sinkLogits, sink)
        let bankValues = try hostMatrix(.attnRelLogitsProj, layer: layer)
        let bankBuffer = try staging(workspace.bank, bankValues)

        // The KV rows this position attends over, staged: the second command
        // buffer, because the attention reads the row the first one wrote.
        let rows = try workspace.stage(state)
        let tailStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        if let profile {
            profile.record(
                stage: .attentionState,
                wallNanoseconds: tailStart - frontWaitEnded,
                buffers: Inkling2StageTiming()
            )
        }
        let (tailBuffer, tailEncoder) = try commandComponents()
        let tailArena = Inkling2ScalarArena(device: device)
        defer { tailArena.clear() }
        var tailEncoded = false
        defer { if !tailEncoded { tailEncoder.endEncoding() } }

        try attention(
            encoder: tailEncoder, workspace: workspace, state: state, rows: rows,
            position: position, relative: relativeBuffer, bank: bankBuffer,
            sink: sinkBuffer, output: workspace.attentionOutput, arena: tailArena,
            dumpsScores: dumpsAttentionScores
        )

        // The output gate, then the grouped output projection.
        try gemm(
            encoder: tailEncoder, role: .attnGate, layer: layer,
            input: queryInputBuffer, inputOffset: 0,
            output: workspace.gate, outputOffset: 0,
            rows: heads * headDim, columns: hidden,
            arena: tailArena
        )
        try gateMultiply(
            encoder: tailEncoder, output: workspace.attentionOutput,
            gate: workspace.gate, count: heads * headDim, arena: tailArena
        )
        try gemm(
            encoder: tailEncoder, role: .attnWoUd, layer: layer,
            input: workspace.attentionOutput, inputOffset: 0,
            output: workspace.projectionOutput, outputOffset: 0,
            rows: hidden, columns: heads * headDim,
            arena: tailArena
        )
        // The post-attention convolution rides this buffer when the caller asks
        // for it: it reads `projectionOutput` (this buffer's own last write) and
        // writes `workspace.hidden`, which nothing here touches, so the tail
        // pays one commit for both. Its history takes the pre-convolution value,
        // pushed by the caller after the wait; the next reader of that history
        // is the next visit.
        if foldsPostAttentionConvolution {
            try encodePostAttentionConvolution(
                encoder: tailEncoder, state: state, values: workspace.projectionOutput,
                valuesCount: hidden, dynamicInput: [],
                layer: layer, output: workspace.hidden, arena: tailArena
            )
        }

        tailEncoder.endEncoding()
        tailEncoded = true
        let tailCommit = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        tailBuffer.commit()
        let tailWaitBegan = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        tailBuffer.waitUntilCompleted()
        let tailWaitEnded = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        if let error = tailBuffer.error { throw error }
        if let profile {
            var billed = Inkling2StageTiming()
            billed.add(
                Inkling2BufferClock(encodeStart: tailStart, commit: tailCommit, buffer: tailBuffer),
                drainNanoseconds: tailWaitEnded - tailWaitBegan
            )
            profile.record(
                stage: .attentionTail,
                wallNanoseconds: tailWaitEnded - tailStart,
                buffers: billed
            )
        }
        if foldsPostAttentionConvolution {
            state.pushConvolution(2, value: readBack(workspace.projectionOutput, count: hidden))
        }
        return readBack(workspace.projectionOutput, count: hidden)
    }

    /// The attention kernel over one visit's staged KV rows, in whatever form
    /// the visit's cache holds them.
    func attention(
        encoder: MTLComputeCommandEncoder,
        workspace: Inkling2DeviceWorkspace,
        state: Inkling2VisitState,
        rows: Int,
        position: Int,
        relative: MTLBuffer,
        bank: MTLBuffer,
        sink: MTLBuffer,
        output: MTLBuffer,
        arena: Inkling2ScalarArena,
        dumpsScores: Bool = false
    ) throws {
        let isFull = state.isFullAttention
        let heads = isFull ? config.numAttentionHeads : config.swaNumAttentionHeads
        let kvHeads = isFull ? config.numKeyValueHeads : config.swaNumKeyValueHeads
        let headDim = isFull ? config.headDim : config.swaHeadDim
        let kvWidth = kvHeads * headDim
        // The staging buffers were laid out for one storage form, and the two
        // are not interchangeable: a byte row read as floats is a plausible
        // number, not a fault. Refuse instead.
        guard workspace.stagedStorage == state.storage else {
            throw Inkling2ForwardError.unsupported(
                "the workspace stages \(workspace.stagedStorage.rawValue) KV rows but this visit "
                    + "holds \(state.storage.rawValue)"
            )
        }
        // The rows are split across threadgroups rather than walked serially in
        // one: one threadgroup per (head, split), then a combine pass. The
        // single-threadgroup kernels (`inkling_attention_f32` and its INT8
        // twin) stay in the shader as the reference this path is checked
        // against, and their per-head form is what the split was measured
        // against -- 24 threadgroups of 32 lanes, a reduction and three
        // barriers per row.
        let plan = Inkling2AttentionSplit.plan(rowCount: rows)
        let values: [UInt32] = [
            UInt32(heads), UInt32(heads / kvHeads), UInt32(headDim), UInt32(kvWidth),
            UInt32(rows), UInt32(state.windowStart), UInt32(position),
            UInt32(config.dRel), UInt32(isFull ? config.relExtent : config.slidingWindowSize),
            UInt32(plan.chunk), UInt32(plan.splits)
        ]
        switch state.storage {
        case .float32:
            encoder.setComputePipelineState(try pipeline("inkling_attention_split_f32"))
            encoder.setBuffer(workspace.query, offset: 0, index: 0)
            encoder.setBuffer(workspace.stagedKeys, offset: 0, index: 1)
            encoder.setBuffer(workspace.stagedValues, offset: 0, index: 2)
            encoder.setBuffer(relative, offset: 0, index: 3)
            encoder.setBuffer(bank, offset: 0, index: 4)
            encoder.setBuffer(sink, offset: 0, index: 5)
            encoder.setBuffer(workspace.attentionPartials, offset: 0, index: 6)
            for (index, value) in values.enumerated() {
                encoder.setBuffer(arena.make(value), offset: 0, index: 7 + index)
            }
            // An optional score dump, into the CALLER's workspace buffer: the host
            // captures `scores_head0`, and comparing them is what separates the bias
            // from the softmax from the value fold. It has to outlive this call for
            // the caller to read it, so it cannot be the arena's. Each split writes
            // its own row range, so head 0's slots are filled exactly once.
            encoder.setBuffer(workspace.scoreDump, offset: 0, index: 18)
            encoder.setBuffer(arena.make(dumpsScores), offset: 0, index: 19)
        case .int8:
            encoder.setComputePipelineState(try pipeline("inkling_attention_split_int8_f32"))
            encoder.setBuffer(workspace.query, offset: 0, index: 0)
            encoder.setBuffer(workspace.stagedKeys, offset: 0, index: 1)
            encoder.setBuffer(workspace.stagedValues, offset: 0, index: 2)
            encoder.setBuffer(workspace.stagedKeyScales, offset: 0, index: 3)
            encoder.setBuffer(workspace.stagedValueScales, offset: 0, index: 4)
            encoder.setBuffer(relative, offset: 0, index: 5)
            encoder.setBuffer(bank, offset: 0, index: 6)
            encoder.setBuffer(sink, offset: 0, index: 7)
            encoder.setBuffer(workspace.attentionPartials, offset: 0, index: 8)
            for (index, value) in values.enumerated() {
                encoder.setBuffer(arena.make(value), offset: 0, index: 9 + index)
            }
            encoder.setBuffer(workspace.scoreDump, offset: 0, index: 20)
            encoder.setBuffer(arena.make(dumpsScores), offset: 0, index: 21)
        }
        // A split stages only its own chunk, so this is 1.2 KB at the widest
        // arm instead of the whole row count -- which is what let the shipped
        // form's 8,160-row threadgroup-memory ceiling disappear.
        encoder.setThreadgroupMemoryLength(
            (32 + plan.chunk) * MemoryLayout<Float>.stride, index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: heads * plan.splits, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
        // The combine runs in the same encoder: compute dispatches on one
        // encoder execute in order, so it reads the splits this encoder just
        // wrote.
        encoder.setComputePipelineState(try pipeline("inkling_attention_split_combine_f32"))
        encoder.setBuffer(workspace.attentionPartials, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(arena.make(UInt32(heads)), offset: 0, index: 2)
        encoder.setBuffer(arena.make(UInt32(headDim)), offset: 0, index: 3)
        encoder.setBuffer(arena.make(UInt32(plan.splits)), offset: 0, index: 4)
        encoder.dispatchThreadgroups(
            MTLSize(width: heads, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1)
        )
    }

    // MARK: The MLP half

    /// The MLP half of one visit: the MLP norm, the router and its selection,
    /// the expert mixture, the post-MLP convolution and the residual add.
    ///
    /// `mlpInput` is the block-read result (and its message gate) the host
    /// computed, and `residual` is the vector the host adds back -- two values,
    /// not one: at a block boundary the read's stack carries the layer's input
    /// while the add is the post-attention value alone. The read itself stays on
    /// the host with the stack it reads; the kernel that would do it exists and
    /// is tested, and moving it here would stage the whole stack to save a
    /// softmax over a few dozen rows.
    public func mlpHalf(
        layer: Int,
        state: Inkling2VisitState,
        mlpInput: [Float],
        residual: [Float],
        workspace: Inkling2DeviceWorkspace,
        profile: Inkling2VisitProfile? = nil
    ) throws -> [Float] {
        let stageStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let hiddenWidth = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        let experts = config.numRoutedExperts
        let topK = config.numExpertsPerToken

        // The front's own bindings, priced before it encodes; the mixture prices
        // itself once its eight experts are resolved.
        try guardDenseBinding(
            "mlp front", roles: [.mlpNorm, .routerWeight, .routerBias], layer: layer
        )

        // 1. The MLP norm, then the router and its selection, in one buffer.
        let mlpInputBuffer = try staging(workspace.hidden, mlpInput)
        let encodeStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let (commandBuffer, encoder) = try commandComponents()
        let arena = Inkling2ScalarArena(device: device)
        defer { arena.clear() }
        var encoded = false
        defer { if !encoded { encoder.endEncoding() } }
        let normGain = try weights.tensor(.mlpNorm, layer: layer)
        try rmsNorm(
            encoder: encoder, input: mlpInputBuffer, inputOffset: 0,
            gain: normGain.buffer, gainOffset: Int(normGain.offset),
            output: workspace.attentionInput, outputOffset: 0, width: hiddenWidth, arena: arena
        )
        try gemm(
            encoder: encoder, role: .routerWeight, layer: layer,
            input: workspace.attentionInput, inputOffset: 0,
            output: workspace.routerScores, outputOffset: 0,
            rows: experts, columns: hiddenWidth, arena: arena
        )
        try sigmoidInPlace(encoder: encoder, vector: workspace.routerScores, count: experts, arena: arena)
        let biasTensor = try weights.tensor(.routerBias, layer: layer)
        try routerTopK(
            encoder: encoder, scores: workspace.routerScores,
            bias: biasTensor.buffer, biasOffset: Int(biasTensor.offset),
            indices: workspace.routeIndices, weights: workspace.routeWeights,
            numExperts: experts, topK: topK, routeScale: config.routeScale, arena: arena
        )
        encoder.endEncoding()
        encoded = true
        let frontCommit = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        let frontWaitBegan = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        commandBuffer.waitUntilCompleted()
        let frontWaitEnded = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        if let error = commandBuffer.error { throw error }
        if let profile {
            var billed = Inkling2StageTiming()
            billed.add(
                Inkling2BufferClock(
                    encodeStart: encodeStart, commit: frontCommit, buffer: commandBuffer
                ),
                drainNanoseconds: frontWaitEnded - frontWaitBegan
            )
            profile.record(
                stage: .mlpFront,
                wallNanoseconds: frontWaitEnded - stageStart,
                buffers: billed
            )
        }

        // The selection is a device value the host needs to address the bank.
        // The mixture's clock starts where the front's ended, so the two
        // read-backs and the accumulator clear are billed to the mixture rather
        // than falling between two stages.
        let mixtureStart = frontWaitEnded
        let indices = readBackIndices(workspace.routeIndices, count: topK)
        let routeWeights = readBack(workspace.routeWeights, count: topK)

        // 2. The experts. Each one is gate/up over the normed input, the
        //    softcapped SwiGLU, the down projection, and a weighted fold into
        //    the accumulator. All of it is ONE command buffer for the whole
        //    mixture: the dispatches of a compute encoder run in the order they
        //    were encoded, so the folds still land in route order, and nothing
        //    reads a value back between experts -- eight separate buffers meant
        //    eight commits and eight waits for arithmetic the queue had already
        //    serialized.
        //
        //    The eight are RESOLVED first (banks, residency, and the binding
        //    guard) so a mixture this device cannot wire is refused before any
        //    of it is encoded -- and so a slot fill runs once per expert.
        let mixture = try resolveMixture(layer: layer, experts: indices)
        let accumulatorPointer = workspace.residualAccumulator.contents()
            .bindMemory(to: Float.self, capacity: hiddenWidth)
        for index in 0..<hiddenWidth { accumulatorPointer[index] = 0 }
        var mixtureBytes: UInt64 = 0
        let mixtureEncodeStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let (mixtureBuffer, mixtureEncoder) = try commandComponents()
        let mixtureArena = Inkling2ScalarArena(device: device)
        var mixtureEncoded = false
        defer {
            if !mixtureEncoded { mixtureEncoder.endEncoding() }
            mixtureArena.clear()
        }
        for (slot, expert) in mixture.enumerated() {
            let routeWeight = routeWeights[slot]
            let gateUp = expert.gateUp
            let down = expert.down
            // A resident slot, when an arena is installed: one buffer holding
            // both banks, so the two projections read it at the slot's own
            // region offsets instead of the fused banks' expert offsets. The
            // layout comes from the MMAP banks (a slot is a byte copy of them),
            // so the two paths decode identically by construction.
            let resident = expert.resident
            if prefaultsExpertBanks {
                touchedChecksum &+= Self.touch(
                    gateUp.buffer, offset: gateUp.baseOffset + gateUp.expertOffset,
                    bytes: UInt64(2 * intermediate) * UInt64(gateUp.rowStride)
                )
                touchedChecksum &+= Self.touch(
                    gateUp.scaleBuffer, offset: gateUp.scaleBaseOffset + gateUp.scaleExpertOffset,
                    bytes: UInt64(2 * intermediate) * UInt64(gateUp.scaleRowStride)
                )
                touchedChecksum &+= Self.touch(
                    down.buffer, offset: down.baseOffset + down.expertOffset,
                    bytes: UInt64(hiddenWidth) * UInt64(down.rowStride)
                )
                touchedChecksum &+= Self.touch(
                    down.scaleBuffer, offset: down.scaleBaseOffset + down.scaleExpertOffset,
                    bytes: UInt64(hiddenWidth) * UInt64(down.scaleRowStride)
                )
            }
            if let resident {
                try slotExpertRow(
                    encoder: mixtureEncoder,
                    slot: resident.buffer, slotBaseOffset: resident.baseOffset,
                    address: resident.address, isGateUp: true,
                    input: workspace.attentionInput, output: workspace.expertGateUp,
                    rows: 2 * intermediate, logicalColumns: hiddenWidth,
                    layout: gateUp.layout,
                    outerScale: gateUp.layout.requiresAmaxDenominator && gateUp.denominator > 0
                        ? gateUp.amax / gateUp.denominator : 1,
                    arena: mixtureArena
                )
            } else {
                try expertRow(
                    encoder: mixtureEncoder,
                    weight: gateUp.buffer, weightOffset: gateUp.baseOffset + gateUp.expertOffset,
                    scales: gateUp.scaleBuffer, scaleOffset: gateUp.scaleBaseOffset + gateUp.scaleExpertOffset,
                    input: workspace.attentionInput, output: workspace.expertGateUp,
                    rows: 2 * intermediate, logicalColumns: hiddenWidth,
                    layout: gateUp.layout,
                    outerScale: gateUp.layout.requiresAmaxDenominator && gateUp.denominator > 0
                        ? gateUp.amax / gateUp.denominator : 1,
                    weightRowStride: gateUp.rowStride, scaleRowStride: gateUp.scaleRowStride,
                    arena: mixtureArena
                )
            }
            try softcappedSwiGLUInterleaved(
                encoder: mixtureEncoder, gateUp: workspace.expertGateUp,
                output: workspace.expertIntermediate, intermediate: intermediate, arena: mixtureArena
            )
            if let resident {
                try slotExpertRow(
                    encoder: mixtureEncoder,
                    slot: resident.buffer, slotBaseOffset: resident.baseOffset,
                    address: resident.address, isGateUp: false,
                    input: workspace.expertIntermediate, output: workspace.expertDown,
                    rows: hiddenWidth, logicalColumns: intermediate,
                    layout: down.layout,
                    outerScale: down.layout.requiresAmaxDenominator && down.denominator > 0
                        ? down.amax / down.denominator : 1,
                    arena: mixtureArena
                )
            } else {
                try expertRow(
                    encoder: mixtureEncoder,
                    weight: down.buffer, weightOffset: down.baseOffset + down.expertOffset,
                    scales: down.scaleBuffer, scaleOffset: down.scaleBaseOffset + down.scaleExpertOffset,
                    input: workspace.expertIntermediate, output: workspace.expertDown,
                    rows: hiddenWidth, logicalColumns: intermediate,
                    layout: down.layout,
                    outerScale: down.layout.requiresAmaxDenominator && down.denominator > 0
                        ? down.amax / down.denominator : 1,
                    weightRowStride: down.rowStride, scaleRowStride: down.scaleRowStride,
                    arena: mixtureArena
                )
            }
            try weightedAccumulate(
                encoder: mixtureEncoder, accumulator: workspace.residualAccumulator,
                source: workspace.expertDown, weight: routeWeight, count: hiddenWidth,
                arena: mixtureArena
            )
            if profile != nil {
                // The bank's own strides: a packed row is `rowStride` bytes and a
                // scale row is `scaleRowStride`, so this is what a residency
                // policy would have to move for this expert, not an estimate
                // from the logical shape.
                mixtureBytes += UInt64(2 * intermediate) &* UInt64(gateUp.rowStride &+ gateUp.scaleRowStride)
                mixtureBytes += UInt64(hiddenWidth) &* UInt64(down.rowStride &+ down.scaleRowStride)
            }
        }
        // 3. The post-MLP convolution rides this buffer too: it reads the
        //    accumulator the last expert wrote and writes `workspace.hidden`,
        //    which no expert touches. Its history takes the mixture
        //    (pre-convolution), pushed after the one wait -- the next reader of
        //    that history is the next visit.
        try convolve(
            encoder: mixtureEncoder, state: state, index: 3,
            values: workspace.residualAccumulator, valuesCount: hiddenWidth,
            dynamicInput: mlpInput,
            layer: layer, isFull: false, headDim: config.headDim,
            output: workspace.hidden, arena: mixtureArena
        )
        mixtureEncoder.endEncoding()
        mixtureEncoded = true
        let mixtureCommit = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        mixtureBuffer.commit()
        let mixtureWaitBegan = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        mixtureBuffer.waitUntilCompleted()
        let mixtureWaitEnded = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        if let error = mixtureBuffer.error { throw error }
        state.pushConvolution(3, value: readBack(workspace.residualAccumulator, count: hiddenWidth))
        let convolved = readBack(workspace.hidden, count: hiddenWidth)
        if let profile {
            var billed = Inkling2StageTiming()
            billed.add(
                Inkling2BufferClock(
                    encodeStart: mixtureEncodeStart, commit: mixtureCommit, buffer: mixtureBuffer
                ),
                drainNanoseconds: mixtureWaitEnded - mixtureWaitBegan
            )
            profile.record(
                stage: .expertMixture,
                wallNanoseconds: DispatchTime.now().uptimeNanoseconds - mixtureStart,
                buffers: billed
            )
            profile.recordExperts(count: indices.count, bytes: mixtureBytes)
            if !expertArenas.isEmpty {
                profile.recordArena(Inkling2ArenaCounters.merged(
                    expertArenas.map { $0.arena.statistics }
                ))
            }
        }
        return zip(residual, convolved).map(+)
    }

    /// `out = rmsnorm(input, gain)`, one threadgroup over the row.
    func rmsNorm(
        encoder: MTLComputeCommandEncoder,
        input: MTLBuffer, inputOffset: Int,
        gain: MTLBuffer, gainOffset: Int,
        output: MTLBuffer, outputOffset: Int,
        width: Int,
        arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_rmsnorm_f32"))
        encoder.setBuffer(input, offset: inputOffset, index: 0)
        // A tensor's offset inside its shard is not buffer-aligned, so the gain
        // binds at zero and its offset travels as a constant.
        encoder.setBuffer(gain, offset: 0, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBuffer(arena.make(UInt32(width)), offset: 0, index: 3)
        encoder.setBuffer(arena.make(config.rmsNormEps), offset: 0, index: 4)
        encoder.setBuffer(arena.make(UInt64(gainOffset)), offset: 0, index: 5)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
        )
    }

    func sigmoidInPlace(
        encoder: MTLComputeCommandEncoder, vector: MTLBuffer, count: Int, arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_sigmoid_inplace_f32"))
        encoder.setBuffer(vector, offset: 0, index: 0)
        encoder.setBuffer(arena.make(UInt32(count)), offset: 0, index: 1)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func weightedAccumulate(
        encoder: MTLComputeCommandEncoder, accumulator: MTLBuffer, source: MTLBuffer,
        weight: Float, count: Int, arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_weighted_accumulate_f32"))
        encoder.setBuffer(accumulator, offset: 0, index: 0)
        encoder.setBuffer(source, offset: 0, index: 1)
        encoder.setBuffer(arena.make(weight), offset: 0, index: 2)
        encoder.setBuffer(arena.make(UInt32(count)), offset: 0, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: count, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func softcappedSwiGLUInterleaved(
        encoder: MTLComputeCommandEncoder, gateUp: MTLBuffer, output: MTLBuffer,
        intermediate: Int, arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_softcapped_swiglu_interleaved_f32"))
        encoder.setBuffer(gateUp, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBuffer(arena.make(UInt32(intermediate)), offset: 0, index: 2)
        encoder.setBuffer(arena.make(config.mlpGateSoftcap), offset: 0, index: 3)
        encoder.setBuffer(arena.make(config.mlpUpSoftcap), offset: 0, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: intermediate, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func routerTopK(
        encoder: MTLComputeCommandEncoder, scores: MTLBuffer, bias: MTLBuffer, biasOffset: Int,
        indices: MTLBuffer, weights: MTLBuffer, numExperts: Int, topK: Int,
        routeScale: Float, arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_router_topk_f32"))
        encoder.setBuffer(scores, offset: 0, index: 0)
        encoder.setBuffer(bias, offset: 0, index: 1)
        encoder.setBuffer(indices, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: 0, index: 3)
        encoder.setBuffer(arena.make(UInt32(numExperts)), offset: 0, index: 4)
        encoder.setBuffer(arena.make(UInt32(topK)), offset: 0, index: 5)
        encoder.setBuffer(arena.make(routeScale), offset: 0, index: 6)
        // The bias binds at zero with its offset as a constant: a tensor's
        // offset inside its shard is not buffer-aligned.
        encoder.setBuffer(arena.make(UInt64(biasOffset)), offset: 0, index: 7)
        encoder.setThreadgroupMemoryLength(
            // 32 warp sums, the steered ranks padded to a threadgroup stride,
            // the chosen ids, then one (rank, id) pair per warp for the
            // cross-warp merge. 64 threads is two warps; the merge reads up to
            // sixteen slots.
            (32 + ((numExperts + 63) / 64) * 64 + 32 + 16 + 16) * MemoryLayout<Float>.stride,
            index: 0
        )
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    func expertRow(
        encoder: MTLComputeCommandEncoder,
        weight: MTLBuffer, weightOffset: UInt64,
        scales: MTLBuffer, scaleOffset: UInt64,
        input: MTLBuffer, output: MTLBuffer, rows: Int, logicalColumns: Int,
        layout: Inkling2ExpertQuantLayout, outerScale: Float,
        weightRowStride: Int, scaleRowStride: Int,
        arena: Inkling2ScalarArena
    ) throws {
        encoder.setComputePipelineState(try pipeline("inkling_expert_row_f32"))
        encoder.setBuffer(weight, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(input, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBuffer(arena.make(UInt32(logicalColumns)), offset: 0, index: 4)
        encoder.setBuffer(arena.make(UInt32(weightRowStride)), offset: 0, index: 5)
        encoder.setBuffer(arena.make(UInt32(scaleRowStride)), offset: 0, index: 6)
        encoder.setBuffer(arena.make(UInt32(layout.blockSize)), offset: 0, index: 7)
        encoder.setBuffer(arena.make(layout == .nvfp4), offset: 0, index: 8)
        encoder.setBuffer(arena.make(outerScale), offset: 0, index: 9)
        encoder.setBuffer(arena.make(weightOffset), offset: 0, index: 10)
        encoder.setBuffer(arena.make(scaleOffset), offset: 0, index: 11)
        encoder.setThreadgroupMemoryLength(32 * MemoryLayout<Float>.stride, index: 0)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
        )
    }

    /// One expert projection straight out of a RESIDENT slot: the arena's
    /// packed buffer, with each region addressed from the slot's own base.
    ///
    /// The binding itself is `Inkling2SlotProjection.encode`, which lives with
    /// the arena's geometry so the packed-slot layout and the way a kernel reads
    /// it cannot drift apart -- and so it can be run against the host rule with
    /// no checkpoint.
    ///
    /// `slotBaseOffset` is the slot index times `geometry.slotBytes`; it travels
    /// as a constant because a packed region's offset is not 256-byte aligned.
    func slotExpertRow(
        encoder: MTLComputeCommandEncoder,
        slot: MTLBuffer, slotBaseOffset: Int,
        address: Inkling2ExpertSlotGeometry.SlotAddress,
        isGateUp: Bool,
        input: MTLBuffer, output: MTLBuffer,
        rows: Int, logicalColumns: Int,
        layout: Inkling2ExpertQuantLayout, outerScale: Float,
        arena: Inkling2ScalarArena
    ) throws {
        Inkling2SlotProjection.encode(
            pipeline: try pipeline("inkling_expert_row_f32"),
            encoder: encoder, slot: slot, slotBaseOffset: slotBaseOffset,
            address: address, isGateUp: isGateUp,
            input: input, output: output, rows: rows, logicalColumns: logicalColumns,
            layout: layout, outerScale: outerScale, arena: arena
        )
    }

    /// The slots holding every routed expert of one mixture, filling the misses
    /// in ONE fan-out.
    ///
    /// Nil for an expert when no arena packs its bank -- the mixture then reads
    /// the shard mmap exactly as it always did. A fill reads the expert's four
    /// ranges out of the fused banks by pread; the offsets are the banks' own, so
    /// a slot is a byte-for-byte copy of what the mmap path would have read.
    ///
    /// The whole visit's routing is known before any expert is read, so the
    /// experts go to their arena in ONE batch: the shipped form filled them one
    /// call at a time, which asks this SSD for four concurrent reads when
    /// thirty-two are available. Slice 34 measured that shape at 4.8 GB/s
    /// against 8.0 GB/s batched over the same slots and the same bytes (1.68x
    /// on the probe, 1.34x inside a 1,032-position walk).
    func residentSlots(
        layer: Int,
        banks: [(
            gateUp: Inkling2DeviceWeights.ExpertBank,
            down: Inkling2DeviceWeights.ExpertBank
        )],
        experts: [Int]
    ) throws -> [(
        buffer: MTLBuffer, baseOffset: Int, address: Inkling2ExpertSlotGeometry.SlotAddress
    )?] {
        var residents = [(
            buffer: MTLBuffer, baseOffset: Int, address: Inkling2ExpertSlotGeometry.SlotAddress
        )?](repeating: nil, count: experts.count)
        guard !expertArenas.isEmpty else { return residents }
        var held = [Bool](repeating: false, count: experts.count)
        // A slot is a byte copy at FIXED strides, so a layer quantized another
        // way needs an arena of its own. The published checkpoint mixes NVFP4 and
        // MXFP8 layers, and a copy at the wrong strides would report hits over
        // the wrong bytes -- so each bank goes to the arena that HOLDS it, and a
        // bank no arena holds is refused (counted) and left on the mmap.
        for installed in expertArenas {
            var indices: [Int] = []
            var requests: [Inkling2ExpertArena.FillRequest] = []
            for (index, bank) in banks.enumerated() where !held[index]
                && installed.geometry.holds(
                    weightRowStride: bank.gateUp.rowStride,
                    scaleRowStride: bank.gateUp.scaleRowStride,
                    downWeightRowStride: bank.down.rowStride,
                    downScaleRowStride: bank.down.scaleRowStride
                ) {
                indices.append(index)
                requests.append(Inkling2ExpertArena.FillRequest(
                    layer: layer, expert: experts[index],
                    gateUpSourceOffset: bank.gateUp.fileWeightOffset,
                    gateUpScaleSourceOffset: bank.gateUp.fileScaleOffset,
                    downSourceOffset: bank.down.fileWeightOffset,
                    downScaleSourceOffset: bank.down.fileScaleOffset
                ))
            }
            guard !requests.isEmpty else { continue }
            for index in indices { held[index] = true }
            let slots = try installed.arena.fill(requests, reader: installed.reader)
            // A bookkeeping-only arena has nowhere to put the bytes: the caller
            // then fills the slots itself, exactly as the serial path allowed.
            guard let buffer = installed.arena.slotBuffer else { continue }
            for (position, index) in indices.enumerated() {
                residents[index] = (
                    buffer,
                    slots[position] * installed.geometry.slotBytes,
                    installed.geometry.slotAddress
                )
            }
        }
        for index in experts.indices where !held[index] {
            expertArenas[0].arena.recordIncompatible()
        }
        return residents
    }

    /// Every routed expert resolved, and the mixture's binding refused when this
    /// device cannot wire it.
    ///
    /// The resolve pass is what makes the guard honest: a resident expert binds
    /// its slot, not the two mmap segments its banks live in, so the two paths
    /// are priced differently. Both are priced AFTER the arena has decided
    /// residency, which is where a miss costs its `pread` -- so the guard never
    /// reports a binding the mixture would not have made.
    func resolveMixture(
        layer: Int, experts: [Int]
    ) throws -> [Inkling2ResolvedExpert] {
        var banks: [(
            gateUp: Inkling2DeviceWeights.ExpertBank,
            down: Inkling2DeviceWeights.ExpertBank
        )] = []
        banks.reserveCapacity(experts.count)
        for expert in experts {
            banks.append((
                try weights.expertBank(.expertW13, layer: layer, expert: expert),
                try weights.expertBank(.expertW2, layer: layer, expert: expert)
            ))
        }
        let residents = try residentSlots(layer: layer, banks: banks, experts: experts)
        var resolved: [Inkling2ResolvedExpert] = []
        resolved.reserveCapacity(experts.count)
        for index in experts.indices {
            resolved.append(Inkling2ResolvedExpert(
                gateUp: banks[index].gateUp, down: banks[index].down,
                resident: residents[index]
            ))
        }
        try guardMixtureBinding(layer: layer, experts: resolved)
        return resolved
    }

    /// The bytes of bank this device will wire in one command buffer.
    ///
    /// `recommendedMaxWorkingSetSize` is Metal's own ceiling for resident
    /// buffers, so it is the line the guards draw rather than a number tuned to
    /// one machine. Measured on a 16 GB M1 Pro: layer 0's mixture binds 17.24 GB
    /// across two segments (8.66 GiB and 8.57 GiB) and reads zeros; each segment
    /// alone reads correctly, so the ceiling is the failure's own boundary and
    /// not an estimate. The dense stages draw the same line -- they bind
    /// segments rather than expert slices, so a segmentation that closes buffers
    /// near `maxBufferLength` can put two of them in one command buffer.
    public var mixtureBindingBudget: UInt64 {
        UInt64(device.recommendedMaxWorkingSetSize)
    }

    /// Prices the segments one DENSE stage's command buffer binds and refuses
    /// them past the device's ceiling.
    ///
    /// Each dense stage resolves its projections inside `gemm`, so the guard
    /// resolves them here first, prices them exactly as the kernels will bind
    /// them, and only then lets the stage encode. The roles passed are one
    /// command buffer's: the attention front and tail are priced separately
    /// because their buffers never coexist, and charging a stage for a binding
    /// it never makes would refuse a graph that runs.
    func guardDenseBinding(
        _ stage: String, roles: [Inkling2TensorRole], layer: Int?
    ) throws {
        var binding = Inkling2MixtureBinding()
        for role in roles {
            let tensor = try weights.tensor(role, layer: layer)
            binding.add(tensor.buffer, label: "the \(role.rawValue) weight")
        }
        try binding.refuse(
            over: mixtureBindingBudget,
            subject: "the \(stage) stage",
            remedy: "a dense stage can only run if the checkpoint's segments are closed "
                + "below this ceiling"
        )
    }

    func guardMixtureBinding(layer: Int, experts: [Inkling2ResolvedExpert]) throws {
        var binding = Inkling2MixtureBinding()
        for expert in experts {
            if let resident = expert.resident {
                binding.add(resident.buffer, label: "one resident slot bank")
            } else {
                binding.add(expert.gateUp.buffer, label: "the w13 bank of layer \(layer)")
                binding.add(expert.gateUp.scaleBuffer, label: "the w13 scales of layer \(layer)")
                binding.add(expert.down.buffer, label: "the w2 bank of layer \(layer)")
                binding.add(expert.down.scaleBuffer, label: "the w2 scales of layer \(layer)")
            }
        }
        try binding.refuse(over: mixtureBindingBudget)
    }

    // MARK: The whole visit

    /// One visit with both halves on the device: the host's own
    /// `Inkling2CPUForward.runVisit` with `attentionHalf` and `mlpHalf` in place
    /// of the two halves it calls.
    ///
    /// The SCHEDULE (the two block-read points, the boundary append, which
    /// residual the mixture is added to, the message gates before the norms) is
    /// transcribed from the host method line for line rather than re-derived:
    /// it is the part the reference is subtle about, and it has exactly one
    /// correct form. Everything this does NOT move -- the block reads, the
    /// message gates, the input norms -- is the host's own code, called on the
    /// same host graph object the device helper was built with.
    ///
    /// `position` is the absolute position, and the visit state it drives is
    /// the sequence's own: after a visit it holds the KV row and the four
    /// convolution histories the host would have left.
    public func runVisit(
        layer: Int,
        pass: Int,
        hidden initial: [Float],
        blockStates: inout [[Float]],
        messages: [[Float]]?,
        position: Int,
        workspace: Inkling2DeviceWorkspace
    ) throws -> (output: [Float], attentionQueries: Int) {
        let profile: Inkling2VisitProfile? = recordsVisitTiming ? visitProfile : nil
        let visitStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds
        let visitIndex = try host.visitIndex(layer: layer, pass: pass)
        let state = host.states[visitIndex]
        let isFull = !config.isLocalLayer(layer)
        let blocks = config.nLayersPerAttnResBlock
        let blockResidual = initial
        let hidden = initial

        // 1. The attention input read, and the message gates on its results.
        var qRead = hidden
        var kRead = hidden
        var vRead = hidden
        var attentionQueries = 0
        if layer > 0 {
            let packed = try weights.hostMatrix(.attnResAttnQuery, layer: layer - 1)
            let queries = host.unpackRows(packed, count: isFull ? 3 : 1)
            attentionQueries = queries.count
            let reads = host.blockRead(blockStates: blockStates, partial: hidden, queries: queries)
            qRead = reads[0]
            kRead = isFull && reads.count > 1 ? reads[1] : reads[0]
            vRead = isFull && reads.count > 2 ? reads[2] : reads[0]
        }
        var queryRead = qRead
        var keyRead = kRead
        var valueRead = vRead
        if let messages, messages.count == 4 {
            queryRead = try host.messageGate(role: .controllerAttnDown, layer: layer, current: queryRead, message: messages[0])
            keyRead = try host.messageGate(role: .controllerKDown, layer: layer, current: keyRead, message: messages[1])
            valueRead = try host.messageGate(role: .controllerVDown, layer: layer, current: valueRead, message: messages[2])
        }
        // 2. The three projection inputs, each normalized as the host does.
        let queryInput: [Float]
        let keyInput: [Float]
        let valueInput: [Float]
        if isFull {
            queryInput = Inkling2Math.rmsNorm(
                queryRead, gain: try weights.hostMatrix(.qInputNorm, layer: layer), eps: config.rmsNormEps
            )
            keyInput = Inkling2Math.rmsNorm(
                keyRead, gain: try weights.hostMatrix(.kInputNorm, layer: layer), eps: config.rmsNormEps
            )
            valueInput = Inkling2Math.rmsNorm(
                valueRead, gain: try weights.hostMatrix(.vInputNorm, layer: layer), eps: config.rmsNormEps
            )
        } else {
            let gain = try weights.hostMatrix(.attnNorm, layer: layer)
            queryInput = Inkling2Math.rmsNorm(queryRead, gain: gain, eps: config.rmsNormEps)
            keyInput = Inkling2Math.rmsNorm(keyRead, gain: gain, eps: config.rmsNormEps)
            valueInput = Inkling2Math.rmsNorm(valueRead, gain: gain, eps: config.rmsNormEps)
        }

        // 3. The attention half, on the device.
        if let profile {
            profile.record(
                stage: .hostPrep,
                wallNanoseconds: DispatchTime.now().uptimeNanoseconds - visitStart,
                buffers: Inkling2StageTiming()
            )
        }
        _ = try attentionHalf(
            layer: layer, state: state, queryInput: queryInput, keyInput: keyInput,
            valueInput: valueInput, position: position, workspace: workspace,
            foldsPostAttentionConvolution: true, profile: profile
        )
        // 4. The post-attention convolution, already run inside the attention
        //    half's last buffer: it read that projection and wrote
        //    `workspace.hidden`, and the history took the pre-convolution value.
        let postAttention = readBack(workspace.hidden, count: config.hiddenSize)
        let mlpReadStart = profile == nil ? 0 : DispatchTime.now().uptimeNanoseconds

        // 5. The residual: the boundary append is LOCAL to the MLP read.
        var residual = postAttention
        var mlpStack = blockStates
        if layer % blocks == 0 {
            mlpStack.append(blockResidual)
        } else {
            residual = zip(blockResidual, postAttention).map(+)
        }
        let mlpPacked = try weights.hostMatrix(.attnResMlpQuery, layer: layer)
        var mlpHidden = host.blockRead(blockStates: mlpStack, partial: residual, queries: [mlpPacked])[0]
        if let messages, messages.count == 4 {
            mlpHidden = try host.messageGate(
                role: .controllerMlpDown, layer: layer, current: mlpHidden, message: messages[3]
            )
        }
        // 6. The MLP half, on the device (it returns residual + convolved mixture).
        if let profile {
            profile.record(
                stage: .mlpRead,
                wallNanoseconds: DispatchTime.now().uptimeNanoseconds - mlpReadStart,
                buffers: Inkling2StageTiming()
            )
        }
        let output = try mlpHalf(
            layer: layer, state: state, mlpInput: mlpHidden, residual: residual,
            workspace: workspace, profile: profile
        )
        if let profile {
            profile.closeVisit(
                wallNanoseconds: DispatchTime.now().uptimeNanoseconds - visitStart,
                attentionQueries: attentionQueries
            )
        }
        return (output, attentionQueries)
    }

    /// The post-attention convolution, encoded into a caller's buffer.
    ///
    /// One dispatch, and the wait and the history push belong to the caller:
    /// folded into the attention tail it shares that buffer's single wait, and
    /// index 2's history is not read again until the NEXT visit, so pushing it
    /// after the tail's wait is the same ordering with one commit fewer.
    func encodePostAttentionConvolution(
        encoder: MTLComputeCommandEncoder,
        state: Inkling2VisitState, values: MTLBuffer, valuesCount: Int,
        dynamicInput: [Float], layer: Int,
        output: MTLBuffer, arena: Inkling2ScalarArena
    ) throws {
        try convolve(
            encoder: encoder, state: state, index: 2,
            values: values, valuesCount: valuesCount, dynamicInput: dynamicInput,
            layer: layer, isFull: false, headDim: config.headDim,
            output: output, arena: arena
        )
    }

    /// The routing indices as integers, which the host needs to address a bank.
    func readBackIndices(_ buffer: MTLBuffer, count: Int) -> [Int] {
        Array(UnsafeBufferPointer(
            start: buffer.contents().bindMemory(to: UInt32.self, capacity: count), count: count
        )).map(Int.init)
    }

    /// Reads every page of a weight range on the host, one byte per page.
    /// Measurement only: a no-op unless `prefaultsExpertBanks` is set.
    ///
    /// The bytes go into `touchedChecksum` rather than being written back: a
    /// shard is a read-only mapping of the checkpoint, so storing into it is a
    /// bus error, and an unused read would be eliminated. The checksum makes
    /// the loads observable without writing anything.
    nonisolated static func touch(_ buffer: MTLBuffer, offset: UInt64, bytes: UInt64) -> UInt8 {
        guard bytes > 0, offset < UInt64(buffer.length) else { return 0 }
        let length = Int(min(bytes, UInt64(buffer.length) - offset))
        let base = buffer.contents().advanced(by: Int(offset))
        let page = max(1, Int(getpagesize()))
        var checksum: UInt8 = 0
        var index = 0
        while index < length {
            checksum = checksum &+ base.load(fromByteOffset: index, as: UInt8.self)
            index += page
        }
        return checksum
    }
}

// MARK: - What a mixture binds

/// One installed arena: its slot layout, the bookkeeping, and the reader that
/// fills a slot from the shard file.
nonisolated public struct Inkling2InstalledExpertArena {
    public let geometry: Inkling2ExpertSlotGeometry
    public let arena: Inkling2ExpertArena
    public let reader: Inkling2ArenaShardReader

    public init(
        geometry: Inkling2ExpertSlotGeometry,
        arena: Inkling2ExpertArena,
        reader: Inkling2ArenaShardReader
    ) {
        self.geometry = geometry
        self.arena = arena
        self.reader = reader
    }

    /// Slots times one slot's bytes: the RAM this arena holds.
    public var bytes: Int { arena.capacity * geometry.slotBytes }
}

/// One routed expert, resolved: its two mmap banks, and the resident slot that
/// will serve it instead when an arena holds it.
///
/// Resolution happens for all eight routings BEFORE the mixture's encoder is
/// used, because the binding guard has to price what the command buffer will
/// bind and a slot fill must not run twice for the same expert.
nonisolated public struct Inkling2ResolvedExpert {
    public let gateUp: Inkling2DeviceWeights.ExpertBank
    public let down: Inkling2DeviceWeights.ExpertBank
    /// `(buffer, byte offset, region address)` of the slot holding this expert,
    /// or nil when the mixture reads the shard mmap for it.
    public let resident: (buffer: MTLBuffer, baseOffset: Int,
                          address: Inkling2ExpertSlotGeometry.SlotAddress)?

    public init(
        gateUp: Inkling2DeviceWeights.ExpertBank,
        down: Inkling2DeviceWeights.ExpertBank,
        resident: (buffer: MTLBuffer, baseOffset: Int,
                   address: Inkling2ExpertSlotGeometry.SlotAddress)?
    ) {
        self.gateUp = gateUp
        self.down = down
        self.resident = resident
    }
}

/// What one mixture command buffer binds, priced the way the driver charges it:
/// every distinct buffer WHOLE, not the slice of it an expert reads.
///
/// A dispatch that binds a bank and its scales binds two buffers, and on a
/// segmented checkpoint those are two segments. The price is therefore the
/// segments' own lengths -- 17.24 GB for layer 0 of the published checkpoint on
/// a 16 GB M1 Pro, which is over `recommendedMaxWorkingSetSize` and, measured,
/// returns ZEROS for every row with no command-buffer error at all (see the
/// plan's Phase 6 slice 19). Segments are deduplicated by buffer identity, so an
/// eight-expert mixture over one layer's two segments is priced once.
nonisolated public struct Inkling2MixtureBinding: Equatable {
    public struct Segment: Equatable, CustomStringConvertible {
        public let label: String
        public let bytes: UInt64
        public var description: String { "\(label) \(Inkling2MixtureBinding.describe(bytes))" }
    }

    public private(set) var segments: [Segment] = []
    private var seen: Set<ObjectIdentifier> = []

    public init() {}

    /// The segments' total, i.e. what the device is asked to wire.
    public var bytes: UInt64 { segments.reduce(0) { $0 + $1.bytes } }

    public mutating func add(_ buffer: MTLBuffer, label: String) {
        guard seen.insert(ObjectIdentifier(buffer)).inserted else { return }
        segments.append(Segment(label: label, bytes: UInt64(buffer.length)))
    }

    /// Refuses a binding past `budget`, naming every segment it would bind.
    ///
    /// `subject` is what the message says was refused -- the mixture of experts,
    /// or one of the dense stages, which bind a projection's segment per command
    /// buffer and fail the same silent way. `remedy` is the clause that follows;
    /// the mixture's is the arena, a dense stage's is a finer segmentation.
    public func refuse(
        over budget: UInt64,
        subject: String = "the expert mixture",
        remedy: String = "an expert arena serves each expert from a slot of its own and "
            + "stays under the line"
    ) throws {
        guard bytes > budget else { return }
        throw Inkling2ForwardError.unsupported(
            "\(subject) would bind \(Self.describe(bytes)) of bank in one command "
                + "buffer (\(segments.map(\.description).joined(separator: " + "))), past the "
                + "\(Self.describe(budget)) this device will wire. A binding that large "
                + "returns zeros for every row and reports no error, so it is refused here "
                + "rather than served as logits; \(remedy)."
        )
    }

    public static func describe(_ bytes: UInt64) -> String {
        let gigabyte = Double(bytes) / 1_073_741_824
        return gigabyte >= 1
            ? String(format: "%.2f GB", gigabyte)
            : String(format: "%.2f MB", Double(bytes) / 1_048_576)
    }
}
