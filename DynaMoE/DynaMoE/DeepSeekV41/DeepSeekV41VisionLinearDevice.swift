//
//  DeepSeekV41VisionLinearDevice.swift
//  DynaMoE
//
//  Phase: DeepSeek-V4.1-Flash port -- the vision tower's block GEMM on the device.
//
//  The tower's projections are 60% of what an image costs once its attention runs on
//  the GPU: the largest admissible image is 41.71 s on the app's own -O build, of
//  which 6.29 s is now the device attention and ~9.74 s is the per-token term -- one
//  `cblas_sgemm` per projection over the whole block of patch tokens ("DSV4.1 #37",
//  "DSV4.1 #41"). That host GEMM is Accelerate, not scalar Swift: 838 MFLOP per
//  patch token at 1.126 ms/token is roughly 744 GFLOP/s, close to what this machine's
//  CPU can do. Moving it to the GPU is therefore a bet on the device's headroom
//  rather than a repair, and it is also what any device-resident tower needs: a
//  token matrix that leaves for the host between blocks costs a transfer per stage.
//
//  This file is the stage and only the stage -- `dsv41_vision_gemm_f32`, its shape
//  validation, and a runner whose return value is the dispatch's GPU time. The tower
//  still calls `linearBatched`; wiring `encode` to this is a separate change, and it
//  should be made only against a measured win at the published shapes.
//

import Foundation
import Metal

/// The device form of the tower's projection stage: `C = A * Bᵀ + bias`.
///
/// The operands are the tower's own layout (`A` is `[rows][depth]`, `B` is
/// `[outputs][depth]`), so a caller hands over the token block and the weight matrix
/// unwidened, and the result is `[rows][outputs]` -- the same rows `linearBatched`
/// returns, in the same order.
nonisolated public final class DeepSeekV41VisionLinearDevice {

    /// One threadgroup owns this many output rows and columns.
    public static let tileRows = 64
    public static let tileColumns = 64
    /// The depth chunk staged in threadgroup memory per step.
    public static let tileDepth = 16
    /// Eight SIMD groups, laid out 2x4 over the tile.
    public static let threadgroupWidth = 256

    /// The three tiles the kernel binds, in bytes: the input chunk, the weight
    /// chunk, and the output tile the accumulators are stored into before the
    /// guarded write-out. The kernel names them rather than declaring them because
    /// the offline compiler duplicates declared tiles -- 49,152 bytes for a kernel
    /// that asks for 24,576 -- which the driver runs and the validation layer
    /// rejects against the device's 32,768 limit.
    public static var threadgroupBytes: Int {
        let stride = MemoryLayout<Float>.stride
        return (tileRows * tileDepth + tileColumns * tileDepth + tileRows * tileColumns) * stride
    }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    private let gemmPipeline: MTLComputePipelineState

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
        guard let function = library.makeFunction(name: "dsv41_vision_gemm_f32") else {
            throw DeepSeekV41VisionDeviceError.unavailable(
                "kernel dsv41_vision_gemm_f32 is missing"
            )
        }
        self.device = device
        self.queue = queue
        self.gemmPipeline = try device.makeComputePipelineState(function: function)
        guard Self.threadgroupBytes <= device.maxThreadgroupMemoryLength else {
            throw DeepSeekV41VisionDeviceError.unavailable(
                "the GEMM needs \(Self.threadgroupBytes) bytes of threadgroup memory and "
                    + "this device allows \(device.maxThreadgroupMemoryLength)"
            )
        }
    }

    // MARK: Stage runner

    /// Runs one projection into a caller-owned output buffer and returns its GPU
    /// time, without the readback.
    ///
    /// The readback is not part of the stage: at the largest image the output is
    /// `rows * outputs` floats, and materializing it as `[[Float]]` costs more than
    /// the dispatch it describes. A measurement of this stage times this call; the
    /// rows are read separately when a caller needs to compare them.
    @discardableResult
    public func runLinear(
        inputs: MTLBuffer,
        weights: MTLBuffer,
        bias: MTLBuffer?,
        output: MTLBuffer,
        rows: Int,
        outputs: Int,
        depth: Int
    ) throws -> Double {
        try validate(
            inputs: inputs, weights: weights, output: output,
            rows: rows, outputs: outputs, depth: depth
        )
        let zeroBias: MTLBuffer?
        if bias == nil {
            guard let zeros = device.makeBuffer(
                length: max(outputs, 1) * MemoryLayout<Float>.stride,
                options: .storageModeShared
            ) else {
                throw DeepSeekV41VisionDeviceError.unavailable("could not allocate a bias buffer")
            }
            memset(zeros.contents(), 0, zeros.length)
            zeroBias = zeros
        } else {
            zeroBias = bias
        }
        guard let biasBuffer = zeroBias,
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw DeepSeekV41VisionDeviceError.unavailable("no command buffer")
        }
        var rowCount = UInt32(rows)
        var outputCount = UInt32(outputs)
        var depthCount = UInt32(depth)
        encoder.setComputePipelineState(gemmPipeline)
        encoder.setBuffer(inputs, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: 0, index: 1)
        encoder.setBuffer(biasBuffer, offset: 0, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBytes(&outputCount, length: MemoryLayout<UInt32>.stride, index: 5)
        encoder.setBytes(&depthCount, length: MemoryLayout<UInt32>.stride, index: 6)
        encoder.setThreadgroupMemoryLength(
            Self.tileRows * Self.tileDepth * MemoryLayout<Float>.stride, index: 0
        )
        encoder.setThreadgroupMemoryLength(
            Self.tileColumns * Self.tileDepth * MemoryLayout<Float>.stride, index: 1
        )
        encoder.setThreadgroupMemoryLength(
            Self.tileRows * Self.tileColumns * MemoryLayout<Float>.stride, index: 2
        )
        let columnGroups = (outputs + Self.tileColumns - 1) / Self.tileColumns
        let rowGroups = (rows + Self.tileRows - 1) / Self.tileRows
        encoder.dispatchThreadgroups(
            MTLSize(width: max(columnGroups, 1), height: max(rowGroups, 1), depth: 1),
            threadsPerThreadgroup: MTLSize(width: Self.threadgroupWidth, height: 1, depth: 1)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw DeepSeekV41VisionDeviceError.unavailable("vision GEMM failed: \(error)")
        }
        return commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
    }

    /// One projection as rows, the shape `DeepSeekV41VisionForward.linear` returns.
    ///
    /// The rows are validated before anything is copied, because a ragged input row
    /// would make the kernel read another row's values for the missing tail.
    public func linear(
        _ inputs: [[Float]],
        weight: DeepSeekV41Matrix,
        bias: [Float]?
    ) throws -> [[Float]] {
        let rows = inputs.count
        let outputs = weight.rows
        let depth = weight.columns
        guard rows > 0, outputs > 0, depth > 0, weight.values.count >= outputs * depth else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "linear needs a non-empty \(outputs)x\(depth) matrix, got "
                    + "\(weight.values.count) values"
            )
        }
        if let bias, bias.count != outputs {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "bias has \(bias.count) values, expected \(outputs)"
            )
        }
        for (index, row) in inputs.enumerated() where row.count != depth {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "input row \(index) has \(row.count) values, expected \(depth)"
            )
        }
        let flat = inputs.flatMap { $0 }
        let inputBuffer = try makeBuffer(flat)
        let weightBuffer = try makeBuffer(weight.values)
        guard let outputBuffer = device.makeBuffer(
            length: max(rows * outputs, 1) * MemoryLayout<Float>.stride,
            options: .storageModeShared
        ) else {
            throw DeepSeekV41VisionDeviceError.unavailable("could not allocate a GEMM buffer")
        }
        let biasBuffer = try bias.map { try makeBuffer($0) }
        _ = try runLinear(
            inputs: inputBuffer, weights: weightBuffer, bias: biasBuffer,
            output: outputBuffer, rows: rows, outputs: outputs, depth: depth
        )
        return readRows(outputBuffer, rows: rows, columns: outputs)
    }

    // MARK: Buffers

    /// A shared-storage copy of `values`, for the operands and the result.
    public func makeBuffer(_ values: [Float]) throws -> MTLBuffer {
        let length = max(values.count, 1) * MemoryLayout<Float>.stride
        guard let buffer = values.withUnsafeBytes({ bytes -> MTLBuffer? in
            guard let base = bytes.baseAddress else {
                return device.makeBuffer(length: length, options: .storageModeShared)
            }
            return device.makeBuffer(
                bytes: base, length: max(bytes.count, 4), options: .storageModeShared
            )
        }) else {
            throw DeepSeekV41VisionDeviceError.unavailable(
                "could not allocate a buffer of \(values.count) floats"
            )
        }
        return buffer
    }

    /// The rows of a `[rows][columns]` result, copied off the shared buffer.
    public func readRows(_ buffer: MTLBuffer, rows: Int, columns: Int) -> [[Float]] {
        let values = buffer.contents().assumingMemoryBound(to: Float.self)
        return (0..<rows).map { row in
            Array(UnsafeBufferPointer(start: values + row * columns, count: columns))
        }
    }

    private func validate(
        inputs: MTLBuffer,
        weights: MTLBuffer,
        output: MTLBuffer,
        rows: Int,
        outputs: Int,
        depth: Int
    ) throws {
        guard rows > 0, outputs > 0, depth > 0 else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the GEMM needs positive rows, outputs and depth, got "
                    + "\(rows) x \(outputs) x \(depth)"
            )
        }
        let stride = MemoryLayout<Float>.stride
        guard inputs.length >= rows * depth * stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the input buffer is smaller than \(rows) x \(depth)"
            )
        }
        guard weights.length >= outputs * depth * stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the weight buffer is smaller than \(outputs) x \(depth)"
            )
        }
        guard output.length >= rows * outputs * stride else {
            throw DeepSeekV41VisionDeviceError.shapeMismatch(
                "the output buffer is smaller than \(rows) x \(outputs)"
            )
        }
    }
}
