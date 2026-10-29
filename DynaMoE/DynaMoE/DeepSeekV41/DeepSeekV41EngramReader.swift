//
//  DeepSeekV41EngramReader.swift
//  DynaMoE
//
//  Row-granular streaming for the Engram tables.
//
//  Each table is ~100 GB in ONE safetensors tensor, which is far past any Apple
//  GPU's `maxBufferLength` (the segmentation step in `InferenceEngine` refuses
//  such a tensor). So the table is never mapped and never becomes an
//  `MTLBuffer`: a token needs 24 rows of 264 bytes per Engram layer, and those
//  rows are pread straight into a staging buffer that the GPU dequantizes.
//
//  The read pattern is the reason the API takes a batch: the hash ids for a
//  position are known before the layer runs, so all of a layer's rows are
//  issued together and can be serviced with real queue depth. A serial loop of
//  24 small random reads per token would otherwise spend the whole decode step
//  waiting on seek latency.
//

import Foundation
#if canImport(Darwin)
import Darwin
#endif

nonisolated public enum DeepSeekV41EngramReadError: Error, Equatable, CustomStringConvertible {
    case cannotOpen(String)
    case tableTooSmall(expected: Int, actual: Int)
    case rowOutOfRange(offset: Int, length: Int, fileSize: Int)
    case shortRead(offset: Int, wanted: Int, got: Int)
    case layerMismatch(request: Int, table: Int)

    public var description: String {
        switch self {
        case .cannotOpen(let detail):
            return "cannot open Engram table: \(detail)"
        case .tableTooSmall(let expected, let actual):
            return "Engram table is \(actual) bytes, needs at least \(expected) for its row count"
        case .rowOutOfRange(let offset, let length, let fileSize):
            return "Engram row read [\(offset), \(offset + length)) is outside the \(fileSize)-byte table"
        case .shortRead(let offset, let wanted, let got):
            return "Engram row read at \(offset) returned \(got) of \(wanted) bytes"
        case .layerMismatch(let request, let table):
            return "Engram row request for layer \(request) was issued to layer \(table)'s table"
        }
    }
}

/// Values the parallel row reads need. The pointer is only written at
/// disjoint offsets, so sharing it across the concurrent reads is safe.
nonisolated private struct ReadContext: @unchecked Sendable {
    let fd: Int32
    /// Byte offset of the table tensor inside the shard file. Requests carry
    /// table-relative offsets, so every pread adds this base.
    let baseOffset: Int
    let rowBytes: Int
    let requests: [DeepSeekV41EngramRowRequest]
    let destination: UnsafeMutableRawPointer
    let failures: FailureBox

    func read(_ index: Int) {
        let request = requests[index]
        let slice = destination.advanced(by: index * rowBytes)
        var total = 0
        while total < rowBytes {
            let got = pread(
                fd,
                slice.advanced(by: total),
                rowBytes - total,
                off_t(baseOffset + request.byteOffset + total)
            )
            if got < 0 {
                if errno == EINTR { continue }
                failures.record(.shortRead(
                    offset: baseOffset + request.byteOffset + total,
                    wanted: rowBytes - total,
                    got: Int(got)
                ))
                return
            }
            if got == 0 {
                failures.record(.shortRead(
                    offset: baseOffset + request.byteOffset + total,
                    wanted: rowBytes - total,
                    got: 0
                ))
                return
            }
            total += got
        }
    }
}

/// First-error slot for the parallel row reads.
nonisolated private final class FailureBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: DeepSeekV41EngramReadError?

    func record(_ error: DeepSeekV41EngramReadError) {
        lock.lock()
        if stored == nil { stored = error }
        lock.unlock()
    }

    var first: DeepSeekV41EngramReadError? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

/// One table's reader. Synchronous and thread-safe: one batch at a time holds the
/// descriptor, so `closeNow()` cannot recycle the fd under an in-flight `pread`.
nonisolated public final class DeepSeekV41EngramReader: @unchecked Sendable {
    public let spec: DeepSeekV41EngramSpec
    public let fileSize: Int
    /// Byte offset of this table tensor inside the shard file.
    public let baseOffset: Int

    private let descriptor: Int32
    private let lock = NSLock()
    private var isClosed = false

    /// Opens the table file and validates it against the spec, so a truncated
    /// or half-downloaded table fails at load rather than mid-generation.
    public init(
        fileURL: URL,
        spec: DeepSeekV41EngramSpec,
        baseOffset: Int = 0
    ) throws {
        guard baseOffset >= 0 else {
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): negative table offset \(baseOffset)"
            )
        }
        let fd = open(fileURL.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): errno \(errno)"
            )
        }
        var status = stat()
        guard fstat(fd, &status) == 0 else {
            close(fd)
            throw DeepSeekV41EngramReadError.cannotOpen(
                "\(fileURL.lastPathComponent): fstat failed, errno \(errno)"
            )
        }
        let size = Int(status.st_size)
        // The table starts at the tensor's own offset inside the shard; only the
        // bytes after that offset belong to it.
        guard baseOffset <= size, size - baseOffset >= spec.totalBytes else {
            close(fd)
            throw DeepSeekV41EngramReadError.tableTooSmall(
                expected: baseOffset + spec.totalBytes, actual: size
            )
        }
        self.descriptor = fd
        self.spec = spec
        self.fileSize = size
        self.baseOffset = baseOffset
    }

    deinit {
        closeNow()
    }

    public func closeNow() {
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else { return }
        isClosed = true
        close(descriptor)
    }

    /// Reads one batch of rows into `destination`, laid out densely in request
    /// order (`request[i]` at `i * spec.rowBytes`). Returns the bytes read.
    ///
    /// Reads are issued in parallel once there are enough of them for queue
    /// depth to matter; each request writes a disjoint destination slice, so
    /// the concurrent writes are safe.
    ///
    /// The lock is held for the whole batch rather than just around the
    /// `isClosed` check: `closeNow()` closes the descriptor, and a descriptor
    /// number recycled by an unrelated `open()` would turn a concurrent `pread`
    /// into a silent read of another file's bytes. Holding the lock makes close
    /// wait for the in-flight rows instead.
    @discardableResult
    public func readRows(
        _ requests: [DeepSeekV41EngramRowRequest],
        into destination: UnsafeMutableRawPointer
    ) throws -> Int {
        guard !requests.isEmpty else { return 0 }
        lock.lock()
        defer { lock.unlock() }
        guard !isClosed else {
            throw DeepSeekV41EngramReadError.cannotOpen("table reader was closed")
        }
        let fd = descriptor

        let rowBytes = spec.rowBytes
        for request in requests {
            guard request.layer == spec.layer else {
                throw DeepSeekV41EngramReadError.layerMismatch(
                    request: request.layer, table: spec.layer
                )
            }
            guard request.byteCount == rowBytes else {
                throw DeepSeekV41EngramReadError.rowOutOfRange(
                    offset: request.byteOffset,
                    length: request.byteCount,
                    fileSize: fileSize
                )
            }
            guard request.byteOffset >= 0,
                  baseOffset + request.byteOffset + rowBytes <= fileSize else {
                throw DeepSeekV41EngramReadError.rowOutOfRange(
                    offset: baseOffset + request.byteOffset,
                    length: rowBytes,
                    fileSize: fileSize
                )
            }
        }

        // Concurrent rows must not race on the first-error slot, and the raw
        // destination pointer has to cross the concurrency boundary explicitly.
        let failures = FailureBox()
        let shared = ReadContext(
            fd: fd,
            baseOffset: baseOffset,
            rowBytes: rowBytes,
            requests: requests,
            destination: destination,
            failures: failures
        )

        let readOne: @Sendable (Int) -> Void = { index in
            shared.read(index)
        }

        if requests.count >= 8 {
            DispatchQueue.concurrentPerform(iterations: requests.count, execute: readOne)
        } else {
            for index in requests.indices { readOne(index) }
        }

        if let firstError = failures.first { throw firstError }
        return requests.count * rowBytes
    }
}
