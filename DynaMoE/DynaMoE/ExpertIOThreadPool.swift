//
//  ExpertIOThreadPool.swift
//  DynaMoE
//
//  High-Performance POSIX pread Thread Pool for Active Expert Streaming
//  Eliminates Darwin VM mmap page faults by streaming active expert slices directly
//  into pre-allocated Metal shared memory staging buffers via parallel NVMe DMA reads.
//

import Foundation
import Metal
import Darwin

public struct ExpertPreadTask {
    public var fd: Int32
    public var dst: UnsafeMutableRawPointer
    public var offset: off_t
    public var size: Int
    public var result: Int = 0

    public init(fd: Int32, dst: UnsafeMutableRawPointer, offset: off_t, size: Int) {
        self.fd = fd
        self.dst = dst
        self.offset = offset
        self.size = size
        self.result = 0
    }
}

public final class ExpertIOThreadPool {
    public static let shared = ExpertIOThreadPool()

    public static let defaultNumThreads: Int = 8

    // Cached open file descriptors: layerIndex -> open fd
    public internal(set) var layerFDs: [Int: Int32] = [:]
    private var layerFDLock = NSRecursiveLock()

    private init() {}

    deinit {
        shutdown()
    }

    public func initialize(numThreads: Int = 8) {
        // Reset any descriptors cached by a previously loaded packed model
        closeAllLayerFDs()
    }

    /// Synchronously dispatches pread tasks in parallel across all CPU cores via GCD
    @discardableResult
    public func dispatchSync(tasks: [ExpertPreadTask]) -> [ExpertPreadTask] {
        guard !tasks.isEmpty else { return [] }
        var resultTasks = tasks
        resultTasks.withUnsafeMutableBufferPointer { buffer in
            DispatchQueue.concurrentPerform(iterations: buffer.count) { i in
                let task = buffer[i]
                let bytesRead = pread(task.fd, task.dst, task.size, task.offset)
                buffer[i].result = bytesRead
            }
        }
        return resultTasks
    }

    /// Synchronously dispatches and updates inout tasks array
    public func dispatchSync(tasks: inout [ExpertPreadTask]) {
        tasks = dispatchSync(tasks: tasks)
    }

    private let asyncQueue = DispatchQueue(label: "dynamoe.expertio.async", qos: .userInitiated)

    /// Asynchronously dispatches pread tasks on a background queue and signals
    /// `done` exactly once after ALL tasks have completed (or immediately if empty).
    /// When `onTaskDone` is provided it is invoked with the task index as soon as
    /// that individual pread finishes, enabling per-slot consumption of
    /// speculative prefetch kicks: consumers wait on individual slots instead of
    /// the whole batch, so slow false-positive reads never block the critical path.
    public func dispatchAsync(tasks: [ExpertPreadTask], onTaskDone: ((Int) -> Void)? = nil, done: DispatchSemaphore? = nil) {
        guard !tasks.isEmpty else {
            done?.signal()
            return
        }
        asyncQueue.async { [weak self] in
            guard let self else {
                done?.signal()
                return
            }
            var resultTasks = tasks
            resultTasks.withUnsafeMutableBufferPointer { buffer in
                DispatchQueue.concurrentPerform(iterations: buffer.count) { i in
                    let task = buffer[i]
                    let bytesRead = pread(task.fd, task.dst, task.size, task.offset)
                    buffer[i].result = bytesRead
                    onTaskDone?(i)
                }
            }
            done?.signal()
        }
    }

    /// Streams an entire file into a pre-allocated anonymous buffer (typically a
    /// device.makeBuffer(length:options:.storageModeShared)) using an N-way parallel
    /// POSIX pread across disjoint byte ranges. Used to pin invariant weights
    /// (e.g. FlashMoE model_weights.bin backbone) into resident anonymous memory so the
    /// GPU never re-faults them from the file-backed mmap during decode.
    ///
    /// Returns elapsed wall time in seconds on success (full `length` bytes read),
    /// or nil if the open/read failed or the transfer was incomplete (partial data
    /// would silently corrupt weights, so callers must fall back to the mmap path).
    @discardableResult
    public static func preadFileIntoBuffer(fd: Int32, dst: UnsafeMutableRawPointer, length: Int, threads: Int = 8) -> Double? {
        guard fd >= 0, length > 0 else { return nil }
        let workerCount = max(1, min(threads, length / 262144))
        let rangeLen = length / workerCount
        var totalRead: Int64 = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        DispatchQueue.concurrentPerform(iterations: workerCount) { i in
            var done = 0
            let end = (i == workerCount - 1) ? length - (workerCount - 1) * rangeLen : rangeLen
            let base = dst.advanced(by: i * rangeLen)
            var fileOff = off_t(i) * off_t(rangeLen)
            while done < end {
                let toRead = min(262144, end - done)
                let n = pread(fd, base.advanced(by: done), toRead, fileOff)
                if n <= 0 { break }
                done += n
                fileOff += off_t(n)
            }
            OSAtomicAdd64(Int64(done), &totalRead)
        }
        guard totalRead == Int64(length) else { return nil }
        return CFAbsoluteTimeGetCurrent() - t0
    }

    // MARK: - File Descriptor Management

    /// Opens and caches the file descriptor for a packed layer file (e.g. packed_experts/layer_00.bin)
    public func getOrOpenLayerFD(layerIndex: Int, packedExpertsDir: URL) -> Int32? {
        layerFDLock.lock()
        defer { layerFDLock.unlock() }

        if let cached = layerFDs[layerIndex] {
            return cached
        }

        let fileName = String(format: "layer_%02d.bin", layerIndex)
        let filePath = packedExpertsDir.appendingPathComponent(fileName).path

        let fd = open(filePath, O_RDONLY | O_CLOEXEC)
        if fd >= 0 {
            // Enable OS page cache for these file descriptors
            _ = fcntl(fd, F_NOCACHE, 0)
            layerFDs[layerIndex] = fd
            return fd
        }
        return nil
    }

    /// Closes and resets all cached layer file descriptors
    public func closeAllLayerFDs() {
        layerFDLock.lock()
        defer { layerFDLock.unlock() }

        for (_, fd) in layerFDs {
            if fd >= 0 {
                close(fd)
            }
        }
        layerFDs.removeAll()
    }

    public func shutdown() {
        closeAllLayerFDs()
    }
}

