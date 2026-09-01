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
    private var layerFDs: [Int: Int32] = [:]
    private var layerFDLock = NSRecursiveLock()

    private init() {}

    deinit {
        shutdown()
    }

    public func initialize(numThreads: Int = 8) {
        // GCD manages thread pool automatically
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
            // Allows OS page cache to be used without aggressive prefetching
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

