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

    private var threads: [pthread_t] = []
    private var mutex = pthread_mutex_t()
    private var workReadyCond = pthread_cond_t()
    private var workDoneCond = pthread_cond_t()

    private var activeTasks: [ExpertPreadTask] = []
    private var tasksCount: Int = 0
    private var tasksCompleted: Int = 0
    private var currentGeneration: Int = 0
    private var completedGeneration: Int = 0
    private var isShutdown: Bool = false
    private var isInitialized: Bool = false

    // Cached open file descriptors: layerIndex -> open fd
    private var layerFDs: [Int: Int32] = [:]
    private var layerFDLock = NSLock()

    private init() {
        initialize(numThreads: Self.defaultNumThreads)
    }

    deinit {
        shutdown()
    }

    public func initialize(numThreads: Int = 8) {
        guard !isInitialized else { return }

        pthread_mutex_init(&mutex, nil)
        pthread_cond_init(&workReadyCond, nil)
        pthread_cond_init(&workDoneCond, nil)

        isShutdown = false
        currentGeneration = 0
        completedGeneration = 0
        tasksCount = 0
        tasksCompleted = 0
        threads.removeAll()

        for i in 0..<numThreads {
            var thread: pthread_t?
            let threadId = UnsafeMutablePointer<Int>.allocate(capacity: 1)
            threadId.pointee = i

            let createResult = pthread_create(&thread, nil, { arg -> UnsafeMutableRawPointer? in
                let tid = arg.assumingMemoryBound(to: Int.self).pointee
                arg.deallocate()
                ExpertIOThreadPool.shared.workerLoop(threadId: tid)
                return nil
            }, threadId)

            if createResult == 0, let t = thread {
                threads.append(t)
            }
        }

        isInitialized = true
    }

    private func workerLoop(threadId: Int) {
        var myGeneration = 0

        pthread_mutex_lock(&mutex)
        while true {
            while currentGeneration == myGeneration && !isShutdown {
                pthread_cond_wait(&workReadyCond, &mutex)
            }

            if isShutdown {
                pthread_mutex_unlock(&mutex)
                break
            }

            myGeneration = currentGeneration
            let count = tasksCount

            // Work stealing across threadId
            var localTasksToProcess: [(index: Int, task: ExpertPreadTask)] = []
            var idx = threadId
            while idx < count {
                localTasksToProcess.append((index: idx, task: activeTasks[idx]))
                idx += threads.count
            }

            pthread_mutex_unlock(&mutex)

            // Perform pread outside the mutex lock
            for item in localTasksToProcess {
                var task = item.task
                let bytesRead = pread(task.fd, task.dst, task.size, task.offset)
                task.result = bytesRead

                pthread_mutex_lock(&mutex)
                if item.index < activeTasks.count {
                    activeTasks[item.index].result = bytesRead
                }
                tasksCompleted += 1
                if tasksCompleted >= count {
                    completedGeneration = myGeneration
                    pthread_cond_broadcast(&workDoneCond)
                }
                pthread_mutex_unlock(&mutex)
            }

            pthread_mutex_lock(&mutex)
        }
    }

    /// Asynchronously dispatches pread tasks across the 8-thread pool, returning a generation token
    @discardableResult
    public func dispatchAsync(tasks: [ExpertPreadTask]) -> Int {
        guard !tasks.isEmpty else { return 0 }

        pthread_mutex_lock(&mutex)
        activeTasks = tasks
        tasksCount = tasks.count
        tasksCompleted = 0
        currentGeneration += 1
        let gen = currentGeneration
        pthread_cond_broadcast(&workReadyCond)
        pthread_mutex_unlock(&mutex)

        return gen
    }

    /// Waits for a dispatched generation token to complete and returns results
    @discardableResult
    public func wait(generation: Int) -> [ExpertPreadTask] {
        guard generation > 0 else { return [] }

        pthread_mutex_lock(&mutex)
        while completedGeneration < generation && !isShutdown {
            pthread_cond_wait(&workDoneCond, &mutex)
        }
        let completed = activeTasks
        pthread_mutex_unlock(&mutex)
        return completed
    }

    /// Synchronously dispatches and waits for all pread tasks to complete
    @discardableResult
    public func dispatchSync(tasks: [ExpertPreadTask]) -> [ExpertPreadTask] {
        guard !tasks.isEmpty else { return [] }
        let gen = dispatchAsync(tasks: tasks)
        return wait(generation: gen)
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
        guard isInitialized else { return }

        pthread_mutex_lock(&mutex)
        isShutdown = true
        pthread_cond_broadcast(&workReadyCond)
        pthread_mutex_unlock(&mutex)

        for t in threads {
            pthread_join(t, nil)
        }
        threads.removeAll()

        closeAllLayerFDs()

        pthread_mutex_destroy(&mutex)
        pthread_cond_destroy(&workReadyCond)
        pthread_cond_destroy(&workDoneCond)
        isInitialized = false
    }
}
