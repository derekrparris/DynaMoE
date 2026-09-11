import SwiftUI
import UniformTypeIdentifiers
import Metal
import Foundation
import Accelerate

struct ExpertRoutingBadgeView: View {
    let rank: Int
    let expertId: Int
    let weight: Float

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("#\(rank + 1)")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                Text("Expert #\(expertId)")
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.bold)
                    .foregroundColor(.purple)
                Spacer()
                Text(String(format: "%.1f%%", weight * 100.0))
                    .font(.system(.caption, design: .monospaced))
                    .fontWeight(.semibold)
            }
            
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.purple.opacity(0.15))
                        .frame(height: 6)
                    Capsule()
                        .fill(LinearGradient(colors: [.purple, .blue], startPoint: .leading, endPoint: .trailing))
                        .frame(width: max(4, geo.size.width * CGFloat(weight)), height: 6)
                }
            }
            .frame(height: 6)
        }
        .padding(8)
        .background(Color(NSColor.controlBackgroundColor))
        .cornerRadius(8)
    }
}

struct LayerTelemetry: Identifiable {
    var id: Int { layerIndex }
    let layerIndex: Int
    let durationMs: Double
    let topExperts: [Int]
    let l2Norm: Float
}

struct TokenPrediction: Identifiable {
    var id: Int { rank }
    let rank: Int
    let tokenId: UInt32
    let tokenString: String
    let logit: Float
    let probability: Float
}

enum KVCachePrecision: String, CaseIterable, Identifiable {
    case fp16 = "FP16 (Half - 50% Memory)"
    case fp8 = "FP8 (8-Bit Quantized - 75% Memory)"
    case fp32 = "FP32 (Full Precision)"

    var id: String { rawValue }

    var bytesPerElement: Int {
        switch self {
        case .fp16: return 2
        case .fp8: return 1
        case .fp32: return 4
        }
    }
}

final class KVCacheManager {
    static let shared = KVCacheManager()

    var kCacheBuffer: MTLBuffer?
    var vCacheBuffer: MTLBuffer?
    var kScaleBuffer: MTLBuffer?
    var vScaleBuffer: MTLBuffer?
    var linearStateBuffer: MTLBuffer?
    var convStateBuffer: MTLBuffer?
    var allocatedSeqLen: Int = 0
    var allocatedKvBytes: Int = 0
    var activePrecision: KVCachePrecision = .fp16

    func reset(
        device: MTLDevice,
        config: ModelConfig? = nil,
        actualLayers: Int = 40,
        totalLoops: Int = 1,
        numKvHeads: Int = 8,
        headDim: Int = 128,
        maxSeqLen: Int = 2048,
        precision: KVCachePrecision = .fp16,
        preservePrefixCount: Int = 0
    ) {
        let oldMaxSeq = self.allocatedSeqLen
        let wasAllocated = (kCacheBuffer != nil)
        self.allocatedSeqLen = maxSeqLen
        self.activePrecision = precision

        let isLing = (config?.isLingModel == true)
        let loops = max(totalLoops, config?.effectiveNumLoops ?? 1)
        let kvHeads = max(numKvHeads, config?.effectiveNumKeyValueHeads ?? 8)
        let hDim = max(headDim, config?.effectiveHeadDim ?? 128)
        let totalSlots = isLing ? (6 * loops) : (actualLayers * loops)
        let actualSlotCapacity = isLing ? totalSlots : max(totalSlots, 44)
        let kvStride = kvHeads * hDim
        let isMla = (isLing || config?.kvLoraRank != nil)
        let effectiveKvStride = isMla ? 4096 : max(kvStride, 1024)
        let elementBytes = precision.bytesPerElement

        // For Ling MLA, K stride is 16 heads * 192 = 3072 halfs, V stride is 16 heads * 128 = 2048 halfs
        let kStride = isLing ? 3072 : effectiveKvStride
        let vStride = isLing ? 2048 : effectiveKvStride
        let kBytes = actualSlotCapacity * maxSeqLen * kStride * elementBytes
        let vBytes = actualSlotCapacity * maxSeqLen * vStride * elementBytes
        let requiredKvBytes = max(kBytes, vBytes)

        let oldK = self.kCacheBuffer
        let oldV = self.vCacheBuffer
        let needsRealloc = (kCacheBuffer == nil || vCacheBuffer == nil || allocatedKvBytes < requiredKvBytes)

        if needsRealloc {
            self.kCacheBuffer = device.makeBuffer(length: kBytes, options: .storageModeShared)
            self.vCacheBuffer = device.makeBuffer(length: vBytes, options: .storageModeShared)
            self.allocatedKvBytes = requiredKvBytes
        }

        if preservePrefixCount > 0 && wasAllocated {
            for slot in 0..<totalSlots {
                let newKLayerByteOffset = slot * maxSeqLen * kStride * elementBytes
                let newVLayerByteOffset = slot * maxSeqLen * vStride * elementBytes
                let prefixKBytes = min(preservePrefixCount, maxSeqLen) * kStride * elementBytes
                let prefixVBytes = min(preservePrefixCount, maxSeqLen) * vStride * elementBytes

                if needsRealloc, let oldKBuf = oldK, let oldVBuf = oldV, let newKBuf = kCacheBuffer, let newVBuf = vCacheBuffer {
                    let oldKLayerByteOffset = slot * oldMaxSeq * kStride * elementBytes
                    let oldVLayerByteOffset = slot * oldMaxSeq * vStride * elementBytes
                    let copyKBytes = min(prefixKBytes, max(0, oldKBuf.length - oldKLayerByteOffset), max(0, newKBuf.length - newKLayerByteOffset))
                    let copyVBytes = min(prefixVBytes, max(0, oldVBuf.length - oldVLayerByteOffset), max(0, newVBuf.length - newVLayerByteOffset))
                    if copyKBytes > 0 {
                        memcpy(newKBuf.contents().advanced(by: newKLayerByteOffset), oldKBuf.contents().advanced(by: oldKLayerByteOffset), copyKBytes)
                    }
                    if copyVBytes > 0 {
                        memcpy(newVBuf.contents().advanced(by: newVLayerByteOffset), oldVBuf.contents().advanced(by: oldVLayerByteOffset), copyVBytes)
                    }
                }

                let tailKByteOffset = newKLayerByteOffset + prefixKBytes
                let tailKBytes = max(0, (maxSeqLen - preservePrefixCount) * kStride * elementBytes)
                if let kBuf = kCacheBuffer, tailKByteOffset + tailKBytes <= kBuf.length {
                    memset(kBuf.contents().advanced(by: tailKByteOffset), 0, tailKBytes)
                }

                let tailVByteOffset = newVLayerByteOffset + prefixVBytes
                let tailVBytes = max(0, (maxSeqLen - preservePrefixCount) * vStride * elementBytes)
                if let vBuf = vCacheBuffer, tailVByteOffset + tailVBytes <= vBuf.length {
                    memset(vBuf.contents().advanced(by: tailVByteOffset), 0, tailVBytes)
                }
            }
        } else {
            if let kBuf = kCacheBuffer { memset(kBuf.contents(), 0, min(kBytes, kBuf.length)) }
            if let vBuf = vCacheBuffer { memset(vBuf.contents(), 0, min(vBytes, vBuf.length)) }
        }

        if precision == .fp8 {
            let scaleCount = actualSlotCapacity * maxSeqLen * kvHeads
            let scaleBytes = scaleCount * MemoryLayout<UInt16>.stride // half precision per-head scales
            if kScaleBuffer == nil || kScaleBuffer!.length < scaleBytes {
                self.kScaleBuffer = device.makeBuffer(length: scaleBytes, options: .storageModeShared)
                self.vScaleBuffer = device.makeBuffer(length: scaleBytes, options: .storageModeShared)
            }
            if preservePrefixCount == 0 {
                if let ksBuf = kScaleBuffer { memset(ksBuf.contents(), 0, min(scaleBytes, ksBuf.length)) }
                if let vsBuf = vScaleBuffer { memset(vsBuf.contents(), 0, min(scaleBytes, vsBuf.length)) }
            }
        }

        let linLayers = isLing ? 18 : actualLayers
        let linValHeads = isLing ? 16 : max(48, config?.effectiveLinearNumValueHeads ?? 32)
        let linHeadDim = max(128, config?.effectiveLinearValueHeadDim ?? 128)
        let linStateBytes = (isLing ? linLayers : max(linLayers, 48)) * linValHeads * linHeadDim * linHeadDim * MemoryLayout<Float>.stride
        if linearStateBuffer == nil || linearStateBuffer!.length < linStateBytes {
            self.linearStateBuffer = device.makeBuffer(length: linStateBytes, options: .storageModeShared)
        }
        if isLing || config?.hasLinearRecurrence == true || preservePrefixCount == 0 {
            if let sBuf = linearStateBuffer { memset(sBuf.contents(), 0, min(linStateBytes, sBuf.length)) }
        }

        let convChannels = isLing ? 6144 : max(10240, (config?.effectiveLinearNumValueHeads ?? 32) > 32 ? 10240 : 8192)
        let convBytes = (isLing ? linLayers : max(linLayers, 48)) * convChannels * 4 * MemoryLayout<Float>.stride
        if convStateBuffer == nil || convStateBuffer!.length < convBytes {
            self.convStateBuffer = device.makeBuffer(length: convBytes, options: .storageModeShared)
        }
        if isLing || config?.hasLinearRecurrence == true || preservePrefixCount == 0 {
            if let cBuf = convStateBuffer { memset(cBuf.contents(), 0, min(convBytes, cBuf.length)) }
        }
    }
}

#if canImport(Darwin)
import Darwin

func getProcessResidentMemoryGB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    if kerr == KERN_SUCCESS {
        return Double(info.resident_size) / (1024.0 * 1024.0 * 1024.0)
    }
    return 0.0
}
#else
func getProcessResidentMemoryGB() -> Double {
    return 0.0
}
#endif

enum MemoryBudgetMode: String, CaseIterable, Identifiable {
    case lowMemory8GB = "8 GB"
    case balanced16GB = "16 GB"
    case unrestricted = "Unrestricted"

    var id: String { rawValue }

    var maxResidentExperts: Int {
        switch self {
        case .lowMemory8GB: return 480       // Up to 480 active resident experts (~2.36 GB FP8)
        case .balanced16GB: return 1280      // Up to 1,280 active resident experts (~6.29 GB FP8)
        case .unrestricted: return 65536     // All experts resident in RAM
        }
    }

    var targetMaxRssGB: Double {
        switch self {
        case .lowMemory8GB: return 6.5
        case .balanced16GB: return 11.5
        case .unrestricted: return 36.6
        }
    }
}

struct ExpertSlice {
    let shardIndex: UInt32
    let offset: UInt64
    let length: UInt64
}

struct ExpertKey: Hashable {
    let layer: Int
    let expertId: Int
}

final class ExpertTransitionTracker {
    // [sourceLayer: [sourceExpertId: [targetExpertId: count]]]
    private var transitions: [Int: [Int: [Int: Int]]] = [:]
    private let lock = NSRecursiveLock()

    func recordTransition(fromLayer: Int, fromExperts: [Int], toLayer: Int, toExperts: [Int]) {
        guard !fromExperts.isEmpty && !toExperts.isEmpty else { return }
        lock.lock()
        defer { lock.unlock() }
        for src in fromExperts {
            for dst in toExperts {
                transitions[fromLayer, default: [:]][src, default: [:]][dst, default: 0] += 1
            }
        }
    }

    func predictNextExperts(currentLayer: Int, currentExperts: [Int], topN: Int) -> [Int] {
        guard !currentExperts.isEmpty else { return [] }
        lock.lock()
        defer { lock.unlock() }
        guard let layerMap = transitions[currentLayer] else { return [] }
        var scoreMap: [Int: Int] = [:]
        for exp in currentExperts {
            if let dstCounts = layerMap[exp] {
                for (dst, count) in dstCounts {
                    scoreMap[dst, default: 0] += count
                }
            }
        }
        return scoreMap.sorted(by: { $0.value > $1.value }).prefix(topN).map { $0.key }
    }

    func reset() {
        lock.lock()
        transitions.removeAll()
        lock.unlock()
    }
}

final class WorkingSetManager {
    static let shared = WorkingSetManager()

    private let lock = NSRecursiveLock()
    private var expertSlices: [ExpertKey: [ExpertSlice]] = [:]
    private var denseSlices: [ExpertSlice] = []
    private var denseBytes: UInt64 = 0
    private var residentExpertBytes: UInt64 = 0
    private var residentExperts: Set<ExpertKey> = []
    private var prefetchedKeys: Set<ExpertKey> = []
    private var accessOrder: [ExpertKey: UInt64] = [:]
    private var accessCounter: UInt64 = 0
    private let prefetchQueue = DispatchQueue(label: "com.dynamoe.prefetch", qos: .userInitiated)
    private let evictionQueue = DispatchQueue(label: "com.dynamoe.eviction", qos: .utility)
    private var prefetchedBackboneLayers: Set<UInt32> = []
    public let transitionTracker = ExpertTransitionTracker()

    private var totalAccesses: Int = 0
    private var cacheHits: Int = 0
    private var cacheMisses: Int = 0
    private var prefetchCount: Int = 0
    private var prefetchHits: Int = 0
    private var lastPagingLatencyMs: Double = 0.0
    private var expertOnlyShardIndices: Set<UInt32> = []
    private var shardFDs: [UInt32: Int32] = [:]
    private var shardFilePaths: [UInt32: String] = [:]

    var totalExpertKeysCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return expertSlices.count
    }

    var residentExpertsCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return residentExperts.count
    }

    var cacheHitRatePercent: Double {
        lock.lock()
        defer { lock.unlock() }
        guard totalAccesses > 0 else { return 100.0 }
        return (Double(cacheHits) / Double(totalAccesses)) * 100.0
    }

    var prefetchEfficiencyPercent: Double {
        lock.lock()
        defer { lock.unlock() }
        guard prefetchCount > 0 else { return 100.0 }
        return (Double(prefetchHits) / Double(prefetchCount)) * 100.0
    }

    var currentPagingLatencyMs: Double {
        lock.lock()
        defer { lock.unlock() }
        return lastPagingLatencyMs
    }

    var effectiveResidentMemoryGB: Double {
        lock.lock()
        let weightsGB = Double(denseBytes + residentExpertBytes) / (1024.0 * 1024.0 * 1024.0)
        lock.unlock()
        let heapGB = getProcessResidentMemoryGB()
        return heapGB + weightsGB
    }

    private static var pageFaultSink: Int64 = 0

    func closeAllFileDescriptors() {
        lock.lock()
        defer { lock.unlock() }
        for (_, fd) in shardFDs {
            close(fd)
        }
        shardFDs.removeAll()
        shardFilePaths.removeAll()
    }

    /// Coordinated parallel bulk POSIX pread priming across CPU cores.
    /// Uses sequential NVMe DMA block transfers to fault entire contiguous slices directly
    /// into Darwin's Unified Memory Buffer Cache at line rate (>2,500 MB/s), completely
    /// eliminating the ~460,000+ random 16 KB CPU/GPU page fault traps.
    func primeSlices(_ slices: [ExpertSlice], shardBuffers: [UInt32: MTLBuffer]) {
        guard !slices.isEmpty else { return }

        lock.lock()
        let fds = self.shardFDs
        lock.unlock()

        if !fds.isEmpty {
            DispatchQueue.concurrentPerform(iterations: slices.count) { i in
                let slice = slices[i]
                guard let fd = fds[slice.shardIndex] else { return }
                let len = Int(slice.length)
                let offset = off_t(slice.offset)
                guard len > 0 else { return }

                // 1. Issue kernel readahead advisory for the entire contiguous slice
                var radv = radvisory(ra_offset: offset, ra_count: Int32(len))
                _ = fcntl(fd, F_RDADVISE, &radv)

                // 2. Synchronously pread in 256KB chunks to populate the macOS Unified Buffer Cache
                var scratch = [UInt8](repeating: 0, count: min(len, 262144))
                var bytesRead = 0
                while bytesRead < len {
                    let toRead = min(len - bytesRead, scratch.count)
                    let n = pread(fd, &scratch, toRead, offset + off_t(bytesRead))
                    if n <= 0 { break }
                    bytesRead += n
                }
            }
            return
        }

        // Fallback: If shard file descriptors are not available, fault via mmap pointer
        let pageSize = Int(vm_page_size) // 16384 bytes on Apple Silicon
        DispatchQueue.concurrentPerform(iterations: slices.count) { i in
            let slice = slices[i]
            guard let buf = shardBuffers[slice.shardIndex] else { return }
            let len = Int(slice.length)
            let offset = Int(slice.offset)
            guard len > 0, offset + len <= buf.length else { return }
            let rawPtr = buf.contents().advanced(by: offset)
            posix_madvise(rawPtr, len, POSIX_MADV_WILLNEED)
            let bytePtr = rawPtr.assumingMemoryBound(to: UInt8.self)
            var dummy: UInt64 = 0
            for off in stride(from: 0, to: len, by: pageSize) {
                dummy &+= UInt64(bytePtr[off])
            }
            dummy &+= UInt64(bytePtr[len - 1])
            OSAtomicAdd64(Int64(bitPattern: dummy), &WorkingSetManager.pageFaultSink)
        }
    }

    func initialize(summary: ModelSummary, shardBuffers: [UInt32: MTLBuffer], mode: MemoryBudgetMode, modelDir: URL? = nil) {
        lock.lock()
        for (_, fd) in shardFDs {
            close(fd)
        }
        shardFDs.removeAll()
        shardFilePaths.removeAll()

        if let dir = modelDir {
            for shard in summary.shards {
                let shardPath = dir.appendingPathComponent(shard.filename).path
                shardFilePaths[shard.index] = shardPath
                let fd = open(shardPath, O_RDONLY)
                if fd >= 0 {
                    shardFDs[shard.index] = fd
                }
            }
        }

        expertSlices.removeAll()
        denseSlices.removeAll()
        denseBytes = 0
        residentExpertBytes = 0
        residentExperts.removeAll()
        prefetchedKeys.removeAll()
        prefetchedBackboneLayers.removeAll()
        accessOrder.removeAll()
        accessCounter = 0
        transitionTracker.reset()
        totalAccesses = 0
        cacheHits = 0
        cacheMisses = 0
        prefetchCount = 0
        prefetchHits = 0
        lastPagingLatencyMs = 0.0

        var shardHasDense: [UInt32: Bool] = [:]
        for tensor in summary.tensors {
            let name = tensor.name.lowercased()
            if name.contains("ple.") || name.contains("ngram_embedding") || name.contains("mtp.") || name.contains("visual.") {
                continue
            }
            let length = tensor.offsetEnd - tensor.offsetStart
            let slice = ExpertSlice(
                shardIndex: tensor.shardIndex,
                offset: tensor.offsetStart,
                length: length
            )

            if let l = tensor.layerIndex, let exp = tensor.expertId {
                let key = ExpertKey(layer: Int(l), expertId: Int(exp))
                expertSlices[key, default: []].append(slice)
            } else {
                denseSlices.append(slice)
                denseBytes += length
                shardHasDense[tensor.shardIndex] = true
            }
        }
        let allShardIndices = Set(summary.shards.map { $0.index })
        expertOnlyShardIndices = allShardIndices.filter { shardHasDense[$0] != true }
        lock.unlock()

        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        }
    }

    func preFaultAll(shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        lock.lock()
        let fds = self.shardFDs
        for key in expertSlices.keys {
            residentExperts.insert(key)
        }
        residentExpertBytes = expertSlices.reduce(0) { sum, pair in
            sum + pair.value.reduce(0) { $0 + $1.length }
        }
        lock.unlock()

        let shards = summary.shards
        let pageSize = Int(vm_page_size) // 16384 bytes on Apple Silicon

        // Fast parallel sequential priming across CPU cores
        DispatchQueue.concurrentPerform(iterations: shards.count) { i in
            let shard = shards[i]
            let len = Int(shard.length)
            guard len > 0 else { return }

            if let fd = fds[shard.index] {
                // 1. Advise kernel of full sequential readahead
                var radv = radvisory(ra_offset: 0, ra_count: Int32(len))
                _ = fcntl(fd, F_RDADVISE, &radv)

                // 2. Coordinated parallel pread DMA transfers in 1 MB blocks
                let chunkSize = min(len, 1048576)
                var scratch = [UInt8](repeating: 0, count: chunkSize)
                var bytesRead = 0
                while bytesRead < len {
                    let toRead = min(len - bytesRead, chunkSize)
                    let n = pread(fd, &scratch, toRead, off_t(bytesRead))
                    if n <= 0 { break }
                    bytesRead += n
                }
            }

            // 3. Stride touch through mmap MTLBuffer to map hardware page tables directly
            if let buf = shardBuffers[shard.index] {
                let rawPtr = buf.contents()
                posix_madvise(rawPtr, len, POSIX_MADV_WILLNEED)
                let bytePtr = rawPtr.assumingMemoryBound(to: UInt8.self)
                var dummy: UInt64 = 0
                for off in stride(from: 0, to: len, by: pageSize) {
                    dummy &+= UInt64(bytePtr[off])
                }
                dummy &+= UInt64(bytePtr[len - 1])
                OSAtomicAdd64(Int64(bitPattern: dummy), &WorkingSetManager.pageFaultSink)
            }
        }
    }

    func flushAllExperts(shardBuffers: [UInt32: MTLBuffer]) {
        var evictSlices: [ExpertSlice] = []
        lock.lock()
        for (_, slices) in expertSlices {
            for slice in slices {
                if expertOnlyShardIndices.contains(slice.shardIndex) {
                    evictSlices.append(slice)
                }
            }
        }
        residentExperts.removeAll()
        prefetchedKeys.removeAll()
        accessOrder.removeAll()
        accessCounter = 0
        residentExpertBytes = 0
        lock.unlock()

        if !evictSlices.isEmpty {
            prefetchQueue.async {
                for slice in evictSlices {
                    if let buf = shardBuffers[slice.shardIndex] {
                        let ptr = buf.contents().advanced(by: Int(slice.offset))
                        posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                    }
                }
            }
        }
    }

    func trimToBudget(mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer]) {
        if mode == .unrestricted { return }

        var evictSlices: [ExpertSlice] = []
        lock.lock()
        let maxAllowed = mode.maxResidentExperts
        var excess = residentExperts.count - maxAllowed
        let weightsGB = Double(denseBytes + residentExpertBytes) / (1024.0 * 1024.0 * 1024.0)
        let currentRss = getProcessResidentMemoryGB() + weightsGB
        if currentRss > mode.targetMaxRssGB {
            let overRssGB = currentRss - mode.targetMaxRssGB
            let extraExpertsToTrim = Int(ceil(overRssGB * 1024.0 / 4.9))
            excess = max(excess, extraExpertsToTrim)
        }

        if excess > 0 {
            let sortedOldest = accessOrder.sorted(by: { $0.value < $1.value }).prefix(excess)
            for item in sortedOldest {
                let evictKey = item.key
                residentExperts.remove(evictKey)
                prefetchedKeys.remove(evictKey)
                accessOrder.removeValue(forKey: evictKey)
                if let slices = expertSlices[evictKey] {
                    let expertSize = slices.reduce(0) { $0 + $1.length }
                    residentExpertBytes = residentExpertBytes >= expertSize ? (residentExpertBytes - expertSize) : 0
                    for slice in slices {
                        if expertOnlyShardIndices.contains(slice.shardIndex) {
                            evictSlices.append(slice)
                        }
                    }
                }
            }
        }
        lock.unlock()

        if !evictSlices.isEmpty {
            evictionQueue.async {
                for slice in evictSlices {
                    if let buf = shardBuffers[slice.shardIndex] {
                        let ptr = buf.contents().advanced(by: Int(slice.offset))
                        posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                    }
                }
            }
        }
    }

    func trimAfterPrefill(shardBuffers: [UInt32: MTLBuffer], mode: MemoryBudgetMode) {
        trimToBudget(mode: mode, shardBuffers: shardBuffers)
    }

    func prefetchLayerBackbone(layer: CachedLayer, shardBuffers: [UInt32: MTLBuffer]) {
        lock.lock()
        if prefetchedBackboneLayers.contains(layer.layerIndex) {
            lock.unlock()
            return
        }
        prefetchedBackboneLayers.insert(layer.layerIndex)
        lock.unlock()

        let tensors = layer.backboneTensors
        guard !tensors.isEmpty else { return }
        let slices = tensors.map { t in
            ExpertSlice(shardIndex: t.shardIndex, offset: t.offsetStart, length: t.offsetEnd - t.offsetStart)
        }
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }
            self.primeSlices(slices, shardBuffers: shardBuffers)
        }
    }

    func prefetchLayerExperts(layer: Int, expertIds: [Int], shardBuffers: [UInt32: MTLBuffer]) {
        guard !expertIds.isEmpty else { return }
        var slicesToPrefetch: [ExpertSlice] = []

        lock.lock()
        for expId in expertIds {
            let key = ExpertKey(layer: layer, expertId: expId)
            if !residentExperts.contains(key) && !prefetchedKeys.contains(key) {
                prefetchedKeys.insert(key)
                prefetchCount += 1
                if let slices = expertSlices[key] {
                    slicesToPrefetch.append(contentsOf: slices)
                }
            }
        }
        lock.unlock()

        guard !slicesToPrefetch.isEmpty else { return }
        prefetchQueue.async { [weak self] in
            guard let self = self else { return }
            self.primeSlices(slicesToPrefetch, shardBuffers: shardBuffers)
        }
    }

    func predictNextLayerExperts(currentLayer: Int, currentActiveExperts: [Int], topN: Int = 4) -> [Int] {
        return transitionTracker.predictNextExperts(currentLayer: currentLayer, currentExperts: currentActiveExperts, topN: topN)
    }

    func touchAndEvict(layer: Int, activeExpertIds: [Int], mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer], isPrefill: Bool = false) {
        // In unrestricted full-RAM mode, bypass eviction overhead completely
        if mode == .unrestricted {
            return
        }

        var demandSlices: [ExpertSlice] = []

        lock.lock()
        for expId in activeExpertIds {
            let key = ExpertKey(layer: layer, expertId: expId)
            totalAccesses += 1
            accessCounter &+= 1
            accessOrder[key] = accessCounter

            if residentExperts.contains(key) {
                cacheHits += 1
            } else {
                cacheMisses += 1
                if prefetchedKeys.contains(key) {
                    prefetchHits += 1
                }
                residentExperts.insert(key)
                if let slices = expertSlices[key] {
                    let expertSize = slices.reduce(0) { $0 + $1.length }
                    residentExpertBytes += expertSize
                    demandSlices.append(contentsOf: slices)
                }
            }
        }
        lock.unlock()

        if !demandSlices.isEmpty {
            let t0 = CFAbsoluteTimeGetCurrent()
            primeSlices(demandSlices, shardBuffers: shardBuffers)
            let lat = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
            lock.lock()
            lastPagingLatencyMs = lat
            lock.unlock()
        }
    }

    func setBudgetMode(mode: MemoryBudgetMode, shardBuffers: [UInt32: MTLBuffer], summary: ModelSummary) {
        if mode == .unrestricted {
            preFaultAll(shardBuffers: shardBuffers, summary: summary)
        } else {
            var evictSlices: [ExpertSlice] = []
            lock.lock()
            let maxAllowed = mode.maxResidentExperts
            let excess = residentExperts.count - maxAllowed
            if excess > 0 {
                let sortedOldest = accessOrder.sorted(by: { $0.value < $1.value }).prefix(excess)
                for item in sortedOldest {
                    let evictKey = item.key
                    residentExperts.remove(evictKey)
                    prefetchedKeys.remove(evictKey)
                    accessOrder.removeValue(forKey: evictKey)
                    if let slices = expertSlices[evictKey] {
                        let expertSize = slices.reduce(0) { $0 + $1.length }
                        residentExpertBytes = residentExpertBytes >= expertSize ? (residentExpertBytes - expertSize) : 0
                        for slice in slices {
                            if expertOnlyShardIndices.contains(slice.shardIndex) {
                                evictSlices.append(slice)
                            }
                        }
                    }
                }
            }
            lock.unlock()

            if !evictSlices.isEmpty {
                evictionQueue.async {
                    for slice in evictSlices {
                        if let buf = shardBuffers[slice.shardIndex] {
                            let ptr = buf.contents().advanced(by: Int(slice.offset))
                            posix_madvise(ptr, Int(slice.length), POSIX_MADV_DONTNEED)
                        }
                    }
                }
            }
        }
    }
}

extension TensorMetadata: Identifiable {
    public var id: String { name }
}

struct ContentView: View {
    @State private var engine: DynaMoeEngine? = nil
    @State private var tokenizer: DynaMoeTokenizer? = nil
    @State private var summary: ModelSummary? = nil
    @State private var errorMessage: String? = nil
    @State private var metalStatus: String = "GPU Status: Waiting for weights..."
    @State private var searchText: String = ""
    @State private var selectedCategory: String = "All"
    
    // Multi-Shard Metal Buffers
    @State private var shardBuffers: [UInt32: MTLBuffer] = [:]
    
    // File Importers
    @State private var isWeightImporterPresented: Bool = false
    @State private var isTokenizerImporterPresented: Bool = false
    
    // Tokenizer Playground State
    @State private var promptInput: String = "Hello DynaMoE, routing tokens to experts..."
    @State private var tokenIDsOutput: String = "Load a tokenizer.json file to tokenize text"
    
    @State private var selectedTensorID: String? = nil
    @State private var gpuComputeOutput: String? = nil

    // Active Token Embedding & MoE Routing State
    @State private var activeH0Buffer: MTLBuffer? = nil
    @State private var activeTokenCount: Int = 0
    @State private var activeHiddenDim: Int = 2048
    
    @State private var selectedLayerForRouting: Int = 0
    @State private var routedExperts: [(id: Int, weight: Float)] = []
    @State private var sharedExpertWeight: Float? = nil
    @State private var routerStatusText: String? = nil

    // MoE Layer MLP Output State
    @State private var activeHmlpBuffer: MTLBuffer? = nil
    @State private var layerMlpStatusText: String? = nil
    @State private var layerMlpSampleOutput: String? = nil
    @State private var isExecutingMlp: Bool = false

    // Full Transformer Layer Execution State (h_l -> h_l+1)
    @State private var activeH1Buffer: MTLBuffer? = nil
    @State private var fullLayerStatusText: String? = nil
    @State private var fullLayerSampleOutput: String? = nil
    @State private var isExecutingFullLayer: Bool = false

    // Multi-Layer Backbone Execution State (h_0 -> h_N)
    @State private var targetLayerCount: Int = 40
    @State private var activeHFinalBuffer: MTLBuffer? = nil
    @State private var multiLayerStatusText: String? = nil
    @State private var multiLayerTelemetry: [LayerTelemetry] = []
    @State private var isExecutingMultiLayer: Bool = false

    // Final RMSNorm & LM Head Vocabulary Projection State
    @State private var activeLogitsBuffer: MTLBuffer? = nil
    @State private var topTokenPredictions: [TokenPrediction] = []
    @State private var lmHeadStatusText: String? = nil
    @State private var isExecutingLMHead: Bool = false

    // Autoregressive Text Generation State
    @State private var temperature: Float = 0.0
    @State private var topP: Float = 0.9
    @State private var minP: Float = 0.05
    @State private var topK: Int = 50
    @State private var repetitionPenalty: Float = 1.1
    @State private var presencePenalty: Float = 0.0
    @AppStorage("dynamoe_max_tokens") private var maxNewTokens: Int = 8192
    @State private var activeProfile: ModelProfileType = .coder
    @State private var isGeneratingText: Bool = false
    @State private var generatedStreamText: String = ""
    @State private var thinkingText: String = ""
    @State private var responseText: String = ""
    @State private var isThinking: Bool = false
    @State private var isThinkingExpanded: Bool = true
    @State private var generationSpeedTokPerSec: Double = 0.0
    @State private var generationTotalTokens: Int = 0
    @State private var generationElapsedMs: Double = 0.0
    @State private var generationStatusText: String? = nil
    @State private var generationTask: Task<Void, Never>? = nil

    // Working Set & Dynamic SSD Expert Paging State
    @State private var memoryExecutionMode: MemoryExecutionMode = .autoDetect
    @State private var memoryBudgetMode: MemoryBudgetMode = .balanced16GB
    @AppStorage("dynamoe_kv_cache_precision") private var kvCachePrecisionRaw: String = KVCachePrecision.fp16.rawValue
    @AppStorage("dynamoe_speculative_prefetch_enabled") private var speculativePrefetchEnabled: Bool = true
    @AppStorage("dynamoe_prefetch_lookahead_depth") private var prefetchLookaheadDepth: Int = 1
    @AppStorage("dynamoe_jetspec_enabled") private var jetSpecEnabled: Bool = false
    @AppStorage("dynamoe_jetspec_depth") private var jetSpecMaxDepth: Int = 3
    @AppStorage("dynamoe_jetspec_branching") private var jetSpecBranchingFactor: Int = 2
    @AppStorage("dynamoe_jetspec_expert_cap") private var jetSpecMaxExpertCap: Int = 8
    @State private var jetSpecMeanTau: Double = 1.0
    @State private var jetSpecTotalDraftAccepted: Int = 0
    @State private var jetSpecTotalDraftProposed: Int = 0

    var kvCachePrecisionBinding: Binding<KVCachePrecision> {
        Binding(
            get: { KVCachePrecision(rawValue: kvCachePrecisionRaw) ?? .fp16 },
            set: { kvCachePrecisionRaw = $0.rawValue }
        )
    }
    var kvCachePrecision: KVCachePrecision {
        KVCachePrecision(rawValue: kvCachePrecisionRaw) ?? .fp16
    }

    @State private var modelConfig: ModelConfig? = nil
    @State private var detectedArchitecture: ModelArchitectureType = .hybridSsmMoe
    @State private var currentRssGB: Double = 0.0
    @State private var residentExpertCount: Int = 0
    @State private var totalExpertCount: Int = 0
    @State private var cacheHitRate: Double = 100.0
    @State private var prefetchEfficiency: Double = 100.0
    @State private var lastPagingLatencyMs: Double = 0.0
    @State private var pagingStatusMessage: String? = nil

    // Multi-Session Chat UI State (Antigravity Style)
    @ObservedObject private var zoomManager = AppZoomManager.shared
    @ObservedObject var localModelManager: LocalModelManager = LocalModelManager.shared
    @State private var activeLoadedModelPath: String? = nil
    @State private var isLoadingModel: Bool = false
    @State private var columnVisibility: NavigationSplitViewVisibility = .all
    @State private var sessions: [ChatSession] = [
        ChatSession(title: "New Chat")
    ]
    @State private var selectedSessionId: UUID? = nil
    @State private var isSettingsPresented: Bool = false
    @State private var chatPromptText: String = ""
    @State private var systemPrompt: String = ModelConfig.getUserDefaultSystemPrompt()
    @AppStorage("dynamoe_thinking_enabled") private var defaultThinkingEnabled: Bool = true
    @AppStorage("dynamoe_agent_tools_enabled") private var defaultAgentToolsEnabled: Bool = false
    @AppStorage("dynamoe_agent_working_directory") private var agentWorkingDirectory: String = ""
    @AppStorage("dynamoe_max_tool_output_length") private var maxToolOutputLength: Int = 4000
    @AppStorage("dynamoe_max_agent_steps") private var maxAgentSteps: Int = 15

    var isAgentToolsEnabledForActiveSession: Bool {
        if let active = activeSessionBinding.wrappedValue, let enabled = active.isAgentToolsEnabled {
            return enabled
        }
        return defaultAgentToolsEnabled
    }

    var isStreamingOffDisk: Bool {
        guard let summary = summary else { return false }
        let isMoE = (modelConfig?.isMoE ?? (totalExpertCount > 0))
        if !isMoE {
            // Dense models always run purely in unified RAM -> Fast Bunny
            return false
        }
        let eff = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: summary.sizeGb)
        if eff == .residentRAM {
            return false
        }
        // In SSD streaming mode, if speed is actively fast (> 15 tok/s), show bunny; otherwise tortoise
        if generationSpeedTokPerSec >= 15.0 {
            return false
        }
        return true
    }

    var activeModelDisplayName: String? {
        if let session = activeSessionBinding.wrappedValue,
           let name = session.selectedModelName, !name.isEmpty {
            return name
        }
        if let activePath = activeLoadedModelPath {
            if let dm = localModelManager.discoveredModels.first(where: { $0.snapshotPath == activePath }) {
                return dm.displayName
            }
            if activePath.lowercased().contains("ornith") {
                return "Ornith 1.5 9B OptiQ-4bit"
            }
        }
        if let summary = summary {
            if let type = modelConfig?.modelType, !type.isEmpty {
                if type.lowercased().contains("ornith") {
                    return "Ornith 1.5 35B A3B FP8"
                }
                let isOrnithGDN = summary.tensors.contains(where: { $0.name.contains("linear_attn") }) &&
                                 !summary.tensors.contains(where: { $0.name.contains("hc_norm") || $0.name.contains("hyper_connection") })
                if isOrnithGDN {
                    return "Ornith 1.5 9B OptiQ-4bit"
                }
                return type
            }
            return detectedArchitecture.shortName
        }
        return nil
    }

    var activeModelSupportsThinking: Bool {
        // 1. Check loaded model config / summary / detected architecture
        if ModelConfig.supportsThinking(
            config: modelConfig,
            summary: summary,
            modelName: activeModelDisplayName,
            modelPath: activeLoadedModelPath
        ) {
            return true
        }
        // 2. Check active session selected model
        if let session = activeSessionBinding.wrappedValue {
            if let model = localModelManager.discoveredModels.first(where: { $0.id == (session.selectedModelId ?? "") || $0.snapshotPath == (session.selectedModelPath ?? "") }) {
                if model.supportsThinking { return true }
            }
            if ModelConfig.supportsThinking(
                modelName: session.selectedModelName,
                modelPath: session.selectedModelPath
            ) {
                return true
            }
        }
        return false
    }

    var isThinkingEnabledForActiveSession: Bool {
        if let session = activeSessionBinding.wrappedValue, let enabled = session.isThinkingEnabled {
            return enabled
        }
        return defaultThinkingEnabled
    }

    var activeSessionBinding: Binding<ChatSession?> {
        Binding<ChatSession?>(
            get: {
                if let id = selectedSessionId {
                    return sessions.first(where: { $0.id == id }) ?? sessions.first
                }
                return sessions.first
            },
            set: { updated in
                guard let updated = updated else { return }
                if let idx = sessions.firstIndex(where: { $0.id == updated.id }) {
                    sessions[idx] = updated
                }
            }
        )
    }

    let categoryFilters = ["All", "Self-Attention", "MoE Router", "Routed Expert", "Shared Expert", "Embedding", "LM Head"]

    var selectedTensor: TensorMetadata? {
        summary?.tensors.first(where: { $0.name == selectedTensorID })
    }

    private func applyProfile(_ profile: ModelProfileType, for modelIdentifier: String? = nil) {
        let modelKey = modelIdentifier ?? activeModelDisplayName ?? localModelManager.defaultModelId
        let settings = ModelProfileManager.shared.getProfile(for: modelKey, type: profile)
        self.activeProfile = profile
        ModelProfileManager.shared.setActiveProfile(for: modelKey, type: profile)

        self.temperature = settings.temperature
        self.topP = settings.topP
        self.minP = settings.minP
        self.topK = settings.topK
        self.repetitionPenalty = settings.repetitionPenalty
        self.presencePenalty = settings.presencePenalty
        self.maxNewTokens = settings.maxNewTokens
        self.systemPrompt = settings.systemPrompt
        self.jetSpecEnabled = settings.jetSpecEnabled
        self.jetSpecMaxDepth = settings.jetSpecMaxDepth
        self.jetSpecBranchingFactor = settings.jetSpecBranchingFactor
        self.jetSpecMaxExpertCap = settings.jetSpecMaxExpertCap
    }

    private func saveCurrentSettingsToProfile(_ profile: ModelProfileType, for modelIdentifier: String? = nil) {
        let modelKey = modelIdentifier ?? activeModelDisplayName ?? localModelManager.defaultModelId
        let settings = GenerationProfileSettings(
            temperature: self.temperature,
            topP: self.topP,
            minP: self.minP,
            topK: self.topK,
            repetitionPenalty: self.repetitionPenalty,
            presencePenalty: self.presencePenalty,
            maxNewTokens: self.maxNewTokens,
            systemPrompt: self.systemPrompt,
            jetSpecEnabled: self.jetSpecEnabled,
            jetSpecMaxDepth: self.jetSpecMaxDepth,
            jetSpecBranchingFactor: self.jetSpecBranchingFactor,
            jetSpecMaxExpertCap: self.jetSpecMaxExpertCap
        )
        ModelProfileManager.shared.saveProfile(for: modelKey, type: profile, settings: settings)
    }

    private func switchModel(to model: DiscoveredModel) {
        PrefixCacheManager.shared.invalidate()
        loadAndBridgeToMetal(filePath: model.snapshotPath)
        activeLoadedModelPath = model.snapshotPath
        localModelManager.setLastUsedModel(id: model.id)
        if let sid = selectedSessionId ?? sessions.first?.id,
           let idx = sessions.firstIndex(where: { $0.id == sid }) {
            sessions[idx].selectedModelId = model.id
            sessions[idx].selectedModelName = model.displayName
            sessions[idx].selectedModelPath = model.snapshotPath
        }
        let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: model.id)
        applyProfile(preferredProfile, for: model.id)
    }

    var body: some View {
        ZStack {
            NavigationSplitView(columnVisibility: $columnVisibility) {
                SidebarView(
                    sessions: $sessions,
                    selectedSessionId: $selectedSessionId,
                    isSettingsPresented: $isSettingsPresented,
                    modelName: activeModelDisplayName,
                    metalStatus: metalStatus,
                    currentRssGB: currentRssGB,
                    isGenerating: isGeneratingText,
                    onNewChat: {
                        let defModel = localModelManager.getDefaultOrFirstModel()
                        let newSession = ChatSession(
                            title: "New Chat",
                            selectedModelId: defModel?.id,
                            selectedModelName: defModel?.displayName,
                            selectedModelPath: defModel?.snapshotPath
                        )
                        sessions.insert(newSession, at: 0)
                        selectedSessionId = newSession.id
                        if let model = defModel, activeLoadedModelPath != model.snapshotPath {
                            switchModel(to: model)
                        }
                    },
                    onDeleteSession: { id in
                        PrefixCacheManager.shared.invalidate(sessionId: id)
                        sessions.removeAll(where: { $0.id == id })
                        if sessions.isEmpty {
                            let defModel = localModelManager.getDefaultOrFirstModel()
                            let newSession = ChatSession(
                                title: "New Chat",
                                selectedModelId: defModel?.id,
                                selectedModelName: defModel?.displayName,
                                selectedModelPath: defModel?.snapshotPath
                            )
                            sessions.append(newSession)
                            selectedSessionId = newSession.id
                        } else if selectedSessionId == id {
                            selectedSessionId = sessions.first?.id
                        }
                    }
                )
                .navigationSplitViewColumnWidth(min: max(180, 220 * zoomManager.zoomScale), ideal: max(210, 260 * zoomManager.zoomScale), max: max(260, 320 * zoomManager.zoomScale))
            } detail: {
                ChatDetailView(
                    session: activeSessionBinding,
                    promptText: $chatPromptText,
                    isGenerating: isGeneratingText,
                    isStreamingOffDisk: isStreamingOffDisk,
                    generationSpeed: generationSpeedTokPerSec,
                    generationTokens: generationTotalTokens,
                    jetSpecEnabled: jetSpecEnabled,
                    jetSpecMeanTau: jetSpecMeanTau,
                    jetSpecDraftAccepted: jetSpecTotalDraftAccepted,
                    modelName: activeModelDisplayName,
                    tokenizer: tokenizer,
                    activeProfile: activeProfile,
                    onSelectProfile: { profile in
                        applyProfile(profile, for: activeLoadedModelPath ?? localModelManager.defaultModelId)
                    },
                    supportsThinking: activeModelSupportsThinking,
                    isThinkingEnabled: isThinkingEnabledForActiveSession,
                    isAgentToolsEnabled: isAgentToolsEnabledForActiveSession,
                    onSendMessage: { prompt in
                        handleSendMessage(prompt)
                    },
                    onStopGeneration: {
                        stopAutoregressiveGeneration()
                    },
                    onQueuePrompt: { text in
                        handleQueuePrompt(text)
                    },
                    onSendImmediate: { text in
                        interruptAndSendMessage(text)
                    },
                    onRemoveQueuedPrompt: { id in
                        handleRemoveQueuedPrompt(id: id)
                    },
                    onSelectPromptStarter: { starter in
                        chatPromptText = starter
                        handleSendMessage(starter)
                        chatPromptText = ""
                    },
                    onSelectDiscoveredModel: { dm in
                        switchModel(to: dm)
                    },
                    onOpenSettings: {
                        openSettingsWindow()
                    },
                    onToggleThinking: { enabled in
                        defaultThinkingEnabled = enabled
                        if let sid = selectedSessionId ?? sessions.first?.id,
                           let idx = sessions.firstIndex(where: { $0.id == sid }) {
                            sessions[idx].isThinkingEnabled = enabled
                        }
                    },
                    onToggleAgentTools: { enabled in
                        defaultAgentToolsEnabled = enabled
                        if let sid = selectedSessionId ?? sessions.first?.id,
                           let idx = sessions.firstIndex(where: { $0.id == sid }) {
                            sessions[idx].isAgentToolsEnabled = enabled
                        }
                    },
                    onToggleSidebar: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            columnVisibility = (columnVisibility == .detailOnly) ? .all : .detailOnly
                        }
                    }
                )
            }

            if let hud = zoomManager.hudText {
                VStack {
                    Spacer()
                    HStack {
                        Spacer()
                        Text("Zoom: \(hud)")
                            .font(.system(size: 13, weight: .semibold, design: .rounded))
                            .foregroundColor(.white)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 7)
                            .background(Color.black.opacity(0.8))
                            .clipShape(Capsule())
                            .shadow(color: Color.black.opacity(0.2), radius: 6, x: 0, y: 3)
                            .padding(20)
                            .transition(.opacity.combined(with: .scale(scale: 0.9)))
                    }
                }
                .allowsHitTesting(false)
                .animation(.easeInOut(duration: 0.2), value: zoomManager.hudText)
            }
        }
        .onChange(of: isSettingsPresented) { isPresented in
            if isPresented {
                openSettingsWindow()
                isSettingsPresented = false
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .openDynaMoESettings)) { _ in
            openSettingsWindow()
        }
        .onChange(of: currentRssGB) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: residentExpertCount) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: cacheHitRate) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: activeLoadedModelPath) { _ in
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: memoryExecutionMode) { newMode in
            applyMemoryExecutionMode(newMode)
            syncSettingsWindowIfNeeded()
        }
        .onChange(of: memoryBudgetMode) { newBudget in
            applyMemoryBudgetMode(newBudget)
            syncSettingsWindowIfNeeded()
        }
        .onAppear {
            if selectedSessionId == nil {
                selectedSessionId = sessions.first?.id
            }
            if summary == nil {
                let isTesting = ProcessInfo.processInfo.environment["XCTestConfigurationFilePath"] != nil || NSClassFromString("XCTestCase") != nil
                if !isTesting, let initialModel = localModelManager.getDefaultOrFirstModel() {
                    switchModel(to: initialModel)
                }
            } else {
                let initialProfile = ModelProfileManager.shared.getActiveProfile(for: activeLoadedModelPath ?? localModelManager.defaultModelId)
                applyProfile(initialProfile, for: activeLoadedModelPath ?? localModelManager.defaultModelId)
            }
            updatePagingStats()
        }
        .onChange(of: selectedSessionId) { newId in
            guard let newId = newId, let session = sessions.first(where: { $0.id == newId }) else { return }
            if let targetPath = session.selectedModelPath, !targetPath.isEmpty, activeLoadedModelPath != targetPath {
                if let model = localModelManager.getModel(byId: targetPath) ?? localModelManager.getModel(byId: session.selectedModelId ?? "") {
                    switchModel(to: model)
                } else {
                    loadAndBridgeToMetal(filePath: targetPath)
                    activeLoadedModelPath = targetPath
                    let preferredProfile = ModelProfileManager.shared.getActiveProfile(for: targetPath)
                    applyProfile(preferredProfile, for: targetPath)
                }
            }
        }
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                updatePagingStats()
            }
        }
    }

    // MARK: - Independent Settings Window Management
    @ViewBuilder
    private func buildSettingsSheetView() -> some View {
        SettingsSheetView(
            summary: summary,
            modelConfig: modelConfig,
            tokenizer: tokenizer,
            activeModelPath: activeLoadedModelPath,
            metalStatus: metalStatus,
            detectedArchitecture: detectedArchitecture,
            onSelectModel: {
                #if os(macOS)
                selectModelWithOpenPanel()
                #else
                isWeightImporterPresented = true
                #endif
            },
            onSelectTokenizer: {
                #if os(macOS)
                selectTokenizerWithOpenPanel()
                #else
                isTokenizerImporterPresented = true
                #endif
            },
            onLoadDiscoveredModel: { dm in
                switchModel(to: dm)
            },
            temperature: $temperature,
            topP: $topP,
            minP: $minP,
            topK: $topK,
            repetitionPenalty: $repetitionPenalty,
            presencePenalty: $presencePenalty,
            maxNewTokens: $maxNewTokens,
            systemPrompt: $systemPrompt,
            targetLayerCount: $targetLayerCount,
            activeProfile: $activeProfile,
            memoryExecutionMode: $memoryExecutionMode,
            memoryBudgetMode: $memoryBudgetMode,
            kvCachePrecision: kvCachePrecisionBinding,
            speculativePrefetchEnabled: $speculativePrefetchEnabled,
            prefetchLookaheadDepth: $prefetchLookaheadDepth,
            currentRssGB: currentRssGB,
            residentExpertCount: residentExpertCount,
            totalExpertCount: totalExpertCount,
            cacheHitRate: cacheHitRate,
            prefetchEfficiency: prefetchEfficiency,
            lastPagingLatencyMs: lastPagingLatencyMs,
            pagingStatusMessage: pagingStatusMessage,
            onFlushCache: { flushExpertCache() },
            onPreFaultAll: { preFaultAllWeights() },
            searchText: $searchText,
            selectedCategory: $selectedCategory,
            categoryFilters: categoryFilters,
            selectedTensorID: $selectedTensorID,
            selectedTensor: selectedTensor,
            onExecuteMoERouter: { l in executeMoERouter(layerIndex: l) },
            onExecuteFullLayer: { l in executeFullLayerForward(layerIndex: l) },
            onExecuteMultiLayer: { n in executeMultiLayerForward(numLayers: n) },
            isExecutingMlp: isExecutingMlp,
            isExecutingFullLayer: isExecutingFullLayer,
            isExecutingMultiLayer: isExecutingMultiLayer
        )
    }

    private func openSettingsWindow() {
        SettingsWindowManager.shared.show(title: "Settings & Diagnostics") {
            buildSettingsSheetView()
        }
    }

    private func syncSettingsWindowIfNeeded() {
        if SettingsWindowManager.shared.isWindowOpen {
            SettingsWindowManager.shared.update {
                buildSettingsSheetView()
            }
        }
    }

    private func handleQueuePrompt(_ text: String) {
        guard let currentSessionId = selectedSessionId ?? sessions.first?.id else { return }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        sessions[sessionIdx].queuedPrompts.append(QueuedPrompt(text: trimmed))
    }

    private func handleRemoveQueuedPrompt(id: UUID) {
        guard let currentSessionId = selectedSessionId ?? sessions.first?.id else { return }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return }
        sessions[sessionIdx].queuedPrompts.removeAll(where: { $0.id == id })
    }

    private func interruptAndSendMessage(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        if isGeneratingText {
            stopAutoregressiveGeneration()
            Task { @MainActor in
                // Brief yield to allow the cancelled generation task to release resources cleanly
                try? await Task.sleep(nanoseconds: 60_000_000)
                self.handleSendMessage(trimmed)
            }
        } else {
            handleSendMessage(trimmed)
        }
    }

    private func dequeueAndRunNextPromptIfNeeded(sessionId: UUID?) {
        guard let currentSessionId = sessionId ?? selectedSessionId ?? sessions.first?.id else { return }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return }
        guard !sessions[sessionIdx].queuedPrompts.isEmpty else { return }
        let next = sessions[sessionIdx].queuedPrompts.removeFirst()
        Task { @MainActor in
            // Smooth transition to next turn
            try? await Task.sleep(nanoseconds: 80_000_000)
            self.handleSendMessage(next.text)
        }
    }

    private func handleSendMessage(_ text: String) {
        guard let currentSessionId = selectedSessionId ?? sessions.first?.id else { return }
        guard let sessionIdx = sessions.firstIndex(where: { $0.id == currentSessionId }) else { return }
        
        let userMsg = ChatMessage(role: .user, content: text)
        sessions[sessionIdx].messages.append(userMsg)
        
        // Auto-title session if it's the first user message
        if sessions[sessionIdx].messages.filter({ $0.role == .user }).count == 1 {
            let cleanTitle = text.prefix(28).trimmingCharacters(in: .whitespacesAndNewlines)
            sessions[sessionIdx].title = cleanTitle.isEmpty ? "Chat" : String(cleanTitle)
            sessions[sessionIdx].createdAt = Date()
        }
        
        let modelSupportsThinking = activeModelSupportsThinking
        let thinkingEnabled = (sessions[sessionIdx].isThinkingEnabled ?? defaultThinkingEnabled) && modelSupportsThinking

        let assistantMsgId = UUID()
        let assistantMsg = ChatMessage(id: assistantMsgId, role: .assistant, content: "", thinkingContent: nil, isThinking: thinkingEnabled)
        sessions[sessionIdx].messages.append(assistantMsg)
        
        // Build prompt formatted with chat template
        var promptString = ""
        let conversationDate = sessions[sessionIdx].createdAt
        var effectiveSystem = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: systemPrompt,
            config: modelConfig,
            summary: summary,
            modelName: activeModelDisplayName,
            modelPath: activeLoadedModelPath,
            currentDate: conversationDate
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        let agentToolsEnabled = (sessions[sessionIdx].isAgentToolsEnabled ?? defaultAgentToolsEnabled)
        if agentToolsEnabled {
            effectiveSystem = AgentHarness.shared.buildSystemPrompt(
                baseSystem: effectiveSystem,
                modelName: activeModelDisplayName,
                currentDate: conversationDate
            )
        }

        let isLing = (modelConfig?.isLingModel == true)
        if isLing {
            let thinkingOption = thinkingEnabled ? "on" : "off"
            if !effectiveSystem.isEmpty {
                promptString += "<role>SYSTEM</role>\(effectiveSystem)\ndetailed thinking \(thinkingOption)<|role_end|>"
            } else {
                promptString += "<role>SYSTEM</role>detailed thinking \(thinkingOption)<|role_end|>"
            }
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<|role_end|>", with: "")
                    .replacingOccurrences(of: "<role>", with: "")
                    .replacingOccurrences(of: "</role>", with: "")
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools else { continue }

                if msg.role == .user {
                    promptString += "<role>HUMAN</role>\(cleanMsg)<|role_end|>"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if let think = msg.thinkingContent, !think.isEmpty {
                        let cleanThink = think
                            .replacingOccurrences(of: "<think>", with: "")
                            .replacingOccurrences(of: "</think>", with: "")
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        assistantBody += "\n<think>\(cleanThink)</think>"
                    } else {
                        assistantBody += "\n<think></think>"
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg
                    }
                    promptString += "<role>ASSISTANT</role>\(assistantBody)<|role_end|>"
                }
            }
            if thinkingEnabled {
                promptString += "<role>ASSISTANT</role>\n<think>"
            } else {
                promptString += "<role>ASSISTANT</role>\n<think></think>"
            }
        } else {
            if !effectiveSystem.isEmpty {
                promptString += "<|im_start|>system\n\(effectiveSystem)<|im_end|>\n"
            }
            for msg in sessions[sessionIdx].messages.dropLast() {
                let cleanMsg = msg.content
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let hasTools = (msg.toolCalls != nil && !msg.toolCalls!.isEmpty)
                guard !cleanMsg.isEmpty || (msg.thinkingContent != nil && !msg.thinkingContent!.isEmpty) || hasTools else { continue }

                if msg.role == .user {
                    promptString += "<|im_start|>user\n\(cleanMsg)<|im_end|>\n"
                } else if msg.role == .assistant {
                    var assistantBody = ""
                    if modelSupportsThinking {
                        if let think = msg.thinkingContent, !think.isEmpty {
                            let cleanThink = think
                                .replacingOccurrences(of: "<think>", with: "")
                                .replacingOccurrences(of: "</think>", with: "")
                                .trimmingCharacters(in: .whitespacesAndNewlines)
                            assistantBody += "<think>\n\(cleanThink)\n</think>\n\n"
                        } else {
                            assistantBody += "<think>\n\n</think>\n\n"
                        }
                    }
                    if !cleanMsg.isEmpty {
                        assistantBody += cleanMsg + "\n"
                    }
                    if let calls = msg.toolCalls, !calls.isEmpty {
                        for call in calls {
                            assistantBody += "<tool_call>\n<function=\(call.name)>\n"
                            for (k, v) in call.arguments {
                                assistantBody += "<parameter=\(k)>\n\(v)\n</parameter>\n"
                            }
                            assistantBody += "</function>\n</tool_call>\n"
                        }
                    }
                    promptString += "<|im_start|>assistant\n\(assistantBody.trimmingCharacters(in: .whitespacesAndNewlines))<|im_end|>\n"

                    if let calls = msg.toolCalls, !calls.isEmpty {
                        var outputs: [String] = []
                        for call in calls {
                            if let out = call.output ?? call.error {
                                outputs.append(out)
                            }
                        }
                        if !outputs.isEmpty {
                            promptString += AgentHarness.shared.formatToolResponseTurn(responses: outputs, includeThinkSuffix: false)
                        }
                    }
                }
            }
            if thinkingEnabled {
                promptString += "<|im_start|>assistant\n<think>\n"
            } else if modelSupportsThinking {
                // Prefill an empty closed <think>\n\n</think>\n\n block to guarantee reasoning models
                // (Nanbeige, DeepSeek-R1, QwQ) bypass reasoning entirely and output the direct answer!
                promptString += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            } else {
                promptString += "<|im_start|>assistant\n"
            }
        }
        
        startAutoregressiveGeneration(customPrompt: promptString, sessionId: currentSessionId, messageId: assistantMsgId)
    }

    // MARK: - Tokenizer Execution
    private func loadTokenizer(filePath: String) {
        do {
            let tok = try DynaMoeTokenizer(tokenizerPath: filePath)
            self.tokenizer = tok
            runTokenization(text: promptInput)
        } catch {
            tokenIDsOutput = "❌ Failed to load tokenizer.json: \(error.localizedDescription)"
        }
    }
    
    private func runTokenization(text: String) {
        guard let tokenizer = tokenizer else {
            tokenIDsOutput = "Load a tokenizer.json file to tokenize text"
            return
        }
        do {
            let ids = try tokenizer.encode(text: text)
            let decoded = try tokenizer.decode(ids: ids)
            tokenIDsOutput = "Tokens (\(ids.count)): \(ids) ➔ Decoded: \"\(decoded)\""
        } catch {
            tokenIDsOutput = "❌ Encoding Error: \(error.localizedDescription)"
        }
    }
    
    // MARK: - Execute Token Embedding Lookup (h_0)
    private func executeEmbeddingLookup(tokenIds: [UInt32]) {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Error setting up Metal embedding pipeline."
            return
        }

        do {
            guard let embedWeight = summary.tensors.first(where: {
                !$0.name.contains("visual") && !$0.name.contains("mtp") &&
                ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
                $0.name.contains("weight") && !$0.name.contains("scale")
            }) else {
                gpuComputeOutput = "❌ Couldn't locate embedding weight tensor."
                return
            }

            guard let rawBaseBuffer = shardBuffers[embedWeight.shardIndex] else {
                gpuComputeOutput = "❌ Shard #\(embedWeight.shardIndex) buffer not loaded."
                return
            }

            var weightOffset = embedWeight.offsetStart
            var hiddenDim: UInt32 = 2048 // Qwen 35B hidden dimension
            let tokenCount = tokenIds.count
            let totalVectorElements = tokenCount * Int(hiddenDim)
            
            guard let tokenBuffer = device.makeBuffer(bytes: tokenIds, length: tokenCount * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let h0OutputBuffer = device.makeBuffer(length: totalVectorElements * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let isBF16 = embedWeight.dtype.contains("BF16") || embedWeight.dtype.contains("BFLOAT16")
            
            if isBF16 {
                guard let kernelFunction = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16") else { return }
                let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                
                computeEncoder.setComputePipelineState(pipelineState)
                computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(h0OutputBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            } else {
                let embedScale = summary.tensors.first(where: {
                    $0.name.contains(embedWeight.name.replacingOccurrences(of: ".weight", with: "")) &&
                    ($0.name.contains("scale") || $0.name.contains("scales"))
                })
                let embedScaleRaw = (embedScale != nil) ? shardBuffers[embedScale!.shardIndex] : rawBaseBuffer
                var scaleOffset = embedScale?.offsetStart ?? 0
                
                if let kernelFunction = defaultLibrary.makeFunction(name: "lookup_embeddings_fp8") {
                    let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                    computeEncoder.setComputePipelineState(pipelineState)
                    computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(tokenBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0OutputBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(embedScaleRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                }
            }
            
            let gridSize = MTLSize(width: totalVectorElements, height: 1, depth: 1)
            let threadgroupSize = MTLSize(width: min(totalVectorElements, 256), height: 1, depth: 1)
            computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let rawFloatPtr = h0OutputBuffer.contents().bindMemory(to: Float.self, capacity: totalVectorElements)
            var h0Sample: [String] = []
            for i in 0..<min(8, totalVectorElements) {
                h0Sample.append(String(format: "%.6f", rawFloatPtr[i]))
            }
            
            self.activeH0Buffer = h0OutputBuffer
            self.activeTokenCount = tokenCount
            self.activeHiddenDim = Int(hiddenDim)
            
            gpuComputeOutput = "🚀 Generated h_0 Vector (\(embedWeight.dtype)) [\(tokenCount) x \(hiddenDim)] from Shard #\(embedWeight.shardIndex)! First 8 dims of Token #0: [\(h0Sample.joined(separator: ", "))]"
            
        } catch {
            gpuComputeOutput = "❌ Embedding Lookup Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute MoE Top-K Router
    private func executeMoERouter(layerIndex: Int, topK: Int = 8) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Please click 'Generate Hidden State h_0' first."
            return
        }

        do {
            // Locate the router gate weight tensor for the target layer
            guard let gateWeight = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) &&
                ($0.category == "MoE Router" || $0.name.hasSuffix("mlp.gate.weight") || $0.name.contains("mlp.gate")) &&
                !$0.name.contains("shared") &&
                !$0.name.contains("scale")
            }) else {
                gpuComputeOutput = "❌ Could not find mlp.gate.weight for Layer #\(layerIndex)."
                return
            }

            guard let rawGateBuffer = shardBuffers[gateWeight.shardIndex] else {
                gpuComputeOutput = "❌ Shard #\(gateWeight.shardIndex) containing Layer #\(layerIndex) gate is not loaded."
                return
            }

            let numExpertsVal: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : 256
            var topKVal: UInt32 = UInt32(topK)
            var hiddenDimVal: UInt32 = UInt32(activeHiddenDim)
            var gateOffset: UInt64 = gateWeight.offsetStart
            var numExperts: UInt32 = numExpertsVal

            guard let outIndicesBuffer = device.makeBuffer(length: topK * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                  let outWeightsBuffer = device.makeBuffer(length: topK * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            guard let kernelFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16") else {
                gpuComputeOutput = "❌ Failed to load moe_router_topk_bf16 kernel."
                return
            }
            let pipelineState = try device.makeComputePipelineState(function: kernelFunction)

            computeEncoder.setComputePipelineState(pipelineState)
            computeEncoder.setBuffer(rawGateBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 1)
            computeEncoder.setBuffer(outIndicesBuffer, offset: 0, index: 2)
            computeEncoder.setBuffer(outWeightsBuffer, offset: 0, index: 3)
            computeEncoder.setBytes(&gateOffset, length: MemoryLayout<UInt64>.stride, index: 4)
            computeEncoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
            computeEncoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 6)
            computeEncoder.setBytes(&topKVal, length: MemoryLayout<UInt32>.stride, index: 7)

            let threadsPerThreadgroup = MTLSize(width: Int(numExpertsVal), height: 1, depth: 1)
            let threadgroups = MTLSize(width: 1, height: 1, depth: 1)
            computeEncoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)

            // Also check for shared expert gate
            var sharedOutBuffer: MTLBuffer? = nil
            if let sharedGateWeight = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("shared_expert_gate") && !$0.name.contains("scale")
            }), let sharedRawBuffer = shardBuffers[sharedGateWeight.shardIndex],
               let sharedKernel = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") {
                
                let sharedPipelineState = try device.makeComputePipelineState(function: sharedKernel)
                var sharedOffset: UInt64 = sharedGateWeight.offsetStart
                sharedOutBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)
                
                if let sharedOutBuffer = sharedOutBuffer {
                    computeEncoder.setComputePipelineState(sharedPipelineState)
                    computeEncoder.setBuffer(sharedRawBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(sharedOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&sharedOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                    
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }
            }

            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            let indicesPtr = outIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: topK)
            let weightsPtr = outWeightsBuffer.contents().bindMemory(to: Float.self, capacity: topK)

            var extractedExperts: [(id: Int, weight: Float)] = []
            var sumProb: Float = 0.0
            for i in 0..<topK {
                let expertId = Int(indicesPtr[i])
                let w = weightsPtr[i]
                extractedExperts.append((id: expertId, weight: w))
                sumProb += w
            }
            self.routedExperts = extractedExperts

            if let sharedOutBuffer = sharedOutBuffer {
                let sharedPtr = sharedOutBuffer.contents().bindMemory(to: Float.self, capacity: 1)
                self.sharedExpertWeight = sharedPtr[0]
            } else {
                self.sharedExpertWeight = nil
            }

            let topExpertSummary = extractedExperts.map { "#\($0.id) (\(String(format: "%.1f%%", $0.weight * 100)))" }.joined(separator: ", ")
            let status = "⚡ Layer #\(layerIndex) Gated on GPU in \(String(format: "%.3f", elapsedMs)) ms! Top-\(topK): [\(topExpertSummary)] (Sum: \(String(format: "%.4f", sumProb)))"
            self.routerStatusText = status
            self.gpuComputeOutput = status

        } catch {
            gpuComputeOutput = "❌ MoE Router Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute MoE Layer MLP (SwiGLU across Top-K + Shared Experts)
    private func executeMoELayerMLP(layerIndex: Int, topK: Int = 8) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingMlp = true
        defer { self.isExecutingMlp = false }

        do {
            // 1. Ensure Routing has been performed for this layer
            if routedExperts.isEmpty {
                executeMoERouter(layerIndex: layerIndex, topK: topK)
            }
            guard !routedExperts.isEmpty else {
                gpuComputeOutput = "❌ MoE routing failed for Layer #\(layerIndex)."
                return
            }

            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var intermediateDim: UInt32 = 512 // Default intermediate size for Qwen/Ornith 35B
            
            // Check if any expert tensor reveals intermediate dimension
            if let sampleGate = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
            }) {
                let dims = sampleGate.shapeDisplay
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                    .components(separatedBy: ",")
                    .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                if dims.count >= 2 {
                    intermediateDim = dims[0]
                    hiddenDim = dims[1]
                }
            }

            // 2. Allocate output accumulator h_mlp and intermediate buffer
            guard let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                gpuComputeOutput = "❌ Failed to allocate GPU command encoder/buffers."
                return
            }

            // 3. Clear h_mlp accumulator vector to 0.0
            if let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32") {
                let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
                let gridSize = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                let threadgroupSize = MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
            }

            // Helper to lookup tensor by query
            let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) &&
                    t.name.contains(sub1) &&
                    t.name.contains(sub2) &&
                    (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                })
            }

            // Load pipeline states for FP8 and BF16 SwiGLU & Down Projections
            guard let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to locate SwiGLU / DownProj Metal kernels."
                return
            }

            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
            let bf16GateUpPipeline = try device.makeComputePipelineState(function: bf16GateUpKernel)
            let bf16DownPipeline = try device.makeComputePipelineState(function: bf16DownKernel)

            var executedExpertCount = 0

            // 4. Dispatch each of the Top-K Routed Experts
            for expert in routedExperts {
                let expId = expert.id
                var p_k = expert.weight
                if p_k <= 0.00001 { continue }

                let expTag = "experts.\(expId)"
                guard let gateWeight = findTensor(expTag, "gate_proj", false),
                      let upWeight   = findTensor(expTag, "up_proj", false),
                      let downWeight = findTensor(expTag, "down_proj", false),
                      let gateRaw    = shardBuffers[gateWeight.shardIndex],
                      let upRaw      = shardBuffers[upWeight.shardIndex],
                      let downRaw    = shardBuffers[downWeight.shardIndex] else {
                    continue
                }

                let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                          let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                          let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                        continue
                    }

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale!.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale!.offsetStart
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale!.offsetStart

                    // Dispatch SwiGLU (Gate & Up Proj)
                    computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                    computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    // Dispatch Down Proj with Weighted Accumulation
                    computeEncoder.setComputePipelineState(fp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)

                } else {
                    var gateWeightOffset = gateWeight.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var downWeightOffset = downWeight.offsetStart

                    // Dispatch BF16 SwiGLU
                    computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    // Dispatch BF16 Down Proj
                    computeEncoder.setComputePipelineState(bf16DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                }

                executedExpertCount += 1
            }

            // 5. Dispatch Shared Expert if present
            var sharedDispatched = false
            if var sharedWeight = sharedExpertWeight, sharedWeight > 0.0001 {
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
                    } else {
                        var gateWeightOffset = gateWeight.offsetStart
                        var upWeightOffset   = upWeight.offsetStart
                        var downWeightOffset = downWeight.offsetStart

                        computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(h0Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(bf16DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 6)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                    sharedDispatched = true
                }
            }

            // 6. Commit and Measure Execution Time
            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            // 7. Calculate L2 Norm & Sample Output
            let outPtr = hMlpBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var sumSquares: Float = 0.0
            var sampleValues: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = outPtr[i]
                sumSquares += v * v
                if i < 8 {
                    sampleValues.append(String(format: "%.5f", v))
                }
            }
            let l2Norm = sqrt(sumSquares)

            self.activeHmlpBuffer = hMlpBuffer
            let sharedDesc = sharedDispatched ? "+ 1 Shared Expert" : ""
            let status = "⚡ Layer #\(layerIndex) MoE MLP Computed in \(String(format: "%.3f", elapsedMs)) ms! (\(executedExpertCount) Routed Experts \(sharedDesc)) | L2 Norm: \(String(format: "%.4f", l2Norm))"
            self.layerMlpStatusText = status
            self.layerMlpSampleOutput = "First 8 dims of h_mlp: [\(sampleValues.joined(separator: ", "))]"
            self.gpuComputeOutput = "\(status)\n\(self.layerMlpSampleOutput!)"

        } catch {
            gpuComputeOutput = "❌ MoE MLP Execution Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute Full Layer Forward Pass (h_l -> h_l+1)
    private func executeFullLayerForward(layerIndex: Int) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingFullLayer = true
        defer { self.isExecutingFullLayer = false }

        do {
            // Ensure routing is calculated
            if routedExperts.isEmpty {
                executeMoERouter(layerIndex: layerIndex, topK: 8)
            }

            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var eps: Float = 1e-6

            let findTensorInLayer = { (query: String) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) && t.name.contains(query)
                })
            }

            guard let rmsKernel = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
                  let addKernel = defaultLibrary.makeFunction(name: "vector_add_f32"),
                  let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32"),
                  let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let bf16GateUpKernel = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up"),
                  let bf16DownKernel = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for full layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
            let bf16GateUpPipeline = try device.makeComputePipelineState(function: bf16GateUpKernel)
            let bf16DownPipeline = try device.makeComputePipelineState(function: bf16DownKernel)

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                gpuComputeOutput = "❌ Failed to allocate GPU command encoder."
                return
            }

            let byteLength = Int(hiddenDim) * MemoryLayout<Float>.stride
            guard let xNorm1Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let attnOutBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMidBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm2Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMlpBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hNextBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate GPU layer buffers."
                return
            }

            // -------------------------------------------------------------
            // Step 1: Pre-Attention RMSNorm (h_l -> xNorm1)
            // -------------------------------------------------------------
            if let norm1Tensor = findTensorInLayer("input_layernorm"),
               let norm1Raw = shardBuffers[norm1Tensor.shardIndex] {
                var gammaOffset = norm1Tensor.offsetStart
                computeEncoder.setComputePipelineState(rmsPipeline)
                computeEncoder.setBuffer(h0Buffer, offset: 0, index: 0)
                computeEncoder.setBuffer(norm1Raw, offset: 0, index: 1)
                computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
            }

            // -------------------------------------------------------------
            // Step 2: Attention Computation (xNorm1 -> attnOut)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(clearPipeline)
            computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            if let oProjTensor = findTensorInLayer("o_proj") ?? findTensorInLayer("out_proj"),
               let oProjRaw = shardBuffers[oProjTensor.shardIndex] {
                let isFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                var oProjOffset = oProjTensor.offsetStart
                var inAttnDim = hiddenDim

                if isFP8, let gemvKernel = defaultLibrary.makeFunction(name: "fp8_gemv") {
                    let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                    let oScaleTensor = summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(layerIndex) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                    })
                    let oScaleRaw = (oScaleTensor != nil) ? shardBuffers[oScaleTensor!.shardIndex] : oProjRaw
                    var oScaleOffset = oScaleTensor?.offsetStart ?? 0

                    computeEncoder.setComputePipelineState(gemvPipeline)
                    computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(oScaleRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), gemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                } else if let bf16GemvKernel = defaultLibrary.makeFunction(name: "bf16_gemv") {
                    let bf16GemvPipeline = try device.makeComputePipelineState(function: bf16GemvKernel)
                    computeEncoder.setComputePipelineState(bf16GemvPipeline)
                    computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), bf16GemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
            }

            // -------------------------------------------------------------
            // Step 3: Residual Connection 1 (h_mid = h_l + attnOut)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(addPipeline)
            computeEncoder.setBuffer(h0Buffer, offset: 0, index: 0)
            computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            // -------------------------------------------------------------
            // Step 4: Post-Attention RMSNorm (h_mid -> xNorm2)
            // -------------------------------------------------------------
            if let norm2Tensor = findTensorInLayer("post_attention_layernorm"),
               let norm2Raw = shardBuffers[norm2Tensor.shardIndex] {
                var gammaOffset = norm2Tensor.offsetStart
                computeEncoder.setComputePipelineState(rmsPipeline)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(norm2Raw, offset: 0, index: 1)
                computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
            }

            // -------------------------------------------------------------
            // Step 5: MoE Router & SwiGLU Feed-Forward Compute (xNorm2 -> hMlp)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(clearPipeline)
            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            var intermediateDim: UInt32 = 512
            if let sampleGate = summary.tensors.first(where: {
                $0.layerIndex == UInt32(layerIndex) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
            }) {
                let dims = sampleGate.shapeDisplay
                    .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                    .components(separatedBy: ",")
                    .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                if dims.count >= 2 {
                    intermediateDim = dims[0]
                }
            }

            guard let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                return
            }

            let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                return summary.tensors.first(where: { t in
                    t.layerIndex == UInt32(layerIndex) &&
                    t.name.contains(sub1) &&
                    t.name.contains(sub2) &&
                    (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                })
            }

            for expert in routedExperts {
                let expId = expert.id
                var p_k = expert.weight
                if p_k <= 0.00001 { continue }

                let expTag = "experts.\(expId)"
                guard let gateWeight = findTensor(expTag, "gate_proj", false),
                      let upWeight   = findTensor(expTag, "up_proj", false),
                      let downWeight = findTensor(expTag, "down_proj", false),
                      let gateRaw    = shardBuffers[gateWeight.shardIndex],
                      let upRaw      = shardBuffers[upWeight.shardIndex],
                      let downRaw    = shardBuffers[downWeight.shardIndex] else {
                    continue
                }

                let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                if isFP8 {
                    let gateScale = findTensor(expTag, "gate_proj", true)
                    let upScale   = findTensor(expTag, "up_proj", true)
                    let downScale = findTensor(expTag, "down_proj", true)

                    guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                          let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                          let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                        continue
                    }

                    var gateWeightOffset = gateWeight.offsetStart
                    var gateScaleOffset  = gateScale!.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var upScaleOffset    = upScale!.offsetStart
                    var downWeightOffset = downWeight.offsetStart
                    var downScaleOffset  = downScale!.offsetStart

                    computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                    computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                    computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                    computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    computeEncoder.setComputePipelineState(fp8DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                } else {
                    var gateWeightOffset = gateWeight.offsetStart
                    var upWeightOffset   = upWeight.offsetStart
                    var downWeightOffset = downWeight.offsetStart

                    computeEncoder.setComputePipelineState(bf16GateUpPipeline)
                    computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                    computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 6)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 7)

                    let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                    let interTg = MTLSize(width: min(Int(intermediateDim), bf16GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                    computeEncoder.setComputePipelineState(bf16DownPipeline)
                    computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)

                    let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                    let hiddenTg = MTLSize(width: min(Int(hiddenDim), bf16DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                    computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                }
            }

            if var sharedWeight = sharedExpertWeight, sharedWeight > 0.0001 {
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedWeight, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
                    }
                }
            }

            // -------------------------------------------------------------
            // Step 6: Residual Connection 2 (h_next = h_mid + hMlp)
            // -------------------------------------------------------------
            computeEncoder.setComputePipelineState(addPipeline)
            computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(hNextBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            // -------------------------------------------------------------
            // Step 7: Commit & Measure Full Block Latency
            // -------------------------------------------------------------
            let startTime = CFAbsoluteTimeGetCurrent()
            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            let elapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            let outPtr = hNextBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var sumSquares: Float = 0.0
            var sampleValues: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = outPtr[i]
                sumSquares += v * v
                if i < 8 {
                    sampleValues.append(String(format: "%.5f", v))
                }
            }
            let l2Norm = sqrt(sumSquares)

            self.activeH1Buffer = hNextBuffer
            let status = "⚡ Layer #\(layerIndex) Full Block (RMSNorm + Attn + Res + MoE + Res) Executed in \(String(format: "%.3f", elapsedMs)) ms! | Output h_\(layerIndex + 1) L2 Norm: \(String(format: "%.4f", l2Norm))"
            self.fullLayerStatusText = status
            self.fullLayerSampleOutput = "First 8 dims of h_\(layerIndex + 1): [\(sampleValues.joined(separator: ", "))]"
            self.gpuComputeOutput = "\(status)\n\(self.fullLayerSampleOutput!)"

        } catch {
            gpuComputeOutput = "❌ Full Layer Forward Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Execute Sequential Multi-Layer Forward Pass (h_0 -> h_N)
    private func executeMultiLayerForward(numLayers: Int) {
        guard let summary = summary,
              let h0Buffer = activeH0Buffer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            gpuComputeOutput = "❌ Missing active h_0 buffer or Metal device. Generate h_0 first."
            return
        }

        self.isExecutingMultiLayer = true
        defer { self.isExecutingMultiLayer = false }

        do {
            var hiddenDim: UInt32 = UInt32(activeHiddenDim)
            var eps: Float = 1e-6

            guard let rmsKernel = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
                  let addKernel = defaultLibrary.makeFunction(name: "vector_add_f32"),
                  let clearKernel = defaultLibrary.makeFunction(name: "clear_vector_f32"),
                  let fp8GateUpKernel = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up"),
                  let fp8DownKernel = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate"),
                  let routerKernel = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
                  let sharedGateKernel = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") else {
                gpuComputeOutput = "❌ Failed to load required Metal compute functions for multi-layer forward."
                return
            }

            let rmsPipeline = try device.makeComputePipelineState(function: rmsKernel)
            let addPipeline = try device.makeComputePipelineState(function: addKernel)
            let clearPipeline = try device.makeComputePipelineState(function: clearKernel)
            let fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpKernel)
            let fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownKernel)
            let routerPipeline = try device.makeComputePipelineState(function: routerKernel)
            let sharedGatePipeline = try device.makeComputePipelineState(function: sharedGateKernel)

            let byteLength = Int(hiddenDim) * MemoryLayout<Float>.stride
            guard let hCurrBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hNextBuffer  = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm1Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let attnOutBuffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMidBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let xNorm2Buffer = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let hMlpBuffer   = device.makeBuffer(length: byteLength, options: .storageModeShared),
                  let routerScoresBuffer = device.makeBuffer(length: 256 * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate GPU multi-layer buffers."
                return
            }

            // Copy initial h0 into hCurrBuffer
            memcpy(hCurrBuffer.contents(), h0Buffer.contents(), byteLength)

            var telemetryList: [LayerTelemetry] = []
            let totalStartTime = CFAbsoluteTimeGetCurrent()
            let actualLayers = min(numLayers, Int(summary.layerCount))

            for l in 0..<actualLayers {
                let layerStartTime = CFAbsoluteTimeGetCurrent()

                let findTensorInLayer = { (query: String) -> TensorMetadata? in
                    return summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(l) && t.name.contains(query)
                    })
                }

                guard let commandBuffer = commandQueue.makeCommandBuffer(),
                      let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                    continue
                }

                // ---------------------------------------------------------
                // Step 0: Dynamic Routing for Layer l
                // ---------------------------------------------------------
                var activeExperts: [(id: Int, weight: Float)] = []

                if let routerTensor = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("mlp.gate.weight")
                }), let routerRaw = shardBuffers[routerTensor.shardIndex] {
                    var routerOffset = routerTensor.offsetStart
                    var numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : 256

                    computeEncoder.setComputePipelineState(routerPipeline)
                    computeEncoder.setBuffer(routerRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(routerScoresBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&routerOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&numExperts, length: MemoryLayout<UInt32>.stride, index: 5)
                    computeEncoder.dispatchThreads(MTLSize(width: Int(numExperts), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numExperts), routerPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                if let sharedGateTensor = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("shared_expert_gate.weight")
                }), let sharedGateRaw = shardBuffers[sharedGateTensor.shardIndex] {
                    var gateOffset = sharedGateTensor.offsetStart
                    computeEncoder.setComputePipelineState(sharedGatePipeline)
                    computeEncoder.setBuffer(sharedGateRaw, offset: 0, index: 0)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(sharedScoreBuffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gateOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 1: Pre-Attention RMSNorm (hCurr -> xNorm1)
                // ---------------------------------------------------------
                if let norm1Tensor = findTensorInLayer("input_layernorm"),
                   let norm1Raw = shardBuffers[norm1Tensor.shardIndex] {
                    var gammaOffset = norm1Tensor.offsetStart
                    computeEncoder.setComputePipelineState(rmsPipeline)
                    computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(norm1Raw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                    computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 2: Attention Computation (xNorm1 -> attnOut)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 0)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                if let oProjTensor = findTensorInLayer("o_proj") ?? findTensorInLayer("out_proj"),
                   let oProjRaw = shardBuffers[oProjTensor.shardIndex] {
                    let isFP8 = !oProjTensor.dtype.contains("BF16") && !oProjTensor.dtype.contains("FLOAT")
                    var oProjOffset = oProjTensor.offsetStart
                    var inAttnDim = hiddenDim

                    if isFP8, let gemvKernel = defaultLibrary.makeFunction(name: "fp8_gemv") {
                        let gemvPipeline = try device.makeComputePipelineState(function: gemvKernel)
                        let oScaleTensor = summary.tensors.first(where: { t in
                            t.layerIndex == UInt32(l) && (t.name.contains("o_proj") || t.name.contains("out_proj")) && (t.name.contains("scale") || t.name.contains("scales"))
                        })
                        let oScaleRaw = (oScaleTensor != nil) ? shardBuffers[oScaleTensor!.shardIndex] : oProjRaw
                        var oScaleOffset = oScaleTensor?.offsetStart ?? 0

                        computeEncoder.setComputePipelineState(gemvPipeline)
                        computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(oScaleRaw, offset: 0, index: 3)
                        computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&oScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                        computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), gemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if let bf16GemvKernel = defaultLibrary.makeFunction(name: "bf16_gemv") {
                        let bf16GemvPipeline = try device.makeComputePipelineState(function: bf16GemvKernel)
                        computeEncoder.setComputePipelineState(bf16GemvPipeline)
                        computeEncoder.setBuffer(oProjRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(xNorm1Buffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 2)
                        computeEncoder.setBytes(&oProjOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        computeEncoder.setBytes(&inAttnDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 5)
                        computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), bf16GemvPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                // ---------------------------------------------------------
                // Step 3: Residual Connection 1 (hMid = hCurr + attnOut)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(addPipeline)
                computeEncoder.setBuffer(hCurrBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(attnOutBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                // ---------------------------------------------------------
                // Step 4: Post-Attention RMSNorm (hMid -> xNorm2)
                // ---------------------------------------------------------
                if let norm2Tensor = findTensorInLayer("post_attention_layernorm"),
                   let norm2Raw = shardBuffers[norm2Tensor.shardIndex] {
                    var gammaOffset = norm2Tensor.offsetStart
                    computeEncoder.setComputePipelineState(rmsPipeline)
                    computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(norm2Raw, offset: 0, index: 1)
                    computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                    computeEncoder.setBytes(&gammaOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)
                    computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    computeEncoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                }

                // ---------------------------------------------------------
                // Step 5: MoE Router & SwiGLU Feed-Forward Compute (xNorm2 -> hMlp)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(clearPipeline)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 0)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                var intermediateDim: UInt32 = 512
                if let sampleGate = summary.tensors.first(where: {
                    $0.layerIndex == UInt32(l) && $0.name.contains("gate_proj") && !$0.name.contains("scale")
                }) {
                    let dims = sampleGate.shapeDisplay
                        .trimmingCharacters(in: CharacterSet(charactersIn: "[]() "))
                        .components(separatedBy: ",")
                        .compactMap { UInt32($0.trimmingCharacters(in: .whitespaces)) }
                    if dims.count >= 2 {
                        intermediateDim = dims[0]
                    }
                }

                guard let interBuffer = device.makeBuffer(length: Int(intermediateDim) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                    continue
                }

                let findTensor = { (sub1: String, sub2: String, isScale: Bool) -> TensorMetadata? in
                    return summary.tensors.first(where: { t in
                        t.layerIndex == UInt32(l) &&
                        t.name.contains(sub1) &&
                        t.name.contains(sub2) &&
                        (isScale ? (t.name.contains("scale") || t.name.contains("scales")) : (!t.name.contains("scale") && !t.name.contains("scales")))
                    })
                }

                let rawScores = routerScoresBuffer.contents().bindMemory(to: Float.self, capacity: 256)
                var scorePairs: [(id: Int, score: Float)] = []
                for idx in 0..<256 {
                    scorePairs.append((id: idx, score: rawScores[idx]))
                }
                scorePairs.sort(by: { $0.score > $1.score })
                let top8 = Array(scorePairs.prefix(8))
                var expSum: Float = 0.0
                for item in top8 {
                    expSum += Darwin.exp(item.score)
                }
                if expSum > 0.0 {
                    activeExperts = top8.map { (id: $0.id, weight: Darwin.exp($0.score) / expSum) }
                } else {
                    activeExperts = (0..<8).map { (id: $0, weight: 1.0 / 8.0) }
                }

                // Demand Paging & Working Set LRU Eviction for Active Experts
                let activeIds = activeExperts.map { $0.id }
                WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: memoryBudgetMode, shardBuffers: shardBuffers)

                // Async Lookahead Speculative Prefetching for layer l + 1
                if l + 1 < actualLayers {
                    let predicted = WorkingSetManager.shared.predictNextLayerExperts(currentLayer: l, currentActiveExperts: activeIds, topN: 8)
                    let prefetchIds = predicted.isEmpty ? [0, 1, 2, 3, 4, 5, 6, 7] : predicted
                    WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: prefetchIds, shardBuffers: shardBuffers)
                }

                for expert in activeExperts {
                    let expId = expert.id
                    var p_k = expert.weight
                    if p_k <= 0.00001 { continue }

                    let expTag = "experts.\(expId)"
                    guard let gateWeight = findTensor(expTag, "gate_proj", false),
                          let upWeight   = findTensor(expTag, "up_proj", false),
                          let downWeight = findTensor(expTag, "down_proj", false),
                          let gateRaw    = shardBuffers[gateWeight.shardIndex],
                          let upRaw      = shardBuffers[upWeight.shardIndex],
                          let downRaw    = shardBuffers[downWeight.shardIndex] else {
                        continue
                    }

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")

                    if isFP8 {
                        let gateScale = findTensor(expTag, "gate_proj", true)
                        let upScale   = findTensor(expTag, "up_proj", true)
                        let downScale = findTensor(expTag, "down_proj", true)

                        guard let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                              let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                              let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil else {
                            continue
                        }

                        var gateWeightOffset = gateWeight.offsetStart
                        var gateScaleOffset  = gateScale!.offsetStart
                        var upWeightOffset   = upWeight.offsetStart
                        var upScaleOffset    = upScale!.offsetStart
                        var downWeightOffset = downWeight.offsetStart
                        var downScaleOffset  = downScale!.offsetStart

                        computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                        computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                        computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                        computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                        computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                        computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                        computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                        computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                        computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                        let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                        let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                        computeEncoder.setComputePipelineState(fp8DownPipeline)
                        computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                        computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                        computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                        computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                        computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                        computeEncoder.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)

                        let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                        let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                        computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                    }
                }

                // Shared expert
                let sharedTag = "shared_expert"
                if let gateWeight = findTensor(sharedTag, "gate_proj", false),
                   let upWeight   = findTensor(sharedTag, "up_proj", false),
                   let downWeight = findTensor(sharedTag, "down_proj", false),
                   let gateRaw    = shardBuffers[gateWeight.shardIndex],
                   let upRaw      = shardBuffers[upWeight.shardIndex],
                   let downRaw    = shardBuffers[downWeight.shardIndex] {

                    let isFP8 = !gateWeight.dtype.contains("BF16") && !gateWeight.dtype.contains("FLOAT")
                    if isFP8 {
                        let gateScale = findTensor(sharedTag, "gate_proj", true)
                        let upScale   = findTensor(sharedTag, "up_proj", true)
                        let downScale = findTensor(sharedTag, "down_proj", true)

                        if let gateSRaw = (gateScale != nil) ? shardBuffers[gateScale!.shardIndex] : nil,
                           let upSRaw = (upScale != nil) ? shardBuffers[upScale!.shardIndex] : nil,
                           let downSRaw = (downScale != nil) ? shardBuffers[downScale!.shardIndex] : nil {

                            var gateWeightOffset = gateWeight.offsetStart
                            var gateScaleOffset  = gateScale!.offsetStart
                            var upWeightOffset   = upWeight.offsetStart
                            var upScaleOffset    = upScale!.offsetStart
                            var downWeightOffset = downWeight.offsetStart
                            var downScaleOffset  = downScale!.offsetStart
                            var sharedW: Float = 0.5

                            computeEncoder.setComputePipelineState(fp8GateUpPipeline)
                            computeEncoder.setBuffer(gateRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(upRaw, offset: 0, index: 1)
                            computeEncoder.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 3)
                            computeEncoder.setBuffer(gateSRaw, offset: 0, index: 4)
                            computeEncoder.setBuffer(upSRaw, offset: 0, index: 5)
                            computeEncoder.setBytes(&gateWeightOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            computeEncoder.setBytes(&gateScaleOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            computeEncoder.setBytes(&upWeightOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                            computeEncoder.setBytes(&upScaleOffset, length: MemoryLayout<UInt64>.stride, index: 9)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 10)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 11)

                            let interGrid = MTLSize(width: Int(intermediateDim), height: 1, depth: 1)
                            let interTg = MTLSize(width: min(Int(intermediateDim), fp8GateUpPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(interGrid, threadsPerThreadgroup: interTg)

                            computeEncoder.setComputePipelineState(fp8DownPipeline)
                            computeEncoder.setBuffer(downRaw, offset: 0, index: 0)
                            computeEncoder.setBuffer(interBuffer, offset: 0, index: 1)
                            computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 2)
                            computeEncoder.setBuffer(downSRaw, offset: 0, index: 3)
                            computeEncoder.setBytes(&downWeightOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                            computeEncoder.setBytes(&downScaleOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            computeEncoder.setBytes(&intermediateDim, length: MemoryLayout<UInt32>.stride, index: 6)
                            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 7)
                            computeEncoder.setBytes(&sharedW, length: MemoryLayout<Float>.stride, index: 8)

                            let hiddenGrid = MTLSize(width: Int(hiddenDim), height: 1, depth: 1)
                            let hiddenTg = MTLSize(width: min(Int(hiddenDim), fp8DownPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
                            computeEncoder.dispatchThreads(hiddenGrid, threadsPerThreadgroup: hiddenTg)
                        }
                    }
                }

                // ---------------------------------------------------------
                // Step 6: Residual Connection 2 (hNext = hMid + hMlp)
                // ---------------------------------------------------------
                computeEncoder.setComputePipelineState(addPipeline)
                computeEncoder.setBuffer(hMidBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(hMlpBuffer, offset: 0, index: 1)
                computeEncoder.setBuffer(hNextBuffer, offset: 0, index: 2)
                computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 3)
                computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                // End encoding and commit layer
                computeEncoder.endEncoding()
                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()

                let layerElapsedMs = (CFAbsoluteTimeGetCurrent() - layerStartTime) * 1000.0

                // Read L2 norm
                let outPtr = hNextBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
                var sumSquares: Float = 0.0
                for i in 0..<Int(hiddenDim) {
                    let v = outPtr[i]
                    sumSquares += v * v
                }
                let l2Norm = sqrt(sumSquares)

                telemetryList.append(LayerTelemetry(
                    layerIndex: l,
                    durationMs: layerElapsedMs,
                    topExperts: activeExperts.map { $0.id },
                    l2Norm: l2Norm
                ))

                // Ping-pong copy for next layer
                memcpy(hCurrBuffer.contents(), hNextBuffer.contents(), byteLength)
            }

            let totalElapsedMs = (CFAbsoluteTimeGetCurrent() - totalStartTime) * 1000.0
            let avgMsPerLayer = totalElapsedMs / Double(max(1, actualLayers))
            let layersPerSec = 1000.0 / max(0.001, avgMsPerLayer)

            // Final L2 norm
            let finalPtr = hCurrBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
            var finalSumSq: Float = 0.0
            var finalSamples: [String] = []
            for i in 0..<Int(hiddenDim) {
                let v = finalPtr[i]
                finalSumSq += v * v
                if i < 8 {
                    finalSamples.append(String(format: "%.5f", v))
                }
            }
            let finalL2Norm = sqrt(finalSumSq)

            self.activeHFinalBuffer = hCurrBuffer
            self.multiLayerTelemetry = telemetryList

            let status = "🚀 Full \(actualLayers)-Layer Backbone Executed in \(String(format: "%.2f", totalElapsedMs)) ms! (Avg \(String(format: "%.3f", avgMsPerLayer)) ms/layer | \(String(format: "%.0f", layersPerSec)) layers/sec) | Final h_\(actualLayers) L2 Norm: \(String(format: "%.4f", finalL2Norm))"
            self.multiLayerStatusText = status
            self.gpuComputeOutput = "\(status)\nFirst 8 dims of h_\(actualLayers): [\(finalSamples.joined(separator: ", "))]"
            updatePagingStats()

        } catch {
            gpuComputeOutput = "❌ Multi-Layer Forward Error: \(error.localizedDescription)"
        }
    }

    private func executeLMHeadProjection() {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvFunction = defaultLibrary.makeFunction(name: "bf16_gemv") else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline or missing shaders."
            return
        }

        let inputBuffer: MTLBuffer
        let sourceDescription: String
        if let hFinal = activeHFinalBuffer {
            inputBuffer = hFinal
            sourceDescription = "h_40 (Backbone Output)"
        } else if let h1 = activeH1Buffer {
            inputBuffer = h1
            sourceDescription = "h_1 (Single Block Output)"
        } else if let h0 = activeH0Buffer {
            inputBuffer = h0
            sourceDescription = "h_0 (Initial Embedding)"
        } else {
            gpuComputeOutput = "⚠️ Please run 'Embed & Route' or 'Execute All 40 Layers' first to generate hidden activations."
            return
        }

        // Find Final RMSNorm Tensor
        guard let normTensor = summary.tensors.first(where: {
            $0.name == "model.language_model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers."))
        }), let normShardBuffer = shardBuffers[normTensor.shardIndex] else {
            gpuComputeOutput = "❌ Final RMSNorm weight (model.language_model.norm.weight) not found."
            return
        }

        // Find LM Head Weight Tensor
        guard let lmHeadTensor = summary.tensors.first(where: {
            $0.name == "lm_head.weight" ||
            $0.name == "language_model.lm_head.weight" ||
            $0.name == "model.lm_head.weight" ||
            $0.name == "lm_head"
        }) ?? summary.tensors.first(where: {
            $0.name.hasSuffix("embed_tokens.weight") || $0.name == "embed_tokens" || $0.name.contains("word_embeddings") || $0.category == "Embedding"
        }), let lmHeadShardBuffer = shardBuffers[lmHeadTensor.shardIndex] else {
            gpuComputeOutput = "❌ LM Head weight not found."
            return
        }

        isExecutingLMHead = true
        topTokenPredictions = []
        lmHeadStatusText = "Projecting 248,320 vocabulary logits..."

        do {
            let rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)
            let gemvPipeline = try device.makeComputePipelineState(function: gemvFunction)

            var hiddenDim = UInt32(activeHiddenDim)
            var eps: Float = 1e-6
            var normOffset = normTensor.offsetStart

            // Parse or set vocab size
            var vocabSize: UInt32 = 248320
            let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
            let shapeParts = cleanShape.split(separator: ",")
            if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
                vocabSize = parsed
            }

            var lmHeadOffset = lmHeadTensor.offsetStart

            // Allocate buffers
            guard let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
                  let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
                gpuComputeOutput = "❌ Failed to allocate xFinal or Logits buffers."
                isExecutingLMHead = false
                return
            }

            let startTime = CFAbsoluteTimeGetCurrent()

            guard let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
                isExecutingLMHead = false
                return
            }

            // 1. Dispatch Final RMSNorm (input -> xFinal)
            computeEncoder.setComputePipelineState(rmsnormPipeline)
            computeEncoder.setBuffer(inputBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(normShardBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(xFinalBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&normOffset, length: MemoryLayout<UInt64>.stride, index: 3)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            computeEncoder.setBytes(&eps, length: MemoryLayout<Float>.stride, index: 5)

            let normTgSize = min(1024, rmsnormPipeline.maxTotalThreadsPerThreadgroup)
            computeEncoder.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
            computeEncoder.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: normTgSize, height: 1, depth: 1))

            // 2. Dispatch LM Head GEMV (xFinal -> logits)
            computeEncoder.setComputePipelineState(gemvPipeline)
            computeEncoder.setBuffer(lmHeadShardBuffer, offset: 0, index: 0)
            computeEncoder.setBuffer(xFinalBuffer, offset: 0, index: 1)
            computeEncoder.setBuffer(logitsBuffer, offset: 0, index: 2)
            computeEncoder.setBytes(&lmHeadOffset, length: MemoryLayout<UInt64>.stride, index: 3)
            computeEncoder.setBytes(&hiddenDim, length: MemoryLayout<UInt32>.stride, index: 4)
            computeEncoder.setBytes(&vocabSize, length: MemoryLayout<UInt32>.stride, index: 5)

            let gemvTgSize = min(256, gemvPipeline.maxTotalThreadsPerThreadgroup)
            computeEncoder.dispatchThreads(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: gemvTgSize, height: 1, depth: 1))

            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let gpuElapsedMs = (CFAbsoluteTimeGetCurrent() - startTime) * 1000.0

            // 3. Extract Top-10 Vocabulary Candidates
            let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
            var topCandidates: [(id: Int, logit: Float)] = []
            
            for v in 0..<Int(vocabSize) {
                let val = logitsPtr[v]
                if topCandidates.count < 10 {
                    topCandidates.append((id: v, logit: val))
                    if topCandidates.count == 10 {
                        topCandidates.sort(by: { $0.logit > $1.logit })
                    }
                } else if val > topCandidates.last!.logit {
                    topCandidates[9] = (id: v, logit: val)
                    topCandidates.sort(by: { $0.logit > $1.logit })
                }
            }

            // Softmax over top candidates
            let maxLogit = topCandidates.first?.logit ?? 0.0
            var expSum: Float = 0.0
            for c in topCandidates {
                expSum += Darwin.exp(c.logit - maxLogit)
            }

            var predictions: [TokenPrediction] = []
            for (idx, item) in topCandidates.enumerated() {
                let prob = expSum > 0.0 ? (Darwin.exp(item.logit - maxLogit) / expSum) : (1.0 / Float(topCandidates.count))
                var tokenStr = "<Token \(item.id)>"
                if let tok = self.tokenizer {
                    do {
                        let decoded = try tok.decode(ids: [UInt32(item.id)])
                        if !decoded.isEmpty {
                            tokenStr = decoded
                        }
                    } catch {}
                }
                predictions.append(TokenPrediction(
                    rank: idx + 1,
                    tokenId: UInt32(item.id),
                    tokenString: tokenStr,
                    logit: item.logit,
                    probability: prob
                ))
            }

            self.topTokenPredictions = predictions
            self.activeLogitsBuffer = logitsBuffer
            self.isExecutingLMHead = false

            let top1 = predictions.first
            let top1Str = top1?.tokenString.replacingOccurrences(of: "\n", with: "\\n") ?? "N/A"
            let top1Pct = String(format: "%.1f%%", (top1?.probability ?? 0) * 100.0)

            let status = "✨ LM Head Projected \(vocabSize) Vocab Logits in \(String(format: "%.2f", gpuElapsedMs)) ms from \(sourceDescription)! Top Candidate: \"\(top1Str)\" (\(top1Pct) prob)"
            self.lmHeadStatusText = status
            self.gpuComputeOutput = status

        } catch {
            isExecutingLMHead = false
            gpuComputeOutput = "❌ LM Head Pipeline Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Autoregressive Generation & Sampling Engine

    private func buildCachedLayers(summary: ModelSummary) -> [CachedLayer] {
        return InferenceEngine.shared.buildCachedLayers(summary: summary, config: modelConfig)
    }

    private func sampleNextToken(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        contextTokens: [UInt32],
        temperature: Float,
        topP: Float,
        minP: Float,
        topK: Int,
        repetitionPenalty: Float,
        presencePenalty: Float = 0.0,
        grammarMask: ((UnsafeMutablePointer<Float>, Int) -> Void)? = nil
    ) -> UInt32 {
        return InferenceEngine.sampleNextToken(
            logits: logits,
            vocabSize: vocabSize,
            contextTokens: contextTokens,
            temperature: temperature,
            topP: topP,
            minP: minP,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            presencePenalty: presencePenalty,
            grammarMask: grammarMask
        )
    }

    private func stopAutoregressiveGeneration() {
        isGeneratingText = false
        generationTask?.cancel()
        generationTask = nil
        ActionApprovalManager.shared.cancelAll()
        generationStatusText = "⏹ Generation stopped by user."
    }

    private func startAutoregressiveGeneration(customPrompt: String? = nil, sessionId: UUID? = nil, messageId: UUID? = nil, agentStep: Int = 0) {
        guard let summary = summary,
              let tokenizer = tokenizer,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary() else {
            let err = "❌ Metal or Tokenizer not ready for text generation. Please load model and tokenizer in Settings."
            isGeneratingText = false
            generationTask = nil
            if let sId = sessionId, let mId = messageId {
                if let sIdx = sessions.firstIndex(where: { $0.id == sId }),
                   let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                    sessions[sIdx].messages[mIdx].content = err
                    sessions[sIdx].messages[mIdx].isThinking = false
                }
            }
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let prompt: String
        if let custom = customPrompt {
            prompt = custom
        } else {
            prompt = promptInput.trimmingCharacters(in: CharacterSet.whitespacesAndNewlines)
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            let err = "⚠️ Please enter a prompt to generate text."
            isGeneratingText = false
            generationTask = nil
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let isNanbeige = (modelConfig?.modelType?.lowercased().contains("nanbeige") == true) ||
                         (modelConfig?.architectures?.contains(where: { $0.lowercased().contains("nanbeige") }) == true) ||
                         (summary.layerCount == 22 && summary.maxExpertId == 0) ||
                         (summary.tensors.contains(where: { $0.name.contains("dense_gate_up_proj") })) ||
                         (summary.tensors.contains(where: { $0.name.hasPrefix("model.layers.0.mlp.gate_proj") }) && summary.layerCount == 22)

        let generationDate: Date
        if let sId = sessionId, let sess = sessions.first(where: { $0.id == sId }) {
            generationDate = sess.createdAt
        } else {
            generationDate = Date()
        }

        let cleanSystem = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: systemPrompt,
            config: modelConfig,
            summary: summary,
            modelName: activeModelDisplayName,
            modelPath: activeLoadedModelPath,
            currentDate: generationDate
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        let modelSupportsThinking = activeModelSupportsThinking
        let thinkingEnabled = isThinkingEnabledForActiveSession && modelSupportsThinking

        let thinkSuffix: String
        if thinkingEnabled {
            thinkSuffix = "<think>\n"
        } else if modelSupportsThinking {
            thinkSuffix = "<think>\n\n</think>\n\n"
        } else {
            thinkSuffix = ""
        }

        let isLing = (modelConfig?.isLingModel == true)
        let formattedPrompt: String
        if isLing {
            if prompt.contains("<role>") {
                formattedPrompt = prompt
            } else {
                let lingThinking = thinkingEnabled ? "detailed thinking on" : "detailed thinking off"
                let sysPart = cleanSystem.isEmpty ? lingThinking : "\(cleanSystem)\n\(lingThinking)"
                let thinkTag = thinkingEnabled ? "\n<think>" : "\n<think></think>"
                formattedPrompt = "<role>SYSTEM</role>\(sysPart)<|role_end|><role>HUMAN</role>\(prompt)<|role_end|><role>ASSISTANT</role>\(thinkTag)"
            }
        } else if prompt.contains("<|im_start|>") {
            var p = prompt
            if !cleanSystem.isEmpty && !prompt.contains("<|im_start|>system") {
                p = "<|im_start|>system\n\(cleanSystem)<|im_end|>\n" + p
            }
            let trimmed = p.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.hasSuffix("<|im_start|>assistant") {
                if thinkingEnabled {
                    p = trimmed + "\n<think>\n"
                } else if modelSupportsThinking {
                    p = trimmed + "\n<think>\n\n</think>\n\n"
                }
            }
            formattedPrompt = p
        } else if !cleanSystem.isEmpty {
            formattedPrompt = "<|im_start|>system\n\(cleanSystem)<|im_end|>\n<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n\(thinkSuffix)"
        } else {
            formattedPrompt = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n\(thinkSuffix)"
        }

        let promptTokenIds: [UInt32]
        do {
            promptTokenIds = try tokenizer.encode(text: formattedPrompt)
        } catch {
            let err = "❌ Tokenizer failed to encode prompt: \(error.localizedDescription)"
            isGeneratingText = false
            generationTask = nil
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        guard !promptTokenIds.isEmpty else {
            let err = "⚠️ Prompt tokenization produced 0 tokens."
            isGeneratingText = false
            generationTask = nil
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Find Embedding Weight & Affine Scales/Biases
        guard let embedWeight = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }), let embedShardBuffer = shardBuffers[embedWeight.shardIndex] else {
            let err = "❌ Embedding weight tensor not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let embedScale = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            ($0.name.contains("scale") || $0.name.contains("scales"))
        })
        let embedBias = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte") || $0.name.contains("word_embeddings") || $0.category == "Embedding") &&
            ($0.name.contains("bias") || $0.name.contains("biases"))
        })

        // Find Final RMSNorm or Hyper-Connection Mixer
        let finalHcNormWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.hc_norm.weight" ||
            $0.name == "language_model.hyper_connection_mixer.hc_norm.weight"
        })
        let finalHcDownWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight" ||
            $0.name == "language_model.hyper_connection_mixer.input_mix_weight_down.weight"
        })
        let finalHcUpWeight = summary.tensors.first(where: {
            $0.name == "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight" ||
            $0.name == "language_model.hyper_connection_mixer.input_mix_weight_up.weight"
        })

        let normTensor = summary.tensors.first(where: {
            $0.name == "model.language_model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers.") && !$0.name.contains("mixer"))
        })

        let hasFinalHc = (finalHcNormWeight != nil && finalHcDownWeight != nil && finalHcUpWeight != nil)
        guard hasFinalHc || (normTensor != nil && shardBuffers[normTensor!.shardIndex] != nil) else {
            let err = "❌ Final RMSNorm weight / Hyper-Connection mixer not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }
        let normShardBuffer = normTensor != nil ? shardBuffers[normTensor!.shardIndex] : nil

        // Find LM Head Weight & Affine Scales/Biases
        let lmHeadTensorCandidate = summary.tensors.first(where: {
            ($0.name == "lm_head.weight" ||
             $0.name == "language_model.lm_head.weight" ||
             $0.name == "model.lm_head.weight" ||
             $0.name == "lm_head") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }) ?? embedWeight

        guard let lmHeadShardBuffer = shardBuffers[lmHeadTensorCandidate.shardIndex] else {
            let err = "❌ LM Head weight not found."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }
        let lmHeadTensor = lmHeadTensorCandidate

        let lmHeadScale = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("scale") || $0.name.contains("scales"))
        }) ?? (lmHeadTensor == embedWeight ? embedScale : nil)
        let lmHeadBias = summary.tensors.first(where: {
            $0.name.contains("lm_head") && ($0.name.contains("bias") || $0.name.contains("biases"))
        }) ?? (lmHeadTensor == embedWeight ? embedBias : nil)

        // Compile Pipelines with correct kernel names
        guard let embedBF16Function = defaultLibrary.makeFunction(name: "lookup_embeddings_bf16"),
              let routerFunction = defaultLibrary.makeFunction(name: "moe_router_topk_bf16"),
              let rmsnormFunction = defaultLibrary.makeFunction(name: "rmsnorm_bf16"),
              let gemvBF16Function = defaultLibrary.makeFunction(name: "bf16_gemv"),
              let addFunction = defaultLibrary.makeFunction(name: "vector_add_f32"),
              let clearFunction = defaultLibrary.makeFunction(name: "clear_vector_f32") else {
            let err = "❌ Failed to load core Metal compute shaders."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let embedPipeline: MTLComputePipelineState
        let embedQ4Pipeline: MTLComputePipelineState?
        let embedQ8Pipeline: MTLComputePipelineState?
        let embedMXFP8Pipeline: MTLComputePipelineState?
        let routerPipeline: MTLComputePipelineState
        let routerQ4Pipeline: MTLComputePipelineState?
        let routerQ8Pipeline: MTLComputePipelineState?
        let rmsnormPipeline: MTLComputePipelineState
        let rmsnormF16Pipeline: MTLComputePipelineState?
        let rmsnormOffsetPipeline: MTLComputePipelineState?
        let rmsnormOffsetF16Pipeline: MTLComputePipelineState?
        let gemvBF16Pipeline: MTLComputePipelineState
        let bf16GemvSimdPipeline: MTLComputePipelineState?
        let fp8GemvPipeline: MTLComputePipelineState?
        let mxfp8GemvPipeline: MTLComputePipelineState?
        let mxfp8GemvSimdPipeline: MTLComputePipelineState?
        let q4GemvPipeline: MTLComputePipelineState?
        let q8GemvPipeline: MTLComputePipelineState?
        let addPipeline: MTLComputePipelineState
        let clearPipeline: MTLComputePipelineState
        let fp8GateUpPipeline: MTLComputePipelineState?
        let fp8DownPipeline: MTLComputePipelineState?
        let fp8GateUpSimdPipeline: MTLComputePipelineState?
        let fp8DownSimdPipeline: MTLComputePipelineState?
        let mxfp8GateUpPipeline: MTLComputePipelineState?
        let mxfp8DownPipeline: MTLComputePipelineState?
        let mxfp8GateUpSimdPipeline: MTLComputePipelineState?
        let mxfp8DownSimdPipeline: MTLComputePipelineState?
        let bf16GateUpPipeline: MTLComputePipelineState?
        let bf16DownPipeline: MTLComputePipelineState?
        let bf16GateUpSimdPipeline: MTLComputePipelineState?
        let bf16DownSimdPipeline: MTLComputePipelineState?
        let q4GateUpPipeline: MTLComputePipelineState?
        let q4DownPipeline: MTLComputePipelineState?
        let q8GateUpPipeline: MTLComputePipelineState?
        let q8DownPipeline: MTLComputePipelineState?
        let q4GateQ8UpPipeline: MTLComputePipelineState?
        let q8GateQ4UpPipeline: MTLComputePipelineState?
        let headRmsnormPipeline: MTLComputePipelineState?
        let headRmsnormF16Pipeline: MTLComputePipelineState?
        let headRmsnormOffsetPipeline: MTLComputePipelineState?
        let headRmsnormOffsetF16Pipeline: MTLComputePipelineState?
        let ropePipeline: MTLComputePipelineState?
        let storeKvCachePipeline: MTLComputePipelineState?
        let storeKvCacheF16Pipeline: MTLComputePipelineState?
        let storeKvCacheFP8Pipeline: MTLComputePipelineState?
        let gqaDecodePipeline: MTLComputePipelineState?
        let gqaDecodeF16Pipeline: MTLComputePipelineState?
        let gqaDecodeFP8Pipeline: MTLComputePipelineState?
        let gqaStandardPipeline: MTLComputePipelineState?
        let gqaStandardF16Pipeline: MTLComputePipelineState?
        let gqaStandardFP8Pipeline: MTLComputePipelineState?
        let causalConv1dPipeline: MTLComputePipelineState?
        let l2NormQkPipeline: MTLComputePipelineState?
        let linearAttnStepPipeline: MTLComputePipelineState?
        let linearAttnStepSigmoidPipeline: MTLComputePipelineState?
        let sharedGatePipeline: MTLComputePipelineState?
        let sharedGateQ4Pipeline: MTLComputePipelineState?
        let sharedGateQ8Pipeline: MTLComputePipelineState?
        let router512Pipeline: MTLComputePipelineState?
        let router512Q4Pipeline: MTLComputePipelineState?
        let gdnLinearAttnStepPipeline: MTLComputePipelineState?
        let gdnLinearAttnStepSigmoidPipeline: MTLComputePipelineState?
        let qsaMqaIndexerPipeline: MTLComputePipelineState?
        let gatedResidualBlendPipeline: MTLComputePipelineState?
        let fuseNgramPlePipeline: MTLComputePipelineState?
        let fusedInit4StreamsPipeline: MTLComputePipelineState?
        let extractStream0Pipeline: MTLComputePipelineState?
        let hcNormPipeline: MTLComputePipelineState?
        let hcDownProjPipeline: MTLComputePipelineState?
        let hcUpBlendPipeline: MTLComputePipelineState?
        let hcInjectScalePipeline: MTLComputePipelineState?
        let hcInjectPipeline: MTLComputePipelineState?
        let causalConv1dSeqPipeline: MTLComputePipelineState?
        let l2NormQkSeqPipeline: MTLComputePipelineState?
        let gdnLinearAttnSeqPipeline: MTLComputePipelineState?
        let gdnLinearAttnSeqSigmoidPipeline: MTLComputePipelineState?
        let fp8GemvSimdPipeline: MTLComputePipelineState?
        let fp8GateUpBatchedPipeline: MTLComputePipelineState?
        let fp8DownBatchedPipeline: MTLComputePipelineState?
        let fp8BlockGateUpPipeline: MTLComputePipelineState?
        let fp8BlockDownPipeline: MTLComputePipelineState?
        let fp8BlockGateUpSimdPipeline: MTLComputePipelineState?
        let fp8BlockDownSimdPipeline: MTLComputePipelineState?
        let fp8BlockGemvPipeline: MTLComputePipelineState?
        let fp8BlockGemvSimdPipeline: MTLComputePipelineState?
        let fp8BlockGateUpBatchedPipeline: MTLComputePipelineState?
        let fp8BlockDownBatchedPipeline: MTLComputePipelineState?
        let bf16GateUpBatchedPipeline: MTLComputePipelineState?
        let bf16DownBatchedPipeline: MTLComputePipelineState?
        let kdaLinearAttnStepPipeline: MTLComputePipelineState?
        let ropeInterleavedPipeline: MTLComputePipelineState?
        let routerBailingPipeline: MTLComputePipelineState?
        let storeMlaKvCachePipeline: MTLComputePipelineState?
        let storeMlaKvCacheF16Pipeline: MTLComputePipelineState?
        let mlaAttentionDecodePipeline: MTLComputePipelineState?
        let mlaAttentionDecodeF16Pipeline: MTLComputePipelineState?

        do {
            embedPipeline = try device.makeComputePipelineState(function: embedBF16Function)
            routerPipeline = try device.makeComputePipelineState(function: routerFunction)
            rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)

            if let rmsF16Func = defaultLibrary.makeFunction(name: "rmsnorm_f16") {
                rmsnormF16Pipeline = try device.makeComputePipelineState(function: rmsF16Func)
            } else { rmsnormF16Pipeline = nil }

            if let rmsOffsetFunc = defaultLibrary.makeFunction(name: "rmsnorm_offset_bf16") {
                rmsnormOffsetPipeline = try device.makeComputePipelineState(function: rmsOffsetFunc)
            } else { rmsnormOffsetPipeline = nil }

            if let rmsOffsetF16Func = defaultLibrary.makeFunction(name: "rmsnorm_offset_f16") {
                rmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: rmsOffsetF16Func)
            } else { rmsnormOffsetF16Pipeline = nil }
            gemvBF16Pipeline = try device.makeComputePipelineState(function: gemvBF16Function)
            addPipeline = try device.makeComputePipelineState(function: addFunction)
            clearPipeline = try device.makeComputePipelineState(function: clearFunction)

            if let rQ4Func = defaultLibrary.makeFunction(name: "moe_router_topk_q4") {
                routerQ4Pipeline = try device.makeComputePipelineState(function: rQ4Func)
            } else { routerQ4Pipeline = nil }

            if let rQ8Func = defaultLibrary.makeFunction(name: "moe_router_topk_q8") {
                routerQ8Pipeline = try device.makeComputePipelineState(function: rQ8Func)
            } else { routerQ8Pipeline = nil }

            if let embedQ4Func = defaultLibrary.makeFunction(name: "lookup_embeddings_q4") {
                embedQ4Pipeline = try device.makeComputePipelineState(function: embedQ4Func)
            } else { embedQ4Pipeline = nil }

            if let embedQ8Func = defaultLibrary.makeFunction(name: "lookup_embeddings_q8") {
                embedQ8Pipeline = try device.makeComputePipelineState(function: embedQ8Func)
            } else { embedQ8Pipeline = nil }

            if let embedMXFP8Func = defaultLibrary.makeFunction(name: "lookup_embeddings_mxfp8") {
                embedMXFP8Pipeline = try device.makeComputePipelineState(function: embedMXFP8Func)
            } else { embedMXFP8Pipeline = nil }

            if let bGemvSimd = defaultLibrary.makeFunction(name: "bf16_gemv_simd") {
                bf16GemvSimdPipeline = try device.makeComputePipelineState(function: bGemvSimd)
            } else { bf16GemvSimdPipeline = nil }

            if let fp8GemvFunc = defaultLibrary.makeFunction(name: "fp8_gemv") {
                fp8GemvPipeline = try device.makeComputePipelineState(function: fp8GemvFunc)
            } else { fp8GemvPipeline = nil }

            if let mxfp8GemvFunc = defaultLibrary.makeFunction(name: "mxfp8_gemv") {
                mxfp8GemvPipeline = try device.makeComputePipelineState(function: mxfp8GemvFunc)
            } else { mxfp8GemvPipeline = nil }

            if let mxfp8GemvSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_gemv_simd") {
                mxfp8GemvSimdPipeline = try device.makeComputePipelineState(function: mxfp8GemvSimdFunc)
            } else { mxfp8GemvSimdPipeline = nil }

            if let q4GemvFunc = defaultLibrary.makeFunction(name: "q4_gemv") {
                q4GemvPipeline = try device.makeComputePipelineState(function: q4GemvFunc)
            } else { q4GemvPipeline = nil }

            if let q8GemvFunc = defaultLibrary.makeFunction(name: "q8_gemv") {
                q8GemvPipeline = try device.makeComputePipelineState(function: q8GemvFunc)
            } else { q8GemvPipeline = nil }

            if let fp8GateUpFunc = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up") {
                fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpFunc)
            } else { fp8GateUpPipeline = nil }

            if let fp8DownFunc = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate") {
                fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownFunc)
            } else { fp8DownPipeline = nil }

            if let fp8GateUpSimd = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up_simd") {
                fp8GateUpSimdPipeline = try device.makeComputePipelineState(function: fp8GateUpSimd)
            } else { fp8GateUpSimdPipeline = nil }

            if let fp8DownSimd = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate_simd") {
                fp8DownSimdPipeline = try device.makeComputePipelineState(function: fp8DownSimd)
            } else { fp8DownSimdPipeline = nil }

            if let fp8BlockGateFunc = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up") {
                fp8BlockGateUpPipeline = try device.makeComputePipelineState(function: fp8BlockGateFunc)
            } else { fp8BlockGateUpPipeline = nil }

            if let fp8BlockDownFunc = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate") {
                fp8BlockDownPipeline = try device.makeComputePipelineState(function: fp8BlockDownFunc)
            } else { fp8BlockDownPipeline = nil }

            if let fp8BlockGateSimd = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up_simd") {
                fp8BlockGateUpSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGateSimd)
            } else { fp8BlockGateUpSimdPipeline = nil }

            if let fp8BlockDownSimd = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate_simd") {
                fp8BlockDownSimdPipeline = try device.makeComputePipelineState(function: fp8BlockDownSimd)
            } else { fp8BlockDownSimdPipeline = nil }

            if let fp8BlockGemvFunc = defaultLibrary.makeFunction(name: "fp8_block_gemv") {
                fp8BlockGemvPipeline = try device.makeComputePipelineState(function: fp8BlockGemvFunc)
            } else { fp8BlockGemvPipeline = nil }

            if let fp8BlockGemvSimd = defaultLibrary.makeFunction(name: "fp8_block_gemv_simd") {
                fp8BlockGemvSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGemvSimd)
            } else { fp8BlockGemvSimdPipeline = nil }

            if let mxfp8GateFunc = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up") {
                mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateFunc)
            } else { mxfp8GateUpPipeline = nil }

            if let mxfp8DownFunc = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate") {
                mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownFunc)
            } else { mxfp8DownPipeline = nil }

            if let mxfp8GateSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_swiglu_gate_up_simd") {
                mxfp8GateUpSimdPipeline = try device.makeComputePipelineState(function: mxfp8GateSimdFunc)
            } else { mxfp8GateUpSimdPipeline = nil }

            if let mxfp8DownSimdFunc = defaultLibrary.makeFunction(name: "mxfp8_down_proj_accumulate_simd") {
                mxfp8DownSimdPipeline = try device.makeComputePipelineState(function: mxfp8DownSimdFunc)
            } else { mxfp8DownSimdPipeline = nil }

            if let bGateUp = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up") {
                bf16GateUpPipeline = try device.makeComputePipelineState(function: bGateUp)
            } else { bf16GateUpPipeline = nil }

            if let bDown = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate") {
                bf16DownPipeline = try device.makeComputePipelineState(function: bDown)
            } else { bf16DownPipeline = nil }

            if let bGateUpSimd = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up_simd") {
                bf16GateUpSimdPipeline = try device.makeComputePipelineState(function: bGateUpSimd)
            } else { bf16GateUpSimdPipeline = nil }

            if let bDownSimd = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate_simd") {
                bf16DownSimdPipeline = try device.makeComputePipelineState(function: bDownSimd)
            } else { bf16DownSimdPipeline = nil }

            if let q4GateUpFunc = defaultLibrary.makeFunction(name: "q4_swiglu_gate_up") {
                q4GateUpPipeline = try device.makeComputePipelineState(function: q4GateUpFunc)
            } else { q4GateUpPipeline = nil }

            if let q4DownFunc = defaultLibrary.makeFunction(name: "q4_down_proj_accumulate") {
                q4DownPipeline = try device.makeComputePipelineState(function: q4DownFunc)
            } else { q4DownPipeline = nil }

            if let q8GateUpFunc = defaultLibrary.makeFunction(name: "q8_swiglu_gate_up") {
                q8GateUpPipeline = try device.makeComputePipelineState(function: q8GateUpFunc)
            } else { q8GateUpPipeline = nil }

            if let q8DownFunc = defaultLibrary.makeFunction(name: "q8_down_proj_accumulate") {
                q8DownPipeline = try device.makeComputePipelineState(function: q8DownFunc)
            } else { q8DownPipeline = nil }

            if let q4q8Func = defaultLibrary.makeFunction(name: "q4_gate_q8_up_swiglu") {
                q4GateQ8UpPipeline = try device.makeComputePipelineState(function: q4q8Func)
            } else { q4GateQ8UpPipeline = nil }

            if let q8q4Func = defaultLibrary.makeFunction(name: "q8_gate_q4_up_swiglu") {
                q8GateQ4UpPipeline = try device.makeComputePipelineState(function: q8q4Func)
            } else { q8GateQ4UpPipeline = nil }

            if let hNormFunc = defaultLibrary.makeFunction(name: "per_head_rmsnorm_bf16") {
                headRmsnormPipeline = try device.makeComputePipelineState(function: hNormFunc)
            } else { headRmsnormPipeline = nil }

            if let hNormF16Func = defaultLibrary.makeFunction(name: "per_head_rmsnorm_f16") {
                headRmsnormF16Pipeline = try device.makeComputePipelineState(function: hNormF16Func)
            } else { headRmsnormF16Pipeline = nil }

            if let hNormOffsetFunc = defaultLibrary.makeFunction(name: "per_head_rmsnorm_offset_bf16") {
                headRmsnormOffsetPipeline = try device.makeComputePipelineState(function: hNormOffsetFunc)
            } else { headRmsnormOffsetPipeline = nil }

            if let hNormOffsetF16Func = defaultLibrary.makeFunction(name: "per_head_rmsnorm_offset_f16") {
                headRmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: hNormOffsetF16Func)
            } else { headRmsnormOffsetF16Pipeline = nil }

            if let ropeFunc = defaultLibrary.makeFunction(name: "apply_rope_qwen") {
                ropePipeline = try device.makeComputePipelineState(function: ropeFunc)
            } else { ropePipeline = nil }

            if let storeKvFunc = defaultLibrary.makeFunction(name: "store_kv_cache") {
                storeKvCachePipeline = try device.makeComputePipelineState(function: storeKvFunc)
            } else { storeKvCachePipeline = nil }

            if let storeKvF16Func = defaultLibrary.makeFunction(name: "store_kv_cache_f16") {
                storeKvCacheF16Pipeline = try device.makeComputePipelineState(function: storeKvF16Func)
            } else { storeKvCacheF16Pipeline = nil }

            if let storeKvFP8Func = defaultLibrary.makeFunction(name: "store_kv_cache_fp8") {
                storeKvCacheFP8Pipeline = try device.makeComputePipelineState(function: storeKvFP8Func)
            } else { storeKvCacheFP8Pipeline = nil }

            if let gqaFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused") {
                gqaDecodePipeline = try device.makeComputePipelineState(function: gqaFunc)
            } else { gqaDecodePipeline = nil }

            if let gqaF16Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_f16") {
                gqaDecodeF16Pipeline = try device.makeComputePipelineState(function: gqaF16Func)
            } else { gqaDecodeF16Pipeline = nil }

            if let gqaFP8Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_fused_fp8") {
                gqaDecodeFP8Pipeline = try device.makeComputePipelineState(function: gqaFP8Func)
            } else { gqaDecodeFP8Pipeline = nil }

            if let gqaStdFunc = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard") {
                gqaStandardPipeline = try device.makeComputePipelineState(function: gqaStdFunc)
            } else { gqaStandardPipeline = nil }

            if let gqaStdF16Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard_f16") {
                gqaStandardF16Pipeline = try device.makeComputePipelineState(function: gqaStdF16Func)
            } else { gqaStandardF16Pipeline = nil }

            if let gqaStdFP8Func = defaultLibrary.makeFunction(name: "gqa_attention_decode_standard_fp8") {
                gqaStandardFP8Pipeline = try device.makeComputePipelineState(function: gqaStdFP8Func)
            } else { gqaStandardFP8Pipeline = nil }

            if let convFunc = defaultLibrary.makeFunction(name: "causal_conv1d_silu") {
                causalConv1dPipeline = try device.makeComputePipelineState(function: convFunc)
            } else { causalConv1dPipeline = nil }

            if let l2Func = defaultLibrary.makeFunction(name: "l2_norm_qk") {
                l2NormQkPipeline = try device.makeComputePipelineState(function: l2Func)
            } else { l2NormQkPipeline = nil }

            if let linStepFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_step") {
                linearAttnStepPipeline = try device.makeComputePipelineState(function: linStepFunc)
            } else { linearAttnStepPipeline = nil }

            if let linStepSigFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_step_sigmoid") {
                linearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: linStepSigFunc)
            } else { linearAttnStepSigmoidPipeline = nil }

            if let sgFunc = defaultLibrary.makeFunction(name: "moe_shared_gate_bf16") {
                sharedGatePipeline = try device.makeComputePipelineState(function: sgFunc)
            } else { sharedGatePipeline = nil }

            if let sgQ4Func = defaultLibrary.makeFunction(name: "moe_shared_gate_q4") {
                sharedGateQ4Pipeline = try device.makeComputePipelineState(function: sgQ4Func)
            } else { sharedGateQ4Pipeline = nil }

            if let sgQ8Func = defaultLibrary.makeFunction(name: "moe_shared_gate_q8") {
                sharedGateQ8Pipeline = try device.makeComputePipelineState(function: sgQ8Func)
            } else { sharedGateQ8Pipeline = nil }

            if let r512Func = defaultLibrary.makeFunction(name: "moe_router_topk_512_bf16") {
                router512Pipeline = try device.makeComputePipelineState(function: r512Func)
            } else { router512Pipeline = nil }

            if let r512Q4Func = defaultLibrary.makeFunction(name: "moe_router_topk_512_q4") {
                router512Q4Pipeline = try device.makeComputePipelineState(function: r512Q4Func)
            } else { router512Q4Pipeline = nil }

            if let gdnFunc = defaultLibrary.makeFunction(name: "gdn_linear_attention_recurrent_step") {
                gdnLinearAttnStepPipeline = try device.makeComputePipelineState(function: gdnFunc)
            } else { gdnLinearAttnStepPipeline = nil }

            if let gdnSigFunc = defaultLibrary.makeFunction(name: "gdn_linear_attention_recurrent_step_sigmoid") {
                gdnLinearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: gdnSigFunc)
            } else { gdnLinearAttnStepSigmoidPipeline = nil }

            if let qsaIdxFunc = defaultLibrary.makeFunction(name: "qsa_mqa_indexer_score_blocks") {
                qsaMqaIndexerPipeline = try device.makeComputePipelineState(function: qsaIdxFunc)
            } else { qsaMqaIndexerPipeline = nil }

            if let grBlendFunc = defaultLibrary.makeFunction(name: "gated_residual_blend_4stream") {
                gatedResidualBlendPipeline = try device.makeComputePipelineState(function: grBlendFunc)
            } else { gatedResidualBlendPipeline = nil }

            if let pleFunc = defaultLibrary.makeFunction(name: "fuse_ngram_ple_embedding") {
                fuseNgramPlePipeline = try device.makeComputePipelineState(function: pleFunc)
            } else { fuseNgramPlePipeline = nil }

            if let init4StreamsFunc = defaultLibrary.makeFunction(name: "fused_init_4streams") {
                fusedInit4StreamsPipeline = try device.makeComputePipelineState(function: init4StreamsFunc)
            } else { fusedInit4StreamsPipeline = nil }

            if let extract0Func = defaultLibrary.makeFunction(name: "extract_stream0") {
                extractStream0Pipeline = try device.makeComputePipelineState(function: extract0Func)
            } else { extractStream0Pipeline = nil }

            if let hcNormFunc = defaultLibrary.makeFunction(name: "hyper_connection_norm_bf16") {
                hcNormPipeline = try device.makeComputePipelineState(function: hcNormFunc)
            } else { hcNormPipeline = nil }

            if let hcDownFunc = defaultLibrary.makeFunction(name: "hyper_connection_down_proj_bf16") {
                hcDownProjPipeline = try device.makeComputePipelineState(function: hcDownFunc)
            } else { hcDownProjPipeline = nil }

            if let hcUpFunc = defaultLibrary.makeFunction(name: "hyper_connection_up_proj_blend_bf16") {
                hcUpBlendPipeline = try device.makeComputePipelineState(function: hcUpFunc)
            } else { hcUpBlendPipeline = nil }

            if let hcInjScaleFunc = defaultLibrary.makeFunction(name: "hyper_connection_inject_scale_bf16") {
                hcInjectScalePipeline = try device.makeComputePipelineState(function: hcInjScaleFunc)
            } else { hcInjectScalePipeline = nil }

            if let hcInjFunc = defaultLibrary.makeFunction(name: "hyper_connection_inject_bf16") {
                hcInjectPipeline = try device.makeComputePipelineState(function: hcInjFunc)
            } else { hcInjectPipeline = nil }

            if let fp8SimdFunc = defaultLibrary.makeFunction(name: "fp8_gemv_simd") {
                fp8GemvSimdPipeline = try device.makeComputePipelineState(function: fp8SimdFunc)
            } else { fp8GemvSimdPipeline = nil }

            if let convSeqFunc = defaultLibrary.makeFunction(name: "causal_conv1d_sequence_silu") {
                causalConv1dSeqPipeline = try device.makeComputePipelineState(function: convSeqFunc)
            } else { causalConv1dSeqPipeline = nil }

            if let l2SeqFunc = defaultLibrary.makeFunction(name: "l2_norm_qk_sequence") {
                l2NormQkSeqPipeline = try device.makeComputePipelineState(function: l2SeqFunc)
            } else { l2NormQkSeqPipeline = nil }

            if let gdnSeqFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_sequence") {
                gdnLinearAttnSeqPipeline = try device.makeComputePipelineState(function: gdnSeqFunc)
            } else { gdnLinearAttnSeqPipeline = nil }

            if let gdnSeqSigFunc = defaultLibrary.makeFunction(name: "linear_attention_recurrent_sequence_sigmoid") {
                gdnLinearAttnSeqSigmoidPipeline = try device.makeComputePipelineState(function: gdnSeqSigFunc)
            } else { gdnLinearAttnSeqSigmoidPipeline = nil }

            if let fp8GateBatchedFunc = defaultLibrary.makeFunction(name: "fp8_swiglu_gate_up_batched") {
                fp8GateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8GateBatchedFunc)
            } else { fp8GateUpBatchedPipeline = nil }

            if let fp8DownBatchedFunc = defaultLibrary.makeFunction(name: "fp8_down_proj_accumulate_batched") {
                fp8DownBatchedPipeline = try device.makeComputePipelineState(function: fp8DownBatchedFunc)
            } else { fp8DownBatchedPipeline = nil }

            if let fp8BlockGateBatchedFunc = defaultLibrary.makeFunction(name: "fp8_block_swiglu_gate_up_batched") {
                fp8BlockGateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockGateBatchedFunc)
            } else { fp8BlockGateUpBatchedPipeline = nil }

            if let fp8BlockDownBatchedFunc = defaultLibrary.makeFunction(name: "fp8_block_down_proj_accumulate_batched") {
                fp8BlockDownBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockDownBatchedFunc)
            } else { fp8BlockDownBatchedPipeline = nil }

            if let bf16GateBatchedFunc = defaultLibrary.makeFunction(name: "bf16_swiglu_gate_up_batched") {
                bf16GateUpBatchedPipeline = try device.makeComputePipelineState(function: bf16GateBatchedFunc)
            } else { bf16GateUpBatchedPipeline = nil }

            if let bf16DownBatchedFunc = defaultLibrary.makeFunction(name: "bf16_down_proj_accumulate_batched") {
                bf16DownBatchedPipeline = try device.makeComputePipelineState(function: bf16DownBatchedFunc)
            } else { bf16DownBatchedPipeline = nil }

            if let kdaFunc = defaultLibrary.makeFunction(name: "kda_linear_attention_recurrent_step") {
                kdaLinearAttnStepPipeline = try device.makeComputePipelineState(function: kdaFunc)
            } else { kdaLinearAttnStepPipeline = nil }

            if let ropeInterFunc = defaultLibrary.makeFunction(name: "apply_rope_interleaved") {
                ropeInterleavedPipeline = try device.makeComputePipelineState(function: ropeInterFunc)
            } else { ropeInterleavedPipeline = nil }

            if let routerBailingFunc = defaultLibrary.makeFunction(name: "moe_router_bailing_topk_bf16") {
                routerBailingPipeline = try device.makeComputePipelineState(function: routerBailingFunc)
            } else { routerBailingPipeline = nil }

            if let storeMlaFunc = defaultLibrary.makeFunction(name: "store_mla_kv_cache") {
                storeMlaKvCachePipeline = try device.makeComputePipelineState(function: storeMlaFunc)
            } else { storeMlaKvCachePipeline = nil }

            if let storeMlaF16Func = defaultLibrary.makeFunction(name: "store_mla_kv_cache_f16") {
                storeMlaKvCacheF16Pipeline = try device.makeComputePipelineState(function: storeMlaF16Func)
            } else { storeMlaKvCacheF16Pipeline = nil }

            if let mlaDecodeFunc = defaultLibrary.makeFunction(name: "mla_attention_decode") {
                mlaAttentionDecodePipeline = try device.makeComputePipelineState(function: mlaDecodeFunc)
            } else { mlaAttentionDecodePipeline = nil }

            if let mlaDecodeF16Func = defaultLibrary.makeFunction(name: "mla_attention_decode_f16") {
                mlaAttentionDecodeF16Pipeline = try device.makeComputePipelineState(function: mlaDecodeF16Func)
            } else { mlaAttentionDecodeF16Pipeline = nil }
        } catch {
            let err = "❌ Failed to create compute pipelines: \(error.localizedDescription)"
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        // Model Hyperparameters & Dynamic Sizing
        let hiddenDim: UInt32 = UInt32(modelConfig?.effectiveHiddenSize ?? (isNanbeige ? 3072 : (activeHiddenDim > 0 ? activeHiddenDim : 2048)))
        let numHeads: UInt32 = UInt32(modelConfig?.effectiveNumAttentionHeads ?? (isNanbeige ? 48 : 16))
        let numKvHeads: UInt32 = UInt32(modelConfig?.effectiveNumKeyValueHeads ?? (isNanbeige ? 8 : 2))
        let headDim: UInt32 = UInt32(modelConfig?.effectiveHeadDim ?? 128)
        let rotaryDim: UInt32 = UInt32(modelConfig?.effectiveRotaryDim ?? 128)
        let thetaVal: Float = modelConfig?.effectiveRopeTheta ?? (isNanbeige ? 70000000.0 : 10000000.0)
        let totalLoops: Int = max(1, modelConfig?.effectiveNumLoops ?? (isNanbeige ? 2 : 1))
        let eosTokenId: UInt32 = UInt32(modelConfig?.effectiveEosTokenId ?? (isNanbeige ? 166101 : 248044))

        let cachedLayers = buildCachedLayers(summary: summary)
        let actualLayers = cachedLayers.count
        guard actualLayers > 0 else {
            let err = "❌ No layer tensors found in model."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let arch = modelConfig?.resolveArchitectureType(summary: summary) ?? (summary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)
        let isHybridArch = arch.isHybridSsm
        let isRMSNormOffset = modelConfig?.isRMSNormUnitOffset ?? false

        let isGatedQ: Bool = {
            if let configGate = modelConfig?.effectiveAttnOutputGate {
                return configGate
            }
            if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                let gatedDim = numHeads * headDim * 2
                return qTensor.shapeDisplay.contains("\(gatedDim)")
            }
            return false
        }()
        let isStandardGqa = !isGatedQ
        let qOutDim: UInt32 = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
        let kvOutDim: UInt32 = numKvHeads * headDim
        let attnCtxDim: UInt32 = numHeads * headDim
        let kvStride: UInt32 = numKvHeads * headDim

        let numExperts: UInt32 = summary.maxExpertId > 0 ? summary.maxExpertId : UInt32(modelConfig?.effectiveNumExperts ?? 0)
        let embedOffset = embedWeight.offsetStart
        let normOffset = normTensor?.offsetStart ?? 0
        let lmHeadOffset = lmHeadTensor.offsetStart
        let eps: Float = modelConfig?.effectiveRmsNormEps ?? (isNanbeige ? 1e-5 : 1e-6)

        var vocabSize: UInt32 = 248320
        let cleanShape = lmHeadTensor.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
        let shapeParts = cleanShape.split(separator: ",")
        if let first = shapeParts.first, let parsed = UInt32(first), parsed > 0 {
            vocabSize = parsed
        } else if let cfgVocab = modelConfig?.effectiveVocabSize {
            vocabSize = UInt32(cfgVocab)
        }

        let maxInterDim = cachedLayers.map { $0.intermediateDim }.max() ?? 512

        var loadedLayout: FlashMoELayout? = nil
        let packedExpertsDir: URL?
        if let modelPath = activeLoadedModelPath {
            var isDir: ObjCBool = false
            FileManager.default.fileExists(atPath: modelPath, isDirectory: &isDir)
            let baseDir = isDir.boolValue ? URL(fileURLWithPath: modelPath) : URL(fileURLWithPath: modelPath).deletingLastPathComponent()
            let pDir = baseDir.appendingPathComponent("packed_experts")
            let layoutFile = pDir.appendingPathComponent("layout.json")
            if FileManager.default.fileExists(atPath: layoutFile.path),
               let lData = try? Data(contentsOf: layoutFile),
               let lay = try? JSONDecoder().decode(FlashMoELayout.self, from: lData) {
                packedExpertsDir = pDir
                loadedLayout = lay
                ExpertIOThreadPool.shared.initialize(numThreads: 8)
            } else {
                packedExpertsDir = nil
            }
        } else {
            packedExpertsDir = nil
        }

        let expertStagingSize = max(16 * 4194304, 16 * Int(loadedLayout?.expert_size ?? 4718592))

        // Allocate Shared Scratch Buffers (Reused across all generation steps)
        guard let singleTokenBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let hCurrBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hNextBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let qGateBuffer = device.makeBuffer(length: max(Int(qOutDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let kVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 4096) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let vVectorBuffer = device.makeBuffer(length: max(Int(kvOutDim), 6144) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zGateBuffer = device.makeBuffer(length: 8192 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aVectorBuffer = device.makeBuffer(length: max(2048, Int(hiddenDim)) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bVectorBuffer = device.makeBuffer(length: max(2048, Int(hiddenDim)) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnCtxBuffer = device.makeBuffer(length: max(Int(attnCtxDim), 8192) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let attnOutBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMidBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm2Buffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let routerIndicesBuffer = device.makeBuffer(length: 16 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let routerWeightsBuffer = device.makeBuffer(length: 16 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let sharedScoreBuffer = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuffer = device.makeBuffer(length: max(Int(maxInterDim), 512) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let expertStagingBuffer = device.makeBuffer(length: expertStagingSize, options: .storageModeShared),
              let hcStreamsBuffer = device.makeBuffer(length: max(4 * Int(hiddenDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcNormedBuffer = device.makeBuffer(length: max(4 * Int(hiddenDim), 10240) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcBottleneckBuffer = device.makeBuffer(length: 512 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcInjectScaleBuffer = device.makeBuffer(length: 4 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            let err = "❌ Failed to allocate GPU scratch buffers."
            gpuComputeOutput = err
            generationStatusText = err
            return
        }

        let temp = self.temperature
        let topPVal = self.topP
        let minPVal = self.minP
        let topKVal = self.topK
        let repPen = self.repetitionPenalty
        let presPen = self.presencePenalty
        let maxTokens = self.maxNewTokens
        let buffers = self.shardBuffers
        let effMode = self.memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: summary.sizeGb)
        let isFullRAM = (effMode == .residentRAM || self.memoryBudgetMode == .unrestricted)
        let budgetMode: MemoryBudgetMode = isFullRAM ? .unrestricted : self.memoryBudgetMode

        let isAgentSession = (sessionId != nil) ? (self.sessions.first(where: { $0.id == sessionId })?.isAgentToolsEnabled ?? self.defaultAgentToolsEnabled) : self.defaultAgentToolsEnabled
        let minSeq = isAgentSession ? 8192 : 2048
        let neededSeqLen = max(minSeq, min(32768, promptTokenIds.count + maxTokens + 512))
        let kvPrec = self.kvCachePrecision
        let hasRecurrence = (modelConfig?.isLingModel == true || modelConfig?.hasLinearRecurrence == true || cachedLayers.contains { $0.attentionType == .linearAttention })
        let prefixTokensReused = hasRecurrence ? 0 : PrefixCacheManager.shared.findCommonPrefix(promptTokenIds: promptTokenIds, sessionId: sessionId)
        KVCacheManager.shared.reset(
            device: device,
            config: modelConfig,
            actualLayers: actualLayers,
            totalLoops: totalLoops,
            numKvHeads: Int(numKvHeads),
            headDim: Int(headDim),
            maxSeqLen: neededSeqLen,
            precision: kvPrec,
            preservePrefixCount: prefixTokensReused
        )
        GrammarConstrainedSampler.shared.reset()

        isGeneratingText = true
        generatedStreamText = ""
        thinkingText = ""
        responseText = ""
        isThinking = thinkingEnabled
        isThinkingExpanded = true
        generationTotalTokens = 0
        generationSpeedTokPerSec = 0.0
        generationElapsedMs = 0.0
        generationStatusText = thinkingEnabled ? "🧠 Reasoning..." : "⚡ Initializing Autoregressive Generation..."

        var priorThinking: String? = nil
        var priorContent: String? = nil
        if let sId = sessionId, let mId = messageId {
            if let sIdx = sessions.firstIndex(where: { $0.id == sId }),
               let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                priorThinking = sessions[sIdx].messages[mIdx].thinkingContent
                priorContent = sessions[sIdx].messages[mIdx].content
                sessions[sIdx].messages[mIdx].isThinking = thinkingEnabled
            }
        }

        generationTask = Task.detached(priority: .userInitiated) {
            var contextTokens = promptTokenIds
            let startTime = CFAbsoluteTimeGetCurrent()
            var firstTokenTimestamp: Double? = nil
            var thinkingEndTimestamp: Double? = nil
            var tokensGenerated = 0
            var currentStep: UInt32 = 0

            // Helper for Linear/GEMV Projections (handles Q4, FP8, and BF16 SIMD)
            func dispatchLinear(
                enc: MTLComputeCommandEncoder,
                weight: TensorMetadata?,
                scale: TensorMetadata?,
                bias: TensorMetadata?,
                inBuf: MTLBuffer,
                outBuf: MTLBuffer,
                inDim: UInt32,
                outDim: UInt32,
                groupSize: UInt32 = 64,
                inOffset: Int = 0,
                outOffset: Int = 0,
                batchSize: Int = 1
            ) {
                guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
                var wOff = w.offsetStart
                var inD = inDim
                var outD = outDim
                var grp = groupSize

                let hasScale = (scale != nil)
                let hasBias = (bias != nil)
                let isMXFP8 = hasScale && (scale!.dtype.contains("U8") || scale!.dtype.contains("UINT8") || (!hasBias && (w.dtype.contains("U32") || w.dtype.contains("U8") || w.dtype.contains("FP8")) && (scale!.offsetEnd - scale!.offsetStart) >= UInt64((inDim / 32) * outDim)))
                let isQuantizedAffine = (hasBias || w.dtype.contains("Q4")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQuantizedAffine && !w.dtype.contains("BF16") && !w.dtype.contains("F16") && !w.dtype.contains("FLOAT")

                if isMXFP8, let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil {
                    var sOff = scale!.offsetStart
                    if let simdPipe = mxfp8GemvSimdPipeline {
                        enc.setComputePipelineState(simdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let mxfp8Pipe = mxfp8GemvPipeline {
                        enc.setComputePipelineState(mxfp8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                } else if isQuantizedAffine {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : nil
                    let bRaw = (bias != nil) ? buffers[bias!.shardIndex] : nil
                    guard let sBuffer = sRaw, let bBuffer = bRaw else { return }
                    var sOff = scale!.offsetStart
                    var bOff = bias!.offsetStart

                    let is8Bit = (w.offsetEnd - w.offsetStart) >= UInt64(outDim) * UInt64(inDim)
                    if is8Bit, let q8Pipe = q8GemvPipeline {
                        enc.setComputePipelineState(q8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: inOffset, index: 3)
                        enc.setBuffer(outBuf, offset: outOffset, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let q4Pipe = q4GemvPipeline {
                        enc.setComputePipelineState(q4Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(sBuffer, offset: 0, index: 1)
                        enc.setBuffer(bBuffer, offset: 0, index: 2)
                        enc.setBuffer(inBuf, offset: inOffset, index: 3)
                        enc.setBuffer(outBuf, offset: outOffset, index: 4)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    }
                } else if isFP8 {
                    let sRaw = (scale != nil) ? buffers[scale!.shardIndex] : wRaw
                    var sOff = scale?.offsetStart ?? 0
                    let isBlockScale = (scale != nil) && (scale!.name.contains("scale_inv") || ((scale!.offsetEnd - scale!.offsetStart) < UInt64(outDim * 2)))
                    if isBlockScale, let bSimdPipe = fp8BlockGemvSimdPipeline ?? fp8BlockGemvPipeline {
                        enc.setComputePipelineState(bSimdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let simdPipe = fp8GemvSimdPipeline {
                        enc.setComputePipelineState(simdPipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    } else if let fp8Pipe = fp8GemvPipeline {
                        enc.setComputePipelineState(fp8Pipe)
                        enc.setBuffer(wRaw, offset: 0, index: 0)
                        enc.setBuffer(inBuf, offset: inOffset, index: 1)
                        enc.setBuffer(outBuf, offset: outOffset, index: 2)
                        enc.setBuffer(sRaw, offset: 0, index: 3)
                        enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), fp8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                } else if let bSimdPipe = bf16GemvSimdPipeline {
                    enc.setComputePipelineState(bSimdPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else {
                    enc.setComputePipelineState(gemvBF16Pipeline)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: inOffset, index: 1)
                    enc.setBuffer(outBuf, offset: outOffset, index: 2)
                    enc.setBytes(&wOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    enc.setBytes(&inD, length: MemoryLayout<UInt32>.stride, index: 4)
                    enc.setBytes(&outD, length: MemoryLayout<UInt32>.stride, index: 5)
                    enc.dispatchThreads(MTLSize(width: Int(outDim), height: batchSize, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(outDim), gemvBF16Pipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }
                enc.memoryBarrier(scope: .buffers)
            }

            // Helper for MoE Expert MLP (SwiGLU Gate/Up + Down Proj Accumulate)
            func dispatchExpertMlp(
                enc: MTLComputeCommandEncoder,
                gateW: TensorMetadata,
                gateS: TensorMetadata?,
                gateB: TensorMetadata?,
                upW: TensorMetadata,
                upS: TensorMetadata?,
                upB: TensorMetadata?,
                downW: TensorMetadata,
                downS: TensorMetadata?,
                downB: TensorMetadata?,
                inBuf: MTLBuffer,
                interBuf: MTLBuffer,
                accumBuf: MTLBuffer,
                inDim: UInt32,
                interDim: UInt32,
                routingWeight: Float,
                groupSize: UInt32 = 64,
                inOffset: Int = 0,
                interOffset: Int = 0,
                accumOffset: Int = 0
            ) {
                guard let gRaw = buffers[gateW.shardIndex],
                      let uRaw = buffers[upW.shardIndex],
                      let dRaw = buffers[downW.shardIndex] else { return }

                var gWOff = gateW.offsetStart
                var uWOff = upW.offsetStart
                var dWOff = downW.offsetStart
                var hDimVal = inDim
                var interDimVal = interDim
                var p_k = routingWeight
                var grp = groupSize

                let hasGateScale = (gateS != nil)
                let hasGateBias = (gateB != nil)
                let isMXFP8 = hasGateScale && (gateS!.dtype.contains("U8") || gateS!.dtype.contains("UINT8") || (!hasGateBias && (gateW.dtype.contains("U32") || gateW.dtype.contains("U8") || gateW.dtype.contains("FP8")) && (gateS!.offsetEnd - gateS!.offsetStart) >= UInt64((inDim / 32) * interDim)))
                let isQuantizedAffine = (hasGateBias || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8")) && !isMXFP8
                let isFP8 = !isMXFP8 && !isQuantizedAffine && !gateW.dtype.contains("BF16") && !gateW.dtype.contains("F16") && !gateW.dtype.contains("FLOAT")

                if isMXFP8 {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    if let gateSimd = mxfp8GateUpSimdPipeline, let downSimd = mxfp8DownSimdPipeline {
                        enc.setComputePipelineState(gateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(downSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let mxfp8GatePipe = mxfp8GateUpPipeline, let mxfp8DownPipe = mxfp8DownPipeline {
                        enc.setComputePipelineState(mxfp8GatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(mxfp8DownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, mxfp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else if isQuantizedAffine {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let gB = gateB, let gBRaw = buffers[gB.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let uB = upB, let uBRaw = buffers[uB.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex],
                          let dB = downB, let dBRaw = buffers[dB.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var gBOff = gB.offsetStart
                    var uSOff = uS.offsetStart
                    var uBOff = uB.offsetStart
                    var dSOff = dS.offsetStart
                    var dBOff = dB.offsetStart

                    let isGate8Bit = (gateW.offsetEnd - gateW.offsetStart) >= UInt64(interDim) * UInt64(inDim)
                    let isUp8Bit = (upW.offsetEnd - upW.offsetStart) >= UInt64(interDim) * UInt64(inDim)
                    let isDown8Bit = (downW.offsetEnd - downW.offsetStart) >= UInt64(inDim) * UInt64(interDim)

                    let swigluPipe: MTLComputePipelineState?
                    if isGate8Bit && isUp8Bit {
                        swigluPipe = q8GateUpPipeline
                    } else if !isGate8Bit && !isUp8Bit {
                        swigluPipe = q4GateUpPipeline
                    } else if !isGate8Bit && isUp8Bit {
                        swigluPipe = q4GateQ8UpPipeline
                    } else {
                        swigluPipe = q8GateQ4UpPipeline
                    }

                    let downPipe: MTLComputePipelineState?
                    if isDown8Bit {
                        downPipe = q8DownPipeline
                    } else {
                        downPipe = q4DownPipeline
                    }

                    if let sPipe = swigluPipe, let dPipe = downPipe {
                        // Step 1: SwiGLU Gate & Up Proj
                        enc.setComputePipelineState(sPipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(gSRaw, offset: 0, index: 1)
                        enc.setBuffer(gBRaw, offset: 0, index: 2)
                        enc.setBuffer(uRaw, offset: 0, index: 3)
                        enc.setBuffer(uSRaw, offset: 0, index: 4)
                        enc.setBuffer(uBRaw, offset: 0, index: 5)
                        enc.setBuffer(inBuf, offset: inOffset, index: 6)
                        enc.setBuffer(interBuf, offset: interOffset, index: 7)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&gBOff, length: MemoryLayout<UInt64>.stride, index: 10)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 11)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 12)
                        enc.setBytes(&uBOff, length: MemoryLayout<UInt64>.stride, index: 13)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 14)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 15)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 16)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        // Step 2: Down Proj Accumulate
                        enc.setComputePipelineState(dPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(dSRaw, offset: 0, index: 1)
                        enc.setBuffer(dBRaw, offset: 0, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 4)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&dBOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 8)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                        enc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else if isFP8 {
                    guard let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                          let uS = upS, let uSRaw = buffers[uS.shardIndex],
                          let dS = downS, let dSRaw = buffers[dS.shardIndex] else { return }

                    var gSOff = gS.offsetStart
                    var uSOff = uS.offsetStart
                    var dSOff = dS.offsetStart

                    let isBlockScale = ((gS.offsetEnd - gS.offsetStart) < UInt64(interDim * 2)) || gS.name.contains("scale_inv")
                    let gatePipe = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline) : fp8GateUpSimdPipeline
                    let downPipe = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline) : fp8DownSimdPipeline

                    if let fp8GateSimd = gatePipe, let fp8DownSimd = downPipe {
                        enc.setComputePipelineState(fp8GateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(fp8DownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let fp8GatePipe = fp8GateUpPipeline, let fp8DownPipe = fp8DownPipeline {
                        enc.setComputePipelineState(fp8GatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBuffer(gSRaw, offset: 0, index: 4)
                        enc.setBuffer(uSRaw, offset: 0, index: 5)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 6)
                        enc.setBytes(&gSOff, length: MemoryLayout<UInt64>.stride, index: 7)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 8)
                        enc.setBytes(&uSOff, length: MemoryLayout<UInt64>.stride, index: 9)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 10)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 11)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, fp8GatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(fp8DownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBuffer(dSRaw, offset: 0, index: 3)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&dSOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 8)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, fp8DownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                } else {
                    // BF16
                    if let bGateSimd = bf16GateUpSimdPipeline, let bDownSimd = bf16DownSimdPipeline {
                        enc.setComputePipelineState(bGateSimd)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreadgroups(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(bDownSimd)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreadgroups(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    } else if let bGatePipe = bf16GateUpPipeline, let bDownPipe = bf16DownPipeline {
                        enc.setComputePipelineState(bGatePipe)
                        enc.setBuffer(gRaw, offset: 0, index: 0)
                        enc.setBuffer(uRaw, offset: 0, index: 1)
                        enc.setBuffer(inBuf, offset: inOffset, index: 2)
                        enc.setBuffer(interBuf, offset: interOffset, index: 3)
                        enc.setBytes(&gWOff, length: MemoryLayout<UInt64>.stride, index: 4)
                        enc.setBytes(&uWOff, length: MemoryLayout<UInt64>.stride, index: 5)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                        enc.dispatchThreads(MTLSize(width: Int(interDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(interDim), bGatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(bDownPipe)
                        enc.setBuffer(dRaw, offset: 0, index: 0)
                        enc.setBuffer(interBuf, offset: interOffset, index: 1)
                        enc.setBuffer(accumBuf, offset: accumOffset, index: 2)
                        enc.setBytes(&dWOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        enc.setBytes(&interDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                        enc.setBytes(&p_k, length: MemoryLayout<Float>.stride, index: 6)
                        enc.dispatchThreads(MTLSize(width: Int(inDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(inDim), bDownPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                }
            }

            // Helper for Single Token Forward Pass
            var previousLayerActiveExperts: [Int: [Int]] = [:]
            func runTokenForward(tokenId: UInt32, step: UInt32, computeLogits: Bool, wait: Bool = true) -> Bool {
                let singleTokenPtr = singleTokenBuffer.contents().bindMemory(to: UInt32.self, capacity: 1)
                singleTokenPtr[0] = tokenId
                var hDim = hiddenDim
                let isGatedQ: Bool = {
                    if let configGate = modelConfig?.effectiveAttnOutputGate {
                        return configGate
                    }
                    if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                        let gatedDim = numHeads * headDim * 2
                        return qTensor.shapeDisplay.contains("\(gatedDim)")
                    }
                    return false
                }()
                let isStandardGqa = !isGatedQ

                // 2. Multi-Layer Transformer Backbone (0..<actualLayers, looped totalLoops times)
                var currentH = hCurrBuffer
                var nextH = hNextBuffer

                guard var activeCmd = commandQueue.makeCommandBuffer() else { return false }

                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]

                        // Asynchronous Layer Lookahead Backbone Prefetching
                        if speculativePrefetchEnabled {
                            let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                            WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                            if prefetchLookaheadDepth >= 2 {
                                let nextNextL = (l + 2) < actualLayers ? (l + 2) : ((l + 2) % actualLayers)
                                WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextNextL], shardBuffers: buffers)
                            }
                        }

                        // --- Phase A: Attention & Routing Sub-Block ---
                        guard let layerEnc1 = activeCmd.makeComputeCommandEncoder() else { return false }

                        // Step 0: Embed Token on Loop 0, Layer 0
                        if loopIdx == 0 && l == 0 {
                            let hasEmbedScale = (embedScale != nil)
                            let hasEmbedBias = (embedBias != nil)
                            let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                            let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                            if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                               let embedScaleRaw = buffers[embedScale!.shardIndex] {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var hDimVal = hiddenDim
                                layerEnc1.setComputePipelineState(embedMXFP8Pipe)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 3)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                layerEnc1.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if isEmbedAffine,
                               let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                               let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                                var wOffset = embedOffset
                                var sOffset = embedScale!.offsetStart
                                var bOffset = embedBias!.offsetStart
                                var tok = tokenId
                                var grpSize: UInt32 = 64
                                let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4
                                if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                                    layerEnc1.setComputePipelineState(embedQ8Pipe)
                                    layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                    layerEnc1.setBuffer(currentH, offset: 0, index: 3)
                                    layerEnc1.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                    layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                    layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                    layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                } else if let embedQ4Pipe = embedQ4Pipeline {
                                    layerEnc1.setComputePipelineState(embedQ4Pipe)
                                    layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                    layerEnc1.setBuffer(currentH, offset: 0, index: 3)
                                    layerEnc1.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                    layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                    layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                    layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                    layerEnc1.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }
                            } else {
                                var wOffset = embedOffset
                                var tokCount: UInt32 = 1
                                layerEnc1.setComputePipelineState(embedPipeline)
                                layerEnc1.setBuffer(embedShardBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(singleTokenBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 2)
                                layerEnc1.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                            if (cachedLayers.first?.attnHcDownWeight != nil), let initStreams = fusedInit4StreamsPipeline {
                                layerEnc1.memoryBarrier(scope: .buffers)
                                layerEnc1.setComputePipelineState(initStreams)
                                layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                                layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 1)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), initStreams.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        }

                        // Step 1: Pre-Attention RMSNorm or Hyper-Connection Mixing (currentH / hcStreams -> xNorm1)
                        if let hcDown = layer.attnHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.attnHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.attnHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 1a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm1Buffer, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1d. Compute Injection Scale if attnHcInject exists
                            if let attnHcInj = layer.attnHcInjectWeight, let attnHcInjRaw = buffers[attnHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = attnHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(attnHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                            var gammaOff = norm1.offsetStart
                            var epsVal = eps
                            let isNorm1F16 = (norm1.dtype.contains("F16") || norm1.dtype.contains("HALF") || norm1.dtype.contains("FLOAT16")) && !norm1.dtype.contains("BF16") && !norm1.dtype.contains("BFLOAT")
                            let norm1Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm1Pipe = (isNorm1F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm1Pipe = (isNorm1F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm1Pipe)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm1Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm1Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 2: Attention Computation
                        if layer.isMLA {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1. Q projection: hiddenDim (1536) -> q_lora_rank (256) into kVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaQAProj, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 256)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 2. Q LayerNorm (RMSNorm 256) in-place on kVectorBuffer
                            if let qALN = layer.mlaQALayernorm, let qALNRaw = buffers[qALN.shardIndex] {
                                var qLNGammaOff = qALN.offsetStart
                                var qLNDim: UInt32 = 256
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(qALNRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&qLNGammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&qLNDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(qLNDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 3. Q_B projection: 256 -> 3072 (16 heads x 192) into qGateBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaQBProj, scale: nil, bias: nil, inBuf: kVectorBuffer, outBuf: qGateBuffer, inDim: 256, outDim: 3072)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4. Apply interleaved RoPE to Q rotary part (elements 128..191 of each 192-dim head in qGateBuffer)
                            if let ropePipe = ropeInterleavedPipeline {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 16
                                var hDimTotal: UInt32 = 192
                                var rotDim: UInt32 = 64
                                var hStride: UInt32 = 192
                                var thetaVal: Float = Float(modelConfig?.ropeTheta ?? 10000.0)
                                var qRopeOffset: UInt32 = 128
                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hDimTotal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rotDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&thetaVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.setBytes(&qRopeOffset, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 5. KV_A projection with MQA: hiddenDim (1536) -> 576 (512 compressed KV + 64 k_rope) into kVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaKVAProjWithMqa, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 576)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 6. Apply interleaved RoPE to K rotary part (elements 512..575 of kVectorBuffer)
                            if let ropePipe = ropeInterleavedPipeline {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 1
                                var hDimTotal: UInt32 = 64
                                var rotDim: UInt32 = 64
                                var hStride: UInt32 = 64
                                var thetaVal: Float = Float(modelConfig?.ropeTheta ?? 10000.0)
                                var kRopeOffset: UInt32 = 512
                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hDimTotal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rotDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&thetaVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.setBytes(&kRopeOffset, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 7. KV LayerNorm (RMSNorm 512) on first 512 elements of kVectorBuffer into zGateBuffer
                            if let kvALN = layer.mlaKVALayernorm, let kvALNRaw = buffers[kvALN.shardIndex] {
                                var kvLNGammaOff = kvALN.offsetStart
                                var kvLNDim: UInt32 = 512
                                var epsVal = eps
                                layerEnc1.setComputePipelineState(rmsnormPipeline)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kvALNRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&kvLNGammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&kvLNDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(kvLNDim)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 8. KV_B projection: 512 (zGateBuffer) -> 4096 (16 heads x (128 k_nope + 128 v)) into vVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.mlaKVBProj, scale: nil, bias: nil, inBuf: zGateBuffer, outBuf: vVectorBuffer, inDim: 512, outDim: 4096)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 9. Store into MLA KV cache (F16)
                            let isLingMla = (modelConfig?.isLingModel == true)
                            let slot = isLingMla ? ((loopIdx * 6) + layer.fullAttnIndex) : ((loopIdx * actualLayers) + layer.fullAttnIndex)
                            let maxSeq = KVCacheManager.shared.allocatedSeqLen
                            let kStride = isLingMla ? 3072 : 4096
                            let vStride = isLingMla ? 2048 : 4096
                            let kLayerByteOffset = slot * maxSeq * kStride * MemoryLayout<UInt16>.stride
                            let vLayerByteOffset = slot * maxSeq * vStride * MemoryLayout<UInt16>.stride
                            if let storePipe = storeMlaKvCacheF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                var pos = UInt32(step)
                                var nHeads: UInt32 = 16
                                layerEnc1.setComputePipelineState(storePipe)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 512 * MemoryLayout<Float>.stride, index: 1)
                                layerEnc1.setBuffer(kCache, offset: kLayerByteOffset, index: 2)
                                layerEnc1.setBuffer(vCache, offset: vLayerByteOffset, index: 3)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 10. Gate projection (hiddenDim -> 16 heads) into bVectorBuffer
                            if let gateProj = layer.mlaGateProj {
                                dispatchLinear(enc: layerEnc1, weight: gateProj, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: 16)
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 11. MLA Attention Decode (F16)
                            if let mlaPipe = mlaAttentionDecodeF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                var seqLen = UInt32(step + 1)
                                var nHeads: UInt32 = 16
                                var qkHeadDim: UInt32 = 192
                                var vDim: UInt32 = 128
                                layerEnc1.setComputePipelineState(mlaPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kCache, offset: kLayerByteOffset, index: 1)
                                layerEnc1.setBuffer(vCache, offset: vLayerByteOffset, index: 2)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&nHeads, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc1.setBytes(&qkHeadDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc1.setBytes(&vDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(nHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(32, Int(nHeads)), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 12. Output projection: 2048 (16 heads x 128 v_dim) -> hiddenDim (1536) into attnOutBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: nil, bias: nil, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: 2048, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else if layer.attentionType == .fullAttention {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = kvStride

                            dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: currentQDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: currentKvDim)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // Q-Norm (if present)
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex] {
                                let isQNormF16 = (qNorm.dtype.contains("F16") || qNorm.dtype.contains("HALF") || qNorm.dtype.contains("FLOAT16")) && !qNorm.dtype.contains("BF16") && !qNorm.dtype.contains("BFLOAT")
                                let qHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = qHeadNormPipe {
                                    var qNormOff = qNorm.offsetStart
                                    var nQ = numHeads
                                    var hD = headDim
                                    var hStride: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // K-Norm (if present)
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex] {
                                let isKNormF16 = (kNorm.dtype.contains("F16") || kNorm.dtype.contains("HALF") || kNorm.dtype.contains("FLOAT16")) && !kNorm.dtype.contains("BF16") && !kNorm.dtype.contains("BFLOAT")
                                let kHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = kHeadNormPipe {
                                    var kNormOff = kNorm.offsetStart
                                    var nK = numKvHeads
                                    var hD = headDim
                                    var hStride = headDim
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // RoPE on Q and K
                            if let ropePipe = ropePipeline {
                                var pos = step
                                var nQ = numHeads
                                var nK = numKvHeads
                                var hD = headDim
                                var rD = rotaryDim
                                var qStr: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                var kStr = headDim
                                var theta = thetaVal

                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Store KV-Cache and Fused GQA Decode
                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let maxSeq = KVCacheManager.shared.allocatedSeqLen
                                let prec = KVCacheManager.shared.activePrecision
                                let layerByteOffset = slot * maxSeq * Int(kvStride) * prec.bytesPerElement
                                var pos = step
                                var nKv = numKvHeads
                                var hD = headDim
                                var nQ = numHeads
                                var seqLen = step + 1

                                switch prec {
                                case .fp16:
                                    if let storePipe = storeKvCacheF16Pipeline ?? storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    if isStandardGqa, let gqaStdPipe = gqaStandardF16Pipeline ?? gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodeF16Pipeline ?? gqaDecodePipeline {
                                        layerEnc1.setComputePipelineState(gqaPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                case .fp8:
                                    let scaleByteOffset = slot * maxSeq * Int(numKvHeads) * MemoryLayout<UInt16>.stride
                                    if let quantStorePipe = storeKvCacheFP8Pipeline,
                                       let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        layerEnc1.setComputePipelineState(quantStorePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                        layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                        let threadgroups = MTLSize(width: Int(numKvHeads), height: 1, depth: 1)
                                        let threadsPerTG = MTLSize(width: 32, height: 1, depth: 1)
                                        layerEnc1.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                                    } else if let storePipe = storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    if let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        if isStandardGqa, let gqaStdPipe = gqaStandardFP8Pipeline {
                                            layerEnc1.setComputePipelineState(gqaStdPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if let gqaPipe = gqaDecodeFP8Pipeline {
                                            layerEnc1.setComputePipelineState(gqaPipe)
                                            layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                    }

                                case .fp32:
                                    if let storePipe = storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    if isStandardGqa, let gqaStdPipe = gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodePipeline {
                                        layerEnc1.setComputePipelineState(gqaPipe)
                                        layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                }
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // Attention Output Projection (o_proj)
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: attnCtxDim, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else if layer.isKDA {
                            // KDA Linear Attention (Key-Decay Attention with recurrent associative state)
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1. Q, K, V projections (hiddenDim 1536 -> 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaQProj ?? layer.qProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaKProj ?? layer.kProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: kVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            dispatchLinear(enc: layerEnc1, weight: layer.kdaVProj ?? layer.vProjTensor, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: vVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 2. Causal 1D Convolutions with SiLU
                            let linIdx = layer.linAttnIndex
                            if let convPipe = causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                var numChan: UInt32 = 2048
                                // Q conv
                                if let qc = layer.kdaQConv1d, let qcRaw = buffers[qc.shardIndex] {
                                    var cOff = qc.offsetStart
                                    let qStateOffset = (linIdx * 6144 + 0) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: qStateOffset, index: 2)
                                    layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                                // K conv
                                if let kc = layer.kdaKConv1d, let kcRaw = buffers[kc.shardIndex] {
                                    var cOff = kc.offsetStart
                                    let kStateOffset = (linIdx * 6144 + 2048) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: kStateOffset, index: 2)
                                    layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                                // V conv
                                if let vc = layer.kdaVConv1d, let vcRaw = buffers[vc.shardIndex] {
                                    var cOff = vc.offsetStart
                                    let vStateOffset = (linIdx * 6144 + 4096) * 4 * MemoryLayout<Float>.stride
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 0)
                                    layerEnc1.setBuffer(vcRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: vStateOffset, index: 2)
                                    layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChan, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.dispatchThreads(MTLSize(width: 2048, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            // 3. F, B, G projections
                            // F projection: decay gate (2048) into aVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: aVectorBuffer, inDim: hiddenDim, outDim: 2048)
                            // B projection: beta gate (16) into bVectorBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: 16)
                            // G projection: output gate (2048) into zGateBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: nil, bias: nil, inBuf: xNorm1Buffer, outBuf: zGateBuffer, inDim: hiddenDim, outDim: 2048)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4. KDA Recurrent Step
                            if let kdaPipe = kdaLinearAttnStepPipeline,
                               let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let oNorm = layer.linearNormTensor, let oNormRaw = buffers[oNorm.shardIndex] {
                                let stateByteOffset = linIdx * Int(16 * 128 * 128) * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var oNormOff = oNorm.offsetStart
                                var numHeads: UInt32 = 16
                                var headDim: UInt32 = 128
                                var lowerBound: Float = Float(modelConfig?.kdaLowerBound ?? -5.0)
                                var epsVal = eps
                                var isF32Params: UInt32 = (aLog.dtype.contains("F32") || (aLog.dtype.contains("FLOAT") && !aLog.dtype.contains("BF16"))) ? 1 : 0

                                layerEnc1.setComputePipelineState(kdaPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(kVectorBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(vVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(aVectorBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 4)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 5)
                                layerEnc1.setBuffer(aLogRaw, offset: 0, index: 6)
                                layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 7)
                                layerEnc1.setBuffer(oNormRaw, offset: 0, index: 8)
                                layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 9)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 10)
                                layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 12)
                                layerEnc1.setBytes(&oNormOff, length: MemoryLayout<UInt64>.stride, index: 13)
                                layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 14)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 15)
                                layerEnc1.setBytes(&lowerBound, length: MemoryLayout<Float>.stride, index: 16)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 17)
                                layerEnc1.setBytes(&isF32Params, length: MemoryLayout<UInt32>.stride, index: 18)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // 5. Output projection: 2048 -> hiddenDim (1536) into attnOutBuffer
                            dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: nil, bias: nil, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: 2048, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else {
                            // Linear Attention (GatedDeltaNet Recurrent State)
                            let linValHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (layer.inProjA != nil && (layer.inProjA!.offsetEnd - layer.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                            let linKeyHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                            let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                            let zDim: UInt32 = linValHeads * 128
                            let aDim: UInt32 = linValHeads
                            let bDim: UInt32 = linValHeads

                            dispatchLinear(enc: layerEnc1, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buffer, outBuf: qGateBuffer, inDim: hiddenDim, outDim: qkvDim)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                               let convPipe = causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                let convStateByteOffset = layer.linAttnIndex * Int(qkvDim) * 4 * MemoryLayout<Float>.stride
                                var cOff = conv1d.offsetStart
                                var numChannels: UInt32 = qkvDim
                                layerEnc1.setComputePipelineState(convPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(convRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(convState, offset: convStateByteOffset, index: 2)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 3)
                                layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                layerEnc1.setBytes(&numChannels, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            if let l2Pipe = l2NormQkPipeline {
                                var numHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                layerEnc1.setComputePipelineState(l2Pipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buffer, outBuf: zGateBuffer, inDim: hiddenDim, outDim: zDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buffer, outBuf: aVectorBuffer, inDim: hiddenDim, outDim: aDim)
                            dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buffer, outBuf: bVectorBuffer, inDim: hiddenDim, outDim: bDim)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                            let selectedStepPipe = useSigmoidGate
                                ? (gdnLinearAttnStepSigmoidPipeline ?? linearAttnStepSigmoidPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)
                                : (gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)

                            if let linPipe = selectedStepPipe,
                               let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                let linIdx = layer.linAttnIndex
                                let stateByteOffset = linIdx * Int(linValHeads * 128 * 128) * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var linNormOff = linNorm.offsetStart
                                var numValHeads: UInt32 = linValHeads
                                var numKeyHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                var epsVal = eps
                                
                                layerEnc1.setComputePipelineState(linPipe)
                                layerEnc1.setBuffer(qGateBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(zGateBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(aVectorBuffer, offset: 0, index: 2)
                                layerEnc1.setBuffer(bVectorBuffer, offset: 0, index: 3)
                                layerEnc1.setBuffer(aLogRaw, offset: 0, index: 4)
                                layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                layerEnc1.setBuffer(linNormRaw, offset: 0, index: 6)
                                layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                                layerEnc1.setBuffer(attnCtxBuffer, offset: 0, index: 8)
                                layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                layerEnc1.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                layerEnc1.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 12)
                                layerEnc1.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 14)
                                layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 15)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuffer, outBuf: attnOutBuffer, inDim: zDim, outDim: hiddenDim)
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 3: Residual Connection 1 (hMid = currentH + attnOut or 4-stream inject)
                        if (layer.attnHcInjectWeight != nil || layer.attnHcDownWeight != nil),
                           let injPipe = hcInjectPipeline {
                            layerEnc1.setComputePipelineState(injPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        } else {
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(currentH, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 4: Post-Attention RMSNorm / HC Stream Mixing (hMid/hcStreams -> xNorm2)
                        if let hcDown = layer.mlpHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.mlpHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.mlpHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 4a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4d. Compute Injection Scale if mlpHcInject exists
                            if let mlpHcInj = layer.mlpHcInjectWeight, let mlpHcInjRaw = buffers[mlpHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = mlpHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(mlpHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                            var gammaOff = norm2.offsetStart
                            var epsVal = eps
                            let isNorm2F16 = (norm2.dtype.contains("F16") || norm2.dtype.contains("HALF") || norm2.dtype.contains("FLOAT16")) && !norm2.dtype.contains("BF16") && !norm2.dtype.contains("BFLOAT")
                            let norm2Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm2Pipe = (isNorm2F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm2Pipe = (isNorm2F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm2Pipe)
                            layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // Step 5: Dense Feed-Forward OR MoE
                        let intermediateDim = layer.intermediateDim

                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            // Dense SwiGLU Feed-Forward (Unified within layerEnc1)
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight {
                                let gateS = layer.denseGateScale
                                let gateB = layer.denseGateBias
                                let upS = layer.denseUpScale
                                let upB = layer.denseUpBias
                                let downS = layer.denseDownScale
                                let downB = layer.denseDownBias

                                dispatchExpertMlp(
                                    enc: layerEnc1,
                                    gateW: gateW,
                                    gateS: gateS,
                                    gateB: gateB,
                                    upW: upW,
                                    upS: upS,
                                    upB: upB,
                                    downW: downW,
                                    downS: downS,
                                    downB: downB,
                                    inBuf: xNorm2Buffer,
                                    interBuf: interBuffer,
                                    accumBuf: hMlpBuffer,
                                    inDim: hiddenDim,
                                    interDim: intermediateDim,
                                    routingWeight: 1.0
                                )
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                            if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                               let injPipe = hcInjectPipeline {
                                layerEnc1.setComputePipelineState(injPipe)
                                layerEnc1.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            } else {
                                layerEnc1.setComputePipelineState(addPipeline)
                                layerEnc1.setBuffer(hMidBuffer, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                layerEnc1.setBuffer(nextH, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }

                            layerEnc1.endEncoding()
                        } else {
                            // MoE Model with dynamic routing
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))
                                var grp: UInt32 = 64

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        layerEnc1.setComputePipelineState(rQ8)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                    } else if let rQ4 = routerQ4Pipeline {
                                        layerEnc1.setComputePipelineState(rQ4)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 4)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 5)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc1.setComputePipelineState(r512)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        layerEnc1.setComputePipelineState(bailingPipe)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 1)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 3)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 4)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                        layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc1.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc1.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc1.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    } else {
                                        layerEnc1.setComputePipelineState(routerPipeline)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                }
                                if let sg = layer.sharedGateTensor, let sgRaw = buffers[sg.shardIndex] {
                                    var sgOff = sg.offsetStart
                                    var grp: UInt32 = 64
                                    if let sgScale = layer.sharedGateTensorScale, let sgBias = layer.sharedGateTensorBias,
                                       let sgScaleRaw = buffers[sgScale.shardIndex], let sgBiasRaw = buffers[sgBias.shardIndex] {
                                        var sOff = sgScale.offsetStart
                                        var bOff = sgBias.offsetStart
                                        let is8Bit = (sg.offsetEnd - sg.offsetStart) >= UInt64(hiddenDim)
                                        if is8Bit, let pipe = sharedGateQ8Pipeline {
                                            layerEnc1.setComputePipelineState(pipe)
                                            layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        } else if let pipe = sharedGateQ4Pipeline {
                                            layerEnc1.setComputePipelineState(pipe)
                                            layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 3)
                                            layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 4)
                                            layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        }
                                    } else if let sgPipe = sharedGatePipeline {
                                        layerEnc1.setComputePipelineState(sgPipe)
                                        layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer, offset: 0, index: 1)
                                        layerEnc1.setBuffer(sharedScoreBuffer, offset: 0, index: 2)
                                        layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                }
                            }

                            layerEnc1.endEncoding()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()
                            if let err = activeCmd.error {
                                print("❌ [METAL ERROR] activeCmd failed at layer \(l): \(err)")
                                return false
                            }

                            var activeExperts: [(id: Int, weight: Float)] = []
                            let topKCount = Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))
                            if layer.routerTensor != nil {
                                let indPtr = routerIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: topKCount)
                                let wPtr = routerWeightsBuffer.contents().bindMemory(to: Float.self, capacity: topKCount)
                                for i in 0..<topKCount {
                                    let p_k = wPtr[i]
                                    if p_k > 0.00001 {
                                        activeExperts.append((id: Int(indPtr[i]), weight: p_k))
                                    }
                                }
                            } else {
                                activeExperts = (0..<topKCount).map { (id: $0, weight: 1.0 / Float(topKCount)) }
                            }

                            var sharedW: Float = 1.0
                            if layer.sharedGateTensor != nil {
                                sharedW = sharedScoreBuffer.contents().load(as: Float.self)
                            }

                            let activeIds = activeExperts.map { $0.id }
                            if speculativePrefetchEnabled {
                                // 1. Transition correlation tracking
                                if l > 0, let prevIds = previousLayerActiveExperts[l - 1] {
                                    WorkingSetManager.shared.transitionTracker.recordTransition(fromLayer: l - 1, fromExperts: prevIds, toLayer: l, toExperts: activeIds)
                                }
                                previousLayerActiveExperts[l] = activeIds

                                // 2. Speculatively prefetch layer l + 1 experts based on Markov transition prediction
                                if l + 1 < actualLayers {
                                    let prefetchCount = min(10, max(8, activeIds.count))
                                    let predicted = WorkingSetManager.shared.predictNextLayerExperts(currentLayer: l, currentActiveExperts: activeIds, topN: prefetchCount)
                                    if !predicted.isEmpty {
                                        WorkingSetManager.shared.prefetchLayerExperts(layer: l + 1, expertIds: predicted, shardBuffers: buffers)
                                    }
                                }

                                // 3. Lookahead prefetch layer l + 2 dense backbone
                                if l + 2 < actualLayers {
                                    let nextNextL = l + 2
                                    if nextNextL < cachedLayers.count {
                                        WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextNextL], shardBuffers: buffers)
                                    }
                                }
                            }
                            WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeIds, mode: budgetMode, shardBuffers: buffers, isPrefill: !computeLogits)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 1769472)
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                // Dynamic component offset discovery from layout
                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                // 1. Fast parallel pread the active experts directly into unified staging MTLBuffer
                                var tasks: [ExpertPreadTask] = []
                                let rawStagingPtr = expertStagingBuffer.contents()
                                for (slot, exp) in activeExperts.enumerated() {
                                    let offset = off_t(exp.id * expertSize)
                                    let dst = rawStagingPtr.advanced(by: slot * expertSize)
                                    tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                                }
                                ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)

                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else { return false }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                for (slot, expert) in activeExperts.enumerated() {
                                    let pk = expert.weight
                                    if pk <= 0.00001 { continue }
                                    let slotOffset = UInt64(slot * expertSize)
                                    let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                    let gSOff = slotOffset + (compGateS?.offset ?? 524288)
                                    let gBOff = slotOffset + (compGateB?.offset ?? 557056)
                                    let uWOff = slotOffset + (compUpW?.offset ?? 589824)
                                    let uSOff = slotOffset + (compUpS?.offset ?? 1114112)
                                    let uBOff = slotOffset + (compUpB?.offset ?? 1146880)
                                    let dWOff = slotOffset + (compDownW?.offset ?? 1179648)
                                    let dSOff = slotOffset + (compDownS?.offset ?? 1703936)
                                    let dBOff = slotOffset + (compDownB?.offset ?? 1736704)

                                    if isFP8Layout {
                                        let isBlockScale = (compGateS?.size ?? 1024) < (intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                        let gateSimd = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline) : (fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                        let downSimd = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline) : (fp8DownSimdPipeline ?? fp8DownPipeline)

                                        if let gateSimd = gateSimd,
                                           let downSimd = downSimd {
                                            var gWOffU = gWOff
                                            var gSOffU = gSOff
                                            var uWOffU = uWOff
                                            var uSOffU = uSOff
                                            var dWOffU = dWOff
                                            var dSOffU = dSOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(intermediateDim)
                                            var pkVal = pk

                                            layerEnc2.setComputePipelineState(gateSimd)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                            layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                            layerEnc2.setBytes(&gWOffU, length: 8, index: 6)
                                            layerEnc2.setBytes(&gSOffU, length: 8, index: 7)
                                            layerEnc2.setBytes(&uWOffU, length: 8, index: 8)
                                            layerEnc2.setBytes(&uSOffU, length: 8, index: 9)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)

                                            layerEnc2.setComputePipelineState(downSimd)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                            layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                            layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                            layerEnc2.setBytes(&dWOffU, length: 8, index: 4)
                                            layerEnc2.setBytes(&dSOffU, length: 8, index: 5)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                            layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)
                                        }
                                    } else if isQuantizedAffine,
                                               let q4GatePipe = q4GateUpPipeline,
                                               let q4DownPipe = q4DownPipeline {
                                        var gWOffU = gWOff
                                        var gSOffU = gSOff
                                        var gBOffU = gBOff
                                        var uWOffU = uWOff
                                        var uSOffU = uSOff
                                        var uBOffU = uBOff
                                        var dWOffU = dWOff
                                        var dSOffU = dSOff
                                        var dBOffU = dBOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var grp: UInt32 = 64
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(q4GatePipe)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 6)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 7)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 8)
                                        layerEnc2.setBytes(&gSOffU, length: 8, index: 9)
                                        layerEnc2.setBytes(&gBOffU, length: 8, index: 10)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 11)
                                        layerEnc2.setBytes(&uSOffU, length: 8, index: 12)
                                        layerEnc2.setBytes(&uBOffU, length: 8, index: 13)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 14)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 15)
                                        layerEnc2.setBytes(&grp, length: 4, index: 16)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(q4DownPipe)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 4)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 5)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 6)
                                        layerEnc2.setBytes(&dSOffU, length: 8, index: 7)
                                        layerEnc2.setBytes(&dBOffU, length: 8, index: 8)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&grp, length: 4, index: 11)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if let gateSimd = bf16GateUpSimdPipeline,
                                               let downSimd = bf16DownSimdPipeline {
                                        var gWOffU = gWOff
                                        var uWOffU = uWOff
                                        var dWOffU = dWOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(gateSimd)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downSimd)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if let gateUnq = bf16GateUpPipeline,
                                               let downUnq = bf16DownPipeline {
                                        var gWOffU = gWOff
                                        var uWOffU = uWOff
                                        var dWOffU = dWOff
                                        var hDimVal: UInt32 = UInt32(hiddenDim)
                                        var interDimVal: UInt32 = UInt32(intermediateDim)
                                        var pkVal = pk

                                        layerEnc2.setComputePipelineState(gateUnq)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(intermediateDim), gateUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downUnq)
                                        layerEnc2.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }
                                }

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight {
                                    let gateS = layer.sharedGateScale
                                    let gateB = layer.sharedGateBias
                                    let upS = layer.sharedUpScale
                                    let upB = layer.sharedUpBias
                                    let downS = layer.sharedDownScale
                                    let downB = layer.sharedDownBias

                                    dispatchExpertMlp(
                                        enc: layerEnc2,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: sharedW
                                    )
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextH, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                layerEnc2.endEncoding()
                                moeCmd.commit()
                                moeCmd.waitUntilCompleted()
                                if let err = moeCmd.error {
                                    print("❌ [METAL ERROR] moeCmd (packed) failed at layer \(l): \(err)")
                                    return false
                                }
                            } else {
                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else { return false }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                for expert in activeExperts {
                                    let expId = expert.id
                                    let p_k = expert.weight
                                    if p_k <= 0.00001 { continue }

                                    if let gateW = layer.expertGateWeights[expId],
                                       let upW = layer.expertUpWeights[expId],
                                       let downW = layer.expertDownWeights[expId] {
                                        let gateS = layer.expertGateScales[expId]
                                        let gateB = layer.expertGateBiases[expId]
                                        let upS = layer.expertUpScales[expId]
                                        let upB = layer.expertUpBiases[expId]
                                        let downS = layer.expertDownScales[expId]
                                        let downB = layer.expertDownBiases[expId]

                                        dispatchExpertMlp(
                                            enc: layerEnc2,
                                            gateW: gateW,
                                            gateS: gateS,
                                            gateB: gateB,
                                            upW: upW,
                                            upS: upS,
                                            upB: upB,
                                            downW: downW,
                                            downS: downS,
                                            downB: downB,
                                            inBuf: xNorm2Buffer,
                                            interBuf: interBuffer,
                                            accumBuf: hMlpBuffer,
                                            inDim: hiddenDim,
                                            interDim: intermediateDim,
                                            routingWeight: p_k
                                        )
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }
                                }

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight {
                                    let gateS = layer.sharedGateScale
                                    let gateB = layer.sharedGateBias
                                    let upS = layer.sharedUpScale
                                    let upB = layer.sharedUpBias
                                    let downS = layer.sharedDownScale
                                    let downB = layer.sharedDownBias

                                    dispatchExpertMlp(
                                        enc: layerEnc2,
                                        gateW: gateW,
                                        gateS: gateS,
                                        gateB: gateB,
                                        upW: upW,
                                        upS: upS,
                                        upB: upB,
                                        downW: downW,
                                        downS: downS,
                                        downB: downB,
                                        inBuf: xNorm2Buffer,
                                        interBuf: interBuffer,
                                        accumBuf: hMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: intermediateDim,
                                        routingWeight: sharedW
                                    )
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                // Step 6: Residual Connection 2 (nextH = hMid + hMlp or 4-stream inject)
                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextH, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)
                                }

                                layerEnc2.endEncoding()
                                moeCmd.commit()
                                moeCmd.waitUntilCompleted()
                                if let err = moeCmd.error {
                                    print("❌ [METAL ERROR] moeCmd (unpacked) failed at layer \(l): \(err)")
                                    return false
                                }
                            }

                            guard let nextCmd = commandQueue.makeCommandBuffer() else { return false }
                            activeCmd = nextCmd
                        }

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }

                    // Apply intermediate loop final norm if multiple loops exist
                    if loopIdx + 1 < totalLoops, let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                        guard let loopNormEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                        var nOff = normOffset
                        var epsVal = eps
                        let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                        let finalNormPipe: MTLComputePipelineState
                        if isRMSNormOffset {
                            finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                        } else {
                            finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                        }
                        loopNormEnc.setComputePipelineState(finalNormPipe)
                        loopNormEnc.setBuffer(currentH, offset: 0, index: 0)
                        loopNormEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                        loopNormEnc.setBuffer(nextH, offset: 0, index: 2)
                        loopNormEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        loopNormEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        loopNormEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        loopNormEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        loopNormEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                        loopNormEnc.endEncoding()

                        let tempBuf = currentH
                        currentH = nextH
                        nextH = tempBuf
                    }
                }

                guard computeLogits else {
                    activeCmd.commit()
                    activeCmd.waitUntilCompleted()
                    if let err = activeCmd.error {
                        print("❌ [METAL ERROR] activeCmd (prefill) failed: \(err)")
                        return false
                    }
                    return true
                }

                // 3. Final RMSNorm & LM Head
                guard let finalEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                if let finalHcDown = finalHcDownWeight, let finalHcDownRaw = buffers[finalHcDown.shardIndex],
                   let finalHcUp = finalHcUpWeight, let finalHcUpRaw = buffers[finalHcUp.shardIndex],
                   let finalHcNorm = finalHcNormWeight, let finalHcNormRaw = buffers[finalHcNorm.shardIndex],
                   let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                    var nOff = finalHcNorm.offsetStart
                    var dOff = finalHcDown.offsetStart
                    var uOff = finalHcUp.offsetStart
                    var totDim: UInt32 = 4 * hiddenDim
                    var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                    var epsVal = eps

                    // Final 1a. Group RMSNorm on 4 Streams
                    finalEnc.setComputePipelineState(normPipe)
                    finalEnc.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(finalHcNormRaw, offset: 0, index: 1)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)

                    // Final 1b. Down-Projection
                    finalEnc.setComputePipelineState(downPipe)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(finalHcDownRaw, offset: 0, index: 1)
                    finalEnc.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                    finalEnc.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)

                    // Final 1c. Up-Projection and Stream Blend
                    finalEnc.setComputePipelineState(upPipe)
                    finalEnc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(finalHcUpRaw, offset: 0, index: 2)
                    finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 3)
                    finalEnc.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                    finalEnc.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                    finalEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                } else if let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                    if (cachedLayers.first?.attnHcDownWeight != nil), let extractPipe = extractStream0Pipeline {
                        finalEnc.setComputePipelineState(extractPipe)
                        finalEnc.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
                        finalEnc.setBuffer(currentH, offset: 0, index: 1)
                        finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                        finalEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), extractPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        finalEnc.memoryBarrier(scope: .buffers)
                    }

                    var nOff = normOffset
                    var epsVal = eps
                    let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                    let finalNormPipe: MTLComputePipelineState
                    if isRMSNormOffset {
                        finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                    } else {
                        finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                    }
                    finalEnc.setComputePipelineState(finalNormPipe)
                    finalEnc.setBuffer(currentH, offset: 0, index: 0)
                    finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(xFinalBuffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    finalEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                }

                dispatchLinear(enc: finalEnc, weight: lmHeadTensor, scale: lmHeadScale, bias: lmHeadBias, inBuf: xFinalBuffer, outBuf: logitsBuffer, inDim: hiddenDim, outDim: vocabSize)

                finalEnc.endEncoding()
                activeCmd.commit()
                if wait {
                    activeCmd.waitUntilCompleted()
                    if let err = activeCmd.error {
                        print("❌ [METAL ERROR] activeCmd (decode) failed: \(err)")
                        return false
                    }
                }

                return true
            }

            // Helper for Layer-Wise Streaming Prompt Prefill (O(Layers) SSD reads instead of O(Tokens * Layers))
            func runLayerWisePrefill(promptTokens: [UInt32], startPos: UInt32 = 0) -> Bool {
                let P = promptTokens.count
                guard P > 0 else { return true }

                let topKCount = max(Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : (numExperts > 0 ? 8 : 1))), 1)
                let hasHc = (cachedLayers.first?.attnHcDownWeight != nil)
                let totalHBytes = max(P * Int(hiddenDim) * MemoryLayout<Float>.stride, 64)
                let totalHcBytes = hasHc ? max(P * 4 * Int(hiddenDim) * MemoryLayout<Float>.stride, 64) : 64

                let maxInterDim = max(cachedLayers.map { $0.intermediateDim }.max() ?? 2560, 2560)
                let linKeyHeads = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                let linValHeads = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (cachedLayers.first?.inProjA != nil && (cachedLayers.first!.inProjA!.offsetEnd - cachedLayers.first!.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                let gdnQkvDim = (linKeyHeads + linKeyHeads + linValHeads) * 128
                let isGatedQ: Bool = {
                    if let configGate = modelConfig?.effectiveAttnOutputGate {
                        return configGate
                    }
                    if let qTensor = cachedLayers.first(where: { $0.attentionType == .fullAttention })?.qProjTensor {
                        let gatedDim = numHeads * headDim * 2
                        return qTensor.shapeDisplay.contains("\(gatedDim)")
                    }
                    return false
                }()
                let isStandardGqa = !isGatedQ
                let gqaQDim: UInt32 = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                let maxQkvDim = max(gdnQkvDim, gqaQDim, hiddenDim, 2560)
                let maxZDim = max(linValHeads * 128, attnCtxDim, 128)

                guard let hBufA = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let hBufB = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let hMidBuffer_all = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let xNorm1Buffer_all = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let xNorm2Buffer_all = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let attnOutBuffer_all = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let hMlpBuffer_all = device.makeBuffer(length: totalHBytes, options: .storageModeShared),
                      let qGateBuffer_all = device.makeBuffer(length: max(P * Int(maxQkvDim), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let zGateBuffer_all = device.makeBuffer(length: max(P * Int(maxZDim), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let attnCtxBuffer_all = device.makeBuffer(length: max(P * Int(maxZDim), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let kVectorBuffer_all = device.makeBuffer(length: max(P * max(Int(kvStride), 128), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let vVectorBuffer_all = device.makeBuffer(length: max(P * max(Int(kvStride), 128), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let aVectorBuffer_all = device.makeBuffer(length: max(P * 128, 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let bVectorBuffer_all = device.makeBuffer(length: max(P * 128, 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let interBuffer_all = device.makeBuffer(length: max(P * Int(maxInterDim), 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let routerIndicesBuffer_all = device.makeBuffer(length: max(P * topKCount, 16) * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                      let routerWeightsBuffer_all = device.makeBuffer(length: max(P * topKCount, 16) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let sharedScoreBuffer_all = device.makeBuffer(length: max(P, 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let hcStreamsBuffer_all = device.makeBuffer(length: max(totalHcBytes, 64), options: .storageModeShared),
                      let hcNormedBuffer_all = device.makeBuffer(length: max(totalHcBytes, 64), options: .storageModeShared),
                      let hcBottleneckBuffer_all = device.makeBuffer(length: max(P * 512, 64) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let hcInjectScaleBuffer_all = device.makeBuffer(length: max(P * 4, 64) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let denseActiveTokensBuffer = device.makeBuffer(length: max(P, 1) * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                      let denseActiveWeightsBuffer = device.makeBuffer(length: max(P, 1) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let expertActiveTokensBuffer = device.makeBuffer(length: max(P * topKCount, 64) * MemoryLayout<UInt32>.stride, options: .storageModeShared),
                      let expertActiveWeightsBuffer = device.makeBuffer(length: max(P * topKCount, 64) * MemoryLayout<Float>.stride, options: .storageModeShared),
                      let prefillStagingBuffer = device.makeBuffer(length: max(min(max(P * topKCount, 64), numExperts > 0 ? Int(numExperts) : 512) * Int(loadedLayout?.expert_size ?? 3151872), 64), options: .storageModeShared) else {
                    return false
                }

                let denseTokPtr = denseActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: P)
                for i in 0..<P { denseTokPtr[i] = UInt32(i) }
                let denseWgtPtr = denseActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: P)
                for i in 0..<P { denseWgtPtr[i] = 1.0 }

                var hDim = hiddenDim
                let prefillStartTime = CFAbsoluteTimeGetCurrent()
                var lastUIUpdateTime = CFAbsoluteTimeGetCurrent()

                // Step 0: Initial Token Embeddings for all P prompt tokens into hBufA
                guard let embedCmd = commandQueue.makeCommandBuffer(),
                      let embedEnc = embedCmd.makeComputeCommandEncoder() else { return false }

                let hasEmbedScale = (embedScale != nil)
                let hasEmbedBias = (embedBias != nil)
                let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                let promptTokensBuffer = device.makeBuffer(bytes: promptTokens, length: P * MemoryLayout<UInt32>.stride, options: .storageModeShared)

                for p in 0..<P {
                    let tokOffset = p * MemoryLayout<UInt32>.stride
                    let outOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                    let tok = promptTokens[p]

                    if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                       let embedScaleRaw = buffers[embedScale!.shardIndex] {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var hDimVal = hiddenDim
                        embedEnc.setComputePipelineState(embedMXFP8Pipe)
                        embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(promptTokensBuffer, offset: tokOffset, index: 1)
                        embedEnc.setBuffer(hBufA, offset: outOffset, index: 2)
                        embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 3)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        embedEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 6)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if isEmbedAffine,
                       let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                       let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var bOffset = embedBias!.offsetStart
                        var tokVal = tok
                        var grpSize: UInt32 = 64
                        let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4
                        if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                            embedEnc.setComputePipelineState(embedQ8Pipe)
                            embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                            embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                            embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                            embedEnc.setBuffer(hBufA, offset: outOffset, index: 3)
                            embedEnc.setBytes(&tokVal, length: MemoryLayout<UInt32>.stride, index: 4)
                            embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                            embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                            embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else if let embedQ4Pipe = embedQ4Pipeline {
                            embedEnc.setComputePipelineState(embedQ4Pipe)
                            embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                            embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                            embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                            embedEnc.setBuffer(hBufA, offset: outOffset, index: 3)
                            embedEnc.setBytes(&tokVal, length: MemoryLayout<UInt32>.stride, index: 4)
                            embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                            embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                            embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                            embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                            embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                            embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                    } else {
                        var wOffset = embedOffset
                        var tokCount: UInt32 = 1
                        embedEnc.setComputePipelineState(embedPipeline)
                        embedEnc.setBuffer(embedShardBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(promptTokensBuffer, offset: tokOffset, index: 1)
                        embedEnc.setBuffer(hBufA, offset: outOffset, index: 2)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        embedEnc.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }

                if hasHc, let initStreams = fusedInit4StreamsPipeline {
                    var hDimVal = hiddenDim
                    embedEnc.setComputePipelineState(initStreams)
                    embedEnc.setBuffer(hBufA, offset: 0, index: 0)
                    embedEnc.setBuffer(hcStreamsBuffer_all, offset: 0, index: 1)
                    embedEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                    embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), initStreams.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                embedEnc.endEncoding()
                embedCmd.commit()
                embedCmd.waitUntilCompleted()

                var currHBuf = hBufA
                var nextHBuf = hBufB

                // Step 1: Layer-Wise Forward Pass: Outer Loop Layers, Inner Loop Tokens
                let totalPasses = totalLoops * actualLayers
                var passIdx = 0

                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]
                        let intermediateDim = layer.intermediateDim

                        if speculativePrefetchEnabled {
                            let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                            WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                        }

                        // Phase A: RMSNorm1/HC Blend + Attention/GDN + Residual1/HC Inject + RMSNorm2/HC Blend + Router (batched across all P tokens in ONE encoder)
                        guard let activeCmd = commandQueue.makeCommandBuffer(),
                              let layerEnc1 = activeCmd.makeComputeCommandEncoder() else { return false }

                        // 1. RMSNorm 1 OR Pre-Attention Hyper-Connection Mixing (dispatched across all P tokens)
                        if let hcDown = layer.attnHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.attnHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.attnHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 1a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm1Buffer_all, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 1d. Compute Injection Scale if attnHcInject exists
                            if let attnHcInj = layer.attnHcInjectWeight, let attnHcInjRaw = buffers[attnHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = attnHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(attnHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex] {
                            var gammaOff = norm1.offsetStart
                            var epsVal = eps
                            let isNorm1F16 = (norm1.dtype.contains("F16") || norm1.dtype.contains("HALF") || norm1.dtype.contains("FLOAT16")) && !norm1.dtype.contains("BF16") && !norm1.dtype.contains("BFLOAT")
                            let norm1Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm1Pipe = (isNorm1F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm1Pipe = (isNorm1F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm1Pipe)
                            layerEnc1.setBuffer(currHBuf, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm1Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm1Buffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // 2. Attention / GDN Computation
                        if layer.attentionType == .fullAttention {
                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = kvStride

                            dispatchLinear(enc: layerEnc1, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: currentQDim, batchSize: P)
                            dispatchLinear(enc: layerEnc1, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buffer_all, outBuf: kVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P)
                            dispatchLinear(enc: layerEnc1, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buffer_all, outBuf: vVectorBuffer_all, inDim: hiddenDim, outDim: currentKvDim, batchSize: P)
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex] {
                                let isQNormF16 = (qNorm.dtype.contains("F16") || qNorm.dtype.contains("HALF") || qNorm.dtype.contains("FLOAT16")) && !qNorm.dtype.contains("BF16") && !qNorm.dtype.contains("BFLOAT")
                                let qHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    qHeadNormPipe = (isQNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = qHeadNormPipe {
                                    var qNormOff = qNorm.offsetStart
                                    var nQ = numHeads
                                    var hD = headDim
                                    var hStride: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(qNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                }
                            }

                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex] {
                                let isKNormF16 = (kNorm.dtype.contains("F16") || kNorm.dtype.contains("HALF") || kNorm.dtype.contains("FLOAT16")) && !kNorm.dtype.contains("BF16") && !kNorm.dtype.contains("BFLOAT")
                                let kHeadNormPipe: MTLComputePipelineState?
                                if isRMSNormOffset {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormOffsetF16Pipeline != nil) ? headRmsnormOffsetF16Pipeline : (headRmsnormOffsetPipeline ?? headRmsnormPipeline)
                                } else {
                                    kHeadNormPipe = (isKNormF16 && headRmsnormF16Pipeline != nil) ? headRmsnormF16Pipeline : headRmsnormPipeline
                                }
                                if let headNormPipe = kHeadNormPipe {
                                    var kNormOff = kNorm.offsetStart
                                    var nK = numKvHeads
                                    var hD = headDim
                                    var hStride = headDim
                                    var epsVal = eps
                                    layerEnc1.setComputePipelineState(headNormPipe)
                                    layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(kNormRaw, offset: 0, index: 1)
                                    layerEnc1.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                    layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                    layerEnc1.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                }
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let ropePipe = ropePipeline {
                                var pos: UInt32 = startPos
                                var nQ = numHeads
                                var nK = numKvHeads
                                var hD = headDim
                                var rD = rotaryDim
                                var qStr: UInt32 = isStandardGqa ? headDim : (headDim * 2)
                                var kStr = headDim
                                var theta = thetaVal

                                layerEnc1.setComputePipelineState(ropePipe)
                                layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc1.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.setBytes(&rD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc1.setBytes(&theta, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(numKvHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                let maxSeq = KVCacheManager.shared.allocatedSeqLen
                                let prec = KVCacheManager.shared.activePrecision
                                let layerByteOffset = slot * maxSeq * Int(kvStride) * prec.bytesPerElement
                                var pos: UInt32 = startPos
                                var nKv = numKvHeads
                                var hD = headDim
                                var nQ = numHeads
                                var seqLen: UInt32 = (startPos > 0) ? (0x80000000 | startPos) : 0

                                switch prec {
                                case .fp16:
                                    if let storePipe = storeKvCacheF16Pipeline ?? storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    if isStandardGqa, let gqaStdPipe = gqaStandardF16Pipeline ?? gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodeF16Pipeline ?? gqaDecodePipeline {
                                        layerEnc1.setComputePipelineState(gqaPipe)
                                        layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                case .fp8:
                                    let scaleByteOffset = slot * maxSeq * Int(numKvHeads) * MemoryLayout<UInt16>.stride
                                    if let storePipe = storeKvCacheFP8Pipeline,
                                       let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 4)
                                        layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 5)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 8)
                                        let threadgroups = MTLSize(width: Int(numKvHeads), height: P, depth: 1)
                                        let threadsPerTG = MTLSize(width: 32, height: 1, depth: 1)
                                        layerEnc1.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerTG)
                                    }

                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    if let kScale = KVCacheManager.shared.kScaleBuffer,
                                       let vScale = KVCacheManager.shared.vScaleBuffer {
                                        if isStandardGqa, let gqaStdPipe = gqaStandardFP8Pipeline {
                                            layerEnc1.setComputePipelineState(gqaStdPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        } else if let gqaPipe = gqaDecodeFP8Pipeline {
                                            layerEnc1.setComputePipelineState(gqaPipe)
                                            layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                            layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                            layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                            layerEnc1.setBuffer(kScale, offset: scaleByteOffset, index: 3)
                                            layerEnc1.setBuffer(vScale, offset: scaleByteOffset, index: 4)
                                            layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 5)
                                            layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        }
                                    }

                                case .fp32:
                                    if let storePipe = storeKvCachePipeline {
                                        layerEnc1.setComputePipelineState(storePipe)
                                        layerEnc1.setBuffer(kVectorBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(vVectorBuffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                        layerEnc1.setBytes(&pos, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(kvStride), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    if isStandardGqa, let gqaStdPipe = gqaStandardPipeline {
                                        layerEnc1.setComputePipelineState(gqaStdPipe)
                                        layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaStdPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else if let gqaPipe = gqaDecodePipeline {
                                        layerEnc1.setComputePipelineState(gqaPipe)
                                        layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                        layerEnc1.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                        layerEnc1.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                        layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreads(MTLSize(width: Int(numHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }
                                }
                            }

                            layerEnc1.memoryBarrier(scope: .buffers)
                            dispatchLinear(enc: layerEnc1, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuffer_all, outBuf: attnOutBuffer_all, inDim: attnCtxDim, outDim: hiddenDim, batchSize: P)
                        } else {
                                // Linear Attention / GDN (Gated DeltaNet) Sequence Dispatch
                                let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                                let zDim: UInt32 = linValHeads * 128
                                let aDim: UInt32 = linValHeads
                                let bDim: UInt32 = linValHeads

                                dispatchLinear(enc: layerEnc1, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buffer_all, outBuf: qGateBuffer_all, inDim: hiddenDim, outDim: qkvDim, batchSize: P)
                                layerEnc1.memoryBarrier(scope: .buffers)

                                if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                                   let convPipe = causalConv1dSeqPipeline ?? causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                                    let convStateByteOffset = layer.linAttnIndex * Int(qkvDim) * 4 * MemoryLayout<Float>.stride
                                    var cOff = conv1d.offsetStart
                                    var numChannels: UInt32 = qkvDim
                                    var seqLen: UInt32 = UInt32(P)
                                    layerEnc1.setComputePipelineState(convPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(convRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(convState, offset: convStateByteOffset, index: 2)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBytes(&cOff, length: MemoryLayout<UInt64>.stride, index: 4)
                                    layerEnc1.setBytes(&numChannels, length: MemoryLayout<UInt32>.stride, index: 5)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 6)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                if let l2Pipe = l2NormQkSeqPipeline ?? l2NormQkPipeline {
                                    var numHeads: UInt32 = linKeyHeads
                                    var headDimVal: UInt32 = 128
                                    var qkvDimVal: UInt32 = qkvDim
                                    layerEnc1.setComputePipelineState(l2Pipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                    layerEnc1.setBytes(&headDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                                    layerEnc1.setBytes(&qkvDimVal, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc1.dispatchThreads(MTLSize(width: Int(linKeyHeads), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(linKeyHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                dispatchLinear(enc: layerEnc1, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buffer_all, outBuf: zGateBuffer_all, inDim: hiddenDim, outDim: zDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buffer_all, outBuf: aVectorBuffer_all, inDim: hiddenDim, outDim: aDim, batchSize: P)
                                dispatchLinear(enc: layerEnc1, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buffer_all, outBuf: bVectorBuffer_all, inDim: hiddenDim, outDim: bDim, batchSize: P)
                                layerEnc1.memoryBarrier(scope: .buffers)

                                let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                                let selectedSeqPipe = useSigmoidGate
                                    ? (gdnLinearAttnSeqSigmoidPipeline ?? gdnLinearAttnStepSigmoidPipeline ?? linearAttnStepSigmoidPipeline ?? gdnLinearAttnSeqPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)
                                    : (gdnLinearAttnSeqPipeline ?? gdnLinearAttnStepPipeline ?? linearAttnStepPipeline)

                                if let linPipe = selectedSeqPipe,
                                   let sBuf = KVCacheManager.shared.linearStateBuffer,
                                   let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                                   let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                                   let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                    let linIdx = layer.linAttnIndex
                                    let stateByteOffset = linIdx * Int(linValHeads * 128 * 128) * MemoryLayout<Float>.stride
                                    var aLogOff = aLog.offsetStart
                                    var dtBiasOff = dtBias.offsetStart
                                    var linNormOff = linNorm.offsetStart
                                    var numValHeads: UInt32 = linValHeads
                                    var numKeyHeads: UInt32 = linKeyHeads
                                    var headDim: UInt32 = 128
                                    var epsVal = eps
                                    var seqLen: UInt32 = UInt32(P)

                                    layerEnc1.setComputePipelineState(linPipe)
                                    layerEnc1.setBuffer(qGateBuffer_all, offset: 0, index: 0)
                                    layerEnc1.setBuffer(zGateBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(aVectorBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(bVectorBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBuffer(aLogRaw, offset: 0, index: 4)
                                    layerEnc1.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                    layerEnc1.setBuffer(linNormRaw, offset: 0, index: 6)
                                    layerEnc1.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                                    layerEnc1.setBuffer(attnCtxBuffer_all, offset: 0, index: 8)
                                    layerEnc1.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 9)
                                    layerEnc1.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                    layerEnc1.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                    layerEnc1.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 12)
                                    layerEnc1.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                    layerEnc1.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 14)
                                    layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 15)
                                    layerEnc1.setBytes(&seqLen, length: MemoryLayout<UInt32>.stride, index: 16)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }

                                dispatchLinear(enc: layerEnc1, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuffer_all, outBuf: attnOutBuffer_all, inDim: zDim, outDim: hiddenDim, batchSize: P)
                            }
                            layerEnc1.memoryBarrier(scope: .buffers)

                        // 3. Post-Attention HyperConnection Injection / Residual 1
                        if (layer.attnHcInjectWeight != nil || layer.attnHcDownWeight != nil),
                           let injPipe = hcInjectPipeline {
                            layerEnc1.setComputePipelineState(injPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else {
                            layerEnc1.setComputePipelineState(addPipeline)
                            layerEnc1.setBuffer(currHBuf, offset: 0, index: 0)
                            layerEnc1.setBuffer(attnOutBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                        layerEnc1.memoryBarrier(scope: .buffers)

                        // 4. Pre-MLP HyperConnection Mixing / RMSNorm 2
                        if let hcDown = layer.mlpHcDownWeight, let hcDownRaw = buffers[hcDown.shardIndex],
                           let hcUp = layer.mlpHcUpWeight, let hcUpRaw = buffers[hcUp.shardIndex],
                           let hcNorm = layer.mlpHcNorm, let hcNormRaw = buffers[hcNorm.shardIndex],
                           let normPipe = hcNormPipeline, let downPipe = hcDownProjPipeline, let upPipe = hcUpBlendPipeline {
                            var nOff = hcNorm.offsetStart
                            var dOff = hcDown.offsetStart
                            var uOff = hcUp.offsetStart
                            var totDim: UInt32 = 4 * hiddenDim
                            var rank: UInt32 = UInt32(modelConfig?.effectiveHcLowrank ?? 320)
                            var epsVal = eps

                            // 4a. Group RMSNorm on 4 Streams
                            layerEnc1.setComputePipelineState(normPipe)
                            layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcNormRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4b. Down-Projection
                            layerEnc1.setComputePipelineState(downPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcDownRaw, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&dOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(rank), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4c. Up-Projection and Stream Blend
                            layerEnc1.setComputePipelineState(upPipe)
                            layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(hcBottleneckBuffer_all, offset: 0, index: 1)
                            layerEnc1.setBuffer(hcUpRaw, offset: 0, index: 2)
                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 3)
                            layerEnc1.setBytes(&uOff, length: MemoryLayout<UInt64>.stride, index: 4)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                            layerEnc1.setBytes(&rank, length: MemoryLayout<UInt32>.stride, index: 6)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            // 4d. Compute Injection Scale if mlpHcInject exists
                            if let mlpHcInj = layer.mlpHcInjectWeight, let mlpHcInjRaw = buffers[mlpHcInj.shardIndex],
                               let injScalePipe = hcInjectScalePipeline {
                                var iOff = mlpHcInj.offsetStart
                                layerEnc1.setComputePipelineState(injScalePipe)
                                layerEnc1.setBuffer(hcNormedBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(mlpHcInjRaw, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&iOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                layerEnc1.setBytes(&totDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc1.dispatchThreadgroups(MTLSize(width: 4, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc1.memoryBarrier(scope: .buffers)
                            }
                        } else if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex] {
                            var gammaOff = norm2.offsetStart
                            var epsVal = eps
                            let isNorm2F16 = (norm2.dtype.contains("F16") || norm2.dtype.contains("HALF") || norm2.dtype.contains("FLOAT16")) && !norm2.dtype.contains("BF16") && !norm2.dtype.contains("BFLOAT")
                            let norm2Pipe: MTLComputePipelineState
                            if isRMSNormOffset {
                                norm2Pipe = (isNorm2F16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                            } else {
                                norm2Pipe = (isNorm2F16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                            }
                            layerEnc1.setComputePipelineState(norm2Pipe)
                            layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                            layerEnc1.setBuffer(norm2Raw, offset: 0, index: 1)
                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                            layerEnc1.setBytes(&gammaOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc1.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc1.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)
                        }

                        // 5. Dense MLP or MoE Routing
                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            layerEnc1.setComputePipelineState(clearPipeline)
                            layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                            layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc1.memoryBarrier(scope: .buffers)

                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight,
                               let gRaw = buffers[gateW.shardIndex],
                               let uRaw = buffers[upW.shardIndex],
                               let dRaw = buffers[downW.shardIndex] {
                                var gWOff = gateW.offsetStart
                                var uWOff = upW.offsetStart
                                var dWOff = downW.offsetStart
                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                var interDimVal: UInt32 = UInt32(intermediateDim)

                                let gateS = layer.denseGateScale
                                let upS = layer.denseUpScale
                                let downS = layer.denseDownScale
                                let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                let isQuantizedAffine = (layer.denseGateBias != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                if isBlockScale,
                                   let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                   let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                   let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                   let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                   let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                    var gSOff = gateS?.offsetStart ?? 0
                                    var uSOff = upS?.offsetStart ?? 0
                                    var dSOff = downS?.offsetStart ?? 0

                                    layerEnc1.setComputePipelineState(gateBatched)
                                    layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBuffer(gsRaw, offset: 0, index: 4)
                                    layerEnc1.setBuffer(usRaw, offset: 0, index: 5)
                                    layerEnc1.setBytes(&gWOff, length: 8, index: 6)
                                    layerEnc1.setBytes(&gSOff, length: 8, index: 7)
                                    layerEnc1.setBytes(&uWOff, length: 8, index: 8)
                                    layerEnc1.setBytes(&uSOff, length: 8, index: 9)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 10)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 11)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 12)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    layerEnc1.setComputePipelineState(downBatched)
                                    layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(dsRaw, offset: 0, index: 3)
                                    layerEnc1.setBytes(&dWOff, length: 8, index: 4)
                                    layerEnc1.setBytes(&dSOff, length: 8, index: 5)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 6)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 7)
                                    var pkVal: Float = 1.0
                                    layerEnc1.setBytes(&pkVal, length: 4, index: 8)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 9)
                                    layerEnc1.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 10)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else if !isQuantizedAffine,
                                           let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                           let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                    layerEnc1.setComputePipelineState(gateBatched)
                                    layerEnc1.setBuffer(gRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(uRaw, offset: 0, index: 1)
                                    layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 3)
                                    layerEnc1.setBytes(&gWOff, length: 8, index: 4)
                                    layerEnc1.setBytes(&uWOff, length: 8, index: 5)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 6)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 7)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)

                                    layerEnc1.setComputePipelineState(downBatched)
                                    layerEnc1.setBuffer(dRaw, offset: 0, index: 0)
                                    layerEnc1.setBuffer(interBuffer_all, offset: 0, index: 1)
                                    layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                    layerEnc1.setBytes(&dWOff, length: 8, index: 3)
                                    layerEnc1.setBytes(&interDimVal, length: 4, index: 4)
                                    layerEnc1.setBytes(&hDimVal, length: 4, index: 5)
                                    var pkVal: Float = 1.0
                                    layerEnc1.setBytes(&pkVal, length: 4, index: 6)
                                    layerEnc1.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                    layerEnc1.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 8)
                                    layerEnc1.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                } else {
                                    for p in 0..<P {
                                        let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                        let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride
                                        dispatchExpertMlp(
                                            enc: layerEnc1,
                                            gateW: gateW,
                                            gateS: gateS,
                                            gateB: layer.denseGateBias,
                                            upW: upW,
                                            upS: upS,
                                            upB: layer.denseUpBias,
                                            downW: downW,
                                            downS: downS,
                                            downB: layer.denseDownBias,
                                            inBuf: xNorm2Buffer_all,
                                            interBuf: interBuffer_all,
                                            accumBuf: hMlpBuffer_all,
                                            inDim: hiddenDim,
                                            interDim: intermediateDim,
                                            routingWeight: 1.0,
                                            inOffset: tokenOffset,
                                            interOffset: interTokenOffset,
                                            accumOffset: tokenOffset
                                        )
                                    }
                                    layerEnc1.memoryBarrier(scope: .buffers)
                                }
                            }

                            if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                               let injPipe = hcInjectPipeline {
                                layerEnc1.setComputePipelineState(injPipe)
                                layerEnc1.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                layerEnc1.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else {
                                layerEnc1.setComputePipelineState(addPipeline)
                                layerEnc1.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                layerEnc1.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                layerEnc1.setBuffer(nextHBuf, offset: 0, index: 2)
                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc1.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        } else {
                            // MoE Router & Shared Gate
                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(topKCount)
                                var grp: UInt32 = 64

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(rQ8)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 4)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 5)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                        }
                                    } else if let rQ4 = routerQ4Pipeline {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(rQ4)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rScaleRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 2)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 4)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 5)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc1.setComputePipelineState(r512)
                                        layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(routerIndicesBuffer_all, offset: 0, index: 2)
                                        layerEnc1.setBuffer(routerWeightsBuffer_all, offset: 0, index: 3)
                                        layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(bailingPipe)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(rBiasRaw, offset: 0, index: 1)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 2)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 3)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 4)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc1.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc1.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc1.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc1.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    } else {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = p * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = p * topKCount * MemoryLayout<Float>.stride
                                            layerEnc1.setComputePipelineState(routerPipeline)
                                            layerEnc1.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 1)
                                            layerEnc1.setBuffer(routerIndicesBuffer_all, offset: rIndexOffset, index: 2)
                                            layerEnc1.setBuffer(routerWeightsBuffer_all, offset: rWeightOffset, index: 3)
                                            layerEnc1.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                            layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 5)
                                            layerEnc1.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                            layerEnc1.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    }
                                }

                                if let sg = layer.sharedGateTensor, let sgRaw = buffers[sg.shardIndex] {
                                    var sgOff = sg.offsetStart
                                    var grp: UInt32 = 64
                                    if let sgScale = layer.sharedGateTensorScale, let sgBias = layer.sharedGateTensorBias,
                                       let sgScaleRaw = buffers[sgScale.shardIndex], let sgBiasRaw = buffers[sgBias.shardIndex] {
                                        var sOff = sgScale.offsetStart
                                        var bOff = sgBias.offsetStart
                                        let is8Bit = (sg.offsetEnd - sg.offsetStart) >= UInt64(hiddenDim)
                                        if is8Bit, let pipe = sharedGateQ8Pipeline {
                                            for p in 0..<P {
                                                let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let sScoreOffset = p * MemoryLayout<Float>.stride
                                                layerEnc1.setComputePipelineState(pipe)
                                                layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                                layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                                layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                                layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                                layerEnc1.setBuffer(sharedScoreBuffer_all, offset: sScoreOffset, index: 4)
                                                layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                                layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                                layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            }
                                        } else if let pipe = sharedGateQ4Pipeline {
                                            for p in 0..<P {
                                                let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let sScoreOffset = p * MemoryLayout<Float>.stride
                                                layerEnc1.setComputePipelineState(pipe)
                                                layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                                layerEnc1.setBuffer(sgScaleRaw, offset: 0, index: 1)
                                                layerEnc1.setBuffer(sgBiasRaw, offset: 0, index: 2)
                                                layerEnc1.setBuffer(xNorm2Buffer_all, offset: tokenOffset, index: 3)
                                                layerEnc1.setBuffer(sharedScoreBuffer_all, offset: sScoreOffset, index: 4)
                                                layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 5)
                                                layerEnc1.setBytes(&sOff, length: MemoryLayout<UInt64>.stride, index: 6)
                                                layerEnc1.setBytes(&bOff, length: MemoryLayout<UInt64>.stride, index: 7)
                                                layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                                layerEnc1.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 9)
                                                layerEnc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            }
                                        }
                                    } else if let sgPipe = sharedGatePipeline {
                                        layerEnc1.setComputePipelineState(sgPipe)
                                        layerEnc1.setBuffer(sgRaw, offset: 0, index: 0)
                                        layerEnc1.setBuffer(xNorm2Buffer_all, offset: 0, index: 1)
                                        layerEnc1.setBuffer(sharedScoreBuffer_all, offset: 0, index: 2)
                                        layerEnc1.setBytes(&sgOff, length: MemoryLayout<UInt64>.stride, index: 3)
                                        layerEnc1.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                                        layerEnc1.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                    }
                                }
                            }
                        }

                        layerEnc1.endEncoding()
                        activeCmd.commit()
                        activeCmd.waitUntilCompleted()

                        // Phase B: Expert MLPs for MoE layers (Expert-Major Batching)
                        if layer.mlpType != .denseMlp && layer.routerTensor != nil {
                            let indStride = topKCount
                            let indAllPtr = routerIndicesBuffer_all.contents().bindMemory(to: UInt32.self, capacity: P * indStride)
                            let wAllPtr = routerWeightsBuffer_all.contents().bindMemory(to: Float.self, capacity: P * indStride)
                            let hasSharedGate = (layer.sharedGateTensor != nil)
                            let sharedScorePtr = sharedScoreBuffer_all.contents().bindMemory(to: Float.self, capacity: P)

                            var expertTokenMap: [Int: [(tokenIdx: UInt32, weight: Float)]] = [:]
                            for p in 0..<P {
                                let rBase = p * indStride
                                for i in 0..<topKCount {
                                    let expId = Int(indAllPtr[rBase + i])
                                    let pk = wAllPtr[rBase + i]
                                    if pk > 0.00001 {
                                        expertTokenMap[expId, default: []].append((tokenIdx: UInt32(p), weight: pk))
                                    }
                                }
                            }

                            var expertBufferOffsetMap: [Int: Int] = [:]
                            var currentExpOffset = 0
                            let expTokPtr = expertActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: max(P * topKCount, 64))
                            let expWgtPtr = expertActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: max(P * topKCount, 64))
                            for (expId, tokenAssignments) in expertTokenMap {
                                guard !tokenAssignments.isEmpty else { continue }
                                expertBufferOffsetMap[expId] = currentExpOffset * MemoryLayout<UInt32>.stride
                                for item in tokenAssignments {
                                    expTokPtr[currentExpOffset] = item.tokenIdx
                                    expWgtPtr[currentExpOffset] = item.weight
                                    currentExpOffset += 1
                                }
                            }

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                let expertSize = Int(loadedLayout?.expert_size ?? 3151872)
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let uniqueActiveExpIds = Array(expertTokenMap.keys).sorted()
                                let numActive = uniqueActiveExpIds.count

                                if numActive > 0 {
                                    var tasks: [ExpertPreadTask] = []
                                    let rawStagingPtr = prefillStagingBuffer.contents()
                                    var expSlotMap: [Int: Int] = [:]
                                    for (slot, expId) in uniqueActiveExpIds.enumerated() {
                                        expSlotMap[expId] = slot
                                        let offset = off_t(expId * expertSize)
                                        let dst = rawStagingPtr.advanced(by: slot * expertSize)
                                        tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                                    }
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)

                                    guard let moeCmd = commandQueue.makeCommandBuffer(),
                                          let layerEnc2 = moeCmd.makeComputeCommandEncoder() else {
                                        return false
                                    }

                                    // Clear hMlpBuffer_all for all P tokens
                                    layerEnc2.setComputePipelineState(clearPipeline)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    layerEnc2.memoryBarrier(scope: .buffers)

                                    for expId in uniqueActiveExpIds {
                                        guard let tokenAssignments = expertTokenMap[expId], let slot = expSlotMap[expId] else { continue }
                                        let count = tokenAssignments.count
                                        if count == 0 { continue }
                                        let expOffset = expertBufferOffsetMap[expId] ?? 0

                                        let slotOffset = UInt64(slot * expertSize)
                                        let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                        let gSOff = slotOffset + (compGateS?.offset ?? 1048576)
                                        let gBOff = slotOffset + (compGateB?.offset ?? 0)
                                        let uWOff = slotOffset + (compUpW?.offset ?? 1049600)
                                        let uSOff = slotOffset + (compUpS?.offset ?? 2098176)
                                        let uBOff = slotOffset + (compUpB?.offset ?? 0)
                                        let dWOff = slotOffset + (compDownW?.offset ?? 2099200)
                                        let dSOff = slotOffset + (compDownS?.offset ?? 3147776)
                                        let dBOff = slotOffset + (compDownB?.offset ?? 0)
                                        let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                        if isFP8Layout {
                                            let isBlockScale = (compGateS?.size ?? 1024) < (intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                            let gateBatched = isBlockScale ? (fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline) : (fp8GateUpBatchedPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                            let downBatched = isBlockScale ? (fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline) : (fp8DownBatchedPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline)

                                            if let gateBatched = gateBatched,
                                               let downBatched = downBatched {
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(intermediateDim)

                                                layerEnc2.setComputePipelineState(gateBatched)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                                layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 4)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 5)
                                                layerEnc2.setBytes(&gWOffU, length: 8, index: 6)
                                                layerEnc2.setBytes(&gSOffU, length: 8, index: 7)
                                                layerEnc2.setBytes(&uWOffU, length: 8, index: 8)
                                                layerEnc2.setBytes(&uSOffU, length: 8, index: 9)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                                layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 12)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)

                                                layerEnc2.setComputePipelineState(downBatched)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                                layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 3)
                                                layerEnc2.setBytes(&dWOffU, length: 8, index: 4)
                                                layerEnc2.setBytes(&dSOffU, length: 8, index: 5)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                                var pkVal: Float = 1.0
                                                layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                                layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 9)
                                                layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 10)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)
                                            }
                                        } else if !isQuantizedAffine,
                                                  let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                                  let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(intermediateDim)

                                            layerEnc2.setComputePipelineState(gateBatched)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                            layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                            layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                            layerEnc2.setBytes(&gWOffU, length: 8, index: 4)
                                            layerEnc2.setBytes(&uWOffU, length: 8, index: 5)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                            layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)

                                            layerEnc2.setComputePipelineState(downBatched)
                                            layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                            layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                            layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                            layerEnc2.setBytes(&dWOffU, length: 8, index: 3)
                                            layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                            layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                            var pkVal: Float = 1.0
                                            layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                            layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 7)
                                            layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 8)
                                            layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            layerEnc2.memoryBarrier(scope: .buffers)
                                        } else if let q4Gate = q4GateUpPipeline,
                                                  let q4Down = q4DownPipeline {
                                            for item in tokenAssignments {
                                                let tOff = Int(item.tokenIdx) * Int(hiddenDim) * MemoryLayout<Float>.stride
                                                let iOff = Int(item.tokenIdx) * Int(intermediateDim) * MemoryLayout<Float>.stride
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var gBOffU = gBOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var uBOffU = uBOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var dBOffU = dBOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(intermediateDim)
                                                var grp: UInt32 = 64
                                                var pkVal = item.weight

                                                layerEnc2.setComputePipelineState(q4Gate)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 2)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 3)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 4)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 5)
                                                layerEnc2.setBuffer(xNorm2Buffer_all, offset: tOff, index: 6)
                                                layerEnc2.setBuffer(interBuffer_all, offset: iOff, index: 7)
                                                layerEnc2.setBytes(&gWOffU, length: 8, index: 8)
                                                layerEnc2.setBytes(&gSOffU, length: 8, index: 9)
                                                layerEnc2.setBytes(&gBOffU, length: 8, index: 10)
                                                layerEnc2.setBytes(&uWOffU, length: 8, index: 11)
                                                layerEnc2.setBytes(&uSOffU, length: 8, index: 12)
                                                layerEnc2.setBytes(&uBOffU, length: 8, index: 13)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 14)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 15)
                                                layerEnc2.setBytes(&grp, length: 4, index: 16)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)

                                                layerEnc2.setComputePipelineState(q4Down)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 0)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 1)
                                                layerEnc2.setBuffer(prefillStagingBuffer, offset: 0, index: 2)
                                                layerEnc2.setBuffer(interBuffer_all, offset: iOff, index: 3)
                                                layerEnc2.setBuffer(hMlpBuffer_all, offset: tOff, index: 4)
                                                layerEnc2.setBytes(&dWOffU, length: 8, index: 5)
                                                layerEnc2.setBytes(&dSOffU, length: 8, index: 6)
                                                layerEnc2.setBytes(&dBOffU, length: 8, index: 7)
                                                layerEnc2.setBytes(&interDimVal, length: 4, index: 8)
                                                layerEnc2.setBytes(&hDimVal, length: 4, index: 9)
                                                layerEnc2.setBytes(&grp, length: 4, index: 10)
                                                layerEnc2.setBytes(&pkVal, length: 4, index: 11)
                                                layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                layerEnc2.memoryBarrier(scope: .buffers)
                                            }
                                        }
                                    }

                                    if let gateW = layer.sharedGateWeight,
                                       let upW = layer.sharedUpWeight,
                                       let downW = layer.sharedDownWeight {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            let sharedW = hasSharedGate ? sharedScorePtr[p] : 1.0
                                            let gateS = layer.sharedGateScale
                                            let gateB = layer.sharedGateBias
                                            let upS = layer.sharedUpScale
                                            let upB = layer.sharedUpBias
                                            let downS = layer.sharedDownScale
                                            let downB = layer.sharedDownBias

                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: gateB,
                                                upW: upW,
                                                upS: upS,
                                                upB: upB,
                                                downW: downW,
                                                downS: downS,
                                                downB: downB,
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: sharedW,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    }

                                    if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                       let injPipe = hcInjectPipeline {
                                        layerEnc2.setComputePipelineState(injPipe)
                                        layerEnc2.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    } else {
                                        layerEnc2.setComputePipelineState(addPipeline)
                                        layerEnc2.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(nextHBuf, offset: 0, index: 2)
                                        layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                        layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                    }

                                    layerEnc2.endEncoding()
                                    moeCmd.commit()
                                    moeCmd.waitUntilCompleted()
                                }
                            } else {
                                let activeExpIds = Array(expertTokenMap.keys)
                                if speculativePrefetchEnabled {
                                    let nextL = (l + 1) < actualLayers ? (l + 1) : 0
                                    WorkingSetManager.shared.prefetchLayerBackbone(layer: cachedLayers[nextL], shardBuffers: buffers)
                                }
                                WorkingSetManager.shared.touchAndEvict(layer: l, activeExpertIds: activeExpIds, mode: budgetMode, shardBuffers: buffers, isPrefill: true)

                                guard let moeCmd = commandQueue.makeCommandBuffer(),
                                      let layerEnc2 = moeCmd.makeComputeCommandEncoder() else { return false }

                                layerEnc2.setComputePipelineState(clearPipeline)
                                layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 0)
                                layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc2.memoryBarrier(scope: .buffers)

                                for (expId, tokenAssignments) in expertTokenMap {
                                    guard let gateW = layer.expertGateWeights[expId],
                                          let upW = layer.expertUpWeights[expId],
                                          let downW = layer.expertDownWeights[expId],
                                          let gRaw = buffers[gateW.shardIndex],
                                          let uRaw = buffers[upW.shardIndex],
                                          let dRaw = buffers[downW.shardIndex] else { continue }

                                    let count = tokenAssignments.count
                                    if count == 0 { continue }
                                    let expOffset = expertBufferOffsetMap[expId] ?? 0

                                    var gWOff = gateW.offsetStart
                                    var uWOff = upW.offsetStart
                                    var dWOff = downW.offsetStart
                                    var hDimVal: UInt32 = UInt32(hiddenDim)
                                    var interDimVal: UInt32 = UInt32(intermediateDim)

                                    let gateS = layer.expertGateScales[expId]
                                    let upS = layer.expertUpScales[expId]
                                    let downS = layer.expertDownScales[expId]
                                    let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                    let isQuantizedAffine = (layer.expertGateBiases[expId] != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                    if isBlockScale,
                                       let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                       let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                       let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                       let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                       let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                        var gSOff = gateS?.offsetStart ?? 0
                                        var uSOff = upS?.offsetStart ?? 0
                                        var dSOff = downS?.offsetStart ?? 0

                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBuffer(gsRaw, offset: 0, index: 4)
                                        layerEnc2.setBuffer(usRaw, offset: 0, index: 5)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 6)
                                        layerEnc2.setBytes(&gSOff, length: 8, index: 7)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 8)
                                        layerEnc2.setBytes(&uSOff, length: 8, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(dsRaw, offset: 0, index: 3)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&dSOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 9)
                                        layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 10)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if !isQuantizedAffine,
                                              let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                              let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.setBuffer(expertActiveTokensBuffer, offset: expOffset, index: 7)
                                        layerEnc2.setBuffer(expertActiveWeightsBuffer, offset: expOffset, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: count, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else {
                                        for item in tokenAssignments {
                                            let tokenOffset = Int(item.tokenIdx) * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = Int(item.tokenIdx) * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: layer.expertGateBiases[expId],
                                                upW: upW,
                                                upS: upS,
                                                upB: layer.expertUpBiases[expId],
                                                downW: downW,
                                                downS: downS,
                                                downB: layer.expertDownBiases[expId],
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: item.weight,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                    }
                                }
                                layerEnc2.memoryBarrier(scope: .buffers)

                                if let gateW = layer.sharedGateWeight,
                                   let upW = layer.sharedUpWeight,
                                   let downW = layer.sharedDownWeight,
                                   let gRaw = buffers[gateW.shardIndex],
                                   let uRaw = buffers[upW.shardIndex],
                                   let dRaw = buffers[downW.shardIndex] {
                                    var gWOff = gateW.offsetStart
                                    var uWOff = upW.offsetStart
                                    var dWOff = downW.offsetStart
                                    var hDimVal: UInt32 = UInt32(hiddenDim)
                                    var interDimVal: UInt32 = UInt32(intermediateDim)
                                    let sharedWeightsBuf = hasSharedGate ? sharedScoreBuffer_all : denseActiveWeightsBuffer

                                    let gateS = layer.sharedGateScale
                                    let upS = layer.sharedUpScale
                                    let downS = layer.sharedDownScale
                                    let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
                                    let isQuantizedAffine = (layer.sharedGateBias != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

                                    if isBlockScale,
                                       let gateBatched = fp8BlockGateUpBatchedPipeline ?? fp8BlockGateUpSimdPipeline ?? fp8GateUpBatchedPipeline,
                                       let downBatched = fp8BlockDownBatchedPipeline ?? fp8BlockDownSimdPipeline ?? fp8DownBatchedPipeline,
                                       let gsRaw = gateS != nil ? buffers[gateS!.shardIndex] : nil,
                                       let usRaw = upS != nil ? buffers[upS!.shardIndex] : nil,
                                       let dsRaw = downS != nil ? buffers[downS!.shardIndex] : nil {
                                        var gSOff = gateS?.offsetStart ?? 0
                                        var uSOff = upS?.offsetStart ?? 0
                                        var dSOff = downS?.offsetStart ?? 0

                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBuffer(gsRaw, offset: 0, index: 4)
                                        layerEnc2.setBuffer(usRaw, offset: 0, index: 5)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 6)
                                        layerEnc2.setBytes(&gSOff, length: 8, index: 7)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 8)
                                        layerEnc2.setBytes(&uSOff, length: 8, index: 9)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 10)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 11)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 12)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(dsRaw, offset: 0, index: 3)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&dSOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 7)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 8)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 9)
                                        layerEnc2.setBuffer(sharedWeightsBuf, offset: 0, index: 10)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else if !isQuantizedAffine,
                                              let gateBatched = bf16GateUpBatchedPipeline ?? bf16GateUpSimdPipeline,
                                              let downBatched = bf16DownBatchedPipeline ?? bf16DownSimdPipeline {
                                        layerEnc2.setComputePipelineState(gateBatched)
                                        layerEnc2.setBuffer(gRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(uRaw, offset: 0, index: 1)
                                        layerEnc2.setBuffer(xNorm2Buffer_all, offset: 0, index: 2)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 3)
                                        layerEnc2.setBytes(&gWOff, length: 8, index: 4)
                                        layerEnc2.setBytes(&uWOff, length: 8, index: 5)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 6)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 7)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)

                                        layerEnc2.setComputePipelineState(downBatched)
                                        layerEnc2.setBuffer(dRaw, offset: 0, index: 0)
                                        layerEnc2.setBuffer(interBuffer_all, offset: 0, index: 1)
                                        layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 2)
                                        layerEnc2.setBytes(&dWOff, length: 8, index: 3)
                                        layerEnc2.setBytes(&interDimVal, length: 4, index: 4)
                                        layerEnc2.setBytes(&hDimVal, length: 4, index: 5)
                                        var pkVal: Float = 1.0
                                        layerEnc2.setBytes(&pkVal, length: 4, index: 6)
                                        layerEnc2.setBuffer(denseActiveTokensBuffer, offset: 0, index: 7)
                                        layerEnc2.setBuffer(sharedWeightsBuf, offset: 0, index: 8)
                                        layerEnc2.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc2.memoryBarrier(scope: .buffers)
                                    } else {
                                        for p in 0..<P {
                                            let tokenOffset = p * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let interTokenOffset = p * Int(intermediateDim) * MemoryLayout<Float>.stride
                                            let sharedW = hasSharedGate ? sharedScorePtr[p] : 1.0
                                            dispatchExpertMlp(
                                                enc: layerEnc2,
                                                gateW: gateW,
                                                gateS: gateS,
                                                gateB: layer.sharedGateBias,
                                                upW: upW,
                                                upS: upS,
                                                upB: layer.sharedUpBias,
                                                downW: downW,
                                                downS: downS,
                                                downB: layer.sharedDownBias,
                                                inBuf: xNorm2Buffer_all,
                                                interBuf: interBuffer_all,
                                                accumBuf: hMlpBuffer_all,
                                                inDim: hiddenDim,
                                                interDim: intermediateDim,
                                                routingWeight: sharedW,
                                                inOffset: tokenOffset,
                                                interOffset: interTokenOffset,
                                                accumOffset: tokenOffset
                                            )
                                        }
                                    }
                                }
                                layerEnc2.memoryBarrier(scope: .buffers)

                                if (layer.mlpHcInjectWeight != nil || layer.mlpHcDownWeight != nil),
                                   let injPipe = hcInjectPipeline {
                                    layerEnc2.setComputePipelineState(injPipe)
                                    layerEnc2.setBuffer(hcStreamsBuffer_all, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                    layerEnc2.setBuffer(hcInjectScaleBuffer_all, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), injPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                } else {
                                    layerEnc2.setComputePipelineState(addPipeline)
                                    layerEnc2.setBuffer(hMidBuffer_all, offset: 0, index: 0)
                                    layerEnc2.setBuffer(hMlpBuffer_all, offset: 0, index: 1)
                                    layerEnc2.setBuffer(nextHBuf, offset: 0, index: 2)
                                    layerEnc2.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                    layerEnc2.dispatchThreads(MTLSize(width: Int(hiddenDim), height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                }

                                layerEnc2.endEncoding()
                                moeCmd.commit()
                                moeCmd.waitUntilCompleted()
                            }
                        }

                        // Swap ping-pong hidden buffers for next layer
                        let tmp = currHBuf
                        currHBuf = nextHBuf
                        nextHBuf = tmp

                        if packedExpertsDir == nil && WorkingSetManager.shared.residentExpertsCount > budgetMode.maxResidentExperts {
                            WorkingSetManager.shared.trimToBudget(mode: budgetMode, shardBuffers: buffers)
                        }

                        passIdx += 1
                        let now = CFAbsoluteTimeGetCurrent()
                        if now - lastUIUpdateTime >= 0.1 || passIdx == totalPasses {
                            lastUIUpdateTime = now
                            let elapsed = max(0.001, now - prefillStartTime)
                            let effectiveTokensProcessed = Double(P) * (Double(passIdx) / Double(totalPasses))
                            let promptSpeed = effectiveTokensProcessed / elapsed
                            let pct = Int((Double(passIdx) / Double(totalPasses)) * 100)
                            let remainingPasses = totalPasses - passIdx
                            let timePerPass = elapsed / Double(passIdx)
                            let etaSec = Double(remainingPasses) * timePerPass
                            let etaStr = etaSec >= 60 ? String(format: "%dm %02ds", Int(etaSec) / 60, Int(etaSec) % 60) : String(format: "%.0fs", etaSec)
                            let speedStr = promptSpeed >= 10 ? String(format: "%.0f", promptSpeed) : String(format: "%.1f", promptSpeed)
                            let prefillStr = "Ingesting prompt: Layer \(passIdx)/\(totalPasses) (\(pct)%) • \(speedStr) tok/s • ETA: \(etaStr)"
                            let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB

                            Task { @MainActor in
                                self.generationSpeedTokPerSec = promptSpeed
                                self.generationStatusText = "📥 " + prefillStr
                                self.currentRssGB = currentRss
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                    self.sessions[sIdx].messages[mIdx].prefillStatus = prefillStr
                                }
                            }
                        }
                    }

                    // Apply intermediate loop final norm if multiple loops exist
                    if loopIdx + 1 < totalLoops, let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                        guard let loopNormCmd = commandQueue.makeCommandBuffer(),
                              let loopNormEnc = loopNormCmd.makeComputeCommandEncoder() else { return false }

                        var nOff = normOffset
                        var epsVal = eps
                        let isFinalF16 = (normTensor.dtype.contains("F16") || normTensor.dtype.contains("HALF") || normTensor.dtype.contains("FLOAT16")) && !normTensor.dtype.contains("BF16") && !normTensor.dtype.contains("BFLOAT")
                        let finalNormPipe: MTLComputePipelineState
                        if isRMSNormOffset {
                            finalNormPipe = (isFinalF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                        } else {
                            finalNormPipe = (isFinalF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                        }

                        loopNormEnc.setComputePipelineState(finalNormPipe)
                        loopNormEnc.setBuffer(currHBuf, offset: 0, index: 0)
                        loopNormEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                        loopNormEnc.setBuffer(nextHBuf, offset: 0, index: 2)
                        loopNormEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                        loopNormEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        loopNormEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                        loopNormEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                        loopNormEnc.dispatchThreadgroups(MTLSize(width: P, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))

                        loopNormEnc.endEncoding()
                        loopNormCmd.commit()
                        loopNormCmd.waitUntilCompleted()

                        let tmp = currHBuf
                        currHBuf = nextHBuf
                        nextHBuf = tmp
                    }
                }

                if hasHc, let ext0 = extractStream0Pipeline {
                    guard let extCmd = commandQueue.makeCommandBuffer(),
                          let extEnc = extCmd.makeComputeCommandEncoder() else { return false }
                    let lastHcTokenOffset = (P - 1) * 4 * Int(hiddenDim) * MemoryLayout<Float>.stride
                    extEnc.setComputePipelineState(ext0)
                    extEnc.setBuffer(hcStreamsBuffer_all, offset: lastHcTokenOffset, index: 0)
                    extEnc.setBuffer(hCurrBuffer, offset: 0, index: 1)
                    extEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 2)
                    extEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), ext0.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    extEnc.endEncoding()
                    extCmd.commit()
                    extCmd.waitUntilCompleted()
                } else {
                    // Copy final prompt token (P - 1)'s hidden state into hCurrBuffer so generation continues seamlessly
                    let lastTokenOffset = (P - 1) * Int(hiddenDim) * MemoryLayout<Float>.stride
                    guard let copyCmd = commandQueue.makeCommandBuffer(),
                          let blit = copyCmd.makeBlitCommandEncoder() else { return false }
                    blit.copy(from: currHBuf, sourceOffset: lastTokenOffset, to: hCurrBuffer, destinationOffset: 0, size: Int(hiddenDim) * MemoryLayout<Float>.stride)
                    blit.endEncoding()
                    copyCmd.commit()
                    copyCmd.waitUntilCompleted()
                }

                return true
            }

            // Allocate JetSpec Staging Buffers for Tree Drafting & Verification
            // Speculative tree forward pass is enabled for resident in-memory standard attention models.
            // Disk-streamed MoE models (packedExpertsDir != nil) and Linear Recurrence / Gated DeltaNet
            // hybrid models (maintaining continuous causal convolution and O(1) recurrent states)
            // execute via the direct single-token path for maximum throughput and state integrity.
            let hasLinearRecurrence = cachedLayers.contains { $0.attentionType == .linearAttention }
            let effectiveJetSpec = jetSpecEnabled && !hasLinearRecurrence
            let effectiveInterDim = modelConfig?.intermediateSize ?? Int(cachedLayers.first?.intermediateDim ?? 14336)
            let effectiveQkvDim = Int(max(hiddenDim * 2, 8192))
            let effectiveZDim = Int(max(hiddenDim * 2, 8192))
            let effectiveTopK = max(1, Int(modelConfig?.effectiveNumExpertsPerTok ?? 8))
            let linLayers = cachedLayers.filter { $0.attentionType == .linearAttention }.count
            let linValHeads = max(32, modelConfig?.effectiveLinearNumValueHeads ?? 32)
            let jetspecStaging = effectiveJetSpec ? InferenceEngine.shared.allocateJetSpecBuffers(
                device: device,
                maxNodes: 8,
                vocabSize: Int(vocabSize),
                hiddenDim: Int(hiddenDim),
                intermediateDim: effectiveInterDim,
                maxQkvDim: effectiveQkvDim,
                maxZDim: effectiveZDim,
                kvStride: Int(kvStride),
                topK: effectiveTopK,
                maxLinearLayers: max(1, linLayers),
                linValHeads: linValHeads
            ) : nil

            // Multi-Node Parallel Tree Forward Pass (Target Backbone Execution)
            func runJetSpecTreeForward(treeMask: JetSpecTreeMask, nodeScores: [Float], step: UInt32) -> Bool {
                guard let jb = jetspecStaging else { return false }
                let N = Int(treeMask.nodeCount)
                if N == 0 { return false }
                let maxTreeDepth = Int(treeMask.depths.max() ?? 0)

                let getRmsNormPipe = { (dtype: String) -> MTLComputePipelineState in
                    let isF16 = (dtype.contains("F16") || dtype.contains("HALF") || dtype.contains("FLOAT16")) && !dtype.contains("BF16") && !dtype.contains("BFLOAT")
                    if isRMSNormOffset {
                        return (isF16 && rmsnormOffsetF16Pipeline != nil) ? rmsnormOffsetF16Pipeline! : (rmsnormOffsetPipeline ?? rmsnormPipeline)
                    } else {
                        return (isF16 && rmsnormF16Pipeline != nil) ? rmsnormF16Pipeline! : rmsnormPipeline
                    }
                }

                // 1. Stage Tree Metadata in JetSpec buffers
                let maskPtr = jb.treeMaskBuffer.contents().bindMemory(to: Float.self, capacity: N * N)
                for i in 0..<(N * N) {
                    maskPtr[i] = i < treeMask.mask.count ? treeMask.mask[i] : -1e9
                }

                let depthsPtr = jb.depthsBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    depthsPtr[i] = i < treeMask.depths.count ? treeMask.depths[i] : 0
                }

                let parentsPtr = jb.parentIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    parentsPtr[i] = i < treeMask.parentIndices.count ? treeMask.parentIndices[i] : 0
                }

                let tokensPtr = jb.candidateTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: N)
                for i in 0..<N {
                    tokensPtr[i] = i < treeMask.tokenIds.count ? treeMask.tokenIds[i] : 0
                }

                guard var activeCmd = commandQueue.makeCommandBuffer() else { return false }

                // 2. Embed all N candidate tree tokens into jb.treeHiddenBuffer
                guard let embedEnc = activeCmd.makeComputeCommandEncoder() else { return false }
                if let rawBaseBuffer = buffers[embedWeight.shardIndex] {
                    var baseOff = embedWeight.offsetStart
                    var hDim = hiddenDim
                    let hasEmbedScale = (embedScale != nil)
                    let hasEmbedBias = (embedBias != nil)
                    let isEmbedMXFP8 = hasEmbedScale && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8") || (!hasEmbedBias && (embedWeight.dtype.contains("U32") || embedWeight.dtype.contains("U8") || embedWeight.dtype.contains("FP8")) && (embedScale!.offsetEnd - embedScale!.offsetStart) >= UInt64(hiddenDim / 32)))
                    let isEmbedAffine = (hasEmbedBias || embedWeight.dtype.contains("Q4") || embedWeight.dtype.contains("Q8") || hasEmbedScale) && !isEmbedMXFP8

                    if isEmbedMXFP8, let embedMXFP8Pipe = embedMXFP8Pipeline,
                       let embedScaleRaw = buffers[embedScale!.shardIndex] {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        embedEnc.setComputePipelineState(embedMXFP8Pipe)
                        embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(jb.candidateTokensBuffer, offset: 0, index: 1)
                        embedEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                        embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 3)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                        embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 6)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXFP8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    } else if isEmbedAffine,
                              let embedScaleRaw = (embedScale != nil) ? buffers[embedScale!.shardIndex] : nil,
                              let embedBiasRaw = (embedBias != nil) ? buffers[embedBias!.shardIndex] : nil {
                        var wOffset = embedOffset
                        var sOffset = embedScale!.offsetStart
                        var bOffset = embedBias!.offsetStart
                        var grpSize: UInt32 = 64
                        let is8Bit = (embedWeight.offsetEnd - embedWeight.offsetStart) >= (UInt64(vocabSize) * UInt64(hiddenDim) * 3) / 4

                        for n in 0..<N {
                            var tok = treeMask.tokenIds[n]
                            let outOffset = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                            if is8Bit, let embedQ8Pipe = embedQ8Pipeline {
                                embedEnc.setComputePipelineState(embedQ8Pipe)
                                embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                                embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                embedEnc.setBuffer(jb.treeHiddenBuffer, offset: outOffset, index: 3)
                                embedEnc.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ8Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            } else if let embedQ4Pipe = embedQ4Pipeline {
                                embedEnc.setComputePipelineState(embedQ4Pipe)
                                embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                                embedEnc.setBuffer(embedScaleRaw, offset: 0, index: 1)
                                embedEnc.setBuffer(embedBiasRaw, offset: 0, index: 2)
                                embedEnc.setBuffer(jb.treeHiddenBuffer, offset: outOffset, index: 3)
                                embedEnc.setBytes(&tok, length: MemoryLayout<UInt32>.stride, index: 4)
                                embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                embedEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                embedEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 8)
                                embedEnc.setBytes(&grpSize, length: MemoryLayout<UInt32>.stride, index: 9)
                                embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedQ4Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            }
                        }
                    } else {
                        let embedPipe = embedPipeline
                        var wOffset = embedOffset
                        var tokCount: UInt32 = UInt32(N)
                        embedEnc.setComputePipelineState(embedPipe)
                        embedEnc.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                        embedEnc.setBuffer(jb.candidateTokensBuffer, offset: 0, index: 1)
                        embedEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                        embedEnc.setBytes(&wOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                        embedEnc.setBytes(&hDim, length: MemoryLayout<UInt32>.stride, index: 4)
                        embedEnc.setBytes(&tokCount, length: MemoryLayout<UInt32>.stride, index: 5)
                        embedEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    }
                }
                embedEnc.endEncoding()

                // 3. Forward Pass through Backbone Layers
                for loopIdx in 0..<totalLoops {
                    for l in 0..<actualLayers {
                        if Task.isCancelled { return false }
                        let layer = cachedLayers[l]
                        guard let layerEnc = activeCmd.makeComputeCommandEncoder() else { return false }

                        // Step A: Pre-Attention RMSNorm 1
                        if let norm1 = layer.norm1Tensor, let normRaw = buffers[norm1.shardIndex] {
                            let rmsPipe = getRmsNormPipe(norm1.dtype)
                            var nOff = norm1.offsetStart
                            var hD = hiddenDim
                            var epsVal = eps
                            layerEnc.setComputePipelineState(rmsPipe)
                            layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(normRaw, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeXNorm1Buffer, offset: 0, index: 2)
                            layerEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Step B: Attention (Full Attention or Linear Attention)
                        if layer.attentionType == .fullAttention {
                            let currentQDim = isStandardGqa ? (numHeads * headDim) : (numHeads * headDim * 2)
                            let currentKvDim = numKvHeads * headDim
                            let attnCtxDim = numHeads * headDim

                            // Q, K, V Projections for all N candidate tree nodes
                            dispatchLinear(enc: layerEnc, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeQGateBuffer, inDim: hiddenDim, outDim: currentQDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeKVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeVVectorBuffer, inDim: hiddenDim, outDim: currentKvDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            // Q / K Normalization if present
                            if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex],
                               let headNormPipe = headRmsnormPipeline {
                                var qNormOff = qNorm.offsetStart
                                var nQ = numHeads
                                var hD = headDim
                                var hStride = isStandardGqa ? headDim : (headDim * 2)
                                var epsVal = eps
                                layerEnc.setComputePipelineState(headNormPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(qNormRaw, offset: 0, index: 1)
                                layerEnc.setBytes(&qNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }
                            if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex],
                               let headNormPipe = headRmsnormPipeline {
                                var kNormOff = kNorm.offsetStart
                                var nK = numKvHeads
                                var hD = headDim
                                var hStride = headDim
                                var epsVal = eps
                                layerEnc.setComputePipelineState(headNormPipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kNormRaw, offset: 0, index: 1)
                                layerEnc.setBytes(&kNormOff, length: MemoryLayout<UInt64>.stride, index: 2)
                                layerEnc.setBytes(&nK, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&hStride, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 6)
                                layerEnc.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Tree-Aware RoPE Rotation: effectivePos = tokenPos + nodeDepths[nodeIdx]
                            if let ropePipe = InferenceEngine.shared.applyRopeTreePipeline {
                                var tPos = step
                                var rDim = rotaryDim
                                var hDimVal = headDim
                                var thVal = thetaVal
                                var qStr = isStandardGqa ? headDim : (headDim * 2)
                                var numNodesVal = UInt32(N)

                                // Rotate Q
                                layerEnc.setComputePipelineState(ropePipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 1)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc.setBytes(&rDim, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&thVal, length: MemoryLayout<Float>.stride, index: 5)
                                layerEnc.setBytes(&qStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                                // Rotate K
                                var kStr = headDim
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBytes(&kStr, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numKvHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Store K and V vectors into KV Cache at slots [step ..< step + N]
                            let slotIndex = (loopIdx * actualLayers) + layer.fullAttnIndex
                            let maxSeq = KVCacheManager.shared.allocatedSeqLen
                            let prec = KVCacheManager.shared.activePrecision

                            if prec == .fp16, let storePipe = storeKvCacheF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float16>.stride
                                var tPos = step
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(storePipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(kvStride), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            } else if let storePipe = storeKvCachePipeline,
                                      let kCache = KVCacheManager.shared.kCacheBuffer,
                                      let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float>.stride
                                var tPos = step
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(storePipe)
                                layerEnc.setBuffer(jb.treeKVectorBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(jb.treeVVectorBuffer, offset: 0, index: 1)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                                layerEnc.setBytes(&tPos, length: MemoryLayout<UInt32>.stride, index: 4)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.dispatchThreads(MTLSize(width: Int(kvStride), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Grouped-Query Attention Tree Causal Verification
                            if prec == .fp16, let treeAttnPipe = InferenceEngine.shared.gqaAttentionTreeVerifyStandardF16Pipeline,
                               let kCache = KVCacheManager.shared.kCacheBuffer,
                               let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float16>.stride
                                var pLen = step
                                var numNodesVal = UInt32(N)
                                var nQ = numHeads
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(treeAttnPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(jb.treeMaskBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 4)
                                layerEnc.setBytes(&pLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), treeAttnPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            } else if let treeAttnPipe = InferenceEngine.shared.gqaAttentionTreeVerifyStandardPipeline,
                                      let kCache = KVCacheManager.shared.kCacheBuffer,
                                      let vCache = KVCacheManager.shared.vCacheBuffer {
                                let layerByteOffset = slotIndex * maxSeq * Int(kvStride) * MemoryLayout<Float>.stride
                                var pLen = step
                                var numNodesVal = UInt32(N)
                                var nQ = numHeads
                                var nKv = numKvHeads
                                var hD = headDim

                                layerEnc.setComputePipelineState(treeAttnPipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                                layerEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                                layerEnc.setBuffer(jb.treeMaskBuffer, offset: 0, index: 3)
                                layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 4)
                                layerEnc.setBytes(&pLen, length: MemoryLayout<UInt32>.stride, index: 5)
                                layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                layerEnc.setBytes(&nQ, length: MemoryLayout<UInt32>.stride, index: 7)
                                layerEnc.setBytes(&nKv, length: MemoryLayout<UInt32>.stride, index: 8)
                                layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 9)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), treeAttnPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Attention Output Projection (o_proj)
                            dispatchLinear(enc: layerEnc, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: jb.treeAttnCtxBuffer, outBuf: jb.treeAttnOutBuffer, inDim: attnCtxDim, outDim: hiddenDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)
                        } else {
                            // Linear Attention (GatedDeltaNet Recurrent State)
                            let linValHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumValueHeads ?? (layer.inProjA != nil && (layer.inProjA!.offsetEnd - layer.inProjA!.offsetStart) > 32 * 2560 ? 48 : 32))
                            let linKeyHeads: UInt32 = UInt32(modelConfig?.effectiveLinearNumKeyHeads ?? 16)
                            let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128
                            let zDim: UInt32 = linValHeads * 128
                            let aDim: UInt32 = linValHeads
                            let bDim: UInt32 = linValHeads

                            dispatchLinear(enc: layerEnc, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeQGateBuffer, inDim: hiddenDim, outDim: qkvDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            if let l2Pipe = l2NormQkSeqPipeline ?? l2NormQkPipeline {
                                var numHeads: UInt32 = linKeyHeads
                                var headDimVal: UInt32 = 128
                                var qkvDimVal: UInt32 = qkvDim
                                layerEnc.setComputePipelineState(l2Pipe)
                                layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                layerEnc.setBytes(&numHeads, length: MemoryLayout<UInt32>.stride, index: 1)
                                layerEnc.setBytes(&headDimVal, length: MemoryLayout<UInt32>.stride, index: 2)
                                layerEnc.setBytes(&qkvDimVal, length: MemoryLayout<UInt32>.stride, index: 3)
                                layerEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            dispatchLinear(enc: layerEnc, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeZGateBuffer, inDim: hiddenDim, outDim: zDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeAVectorBuffer, inDim: hiddenDim, outDim: aDim, batchSize: N)
                            dispatchLinear(enc: layerEnc, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.treeBVectorBuffer, inDim: hiddenDim, outDim: bDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)

                            if let sBuf = KVCacheManager.shared.linearStateBuffer,
                               let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                               let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                               let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                                let linIdx = layer.linAttnIndex
                                let stateFloatsPerNode = Int(linValHeads * 128 * 128)
                                let baseStateByteOffset = linIdx * stateFloatsPerNode * MemoryLayout<Float>.stride
                                let layerOutStateByteOffset = linIdx * jb.maxNodes * stateFloatsPerNode * MemoryLayout<Float>.stride
                                var aLogOff = aLog.offsetStart
                                var dtBiasOff = dtBias.offsetStart
                                var linNormOff = linNorm.offsetStart
                                var numValHeads: UInt32 = linValHeads
                                var numKeyHeads: UInt32 = linKeyHeads
                                var headDim: UInt32 = 128
                                var epsVal = eps
                                var numNodesVal = UInt32(N)
                                var stateSizeFloats = UInt32(stateFloatsPerNode)

                                for d in 0...maxTreeDepth {
                                    var dVal = UInt32(d)
                                    // 1. Gather parent states for candidate nodes at depth d
                                    if let gatherPipe = InferenceEngine.shared.gatherGdnTreeParentStatesPipeline {
                                        layerEnc.setComputePipelineState(gatherPipe)
                                        layerEnc.setBuffer(sBuf, offset: baseStateByteOffset, index: 0)
                                        layerEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: layerOutStateByteOffset, index: 1)
                                        layerEnc.setBuffer(jb.treeGdnParentStateBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.parentIndicesBuffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 4)
                                        layerEnc.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&numNodesVal, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&stateSizeFloats, length: MemoryLayout<UInt32>.stride, index: 7)
                                        let vec4Count = Int(stateSizeFloats / 4)
                                        layerEnc.dispatchThreads(MTLSize(width: vec4Count, height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatherPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                        layerEnc.memoryBarrier(scope: .buffers)
                                    }

                                    // 2. Compute tree step for candidate nodes at depth d
                                    let useSigmoidGate = (modelConfig?.effectiveOutputGateType.lowercased() == "sigmoid")
                                    let selectedTreeStepPipe = useSigmoidGate
                                        ? (InferenceEngine.shared.gdnLinearAttnTreeStepSigmoidPipeline ?? InferenceEngine.shared.gdnLinearAttnTreeStepPipeline)
                                        : InferenceEngine.shared.gdnLinearAttnTreeStepPipeline

                                    if let treeStepPipe = selectedTreeStepPipe {
                                        layerEnc.setComputePipelineState(treeStepPipe)
                                        layerEnc.setBuffer(jb.treeQGateBuffer, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeZGateBuffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeAVectorBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeBVectorBuffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(aLogRaw, offset: 0, index: 4)
                                        layerEnc.setBuffer(dtBiasRaw, offset: 0, index: 5)
                                        layerEnc.setBuffer(linNormRaw, offset: 0, index: 6)
                                        layerEnc.setBuffer(jb.treeGdnParentStateBuffer, offset: 0, index: 7)
                                        layerEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: layerOutStateByteOffset, index: 8)
                                        layerEnc.setBuffer(jb.treeAttnCtxBuffer, offset: 0, index: 9)
                                        layerEnc.setBytes(&aLogOff, length: MemoryLayout<UInt64>.stride, index: 10)
                                        layerEnc.setBytes(&dtBiasOff, length: MemoryLayout<UInt64>.stride, index: 11)
                                        layerEnc.setBytes(&linNormOff, length: MemoryLayout<UInt64>.stride, index: 12)
                                        layerEnc.setBytes(&numValHeads, length: MemoryLayout<UInt32>.stride, index: 13)
                                        layerEnc.setBytes(&numKeyHeads, length: MemoryLayout<UInt32>.stride, index: 14)
                                        layerEnc.setBytes(&headDim, length: MemoryLayout<UInt32>.stride, index: 15)
                                        layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 16)
                                        layerEnc.setBytes(&dVal, length: MemoryLayout<UInt32>.stride, index: 17)
                                        layerEnc.setBuffer(jb.depthsBuffer, offset: 0, index: 18)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: Int(linValHeads), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                        layerEnc.memoryBarrier(scope: .buffers)
                                    }
                                }
                            }

                            let linAttnCtxDim = linValHeads * 128
                            dispatchLinear(enc: layerEnc, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: jb.treeAttnCtxBuffer, outBuf: jb.treeAttnOutBuffer, inDim: linAttnCtxDim, outDim: hiddenDim, batchSize: N)
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Residual 1: treeHMidBuffer = treeHiddenBuffer + treeAttnOutBuffer
                        let addPipe = addPipeline
                        layerEnc.setComputePipelineState(addPipe)
                        layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                        layerEnc.setBuffer(jb.treeAttnOutBuffer, offset: 0, index: 1)
                        layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 2)
                        var hD = hiddenDim
                        layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 3)
                        layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        layerEnc.memoryBarrier(scope: .buffers)

                        // Step C: Post-Attention RMSNorm 2
                        if let norm2 = layer.norm2Tensor, let normRaw = buffers[norm2.shardIndex] {
                            let rmsPipe = getRmsNormPipe(norm2.dtype)
                            var nOff = norm2.offsetStart
                            var hD = hiddenDim
                            var epsVal = eps
                            layerEnc.setComputePipelineState(rmsPipe)
                            layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(normRaw, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 2)
                            layerEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                            layerEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                            layerEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                            layerEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                            layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)
                        }

                        // Step D: Feed-Forward MLP (Dense or MoE)
                        layerEnc.setComputePipelineState(clearPipeline)
                        layerEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 0)
                        layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), clearPipeline.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        layerEnc.memoryBarrier(scope: .buffers)

                        if layer.mlpType == .denseMlp || layer.routerTensor == nil {
                            if let gateW = layer.denseGateWeight,
                               let upW = layer.denseUpWeight,
                               let downW = layer.denseDownWeight {
                                for n in 0..<N {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    dispatchExpertMlp(
                                        enc: layerEnc,
                                        gateW: gateW,
                                        gateS: layer.denseGateScale,
                                        gateB: layer.denseGateBias,
                                        upW: upW,
                                        upS: layer.denseUpScale,
                                        upB: layer.denseUpBias,
                                        downW: downW,
                                        downS: layer.denseDownScale,
                                        downB: layer.denseDownBias,
                                        inBuf: jb.treeXNorm2Buffer,
                                        interBuf: jb.treeInterBuffer,
                                        accumBuf: jb.treeHMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: layer.intermediateDim,
                                        routingWeight: 1.0,
                                        inOffset: inOff,
                                        interOffset: interOff,
                                        accumOffset: accumOff
                                    )
                                }
                                layerEnc.memoryBarrier(scope: .buffers)
                            }

                            // Residual 2: treeHiddenBuffer = treeHMidBuffer + treeHMlpBuffer
                            layerEnc.setComputePipelineState(addPipe)
                            layerEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            layerEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 1)
                            layerEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                            var hD2 = hiddenDim
                            layerEnc.setBytes(&hD2, length: MemoryLayout<UInt32>.stride, index: 3)
                            layerEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            layerEnc.memoryBarrier(scope: .buffers)

                            layerEnc.endEncoding()
                        } else {
                            // MoE Expert Evaluation across candidate tree nodes using real router pre-pass
                            let topKCount = Int(modelConfig?.effectiveNumExpertsPerTok ?? (numExperts >= 512 ? 10 : 8))

                            if let routerTensor = layer.routerTensor, let routerRaw = buffers[routerTensor.shardIndex] {
                                var rOffset = routerTensor.offsetStart
                                var nExp = numExperts
                                var kVal: UInt32 = UInt32(topKCount)
                                var grp: UInt32 = 64
                                var hDimVal = hiddenDim

                                if let rScale = layer.routerScale, let rBias = layer.routerBias,
                                   let rScaleRaw = buffers[rScale.shardIndex], let rBiasRaw = buffers[rBias.shardIndex] {
                                    var sOffset = rScale.offsetStart
                                    var bOffset = rBias.offsetStart

                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let rQ8 = routerQ8Pipeline {
                                        layerEnc.setComputePipelineState(rQ8)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 4)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 5)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 512, height: 1, depth: 1))
                                    } else if let rQ4 = routerQ4Pipeline {
                                        layerEnc.setComputePipelineState(rQ4)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(rScaleRaw, offset: 0, index: 1)
                                        layerEnc.setBuffer(rBiasRaw, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 3)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 4)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 5)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                        layerEnc.setBytes(&sOffset, length: MemoryLayout<UInt64>.stride, index: 7)
                                        layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 8)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 10)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 11)
                                        layerEnc.setBytes(&grp, length: MemoryLayout<UInt32>.stride, index: 12)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                } else {
                                    if (routerTensor.shapeDisplay.contains("512") || numExperts >= 512), let r512 = router512Pipeline {
                                        layerEnc.setComputePipelineState(r512)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 3)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                                    } else if let rBias = layer.routerExpertBias, let rBiasRaw = buffers[rBias.shardIndex],
                                               let bailingPipe = routerBailingPipeline {
                                        var bOffset = rBias.offsetStart
                                        var nGrp: UInt32 = UInt32(modelConfig?.nGroup ?? 8)
                                        var topkGrp: UInt32 = UInt32(modelConfig?.topkGroup ?? 4)
                                        var scaleFactor: Float = Float(modelConfig?.routedScalingFactor ?? 2.5)

                                        for n in 0..<N {
                                            let tokenOffset = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                            let rIndexOffset = n * topKCount * MemoryLayout<UInt32>.stride
                                            let rWeightOffset = n * topKCount * MemoryLayout<Float>.stride
                                            layerEnc.setComputePipelineState(bailingPipe)
                                            layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                            layerEnc.setBuffer(rBiasRaw, offset: 0, index: 1)
                                            layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: tokenOffset, index: 2)
                                            layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: rIndexOffset, index: 3)
                                            layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: rWeightOffset, index: 4)
                                            layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 5)
                                            layerEnc.setBytes(&bOffset, length: MemoryLayout<UInt64>.stride, index: 6)
                                            layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                            layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 8)
                                            layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 9)
                                            layerEnc.setBytes(&nGrp, length: MemoryLayout<UInt32>.stride, index: 10)
                                            layerEnc.setBytes(&topkGrp, length: MemoryLayout<UInt32>.stride, index: 11)
                                            layerEnc.setBytes(&scaleFactor, length: MemoryLayout<Float>.stride, index: 12)
                                            layerEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                        }
                                    } else {
                                        layerEnc.setComputePipelineState(routerPipeline)
                                        layerEnc.setBuffer(routerRaw, offset: 0, index: 0)
                                        layerEnc.setBuffer(jb.treeXNorm2Buffer, offset: 0, index: 1)
                                        layerEnc.setBuffer(jb.treeRouterIndicesBuffer, offset: 0, index: 2)
                                        layerEnc.setBuffer(jb.treeRouterWeightsBuffer, offset: 0, index: 3)
                                        layerEnc.setBytes(&rOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                                        layerEnc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
                                        layerEnc.setBytes(&nExp, length: MemoryLayout<UInt32>.stride, index: 6)
                                        layerEnc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
                                        layerEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
                                    }
                                }
                            }

                            layerEnc.endEncoding()
                            activeCmd.commit()
                            activeCmd.waitUntilCompleted()

                            let indPtr = jb.treeRouterIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N * topKCount)
                            let wPtr = jb.treeRouterWeightsBuffer.contents().bindMemory(to: Float.self, capacity: N * topKCount)

                            var candidateExpertsPerNode: [[(id: Int, weight: Float)]] = []
                            var uniqueActiveExperts = Set<UInt32>()
                            var flatCandidateExperts: [UInt32] = []

                            for n in 0..<N {
                                var nodeExp: [(id: Int, weight: Float)] = []
                                for k in 0..<topKCount {
                                    let expId = indPtr[n * topKCount + k]
                                    let w = wPtr[n * topKCount + k]
                                    flatCandidateExperts.append(expId)
                                    if w > 0.00001 {
                                        nodeExp.append((id: Int(expId), weight: w))
                                        uniqueActiveExperts.insert(expId)
                                    }
                                }
                                candidateExpertsPerNode.append(nodeExp)
                            }

                            var activeNodeIndices = Array(0..<N)
                            if uniqueActiveExperts.count > jetSpecMaxExpertCap && N > 1 {
                                let prunedMask = pruneJetspecTreeMoe(
                                    treeTokens: treeMask.tokenIds,
                                    parentIndices: treeMask.parentIndices,
                                    draftScores: nodeScores,
                                    candidateExpertsFlat: flatCandidateExperts,
                                    expertsPerNode: UInt32(topKCount),
                                    maxUniqueExperts: UInt32(jetSpecMaxExpertCap)
                                )
                                var keptIndices: [Int] = []
                                var searchStart = 0
                                for tok in prunedMask.tokenIds {
                                    for origIdx in searchStart..<N {
                                        if treeMask.tokenIds[origIdx] == tok {
                                            keptIndices.append(origIdx)
                                            searchStart = origIdx + 1
                                            break
                                        }
                                    }
                                }
                                if !keptIndices.isEmpty {
                                    activeNodeIndices = keptIndices
                                }
                            }

                            let prunedUniqueExperts = Array(Set(activeNodeIndices.flatMap { n in candidateExpertsPerNode[n].map { $0.id } })).sorted()
                            let expertSize = Int(loadedLayout?.expert_size ?? 1769472)

                            if let packedDir = packedExpertsDir,
                               let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) {
                                var tasks: [ExpertPreadTask] = []
                                let rawStagingPtr = expertStagingBuffer.contents()
                                for (slot, expId) in prunedUniqueExperts.enumerated() {
                                    let offset = off_t(expId * expertSize)
                                    let dst = rawStagingPtr.advanced(by: slot * expertSize)
                                    tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                                }
                                if !tasks.isEmpty {
                                    ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)
                                }
                            }

                            guard let nextCmd = commandQueue.makeCommandBuffer(),
                                  let moeEnc = nextCmd.makeComputeCommandEncoder() else { return false }
                            activeCmd = nextCmd

                            if packedExpertsDir != nil {
                                let isFP8Layout = loadedLayout?.components.contains { $0.dtype.contains("F8") || $0.name.contains("weight_scale") } ?? false

                                let compGateW = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compGateS = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compGateB = loadedLayout?.components.first(where: { $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compUpW = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compUpS = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compUpB = loadedLayout?.components.first(where: { $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let compDownW = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") && !$0.name.contains("scale") && !$0.name.contains("bias") })
                                let compDownS = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
                                let compDownB = loadedLayout?.components.first(where: { $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

                                let isQuantizedAffine = (compGateB != nil || compGateW?.name.contains("Q4") == true || compGateW?.name.contains("Q8") == true || compGateW?.dtype.contains("Q4") == true || compGateW?.dtype.contains("Q8") == true || (compGateS != nil && !isFP8Layout))

                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride

                                    for expert in candidateExpertsPerNode[n] {
                                        let expId = expert.id
                                        let p_k = expert.weight
                                        if p_k <= 0.00001 { continue }
                                        guard let slot = prunedUniqueExperts.firstIndex(of: expId) else { continue }

                                        let slotOffset = UInt64(slot * expertSize)
                                        let gWOff = slotOffset + (compGateW?.offset ?? 0)
                                        let gSOff = slotOffset + (compGateS?.offset ?? 524288)
                                        let gBOff = slotOffset + (compGateB?.offset ?? 557056)
                                        let uWOff = slotOffset + (compUpW?.offset ?? 589824)
                                        let uSOff = slotOffset + (compUpS?.offset ?? 1114112)
                                        let uBOff = slotOffset + (compUpB?.offset ?? 1146880)
                                        let dWOff = slotOffset + (compDownW?.offset ?? 1179648)
                                        let dSOff = slotOffset + (compDownS?.offset ?? 1703936)
                                        let dBOff = slotOffset + (compDownB?.offset ?? 1736704)

                                        if isFP8Layout {
                                            let isBlockScale = (compGateS?.size ?? 1024) < (layer.intermediateDim * 2) || (compGateS?.name.contains("scale_inv") ?? false)
                                            let gateSimd = isBlockScale ? (fp8BlockGateUpSimdPipeline ?? fp8GateUpSimdPipeline ?? fp8GateUpPipeline) : (fp8GateUpSimdPipeline ?? fp8GateUpPipeline)
                                            let downSimd = isBlockScale ? (fp8BlockDownSimdPipeline ?? fp8DownSimdPipeline ?? fp8DownPipeline) : (fp8DownSimdPipeline ?? fp8DownPipeline)

                                            if let gateSimd = gateSimd, let downSimd = downSimd {
                                                var gWOffU = gWOff
                                                var gSOffU = gSOff
                                                var uWOffU = uWOff
                                                var uSOffU = uSOff
                                                var dWOffU = dWOff
                                                var dSOffU = dSOff
                                                var hDimVal: UInt32 = UInt32(hiddenDim)
                                                var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                                var pkVal = p_k

                                                moeEnc.setComputePipelineState(gateSimd)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                                moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                                moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                                moeEnc.setBytes(&gWOffU, length: 8, index: 6)
                                                moeEnc.setBytes(&gSOffU, length: 8, index: 7)
                                                moeEnc.setBytes(&uWOffU, length: 8, index: 8)
                                                moeEnc.setBytes(&uSOffU, length: 8, index: 9)
                                                moeEnc.setBytes(&hDimVal, length: 4, index: 10)
                                                moeEnc.setBytes(&interDimVal, length: 4, index: 11)
                                                moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                moeEnc.memoryBarrier(scope: .buffers)

                                                moeEnc.setComputePipelineState(downSimd)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                                moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                                moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                                moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                                moeEnc.setBytes(&dWOffU, length: 8, index: 4)
                                                moeEnc.setBytes(&dSOffU, length: 8, index: 5)
                                                moeEnc.setBytes(&interDimVal, length: 4, index: 6)
                                                moeEnc.setBytes(&hDimVal, length: 4, index: 7)
                                                moeEnc.setBytes(&pkVal, length: 4, index: 8)
                                                moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                                moeEnc.memoryBarrier(scope: .buffers)
                                            }
                                        } else if isQuantizedAffine, let q4GatePipe = q4GateUpPipeline, let q4DownPipe = q4DownPipeline {
                                            var gWOffU = gWOff
                                            var gSOffU = gSOff
                                            var gBOffU = gBOff
                                            var uWOffU = uWOff
                                            var uSOffU = uSOff
                                            var uBOffU = uBOff
                                            var dWOffU = dWOff
                                            var dSOffU = dSOff
                                            var dBOffU = dBOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var grp: UInt32 = 64
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(q4GatePipe)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 6)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 7)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 8)
                                            moeEnc.setBytes(&gSOffU, length: 8, index: 9)
                                            moeEnc.setBytes(&gBOffU, length: 8, index: 10)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 11)
                                            moeEnc.setBytes(&uSOffU, length: 8, index: 12)
                                            moeEnc.setBytes(&uBOffU, length: 8, index: 13)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 14)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 15)
                                            moeEnc.setBytes(&grp, length: 4, index: 16)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(q4DownPipe)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 2)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 4)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 5)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 6)
                                            moeEnc.setBytes(&dSOffU, length: 8, index: 7)
                                            moeEnc.setBytes(&dBOffU, length: 8, index: 8)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 9)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 10)
                                            moeEnc.setBytes(&grp, length: 4, index: 11)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 12)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        } else if let gateSimd = bf16GateUpSimdPipeline, let downSimd = bf16DownSimdPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(gateSimd)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 4)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 5)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 6)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 7)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(downSimd)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 3)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 4)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 5)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 6)
                                            moeEnc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        } else if let gateUnq = bf16GateUpPipeline, let downUnq = bf16DownPipeline {
                                            var gWOffU = gWOff
                                            var uWOffU = uWOff
                                            var dWOffU = dWOff
                                            var hDimVal: UInt32 = UInt32(hiddenDim)
                                            var interDimVal: UInt32 = UInt32(layer.intermediateDim)
                                            var pkVal = p_k

                                            moeEnc.setComputePipelineState(gateUnq)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                                            moeEnc.setBuffer(jb.treeXNorm2Buffer, offset: inOff, index: 2)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 3)
                                            moeEnc.setBytes(&gWOffU, length: 8, index: 4)
                                            moeEnc.setBytes(&uWOffU, length: 8, index: 5)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 6)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 7)
                                            moeEnc.dispatchThreads(MTLSize(width: Int(layer.intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(layer.intermediateDim), gateUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)

                                            moeEnc.setComputePipelineState(downUnq)
                                            moeEnc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                                            moeEnc.setBuffer(jb.treeInterBuffer, offset: interOff, index: 1)
                                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: accumOff, index: 2)
                                            moeEnc.setBytes(&dWOffU, length: 8, index: 3)
                                            moeEnc.setBytes(&interDimVal, length: 4, index: 4)
                                            moeEnc.setBytes(&hDimVal, length: 4, index: 5)
                                            moeEnc.setBytes(&pkVal, length: 4, index: 6)
                                            moeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), downUnq.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                                            moeEnc.memoryBarrier(scope: .buffers)
                                        }
                                    }
                                }
                            } else {
                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride

                                    for expert in candidateExpertsPerNode[n] {
                                        let expId = expert.id
                                        let p_k = expert.weight
                                        if p_k <= 0.00001 { continue }

                                        if let gW = layer.expertGateWeights[expId],
                                           let uW = layer.expertUpWeights[expId],
                                           let dW = layer.expertDownWeights[expId] {
                                            dispatchExpertMlp(
                                                enc: moeEnc,
                                                gateW: gW,
                                                gateS: layer.expertGateScales[expId],
                                                gateB: layer.expertGateBiases[expId],
                                                upW: uW,
                                                upS: layer.expertUpScales[expId],
                                                upB: layer.expertUpBiases[expId],
                                                downW: dW,
                                                downS: layer.expertDownScales[expId],
                                                downB: layer.expertDownBiases[expId],
                                                inBuf: jb.treeXNorm2Buffer,
                                                interBuf: jb.treeInterBuffer,
                                                accumBuf: jb.treeHMlpBuffer,
                                                inDim: hiddenDim,
                                                interDim: layer.intermediateDim,
                                                routingWeight: p_k,
                                                inOffset: inOff,
                                                interOffset: interOff,
                                                accumOffset: accumOff
                                            )
                                        }
                                    }
                                }
                            }
                            moeEnc.memoryBarrier(scope: .buffers)

                            if let gateW = layer.sharedGateWeight,
                               let upW = layer.sharedUpWeight,
                               let downW = layer.sharedDownWeight {
                                for n in activeNodeIndices {
                                    let inOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    let interOff = n * Int(layer.intermediateDim) * MemoryLayout<Float>.stride
                                    let accumOff = n * Int(hiddenDim) * MemoryLayout<Float>.stride
                                    dispatchExpertMlp(
                                        enc: moeEnc,
                                        gateW: gateW,
                                        gateS: layer.sharedGateScale,
                                        gateB: layer.sharedGateBias,
                                        upW: upW,
                                        upS: layer.sharedUpScale,
                                        upB: layer.sharedUpBias,
                                        downW: downW,
                                        downS: layer.sharedDownScale,
                                        downB: layer.sharedDownBias,
                                        inBuf: jb.treeXNorm2Buffer,
                                        interBuf: jb.treeInterBuffer,
                                        accumBuf: jb.treeHMlpBuffer,
                                        inDim: hiddenDim,
                                        interDim: layer.intermediateDim,
                                        routingWeight: 1.0,
                                        inOffset: inOff,
                                        interOffset: interOff,
                                        accumOffset: accumOff
                                    )
                                }
                                moeEnc.memoryBarrier(scope: .buffers)
                            }

                            // Residual 2: treeHiddenBuffer = treeHMidBuffer + treeHMlpBuffer
                            moeEnc.setComputePipelineState(addPipe)
                            moeEnc.setBuffer(jb.treeHMidBuffer, offset: 0, index: 0)
                            moeEnc.setBuffer(jb.treeHMlpBuffer, offset: 0, index: 1)
                            moeEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 2)
                            var hD2 = hiddenDim
                            moeEnc.setBytes(&hD2, length: MemoryLayout<UInt32>.stride, index: 3)
                            moeEnc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                            moeEnc.memoryBarrier(scope: .buffers)

                            moeEnc.endEncoding()
                        }
                    }
                }

                // 4. Final RMSNorm & LM Head Output
                guard let finalEnc = activeCmd.makeComputeCommandEncoder() else { return false }
                if let normTensor = normTensor, let normShardBuffer = normShardBuffer {
                    var nOff = normOffset
                    var epsVal = eps
                    var hD = hiddenDim
                    let finalNormPipe = getRmsNormPipe(normTensor.dtype)
                    finalEnc.setComputePipelineState(finalNormPipe)
                    finalEnc.setBuffer(jb.treeHiddenBuffer, offset: 0, index: 0)
                    finalEnc.setBuffer(normShardBuffer, offset: 0, index: 1)
                    finalEnc.setBuffer(jb.treeXNorm1Buffer, offset: 0, index: 2)
                    finalEnc.setBytes(&nOff, length: MemoryLayout<UInt64>.stride, index: 3)
                    finalEnc.setBytes(&hD, length: MemoryLayout<UInt32>.stride, index: 4)
                    finalEnc.setBytes(&epsVal, length: MemoryLayout<Float>.stride, index: 5)
                    finalEnc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
                    finalEnc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, Int(hiddenDim)), height: 1, depth: 1))
                    finalEnc.memoryBarrier(scope: .buffers)
                }

                // LM Head Projection across all N candidate tree nodes
                let lmHead = lmHeadTensor
                dispatchLinear(enc: finalEnc, weight: lmHead, scale: lmHeadScale, bias: lmHeadBias, inBuf: jb.treeXNorm1Buffer, outBuf: jb.targetLogitsBuffer, inDim: hiddenDim, outDim: vocabSize, batchSize: N)
                finalEnc.endEncoding()

                activeCmd.commit()
                activeCmd.waitUntilCompleted()

                return true
            }

            func runJetSpecTreeStep(rootToken: UInt32, step: UInt32, temperature: Float) -> (acceptedTokens: [UInt32], newStep: UInt32)? {
                guard effectiveJetSpec, let jb = jetspecStaging else { return nil }

                // 1. Obtain draft candidate proposals (Draft Head or N-gram Prompt Lookup + Top-K Logits fallback)
                var topDraftTokens: [UInt32] = []
                var topDraftScores: [Float] = []

                if let dhTensor = summary.tensors.first(where: { $0.name.contains("draft_head") || $0.name.contains("speculative_head") }),
                   let dhRaw = buffers[dhTensor.shardIndex],
                   let dhPipe = InferenceEngine.shared.jetDraftHeadPredictPipeline,
                   let cmd = commandQueue.makeCommandBuffer(),
                   let enc = cmd.makeComputeCommandEncoder() {
                    var hDimVal = hiddenDim
                    var vDimVal = vocabSize
                    enc.setComputePipelineState(dhPipe)
                    enc.setBuffer(hCurrBuffer, offset: 0, index: 0)
                    enc.setBuffer(dhRaw, offset: Int(dhTensor.offsetStart), index: 1)
                    enc.setBuffer(jb.draftLogitsBuffer, offset: 0, index: 2)
                    enc.setBuffer(jb.draftLogitsBuffer, offset: 0, index: 3)
                    enc.setBytes(&hDimVal, length: 4, index: 4)
                    enc.setBytes(&vDimVal, length: 4, index: 5)
                    enc.dispatchThreads(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, dhPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    enc.endEncoding()
                    cmd.commit()
                    cmd.waitUntilCompleted()

                    let dhLogitsPtr = jb.draftLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let kCount = min(8, Int(vocabSize))
                    var topCandidates: [(token: UInt32, score: Float)] = []
                    topCandidates.reserveCapacity(kCount + 1)
                    for v in 0..<Int(vocabSize) {
                        let score = dhLogitsPtr[v]
                        if topCandidates.count < kCount {
                            topCandidates.append((token: UInt32(v), score: score))
                            if topCandidates.count == kCount {
                                topCandidates.sort { $0.score > $1.score }
                            }
                        } else if score > topCandidates.last!.score {
                            topCandidates[kCount - 1] = (token: UInt32(v), score: score)
                            var idx = kCount - 1
                            while idx > 0 && topCandidates[idx].score > topCandidates[idx - 1].score {
                                topCandidates.swapAt(idx, idx - 1)
                                idx -= 1
                            }
                        }
                    }
                    let maxScore = topCandidates.map { $0.score }.max() ?? 0.0
                    let effTemp = max(0.01, temperature)
                    let expScores = topCandidates.map { exp(($0.score - maxScore) / effTemp) }
                    let sumExp = expScores.reduce(0.0, +)
                    for (idx, cand) in topCandidates.enumerated() {
                        topDraftTokens.append(cand.token)
                        let prob = sumExp > 0 ? (expScores[idx] / sumExp) : (1.0 / Float(topCandidates.count))
                        topDraftScores.append(prob)
                    }
                }

                let ctxLen = contextTokens.count
                if ctxLen >= 3 {
                    let ngramSize = min(3, ctxLen - 1)
                    let pattern = Array(contextTokens.suffix(ngramSize))
                    for i in stride(from: ctxLen - ngramSize - 1, through: 0, by: -1) {
                        if Array(contextTokens[i..<(i + ngramSize)]) == pattern {
                            let matchEnd = i + ngramSize
                            let numDraft = min(8, ctxLen - matchEnd)
                            for j in 0..<numDraft {
                                topDraftTokens.append(contextTokens[matchEnd + j])
                                topDraftScores.append(max(0.1, 1.0 - Float(j) * 0.1))
                            }
                            break
                        }
                    }
                }

                if topDraftTokens.isEmpty {
                    let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let kCount = min(8, Int(vocabSize))
                    var topCandidates: [(token: UInt32, score: Float)] = []
                    topCandidates.reserveCapacity(kCount + 1)

                    for v in 0..<Int(vocabSize) {
                        let score = logitsPtr[v]
                        if topCandidates.count < kCount {
                            topCandidates.append((token: UInt32(v), score: score))
                            if topCandidates.count == kCount {
                                topCandidates.sort { $0.score > $1.score }
                            }
                        } else if score > topCandidates.last!.score {
                            topCandidates[kCount - 1] = (token: UInt32(v), score: score)
                            var idx = kCount - 1
                            while idx > 0 && topCandidates[idx].score > topCandidates[idx - 1].score {
                                topCandidates.swapAt(idx, idx - 1)
                                idx -= 1
                            }
                        }
                    }

                    let maxScore = topCandidates.map { $0.score }.max() ?? 0.0
                    let effTemp = max(0.01, temperature)
                    let expScores = topCandidates.map { exp(($0.score - maxScore) / effTemp) }
                    let sumExp = expScores.reduce(0.0, +)
                    for (idx, cand) in topCandidates.enumerated() {
                        topDraftTokens.append(cand.token)
                        let prob = sumExp > 0 ? (expScores[idx] / sumExp) : (1.0 / Float(topCandidates.count))
                        topDraftScores.append(prob)
                    }
                }

                if topDraftTokens.isEmpty { return nil }

                // 2. Build Tree Topology using Rust Engine
                var treeMask = buildJetspecCandidateTree(
                    rootTokenId: rootToken,
                    draftTokens: topDraftTokens,
                    draftScores: topDraftScores,
                    depth: UInt32(jetSpecMaxDepth),
                    branchingFactor: UInt32(jetSpecBranchingFactor),
                    maxNodes: UInt32(jb.maxNodes)
                )

                let nodeCount = treeMask.nodeCount
                if nodeCount <= 1 { return nil }

                // 3. Multi-Node Parallel Tree Forward Pass with real MoE router pre-pass and expert budget enforcement
                var nodeScores: [Float] = [1.0]
                for s in topDraftScores {
                    nodeScores.append(s)
                }
                while nodeScores.count < Int(treeMask.nodeCount) {
                    nodeScores.append(0.5)
                }
                let ok = runJetSpecTreeForward(treeMask: treeMask, nodeScores: nodeScores, step: step)
                if !ok { return nil }

                // 5. Verify acceptance using Rust acceptance oracle (Greedy fast-path or Stochastic Sampling)
                let targetLogitsPtr = jb.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(treeMask.nodeCount) * Int(vocabSize))
                let treeTargetLogits = Array(UnsafeBufferPointer(start: targetLogitsPtr, count: Int(treeMask.nodeCount) * Int(vocabSize)))

                let acceptedResult: JetSpecAcceptedResult
                if temperature <= 0.01 {
                    acceptedResult = verifyJetspecTreeGreedy(
                        treeTokens: treeMask.tokenIds,
                        parentIndices: treeMask.parentIndices,
                        targetLogits: treeTargetLogits,
                        vocabSize: vocabSize
                    )
                } else {
                    let rngSeed = UInt64.random(in: 1...UInt64.max)
                    acceptedResult = verifyJetspecTreeSampling(
                        treeTokens: treeMask.tokenIds,
                        parentIndices: treeMask.parentIndices,
                        draftProbs: [],
                        targetLogits: treeTargetLogits,
                        vocabSize: vocabSize,
                        temperature: temperature,
                        rngSeed: rngSeed
                    )
                }

                // 6. Speculative KV Cache Compaction for accepted branch
                if acceptedResult.acceptedCount > 0,
                   let kCache = KVCacheManager.shared.kCacheBuffer,
                   let vCache = KVCacheManager.shared.vCacheBuffer {
                    let maxSeq = KVCacheManager.shared.allocatedSeqLen
                    let prec = KVCacheManager.shared.activePrecision
                    let slotByteSize = Int(kvStride) * prec.bytesPerElement

                    var needsCompaction = false
                    for (i, nodeIdx) in acceptedResult.acceptedNodeIndices.enumerated() {
                        let destSlot = UInt32(i + 1)
                        if nodeIdx != destSlot {
                            needsCompaction = true
                            break
                        }
                    }

                    if needsCompaction,
                       let copyCmd = commandQueue.makeCommandBuffer(),
                       let blit = copyCmd.makeBlitCommandEncoder() {
                        for loopIdx in 0..<totalLoops {
                            for l in 0..<actualLayers {
                                let layer = cachedLayers[l]
                                if layer.attentionType == .fullAttention {
                                    let slot = (loopIdx * actualLayers) + layer.fullAttnIndex
                                    let layerByteOffset = slot * maxSeq * Int(kvStride) * prec.bytesPerElement
                                    for (i, nodeIdx) in acceptedResult.acceptedNodeIndices.enumerated() {
                                        let destSlot = UInt32(i + 1)
                                        if nodeIdx != destSlot {
                                            let srcOffset = layerByteOffset + Int(step + nodeIdx) * slotByteSize
                                            let dstOffset = layerByteOffset + Int(step + destSlot) * slotByteSize
                                            blit.copy(from: kCache, sourceOffset: srcOffset, to: kCache, destinationOffset: dstOffset, size: slotByteSize)
                                            blit.copy(from: vCache, sourceOffset: srcOffset, to: vCache, destinationOffset: dstOffset, size: slotByteSize)
                                        }
                                    }
                                }
                            }
                        }
                        blit.endEncoding()
                        copyCmd.commit()
                        copyCmd.waitUntilCompleted()
                    }
                }

                // 6.5 Commit winning node linear attention recurrent states to linearStateBuffer
                let winningNodeIdx = acceptedResult.acceptedNodeIndices.last ?? 0
                let linLayerCount = UInt32(cachedLayers.filter { $0.attentionType == .linearAttention }.count)
                if linLayerCount > 0,
                   let sBuf = KVCacheManager.shared.linearStateBuffer,
                   let commitPipe = InferenceEngine.shared.commitGdnTreeWinningStatePipeline,
                   let commitCmd = commandQueue.makeCommandBuffer(),
                   let commitEnc = commitCmd.makeComputeCommandEncoder() {
                    var winNode = UInt32(winningNodeIdx)
                    var maxN = UInt32(jb.maxNodes)
                    var numLin = linLayerCount
                    var stateSizeFloats = UInt32(jb.linValHeads * 128 * 128)

                    commitEnc.setComputePipelineState(commitPipe)
                    commitEnc.setBuffer(jb.treeGdnOutStateBuffer, offset: 0, index: 0)
                    commitEnc.setBuffer(sBuf, offset: 0, index: 1)
                    commitEnc.setBytes(&winNode, length: MemoryLayout<UInt32>.stride, index: 2)
                    commitEnc.setBytes(&maxN, length: MemoryLayout<UInt32>.stride, index: 3)
                    commitEnc.setBytes(&numLin, length: MemoryLayout<UInt32>.stride, index: 4)
                    commitEnc.setBytes(&stateSizeFloats, length: MemoryLayout<UInt32>.stride, index: 5)

                    let vec4Count = Int(stateSizeFloats / 4)
                    commitEnc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numLin), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, commitPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    commitEnc.endEncoding()
                    commitCmd.commit()
                    commitCmd.waitUntilCompleted()
                }

                // 7. Copy winning node logits into logitsBuffer for next token sampling
                let lastNodeIdx = acceptedResult.acceptedNodeIndices.last ?? 0
                let srcOffset = Int(lastNodeIdx) * Int(vocabSize) * MemoryLayout<Float>.stride
                if let copyCmd = commandQueue.makeCommandBuffer(),
                   let blit = copyCmd.makeBlitCommandEncoder() {
                    blit.copy(from: jb.targetLogitsBuffer, sourceOffset: srcOffset, to: logitsBuffer, destinationOffset: 0, size: Int(vocabSize) * MemoryLayout<Float>.stride)
                    blit.endEncoding()
                    copyCmd.commit()
                    copyCmd.waitUntilCompleted()
                }

                // 8. Emit accepted tokens + bonus token
                var emittedTokens = acceptedResult.acceptedTokens
                if let bonus = acceptedResult.bonusToken {
                    emittedTokens.append(bonus)
                }

                if emittedTokens.isEmpty {
                    // Fallback: Sample directly from root node (Node 0) target logits
                    // avoiding redundant target forward pass for root token
                    let rootLogitsPtr = jb.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let rootLogits = Array(UnsafeBufferPointer(start: rootLogitsPtr, count: Int(vocabSize)))
                    let sampledTok: UInt32
                    if temperature <= 0.01 {
                        var bestVal = -Float.greatestFiniteMagnitude
                        var bestTok: UInt32 = 0
                        for (v, val) in rootLogits.enumerated() {
                            if val > bestVal {
                                bestVal = val
                                bestTok = UInt32(v)
                            }
                        }
                        sampledTok = bestTok
                    } else {
                        var maxL = -Float.greatestFiniteMagnitude
                        for l in rootLogits { if l > maxL { maxL = l } }
                        let effT = max(0.01, temperature)
                        let probs = rootLogits.map { exp(($0 - maxL) / effT) }
                        let s = probs.reduce(0.0, +)
                        let r = Float.random(in: 0...1) * s
                        var cum: Float = 0.0
                        var chosen: UInt32 = 0
                        for (idx, p) in probs.enumerated() {
                            cum += p
                            if cum >= r {
                                chosen = UInt32(idx)
                                break
                            }
                        }
                        sampledTok = chosen
                    }
                    emittedTokens = [sampledTok]
                }

                return (acceptedTokens: emittedTokens, newStep: step + UInt32(emittedTokens.count))
            }

            let isAgentEnabled = (sessionId != nil) ? (self.sessions.first(where: { $0.id == sessionId })?.isAgentToolsEnabled ?? self.defaultAgentToolsEnabled) : self.defaultAgentToolsEnabled
            var generatedTokenIds: [UInt32] = []
            var accumulatedDecodedText = ""
            var lastUIUpdateTime = CFAbsoluteTimeGetCurrent()

            if isFullRAM && WorkingSetManager.shared.residentExpertsCount < WorkingSetManager.shared.totalExpertKeysCount {
                await MainActor.run {
                    self.generationStatusText = "⚡ Priming resident weights into RAM..."
                }
                WorkingSetManager.shared.preFaultAll(shardBuffers: buffers, summary: summary)
            }

            // Ingest prompt tokens into KV-cache and recurrent states (skipping pinned prefix)
            let promptCount = promptTokenIds.count - 1
            if promptCount > 0 {
                let startPos = min(prefixTokensReused, promptCount)
                if startPos < promptCount {
                    let prefillTokens = Array(promptTokenIds[startPos..<promptCount])
                    let ok: Bool
                    if modelConfig?.isLingModel == true {
                        var prefillSuccess = true
                        let prefillStartTime = CFAbsoluteTimeGetCurrent()
                        var lastPrefillUIUpdateTime = prefillStartTime
                        let totalTokens = prefillTokens.count

                        let initialPrefillStr = "Ingesting prompt: Token 1/\(totalTokens) (0%) • Initializing..."
                        Task { @MainActor in
                            self.generationStatusText = "📥 " + initialPrefillStr
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].prefillStatus = initialPrefillStr
                            }
                        }

                        for (idx, pTok) in prefillTokens.enumerated() {
                            if Task.isCancelled { prefillSuccess = false; break }
                            let stepOk = autoreleasepool {
                                runTokenForward(tokenId: pTok, step: UInt32(startPos + idx), computeLogits: false)
                            }
                            if !stepOk {
                                prefillSuccess = false
                                break
                            }

                            let now = CFAbsoluteTimeGetCurrent()
                            if now - lastPrefillUIUpdateTime >= 0.08 || idx == totalTokens - 1 {
                                lastPrefillUIUpdateTime = now
                                let elapsed = max(0.001, now - prefillStartTime)
                                let promptSpeed = Double(idx + 1) / elapsed
                                let pct = Int((Double(idx + 1) / Double(totalTokens)) * 100)
                                let remaining = totalTokens - (idx + 1)
                                let etaSec = Double(remaining) * (elapsed / Double(idx + 1))
                                let etaStr = etaSec >= 60 ? String(format: "%dm %02ds", Int(etaSec) / 60, Int(etaSec) % 60) : String(format: "%.0fs", etaSec)
                                let speedStr = promptSpeed >= 10 ? String(format: "%.0f", promptSpeed) : String(format: "%.1f", promptSpeed)
                                let prefillStr = "Ingesting prompt: Token \(idx + 1)/\(totalTokens) (\(pct)%) • \(speedStr) tok/s • ETA: \(etaStr)"
                                let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB

                                Task { @MainActor in
                                    self.generationSpeedTokPerSec = promptSpeed
                                    self.generationStatusText = "📥 " + prefillStr
                                    self.currentRssGB = currentRss
                                    if let sId = sessionId, let mId = messageId,
                                       let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                        self.sessions[sIdx].messages[mIdx].prefillStatus = prefillStr
                                    }
                                }
                            }
                        }
                        ok = prefillSuccess
                    } else {
                        ok = runLayerWisePrefill(promptTokens: prefillTokens, startPos: UInt32(startPos))
                    }
                    if !ok {
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.generationStatusText = Task.isCancelled ? "⏹ Generation stopped by user." : "❌ Ingestion failed during prefill."
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                            }
                        }
                        return
                    }
                }
                currentStep = UInt32(promptCount)
                WorkingSetManager.shared.trimAfterPrefill(shardBuffers: buffers, mode: budgetMode)

                // Clear prefill status once prefill completes
                await MainActor.run {
                    if let sId = sessionId, let mId = messageId,
                       let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                        self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                    }
                }
            }

            // Autoregressive generation loop
            var generationStartTime = CFAbsoluteTimeGetCurrent()

            for _ in 0..<maxTokens {
                if Task.isCancelled { break }
                let currentTokenId = contextTokens.last!

                var acceptedBatch: [UInt32] = []
                if effectiveJetSpec && tokensGenerated > 0,
                   let treeStepResult = runJetSpecTreeStep(rootToken: currentTokenId, step: currentStep, temperature: temp),
                   !treeStepResult.acceptedTokens.isEmpty {
                    acceptedBatch = treeStepResult.acceptedTokens
                    currentStep = treeStepResult.newStep
                    jetSpecTotalDraftAccepted += acceptedBatch.count
                    jetSpecTotalDraftProposed += 1
                    jetSpecMeanTau = Double(jetSpecTotalDraftAccepted) / Double(max(jetSpecTotalDraftProposed, 1))
                } else {
                    let ok = autoreleasepool {
                        runTokenForward(tokenId: currentTokenId, step: currentStep, computeLogits: true, wait: true)
                    }
                    if !ok {
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.generationStatusText = Task.isCancelled ? "⏹ Generation stopped by user." : "❌ Inference forward pass failed."
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                            }
                        }
                        return
                    }
                    currentStep += 1

                    // 4. Sample Next Token
                    let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
                    let isGrammarActive = isAgentEnabled && (UserDefaults.standard.object(forKey: "dynamoe_agent_grammar_masking") == nil ? true : UserDefaults.standard.bool(forKey: "dynamoe_agent_grammar_masking"))
                    let nextToken = sampleNextToken(
                        logits: logitsPtr,
                        vocabSize: Int(vocabSize),
                        contextTokens: contextTokens,
                        temperature: temp,
                        topP: topPVal,
                        minP: minPVal,
                        topK: topKVal,
                        repetitionPenalty: repPen,
                        presencePenalty: presPen,
                        grammarMask: isGrammarActive ? { maskLogits, maskVocab in
                            GrammarConstrainedSampler.shared.updateState(emittedText: accumulatedDecodedText)
                            GrammarConstrainedSampler.shared.applyLogitMask(logits: maskLogits, vocabSize: maskVocab, tokenDecoder: { try? tokenizer.decode(ids: [$0]) })
                        } : nil
                    )
                    if tokensGenerated < 10 {
                        let tokText = (try? tokenizer.decode(ids: [nextToken])) ?? ""
                        print("[Autoregressive] Step \(tokensGenerated): nextToken=\(nextToken) ('\(tokText)') logits[nextToken]=\(logitsPtr[Int(nextToken)])")
                    }
                    acceptedBatch = [nextToken]
                }

                var shouldBreak = false
                for nextToken in acceptedBatch {
                    // 5. Check EOS
                    let isLingEos = (modelConfig?.isLingModel == true) && (nextToken == 156895 || nextToken == 156892)
                    if nextToken == eosTokenId || nextToken == 248044 || nextToken == 248046 || nextToken == 166101 || nextToken == 166102 || isLingEos {
                        shouldBreak = true
                        break
                    }

                    generatedTokenIds.append(nextToken)
                    contextTokens.append(nextToken)
                    tokensGenerated += 1

                    if tokensGenerated == 1 {
                        firstTokenTimestamp = CFAbsoluteTimeGetCurrent()
                    }

                    // 6. Incremental Stream Token Decoding (O(1) per step)
                    let deltaText: String
                    if generatedTokenIds.count > 1 {
                        let slice = Array(generatedTokenIds.suffix(2))
                        let sliceText = (try? tokenizer.decode(ids: slice)) ?? ""
                        let prev1 = (try? tokenizer.decode(ids: [generatedTokenIds[generatedTokenIds.count - 2]])) ?? ""
                        if sliceText.hasPrefix(prev1) {
                            deltaText = String(sliceText.dropFirst(prev1.count))
                        } else {
                            deltaText = (try? tokenizer.decode(ids: [nextToken])) ?? ""
                        }
                    } else {
                        deltaText = (try? tokenizer.decode(ids: [nextToken])) ?? ""
                    }
                    accumulatedDecodedText += deltaText

                    if deltaText.contains("<|im_end|>") || deltaText.contains("<|endoftext|>") || deltaText.contains("<|role_end|>") {
                        shouldBreak = true
                        break
                    }

                    // Pre-Execution Catching: Freeze decoding immediately when </tool_call> closes
                    if isAgentEnabled && StreamingToolParser.shared.shouldFreezeGeneration(accumulatedText: accumulatedDecodedText, deltaText: deltaText) {
                        shouldBreak = true
                        break
                    }
                }

                if shouldBreak { break }

                // Token boundary working set pruning: keep resident set strictly within budget
                WorkingSetManager.shared.trimToBudget(mode: budgetMode, shardBuffers: buffers)

                let genElapsedSec = CFAbsoluteTimeGetCurrent() - generationStartTime
                let tokPerSec = Double(tokensGenerated) / max(genElapsedSec, 0.001)
                let elapsedMs = genElapsedSec * 1000.0
                let currentRss = WorkingSetManager.shared.effectiveResidentMemoryGB
                let resCount = WorkingSetManager.shared.residentExpertsCount
                let totalExp = WorkingSetManager.shared.totalExpertKeysCount
                let hitRate = WorkingSetManager.shared.cacheHitRatePercent
                let prefetchEff = WorkingSetManager.shared.prefetchEfficiencyPercent
                let pageLat = WorkingSetManager.shared.currentPagingLatencyMs

                var updatedRaw = accumulatedDecodedText
                var thinkPart = ""
                var respPart = ""
                var activeThink = false

                let promptTrimmed = formattedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
                let promptRequestsThinking = promptTrimmed.hasSuffix("<think>") && !promptTrimmed.hasSuffix("</think>")

                let containsThinkOpen = updatedRaw.contains("<think>") || updatedRaw.contains("<|thought|>") || updatedRaw.contains("<thought>")
                let containsThinkClose = updatedRaw.contains("</think>") || updatedRaw.contains("</|thought|>") || updatedRaw.contains("</thought>")

                if containsThinkClose {
                    if thinkingEndTimestamp == nil {
                        thinkingEndTimestamp = CFAbsoluteTimeGetCurrent()
                    }
                    let delimiter: String
                    let openTag: String
                    if updatedRaw.contains("</think>") {
                        delimiter = "</think>"
                        openTag = "<think>"
                    } else if updatedRaw.contains("</|thought|>") {
                        delimiter = "</|thought|>"
                        openTag = "<|thought|>"
                    } else {
                        delimiter = "</thought>"
                        openTag = "<thought>"
                    }
                    let parts = updatedRaw.components(separatedBy: delimiter)
                    thinkPart = parts[0].replacingOccurrences(of: openTag, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                    let rawResp = parts.dropFirst().joined(separator: delimiter)
                    respPart = rawResp
                        .replacingOccurrences(of: "<|im_end|>", with: "")
                        .replacingOccurrences(of: "<|endoftext|>", with: "")
                        .replacingOccurrences(of: "<|im_start|>", with: "")
                        .replacingOccurrences(of: "<|role_end|>", with: "")
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    activeThink = false
                } else if containsThinkOpen || promptRequestsThinking {
                    // Inside the thinking block before </think> arrives
                    let openTag = updatedRaw.contains("<|thought|>") ? "<|thought|>" : (updatedRaw.contains("<thought>") ? "<thought>" : "<think>")
                    thinkPart = updatedRaw.replacingOccurrences(of: openTag, with: "").trimmingCharacters(in: .whitespaces)
                    activeThink = true
                    respPart = ""
                } else {
                    // Normal direct response without think tags
                    thinkPart = ""
                    activeThink = false
                    respPart = updatedRaw
                        .replacingOccurrences(of: "<|im_end|>", with: "")
                        .replacingOccurrences(of: "<|endoftext|>", with: "")
                        .replacingOccurrences(of: "<|im_start|>", with: "")
                        .replacingOccurrences(of: "<|role_end|>", with: "")
                }

                let liveTtft = firstTokenTimestamp.map { $0 - startTime }
                let liveThinkDuration = thinkingEndTimestamp.map { $0 - generationStartTime }

                // Throttle MainActor UI updates to 60fps ProMotion frame cadence (16ms) or token interval
                let now = CFAbsoluteTimeGetCurrent()
                let shouldUpdateUI = (tokensGenerated == 1) || (now - lastUIUpdateTime >= 0.016) || (tokensGenerated % 4 == 0)

                if shouldUpdateUI {
                    lastUIUpdateTime = now
                    await MainActor.run {
                        self.generatedStreamText = updatedRaw
                        self.thinkingText = thinkPart
                        self.responseText = respPart
                        self.isThinking = activeThink
                        self.generationTotalTokens = tokensGenerated
                        self.generationElapsedMs = elapsedMs
                        self.generationSpeedTokPerSec = tokPerSec
                        let jetSpecBadge = effectiveJetSpec ? " | 🚀 JetSpec τ=\(String(format: "%.1f", self.jetSpecMeanTau))" : ""
                        self.generationStatusText = activeThink ? "🧠 Reasoning: \(tokensGenerated) tokens | \(String(format: "%.1f", tokPerSec)) tok/s\(jetSpecBadge)" : "⚡ Streaming: \(tokensGenerated) tokens | \(String(format: "%.1f", tokPerSec)) tok/s\(jetSpecBadge)"
                        self.currentRssGB = currentRss
                        self.residentExpertCount = resCount
                        self.totalExpertCount = totalExp
                        self.cacheHitRate = hitRate
                        self.prefetchEfficiency = prefetchEff
                        self.lastPagingLatencyMs = pageLat

                        if let sId = sessionId, let mId = messageId {
                            if let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                let combinedThinking: String?
                                if let prior = priorThinking, !prior.isEmpty {
                                    if !thinkPart.isEmpty {
                                        combinedThinking = prior + "\n\n---\n\n" + thinkPart
                                    } else {
                                        combinedThinking = prior
                                    }
                                } else {
                                    combinedThinking = thinkPart.isEmpty ? nil : thinkPart
                                }

                                let combinedContent: String
                                if let prior = priorContent, !prior.isEmpty {
                                    if !respPart.isEmpty {
                                        combinedContent = prior + "\n\n" + respPart
                                    } else {
                                        combinedContent = prior
                                    }
                                } else {
                                    combinedContent = respPart
                                }

                                self.sessions[sIdx].messages[mIdx].thinkingContent = combinedThinking
                                self.sessions[sIdx].messages[mIdx].content = combinedContent
                                self.sessions[sIdx].messages[mIdx].isThinking = activeThink
                                self.sessions[sIdx].messages[mIdx].tokenCount = tokensGenerated
                                self.sessions[sIdx].messages[mIdx].tokensPerSec = tokPerSec
                                self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = liveTtft
                                self.sessions[sIdx].messages[mIdx].thinkingTimeSeconds = liveThinkDuration
                                self.sessions[sIdx].messages[mIdx].jetSpecTau = (effectiveJetSpec && self.jetSpecTotalDraftProposed > 0) ? self.jetSpecMeanTau : nil
                                self.sessions[sIdx].messages[mIdx].jetSpecDraftAccepted = (effectiveJetSpec && self.jetSpecTotalDraftAccepted > 0) ? self.jetSpecTotalDraftAccepted : nil
                            }
                        }
                    }
                }
            }

            let finalGenElapsedSec = CFAbsoluteTimeGetCurrent() - generationStartTime
            let finalTokPerSec = Double(tokensGenerated) / max(finalGenElapsedSec, 0.001)
            let finalElapsedMs = finalGenElapsedSec * 1000.0
            let finalRss = WorkingSetManager.shared.effectiveResidentMemoryGB
            let finalResCount = WorkingSetManager.shared.residentExpertsCount
            let finalHitRate = WorkingSetManager.shared.cacheHitRatePercent
            let finalPrefetchEff = WorkingSetManager.shared.prefetchEfficiencyPercent
            let finalPageLat = WorkingSetManager.shared.currentPagingLatencyMs
            let finalDecoded = accumulatedDecodedText.isEmpty ? ((try? tokenizer.decode(ids: generatedTokenIds)) ?? "") : accumulatedDecodedText

            var finalThink = ""
            var finalResp = ""
            let promptTrimmedFinal = formattedPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
            let promptRequestsThinkingFinal = promptTrimmedFinal.hasSuffix("<think>") && !promptTrimmedFinal.hasSuffix("</think>")

            let finalContainsThinkClose = finalDecoded.contains("</think>") || finalDecoded.contains("</|thought|>") || finalDecoded.contains("</thought>")
            let finalContainsThinkOpen = finalDecoded.contains("<think>") || finalDecoded.contains("<|thought|>") || finalDecoded.contains("<thought>")

            if finalContainsThinkClose {
                if thinkingEndTimestamp == nil {
                    thinkingEndTimestamp = CFAbsoluteTimeGetCurrent()
                }
                let delimiter: String
                let openTag: String
                if finalDecoded.contains("</think>") {
                    delimiter = "</think>"
                    openTag = "<think>"
                } else if finalDecoded.contains("</|thought|>") {
                    delimiter = "</|thought|>"
                    openTag = "<|thought|>"
                } else {
                    delimiter = "</thought>"
                    openTag = "<thought>"
                }
                let parts = finalDecoded.components(separatedBy: delimiter)
                finalThink = parts[0].replacingOccurrences(of: openTag, with: "").trimmingCharacters(in: .whitespacesAndNewlines)
                let rawFinalResp = parts.dropFirst().joined(separator: delimiter)
                finalResp = rawFinalResp
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|role_end|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            } else if finalContainsThinkOpen || promptRequestsThinkingFinal {
                let openTag = finalDecoded.contains("<|thought|>") ? "<|thought|>" : (finalDecoded.contains("<thought>") ? "<thought>" : "<think>")
                finalThink = finalDecoded.replacingOccurrences(of: openTag, with: "").trimmingCharacters(in: .whitespaces)
                finalResp = ""
            } else {
                finalThink = ""
                finalResp = finalDecoded
                    .replacingOccurrences(of: "<think>", with: "")
                    .replacingOccurrences(of: "<|im_end|>", with: "")
                    .replacingOccurrences(of: "<|endoftext|>", with: "")
                    .replacingOccurrences(of: "<|im_start|>", with: "")
                    .replacingOccurrences(of: "<|role_end|>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }

            let finalTtft = firstTokenTimestamp.map { $0 - startTime }
            let finalThinkDuration = thinkingEndTimestamp.map { $0 - generationStartTime }

            // Agent Harness Multi-Step Tool Check
            let parsedResult = isAgentEnabled ? StreamingToolParser.shared.parseStreamingToolCalls(from: finalDecoded) : (calls: [], brokenFragments: [])
            let hasUncalledIntent = isAgentEnabled && parsedResult.calls.isEmpty && agentStep == 0 && (agentStep + 1 < self.maxAgentSteps) && AgentHarness.shared.detectUncalledActionIntent(content: finalResp, thinking: finalThink)
            let willContinueAgent = (!parsedResult.calls.isEmpty || hasUncalledIntent)

            await MainActor.run {
                if !willContinueAgent {
                    self.isGeneratingText = false
                    self.generationTask = nil
                }
                self.generatedStreamText = finalDecoded
                self.thinkingText = finalThink
                self.responseText = finalResp
                self.isThinking = false
                self.generationTotalTokens = tokensGenerated
                self.generationElapsedMs = finalElapsedMs
                self.generationSpeedTokPerSec = finalTokPerSec
                let finalJetSpecBadge = effectiveJetSpec && self.jetSpecTotalDraftProposed > 0 ? " (JetSpec τ=\(String(format: "%.1f", self.jetSpecMeanTau)), \(self.jetSpecTotalDraftAccepted) draft tokens accepted)" : ""
                if !willContinueAgent {
                    self.generationStatusText = "✨ Generated \(tokensGenerated) tokens in \(String(format: "%.2f", finalElapsedMs)) ms (\(String(format: "%.1f", finalTokPerSec)) tok/s)\(finalJetSpecBadge)"
                } else if !parsedResult.calls.isEmpty {
                    self.generationStatusText = "⚙️ Executing \(parsedResult.calls.count) tool call(s)..."
                } else {
                    self.generationStatusText = "🔄 Continuing agent multi-turn action..."
                }
                self.currentRssGB = finalRss
                self.residentExpertCount = finalResCount
                self.cacheHitRate = finalHitRate
                self.prefetchEfficiency = finalPrefetchEff
                self.lastPagingLatencyMs = finalPageLat

                if let sId = sessionId, let mId = messageId {
                    if let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                        let combinedFinalThinking: String?
                        if let prior = priorThinking, !prior.isEmpty {
                            if !finalThink.isEmpty {
                                combinedFinalThinking = prior + "\n\n---\n\n" + finalThink
                            } else {
                                combinedFinalThinking = prior
                            }
                        } else {
                            combinedFinalThinking = finalThink.isEmpty ? nil : finalThink
                        }

                        let combinedFinalContent: String
                        if let prior = priorContent, !prior.isEmpty {
                            if !finalResp.isEmpty {
                                combinedFinalContent = prior + "\n\n" + finalResp
                            } else {
                                combinedFinalContent = prior
                            }
                        } else {
                            combinedFinalContent = finalResp
                        }

                        self.sessions[sIdx].messages[mIdx].thinkingContent = combinedFinalThinking
                        self.sessions[sIdx].messages[mIdx].content = combinedFinalContent
                        self.sessions[sIdx].messages[mIdx].isThinking = false
                        self.sessions[sIdx].messages[mIdx].tokenCount = tokensGenerated
                        self.sessions[sIdx].messages[mIdx].tokensPerSec = finalTokPerSec
                        self.sessions[sIdx].messages[mIdx].timeToFirstTokenSeconds = finalTtft
                        self.sessions[sIdx].messages[mIdx].thinkingTimeSeconds = finalThinkDuration
                        self.sessions[sIdx].messages[mIdx].jetSpecTau = (effectiveJetSpec && self.jetSpecTotalDraftProposed > 0) ? self.jetSpecMeanTau : nil
                        self.sessions[sIdx].messages[mIdx].jetSpecDraftAccepted = (effectiveJetSpec && self.jetSpecTotalDraftAccepted > 0) ? self.jetSpecTotalDraftAccepted : nil
                    }
                }
                PrefixCacheManager.shared.recordTurn(
                    promptTokenIds: promptTokenIds,
                    generatedTokenIds: generatedTokenIds,
                    sessionId: sessionId
                )
                if !willContinueAgent {
                    self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                }
            }

            // Agent Harness Multi-Step Tool Execution
            if isAgentEnabled {
                if !parsedResult.calls.isEmpty {
                    var initialRecords: [ToolCallRecord] = []
                    for call in parsedResult.calls {
                        var stringArgs: [String: String] = [:]
                        for (k, v) in call.arguments {
                            stringArgs[k] = "\(v)"
                        }
                        initialRecords.append(ToolCallRecord(
                            name: call.name,
                            arguments: stringArgs,
                            rawArguments: call.rawArguments,
                            status: .running
                        ))
                    }

                    // Attach initial records to ChatMessage
                    await MainActor.run {
                        if let sId = sessionId, let mId = messageId,
                           let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                           let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                            var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls ?? []
                            currentCalls.append(contentsOf: initialRecords)
                            self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                            self.generationStatusText = "⚙️ Executing \(initialRecords.count) tool call(s)..."
                        }
                    }

                    let baseWdURL = self.agentWorkingDirectory.isEmpty ? nil : URL(fileURLWithPath: self.agentWorkingDirectory)
                    var toolResponses: [String] = []
                    var anyCompleted = false

                    for (idx, call) in parsedResult.calls.enumerated() {
                        if Task.isCancelled {
                            await MainActor.run {
                                self.isGeneratingText = false
                                self.generationTask = nil
                                self.generationStatusText = "⏹ Tool execution stopped by user."
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls {
                                    for i in idx..<currentCalls.count {
                                        if currentCalls[i].status == .running {
                                            currentCalls[i].status = .error
                                            currentCalls[i].error = "Cancelled by user."
                                        }
                                    }
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                }
                            }
                            return
                        }

                        let recordId = initialRecords[idx].id
                        await MainActor.run {
                            self.generationStatusText = "⚙️ Executing [\(idx + 1)/\(parsedResult.calls.count)]: \(call.name)..."
                        }

                        // Human-In-The-Loop (HITL) Safety & Turbo Mode
                        let isTurbo = UserDefaults.standard.bool(forKey: "dynamoe_agent_turbo_mode")
                        let isDestructive = AgentHarness.isStateChanging(toolName: call.name)

                        if isDestructive && !isTurbo {
                            await MainActor.run {
                                if let sId = sessionId, let mId = messageId,
                                   let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                   let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                   var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                   let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                    currentCalls[callIdx].status = .awaitingApproval
                                    self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                    self.generationStatusText = "🛡️ Action requires confirmation: \(call.name)"
                                }
                            }

                            let approved = await ActionApprovalManager.shared.waitForApproval(id: recordId)
                            if !approved {
                                await MainActor.run {
                                    if let sId = sessionId, let mId = messageId,
                                       let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                                       let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                                       var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                                       let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                        currentCalls[callIdx].status = .rejected
                                        currentCalls[callIdx].error = "Action rejected by user."
                                        self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                                    }
                                }
                                let rejectJSON = AgentHarness.toolErrorJSON(tool: call.name, error: "Action rejected by user.")
                                toolResponses.append(rejectJSON)
                                continue
                            }
                        }

                        let execResult = await AgentHarness.shared.executeTool(
                            call: call,
                            workingDirectory: baseWdURL,
                            maxOutputLength: self.maxToolOutputLength
                        )
                        toolResponses.append(execResult.resultJSON)
                        if execResult.isCompleted {
                            anyCompleted = true
                        }

                        // Update ToolCallRecord in ChatMessage
                        await MainActor.run {
                            if let sId = sessionId, let mId = messageId,
                               let sIdx = self.sessions.firstIndex(where: { $0.id == sId }),
                               let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }),
                               var currentCalls = self.sessions[sIdx].messages[mIdx].toolCalls,
                               let callIdx = currentCalls.firstIndex(where: { $0.id == recordId }) {
                                currentCalls[callIdx].status = execResult.record.status
                                currentCalls[callIdx].output = execResult.record.output
                                currentCalls[callIdx].error = execResult.record.error
                                currentCalls[callIdx].executionDurationSeconds = execResult.record.executionDurationSeconds
                                self.sessions[sIdx].messages[mIdx].toolCalls = currentCalls
                            }
                        }
                    }

                    if Task.isCancelled {
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.generationStatusText = "⏹ Generation stopped by user."
                        }
                        return
                    }

                    // If not finished and steps remaining, invoke next step
                    if !anyCompleted && (agentStep + 1 < self.maxAgentSteps) {
                        let toolResponseTurn = AgentHarness.shared.formatToolResponseTurn(
                            responses: toolResponses,
                            includeThinkSuffix: thinkingEnabled
                        )
                        var assistantTurnText = finalDecoded
                        let endTag = (modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>"
                        if !assistantTurnText.contains(endTag) {
                            assistantTurnText += endTag
                        }
                        let nextPrompt = formattedPrompt + assistantTurnText + "\n" + toolResponseTurn
                        await MainActor.run {
                            let nextAssistantMsgId = UUID()
                            if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                                if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                    self.sessions[sIdx].messages[mIdx].isThinking = false
                                    self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                                }
                                let nextMsg = ChatMessage(
                                    id: nextAssistantMsgId,
                                    role: .assistant,
                                    content: "",
                                    thinkingContent: nil,
                                    isThinking: thinkingEnabled
                                )
                                self.sessions[sIdx].messages.append(nextMsg)
                            }
                            self.startAutoregressiveGeneration(
                                customPrompt: nextPrompt,
                                sessionId: sessionId,
                                messageId: nextAssistantMsgId,
                                agentStep: agentStep + 1
                            )
                        }
                        return
                    } else {
                        // All steps finished or complete tool called
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.generationStatusText = anyCompleted ? "✨ Agent completed task." : "✨ Agent completed maximum allowed steps (\(self.maxAgentSteps))."
                            self.dequeueAndRunNextPromptIfNeeded(sessionId: sessionId)
                        }
                        return
                    }
                } else if agentStep == 0 && (agentStep + 1 < self.maxAgentSteps) && AgentHarness.shared.detectUncalledActionIntent(content: finalResp, thinking: finalThink) {
                    if Task.isCancelled {
                        await MainActor.run {
                            self.isGeneratingText = false
                            self.generationTask = nil
                            self.generationStatusText = "⏹ Generation stopped by user."
                        }
                        return
                    }

                    let continuationTurn = AgentHarness.shared.formatActionContinuationTurn(
                        includeThinkSuffix: thinkingEnabled
                    )
                    var assistantTurnText = finalDecoded
                    let endTag = (modelConfig?.isLingModel == true) ? "<|role_end|>" : "<|im_end|>"
                    if !assistantTurnText.contains(endTag) {
                        assistantTurnText += endTag
                    }
                    let nextPrompt = formattedPrompt + assistantTurnText + "\n" + continuationTurn
                    await MainActor.run {
                        let nextAssistantMsgId = UUID()
                        if let sId = sessionId, let sIdx = self.sessions.firstIndex(where: { $0.id == sId }) {
                            if let mId = messageId, let mIdx = self.sessions[sIdx].messages.firstIndex(where: { $0.id == mId }) {
                                self.sessions[sIdx].messages[mIdx].isThinking = false
                                self.sessions[sIdx].messages[mIdx].prefillStatus = nil
                            }
                            let nextMsg = ChatMessage(
                                id: nextAssistantMsgId,
                                role: .assistant,
                                content: "",
                                thinkingContent: nil,
                                isThinking: thinkingEnabled
                            )
                            self.sessions[sIdx].messages.append(nextMsg)
                        }
                        self.startAutoregressiveGeneration(
                            customPrompt: nextPrompt,
                            sessionId: sessionId,
                            messageId: nextAssistantMsgId,
                            agentStep: agentStep + 1
                        )
                    }
                    return
                }
            }
        }
    }

    private func updatePagingStats() {
        self.currentRssGB = WorkingSetManager.shared.effectiveResidentMemoryGB
        self.residentExpertCount = WorkingSetManager.shared.residentExpertsCount
        self.totalExpertCount = WorkingSetManager.shared.totalExpertKeysCount
        self.cacheHitRate = WorkingSetManager.shared.cacheHitRatePercent
        self.lastPagingLatencyMs = WorkingSetManager.shared.currentPagingLatencyMs
    }

    private func applyMemoryBudget(_ mode: MemoryBudgetMode) {
        guard let summary = summary else { return }
        WorkingSetManager.shared.setBudgetMode(mode: mode, shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "⚡ Budget updated: \(mode.rawValue)"
    }

    private func flushExpertCache() {
        WorkingSetManager.shared.flushAllExperts(shardBuffers: shardBuffers)
        updatePagingStats()
        pagingStatusMessage = "🧹 Expert cache flushed (MADV_DONTNEED)"
    }

    private func preFaultAllWeights() {
        guard let summary = summary else { return }
        WorkingSetManager.shared.preFaultAll(shardBuffers: shardBuffers, summary: summary)
        updatePagingStats()
        pagingStatusMessage = "🚀 All weights pre-faulted into RAM"
    }

    private func applyMemoryExecutionMode(_ mode: MemoryExecutionMode) {
        guard let summary = summary else { return }
        let mappedGB = summary.sizeGb
        let eff = mode.resolveEffectiveMode(modelFootprintGB: mappedGB)
        if eff == .residentRAM {
            pagingStatusMessage = "⚡ Priming resident weights into RAM..."
            DispatchQueue.global(qos: .userInitiated).async {
                WorkingSetManager.shared.preFaultAll(shardBuffers: self.shardBuffers, summary: summary)
                Task { @MainActor in
                    self.pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
                    self.updatePagingStats()
                    self.syncSettingsWindowIfNeeded()
                }
            }
        } else {
            let dirUrl: URL? = activeLoadedModelPath != nil ? {
                let p = activeLoadedModelPath!
                var isD: ObjCBool = false
                FileManager.default.fileExists(atPath: p, isDirectory: &isD)
                return isD.boolValue ? URL(fileURLWithPath: p) : URL(fileURLWithPath: p).deletingLastPathComponent()
            }() : nil
            WorkingSetManager.shared.initialize(summary: summary, shardBuffers: shardBuffers, mode: memoryBudgetMode, modelDir: dirUrl)
            pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
            updatePagingStats()
            syncSettingsWindowIfNeeded()
        }
    }

    private func applyMemoryBudgetMode(_ budget: MemoryBudgetMode) {
        applyMemoryBudget(budget)
        syncSettingsWindowIfNeeded()
    }

    private func loadAndBridgeToMetal(filePath: String) {
        self.isLoadingModel = true
        self.metalStatus = "⏳ Loading model engine & zero-copy weights..."
        
        let memoryExecutionMode = self.memoryExecutionMode
        let memoryBudgetMode = self.memoryBudgetMode
        let currentSystemPrompt = self.systemPrompt

        Task.detached(priority: .userInitiated) {
            do {
                let loadedEngine = try DynaMoeEngine(filePath: filePath)
                let loadedSummary = try loadedEngine.getSummary()
                
                guard let device = MTLCreateSystemDefaultDevice() else {
                    await MainActor.run {
                        self.metalStatus = "❌ Failed to initialize Metal GPU."
                        self.isLoadingModel = false
                    }
                    return
                }

                // Attempt to load and parse HuggingFace config.json if present
                let fileUrl = URL(fileURLWithPath: filePath)
                var isDir: ObjCBool = false
                FileManager.default.fileExists(atPath: filePath, isDirectory: &isDir)
                let dirUrl = isDir.boolValue ? fileUrl : fileUrl.deletingLastPathComponent()
                let cfg = ModelConfig.load(from: dirUrl)
                let arch = cfg?.resolveArchitectureType(summary: loadedSummary) ?? (loadedSummary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)

                let sysPrompt: String
                if currentSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    sysPrompt = ModelConfig.getUserDefaultSystemPrompt()
                } else {
                    sysPrompt = currentSystemPrompt
                }

                // Auto-load tokenizer.json from model directory if present
                let tokUrl = dirUrl.appendingPathComponent("tokenizer.json")
                let tok: DynaMoeTokenizer?
                if FileManager.default.fileExists(atPath: tokUrl.path) {
                    tok = try? DynaMoeTokenizer(tokenizerPath: tokUrl.path)
                } else {
                    tok = nil
                }
                
                // Map every shard into Metal zero-copy space
                var buffers: [UInt32: MTLBuffer] = [:]
                var mappedGB: Double = 0.0
                
                for shard in loadedSummary.shards {
                    let address = UInt(shard.baseAddress)
                    guard let pointer = UnsafeMutableRawPointer(bitPattern: address) else { continue }
                    let length = Int(shard.length)
                    guard length > 0 else {
                        print("ℹ️ [Metal] Skipping zero-length shard #\(shard.index) (\(shard.filename)).")
                        continue
                    }
                    guard length <= Int(device.maxBufferLength) else {
                        print("⚠️ [Metal Warning] Shard #\(shard.index) (\(shard.filename)) size \(Double(length) / (1024*1024*1024)) GB exceeds device maxBufferLength (\(Double(device.maxBufferLength) / (1024*1024*1024)) GB). Skipping single-buffer mapping.")
                        continue
                    }
                    
                    if let buffer = device.makeBuffer(bytesNoCopy: pointer, length: length, options: .storageModeShared, deallocator: nil) {
                        buffers[shard.index] = buffer
                        mappedGB += Double(length) / (1024.0 * 1024.0 * 1024.0)
                    }
                }

                let isFlashMoE = ExpertRepacker.isPackedFormat(dir: dirUrl)
                if isFlashMoE {
                    ExpertIOThreadPool.shared.initialize(numThreads: 8)
                } else {
                    WorkingSetManager.shared.initialize(summary: loadedSummary, shardBuffers: buffers, mode: memoryBudgetMode, modelDir: dirUrl)
                    let effMode = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: mappedGB)
                    if effMode == .residentRAM {
                        WorkingSetManager.shared.preFaultAll(shardBuffers: buffers, summary: loadedSummary)
                    }
                }
                
                await MainActor.run {
                    self.engine = loadedEngine
                    self.summary = loadedSummary
                    self.shardBuffers = buffers
                    self.modelConfig = cfg
                    self.detectedArchitecture = arch
                    self.systemPrompt = sysPrompt
                    if let tok = tok {
                        self.tokenizer = tok
                    }
                    self.errorMessage = nil
                    self.selectedTensorID = nil
                    self.gpuComputeOutput = nil
                    self.activeLoadedModelPath = filePath
                    self.isLoadingModel = false

                    if isFlashMoE {
                        self.pagingStatusMessage = "⚡ Ultra-Fast Parallel POSIX Pread Streaming Active (12+ tok/s)"
                    } else {
                        let effMode = memoryExecutionMode.resolveEffectiveMode(modelFootprintGB: mappedGB)
                        if effMode == .residentRAM {
                            self.pagingStatusMessage = "⚡ Operating in Full RAM Resident Mode (Zero Disk Paging)"
                        } else {
                            self.pagingStatusMessage = "🌊 Operating in Dynamic SSD Streaming Mode"
                        }
                    }
                    self.updatePagingStats()
                    self.metalStatus = "✅ Zero-Copy Active! \(loadedSummary.shards.count) Shards Mapped (\(String(format: "%.2f", mappedGB)) GB)"
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Core Engine Error: \(error.localizedDescription)"
                    self.summary = nil
                    self.engine = nil
                    self.shardBuffers.removeAll()
                    self.isLoadingModel = false
                    self.metalStatus = "❌ Engine Load Error"
                }
            }
        }
    }

    private func executeGpuShader(on tensor: TensorMetadata) {
        guard let summary = summary,
              let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue(),
              let defaultLibrary = device.makeDefaultLibrary(),
              let rawBaseBuffer = shardBuffers[tensor.shardIndex] else {
            gpuComputeOutput = "❌ Error setting up Metal pipeline or missing shard buffer."
            return
        }

        do {
            let isBF16 = tensor.dtype.contains("BF16") || tensor.dtype.contains("FLOAT")
            let sampleCount = 8
            let outputByteLength = sampleCount * MemoryLayout<Float>.stride
            guard let outputBuffer = device.makeBuffer(length: outputByteLength, options: .storageModeShared),
                  let commandBuffer = commandQueue.makeCommandBuffer(),
                  let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }

            let baseName = tensor.name.replacingOccurrences(of: ".weight", with: "").replacingOccurrences(of: ".scales", with: "").replacingOccurrences(of: ".weight_scale", with: "")

            if isBF16 {
                guard let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_bf16_preview") else { return }
                let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                var weightOffset = tensor.offsetStart
                computeEncoder.setComputePipelineState(pipelineState)
                computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
                computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 2)
                computeEncoder.dispatchThreads(MTLSize(width: sampleCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sampleCount, height: 1, depth: 1))
            } else {
                let weightTensor = summary.tensors.first(where: { $0.name == "\(baseName).weight" }) ?? tensor
                let scaleTensor = summary.tensors.first(where: { $0.name == "\(baseName).weight_scale" || $0.name == "\(baseName).scales" || $0.name == "\(baseName).scale" })
                let scaleShardRaw = (scaleTensor != nil) ? shardBuffers[scaleTensor!.shardIndex] : rawBaseBuffer

                var weightOffset = weightTensor.offsetStart
                var scaleOffset = scaleTensor?.offsetStart ?? 0

                if let kernelFunction = defaultLibrary.makeFunction(name: "dequantize_fp8_row_scaled") {
                    let pipelineState = try device.makeComputePipelineState(function: kernelFunction)
                    computeEncoder.setComputePipelineState(pipelineState)
                    computeEncoder.setBuffer(rawBaseBuffer, offset: 0, index: 0)
                    computeEncoder.setBuffer(outputBuffer, offset: 0, index: 1)
                    computeEncoder.setBuffer(scaleShardRaw, offset: 0, index: 2)
                    computeEncoder.setBytes(&weightOffset, length: MemoryLayout<UInt64>.stride, index: 3)
                    computeEncoder.setBytes(&scaleOffset, length: MemoryLayout<UInt64>.stride, index: 4)
                    computeEncoder.dispatchThreads(MTLSize(width: sampleCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: sampleCount, height: 1, depth: 1))
                }
            }

            computeEncoder.endEncoding()
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()

            let rawFloatPtr = outputBuffer.contents().bindMemory(to: Float.self, capacity: sampleCount)
            var sampleValues: [String] = []
            for i in 0..<sampleCount {
                sampleValues.append(String(format: "%.6f", rawFloatPtr[i]))
            }

            gpuComputeOutput = "⚡ Dequantized! First 8 values for '\(tensor.name)' (Shard #\(tensor.shardIndex)): [\(sampleValues.joined(separator: ", "))]"

        } catch {
            gpuComputeOutput = "❌ Pipeline Error: \(error.localizedDescription)"
        }
    }

    #if os(macOS)
    private func selectTokenizerWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data]
        panel.message = "Select tokenizer.json"
        panel.prompt = "Load Tokenizer"
        
        if panel.runModal() == .OK, let url = panel.url {
            loadTokenizer(filePath: url.path)
        }
    }

    private func selectModelWithOpenPanel() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.json, .data, .folder]
        panel.message = "Select Model Snapshot Folder, model_weights.bin, or model.safetensors.index.json"
        panel.prompt = "Load Model"
        
        if panel.runModal() == .OK, let url = panel.url {
            var targetPath = url.path
            if url.hasDirectoryPath {
                let indexPath = url.appendingPathComponent("model.safetensors.index.json").path
                let flashMoeJson = url.appendingPathComponent("model_weights.json").path
                let flashMoeBin = url.appendingPathComponent("model_weights.bin").path
                
                if FileManager.default.fileExists(atPath: indexPath) {
                    targetPath = indexPath
                } else if FileManager.default.fileExists(atPath: flashMoeJson) {
                    targetPath = flashMoeJson
                } else if FileManager.default.fileExists(atPath: flashMoeBin) {
                    targetPath = flashMoeBin
                }
            }
            loadAndBridgeToMetal(filePath: targetPath)
        }
    }
    #endif
}
