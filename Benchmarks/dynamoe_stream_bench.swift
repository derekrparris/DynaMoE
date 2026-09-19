//
//  dynamoe_stream_bench.swift
//  DynaMoE streaming-pipeline microbenchmark
//
//  Replicates the exact packed-FlashMoE decode IO/compute pipeline used by
//  ContentView.swift's runTokenForward() against the REAL packed model files
//  on disk, measuring each stage in isolation plus full serialized and
//  pipelined end-to-end token cost.
//
//  MEASURED RESULTS (M1 Pro 16", 16 GB RAM, Ornith 1.5 35B A3B FP8 packed):
//    T1  single-stream sequential cold read ........... 1.39-1.56 GB/s  <- per-stream ceiling!
//    T2  8-way parallel sequential cold read .......... 3.67-4.30 GB/s  <- device ceiling
//    T3  app decode pread pattern (8 experts/layer) ... 3.75-4.01 GB/s cold, 19-23 GB/s warm
//    T4  GPU MoE phase (16 dispatches + barriers) ..... 1.0-1.5 ms/layer (42-61 ms/token)
//    T4b fused MoE (2 dispatches) ..................... 0.6-1.3 ms/layer (1.2-2x faster)
//    T5  router phase (rmsnorm+topk+sync) ............. 0.42-0.9 ms/layer (sync overhead dominated)
//    T6  full serialized token (current app shape) .... 6.7-7.7 tok/s (warm IO)
//    T7  pipelined upper bound (oracle prefetch) ...... 10.0-10.7 tok/s
//    T8  mmap page-fault path ......................... GPU 2.45-3.6 GB/s cold (big dispatch);
//        CPU stride-touch 0.66 GB/s cold
//    T10 app-shaped loop: per-layer 62MB mmap reads
//        interleaved with other GPU work .............. 0.065 GB/s, ~945 ms/layer, EVERY token
//        (file-backed GPU mappings dropped between dispatches -> ~250 us/page re-fault)
//    T11 prefill pattern (K-way concurrent reads) ..... 3.5-5.6 GB/s at K>=128 (fine)
//    T12 807MB staging alloc + first touch ............ 70-84 ms per prefill
//
//  VERDICT: the expert pread pool is NOT the bottleneck (it hits ~4 GB/s cold).
//  The bottleneck is the backbone (2.30 GB/token) + lm_head (0.95 GB/token)
//  being read EVERY token through mmap page faults at 0.065-2.5 GB/s, plus the
//  serialized router->IO->MoE structure with 2-3 blocking GPU syncs per layer.
//  See Benchmarks/PERF_FINDINGS.md for the full analysis and fix list.
//
//  Sections:
//    T1  Single-stream sequential cold read of one packed layer file (device ceiling)
//    T2  8-way parallel sequential read (max parallel DMA ceiling)
//    T3  The app's exact decode pread pattern (concurrentPerform, 8 x expert_size)
//        pass1 cold-ish, pass2 warm, plus serial variant (GCD parallelism check)
//    T4  GPU cost of the real decode MoE kernels (fp8_swiglu_gate_up_simd /
//        fp8_down_proj_accumulate_simd, 16 dispatches + barriers per layer)
//    T4b Fused variant (2 dispatches per layer) to quantify launch/barrier overhead
//    T5  Router phase GPU cost (rmsnorm_bf16 + moe_router_topk_bf16 + sync)
//    T6  FULL serialized token simulation: router -> IO -> MoE per layer (current app)
//    T7  Pipelined upper bound: layer l+1 experts prefetched during layer l GPU work
//    T8  mmap cold page-fault throughput (file-backed pages)
//    T9  lm_head GEMV GPU cost (resident)
//    T10 Full token sim incl. mmap backbone fault-in (SLOW: ~75 s/token, reproduces
//        the real app's re-faulting collapse)
//    T10b Fault dynamics isolation (A: no eviction, B: churn survival, C: CPU pre-touch)
//    T11 Prefill read pattern (K-way concurrent expert reads)
//    T12 Prefill staging buffer alloc + first-touch cost
//
//  Build:
//    swiftc -O dynamoe_stream_bench.swift -o dynamoe_stream_bench \
//        -framework Metal -framework Foundation
//
//  Usage:
//    ./dynamoe_stream_bench --model <snapshotDir> [--metal <ComputeShaders.metal|default.metallib>] [--tokens N]
//

import Foundation
import Metal
import Darwin

// MARK: - Config

struct FlashMoELayout: Codable {
    struct Component: Codable {
        let name: String
        let offset: UInt64
        let size: UInt64
        let dtype: String
        let shape: [Int]
    }
    let expert_size: UInt64
    let num_layers: Int
    let num_experts: Int
    let components: [Component]
}

// Ornith 1.5 35B A3B FP8 geometry (parsed from layout.json; these are fallbacks)
let HIDDEN = 2048
let INTER = 512
let TOPK = 8
let GB: Double = 1024.0 * 1024.0 * 1024.0

var gExpertSize: Int = 3151872
var gNumLayers: Int = 40
var gNumExperts: Int = 256
var gTokens: Int = 4

let printLock = NSLock()
func log(_ s: String) {
    printLock.lock()
    print(s)
    fflush(stdout)
    printLock.unlock()
}

func pct(_ arr: [Double], _ p: Double) -> Double {
    guard !arr.isEmpty else { return 0 }
    let s = arr.sorted()
    let idx = min(s.count - 1, max(0, Int((p / 100.0) * Double(s.count))))
    return s[idx]
}

// Deterministic LCG for expert id sequences (same sequence across tests)
final class Rand {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    func next(_ bound: Int) -> Int {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return Int((state >> 33) % UInt64(bound))
    }
    func tokenExperts() -> [Int] {
        var ids: [Int] = []
        var seen = Set<Int>()
        while ids.count < TOPK {
            let id = next(gNumExperts)
            if seen.insert(id).inserted { ids.append(id) }
        }
        return ids
    }
}

// App-shaped pread task (mirrors ExpertPreadTask)
struct ExpertTask {
    var fd: Int32
    var dst: UnsafeMutableRawPointer
    var offset: off_t
    var size: Int
    var result: Int = 0
}

func openLayerFD(_ packedDir: URL, layer: Int, noCache: Bool) -> Int32 {
    let path = packedDir.appendingPathComponent(String(format: "layer_%02d.bin", layer)).path
    let fd = open(path, O_RDONLY | O_CLOEXEC)
    if fd >= 0 {
        // ExpertIOThreadPool does fcntl(fd, F_NOCACHE, 0) -> page cache ENABLED
        _ = fcntl(fd, F_NOCACHE, noCache ? 1 : 0)
    } else {
        print("⚠️ open() failed for \(path): errno=\(errno)")
    }
    return fd
}

func preadFull(_ fd: Int32, _ dst: UnsafeMutableRawPointer, _ offset: off_t, _ size: Int) -> Int {
    var done = 0
    while done < size {
        let n = pread(fd, dst.advanced(by: done), size - done, offset + off_t(done))
        if n <= 0 { break }
        done += n
    }
    return done
}

// Runs 8 preads exactly like ExpertIOThreadPool.dispatchSync
func dispatchPreads(_ tasks: inout [ExpertTask]) {
    tasks.withUnsafeMutableBufferPointer { buf in
        DispatchQueue.concurrentPerform(iterations: buf.count) { i in
            let t = buf[i]
            buf[i].result = preadFull(t.fd, t.dst, t.offset, t.size)
        }
    }
}

// MARK: - GPU setup

final class GPU {
    let device: MTLDevice
    let queue: MTLCommandQueue
    var pipelines: [String: MTLComputePipelineState] = [:]

    init?(metalPath: String) {
        guard let dev = MTLCreateSystemDefaultDevice() else { return nil }
        device = dev
        guard let q = dev.makeCommandQueue() else { return nil }
        queue = q

        let t0 = CFAbsoluteTimeGetCurrent()
        var lib: MTLLibrary
        do {
            if metalPath.hasSuffix(".metallib") {
                lib = try dev.makeLibrary(URL: URL(fileURLWithPath: metalPath))
            } else {
                let src = try String(contentsOfFile: metalPath, encoding: .utf8)
                log("   compiling \(src.count / 1024) KB of MSL at runtime...")
                lib = try dev.makeLibrary(source: src, options: nil)
            }
        } catch {
            print("MSL compile error: \(error)")
            return nil
        }
        log("   MSL ready in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - t0))s")

        for name in ["clear_vector_f32", "vector_add_f32", "rmsnorm_bf16", "moe_router_topk_bf16",
                     "fp8_swiglu_gate_up_simd", "fp8_down_proj_accumulate_simd",
                     "bf16_swiglu_gate_up_simd", "bf16_down_proj_accumulate_simd",
                     "bf16_gemv_simd", "bench_fault_read", "bench_read_all",
                     "bench_fp8_moe_gate_up_fused", "bench_fp8_moe_down_fused"] {
            if let f = lib.makeFunction(name: name) {
                do {
                    pipelines[name] = try dev.makeComputePipelineState(function: f)
                } catch {
                    log("   pipeline \(name) failed: \(error)")
                }
            } else {
                log("   kernel \(name) not found in library")
            }
        }
    }

    func pipe(_ name: String) -> MTLComputePipelineState? { pipelines[name] }
}

// MARK: - Kernel encoding (replicates ContentView FP8 decode branch)

struct MoEComponents {
    let gateW: UInt64, gateS: UInt64
    let upW: UInt64, upS: UInt64
    let downW: UInt64, downS: UInt64
}

func findComponent(_ layout: FlashMoELayout, _ proj: String, _ wantScale: Bool) -> UInt64 {
    for c in layout.components {
        if c.name.contains(proj) {
            let isScale = c.name.contains("scale") || c.name.contains("scales")
            let isBias = c.name.contains("bias")
            if wantScale && isScale && !isBias { return c.offset }
            if !wantScale && !isScale && !isBias { return c.offset }
        }
    }
    return 0
}

func encodeClear(_ enc: MTLComputeCommandEncoder, _ gpu: GPU, _ buf: MTLBuffer, _ n: Int) {
    let clear = gpu.pipe("clear_vector_f32")!
    enc.setComputePipelineState(clear)
    enc.setBuffer(buf, offset: 0, index: 0)
    var nVal: UInt32 = UInt32(n)
    enc.setBytes(&nVal, length: 4, index: 1)
    enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: min(n, 1024), height: 1, depth: 1))
    enc.memoryBarrier(scope: .buffers)
}

// One expert's gate+up+down exactly like the FP8 decode branch in runTokenForward
func encodeExpert(enc: MTLComputeCommandEncoder, gpu: GPU, staging: MTLBuffer,
                  inter: MTLBuffer, hMlp: MTLBuffer, xNorm: MTLBuffer,
                  slot: Int, expertSize: Int, comp: MoEComponents, weight: Float) {
    let slotOffset = UInt64(slot * expertSize)
    var gWOff = slotOffset + comp.gateW
    var gSOff = slotOffset + comp.gateS
    var uWOff = slotOffset + comp.upW
    var uSOff = slotOffset + comp.upS
    var dWOff = slotOffset + comp.downW
    var dSOff = slotOffset + comp.downS
    var hDimVal: UInt32 = UInt32(HIDDEN)
    var interDimVal: UInt32 = UInt32(INTER)
    var pkVal = weight

    if let gatePipe = gpu.pipe("fp8_swiglu_gate_up_simd"), let downPipe = gpu.pipe("fp8_down_proj_accumulate_simd") {
        enc.setComputePipelineState(gatePipe)
        enc.setBuffer(staging, offset: 0, index: 0)
        enc.setBuffer(staging, offset: 0, index: 1)
        enc.setBuffer(xNorm, offset: 0, index: 2)
        enc.setBuffer(inter, offset: 0, index: 3)
        enc.setBuffer(staging, offset: 0, index: 4)
        enc.setBuffer(staging, offset: 0, index: 5)
        enc.setBytes(&gWOff, length: 8, index: 6)
        enc.setBytes(&gSOff, length: 8, index: 7)
        enc.setBytes(&uWOff, length: 8, index: 8)
        enc.setBytes(&uSOff, length: 8, index: 9)
        enc.setBytes(&hDimVal, length: 4, index: 10)
        enc.setBytes(&interDimVal, length: 4, index: 11)
        enc.dispatchThreadgroups(MTLSize(width: INTER, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.setComputePipelineState(downPipe)
        enc.setBuffer(staging, offset: 0, index: 0)
        enc.setBuffer(inter, offset: 0, index: 1)
        enc.setBuffer(hMlp, offset: 0, index: 2)
        enc.setBuffer(staging, offset: 0, index: 3)
        enc.setBytes(&dWOff, length: 8, index: 4)
        enc.setBytes(&dSOff, length: 8, index: 5)
        enc.setBytes(&interDimVal, length: 4, index: 6)
        enc.setBytes(&hDimVal, length: 4, index: 7)
        enc.setBytes(&pkVal, length: 4, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: HIDDEN, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }
}

// Fused: all TOPK experts' gate+up in ONE dispatch, then all down projections in ONE dispatch
func encodeExpertsFused(enc: MTLComputeCommandEncoder, gpu: GPU, staging: MTLBuffer,
                        inter: MTLBuffer, hMlp: MTLBuffer, xNorm: MTLBuffer,
                        expertIds: [Int], expertSize: Int, idsBuf: MTLBuffer, weightsBuf: MTLBuffer,
                        weights: [Float]) {
    guard let gatePipe = gpu.pipe("bench_fp8_moe_gate_up_fused"),
          let downPipe = gpu.pipe("bench_fp8_moe_down_fused") else { return }

    let idsPtr = idsBuf.contents().bindMemory(to: UInt32.self, capacity: TOPK)
    for (i, e) in expertIds.enumerated() where i < TOPK { idsPtr[i] = UInt32(e) }
    let wPtr = weightsBuf.contents().bindMemory(to: Float.self, capacity: TOPK)
    for (i, w) in weights.enumerated() where i < TOPK { wPtr[i] = w }

    var es: UInt64 = UInt64(expertSize)
    var hDimVal: UInt32 = UInt32(HIDDEN)
    var interDimVal: UInt32 = UInt32(INTER)
    var topKVal: UInt32 = UInt32(TOPK)

    enc.setComputePipelineState(gatePipe)
    enc.setBuffer(staging, offset: 0, index: 0)
    enc.setBuffer(xNorm, offset: 0, index: 1)
    enc.setBuffer(inter, offset: 0, index: 2)
    enc.setBuffer(idsBuf, offset: 0, index: 3)
    enc.setBytes(&es, length: 8, index: 4)
    enc.setBytes(&hDimVal, length: 4, index: 5)
    enc.setBytes(&interDimVal, length: 4, index: 6)
    enc.setBytes(&topKVal, length: 4, index: 7)
    enc.dispatchThreadgroups(MTLSize(width: INTER, height: TOPK, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.memoryBarrier(scope: .buffers)

    enc.setComputePipelineState(downPipe)
    enc.setBuffer(staging, offset: 0, index: 0)
    enc.setBuffer(inter, offset: 0, index: 1)
    enc.setBuffer(hMlp, offset: 0, index: 2)
    enc.setBuffer(idsBuf, offset: 0, index: 3)
    enc.setBuffer(weightsBuf, offset: 0, index: 4)
    enc.setBytes(&es, length: 8, index: 5)
    enc.setBytes(&interDimVal, length: 4, index: 6)
    enc.setBytes(&hDimVal, length: 4, index: 7)
    enc.setBytes(&topKVal, length: 4, index: 8)
    enc.dispatchThreadgroups(MTLSize(width: HIDDEN, height: TOPK, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    enc.memoryBarrier(scope: .buffers)
}

// Router phase: rmsnorm + topk (replicates the tail of Phase A)
func encodeRouter(enc: MTLComputeCommandEncoder, gpu: GPU, h: MTLBuffer, xNorm: MTLBuffer,
                  gamma: MTLBuffer, routerW: MTLBuffer, indices: MTLBuffer, weights: MTLBuffer,
                  numExperts: Int, topK: Int) {
    var hDimVal: UInt32 = UInt32(HIDDEN)
    var epsVal: Float = 1e-6
    var gOff: UInt64 = 0

    if let rn = gpu.pipe("rmsnorm_bf16") {
        enc.setComputePipelineState(rn)
        enc.setBuffer(h, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(xNorm, offset: 0, index: 2)
        enc.setBytes(&gOff, length: 8, index: 3)
        enc.setBytes(&hDimVal, length: 4, index: 4)
        enc.setBytes(&epsVal, length: 4, index: 5)
        enc.setThreadgroupMemoryLength(1024 * MemoryLayout<Float>.stride, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: min(1024, HIDDEN), height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)
    }

    if let router = gpu.pipe("moe_router_topk_bf16") {
        var wOff: UInt64 = 0
        var nExp: UInt32 = UInt32(numExperts)
        var tK: UInt32 = UInt32(topK)
        enc.setComputePipelineState(router)
        enc.setBuffer(routerW, offset: 0, index: 0)
        enc.setBuffer(xNorm, offset: 0, index: 1)
        enc.setBuffer(indices, offset: 0, index: 2)
        enc.setBuffer(weights, offset: 0, index: 3)
        enc.setBytes(&wOff, length: 8, index: 4)
        enc.setBytes(&hDimVal, length: 4, index: 5)
        enc.setBytes(&nExp, length: 4, index: 6)
        enc.setBytes(&tK, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

func encodeMoEPhase(enc: MTLComputeCommandEncoder, gpu: GPU, staging: MTLBuffer,
                    inter: MTLBuffer, hMlp: MTLBuffer, xNorm: MTLBuffer, ids: [Int],
                    expertSize: Int, comp: MoEComponents, idsBuf: MTLBuffer, weightsBuf: MTLBuffer,
                    fused: Bool) {
    encodeClear(enc, gpu, hMlp, HIDDEN)
    if fused {
        let weights = (0..<TOPK).map { _ in Float.random(in: 0.05...0.3) }
        encodeExpertsFused(enc: enc, gpu: gpu, staging: staging, inter: inter, hMlp: hMlp,
                           xNorm: xNorm, expertIds: ids, expertSize: expertSize,
                           idsBuf: idsBuf, weightsBuf: weightsBuf, weights: weights)
    } else {
        for (slot, _) in ids.enumerated() {
            encodeExpert(enc: enc, gpu: gpu, staging: staging, inter: inter, hMlp: hMlp, xNorm: xNorm,
                         slot: slot, expertSize: expertSize, comp: comp, weight: 0.125)
        }
    }
}

// MARK: - Main

func main() {
    var modelDir: URL? = nil
    var metalPath: String? = nil
    var args = Array(CommandLine.arguments.dropFirst())
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "--model": modelDir = URL(fileURLWithPath: args.removeFirst())
        case "--metal": metalPath = args.removeFirst()
        case "--tokens": gTokens = Int(args.removeFirst()) ?? 4
        default: break
        }
    }
    // default: repo ComputeShaders.metal relative to CWD
    if metalPath == nil {
        let cand = "DynaMoE/ComputeShaders.metal"
        if FileManager.default.fileExists(atPath: cand) { metalPath = cand }
    }
    guard let dir = modelDir, metalPath != nil else {
        print("usage: dynamoe_stream_bench --model <snapshotDir> [--metal <.metallib|.metal>] [--tokens N]")
        exit(1)
    }
    let packedDir = dir.appendingPathComponent("packed_experts")
    guard let lData = try? Data(contentsOf: packedDir.appendingPathComponent("layout.json")),
          let layout = try? JSONDecoder().decode(FlashMoELayout.self, from: lData) else {
        print("no packed_experts/layout.json under \(dir.path)")
        exit(1)
    }
    gExpertSize = Int(layout.expert_size)
    gNumLayers = layout.num_layers
    gNumExperts = layout.num_experts

    print("=== DynaMoE Streaming Benchmark ===")
    print("model: \(dir.path)")
    print("layout: layers=\(gNumLayers) experts=\(gNumExperts) expert_size=\(gExpertSize) (\(String(format: "%.2f", Double(gExpertSize) / 1048576.0)) MB) topK=\(TOPK)")
    print("per-token expert traffic: \(String(format: "%.3f", Double(gNumLayers * TOPK * gExpertSize) / GB)) GB")
    print("")

    // ---- GPU init ----
    print("initializing Metal...")

    // If given a .metal source, append the fused benchmark kernels so helpers
    // (bf16_to_fp32, unpack_e4m3) resolve, then compile the combined source.
    var gpuMetalPath = metalPath!
    if metalPath!.hasSuffix(".metal"), let src = try? String(contentsOfFile: metalPath!, encoding: .utf8) {
        let fused = """

        kernel void bench_read_all(
            device const uchar* src [[buffer(0)]],
            device atomic_uint* sink [[buffer(1)]],
            constant uint32_t& nChunks [[buffer(2)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= nChunks) return;
            device const uchar4* base = (device const uchar4*)(src + tid * 256);
            uint acc = 0;
            for (uint i = 0; i < 64; i++) {
                uchar4 v = base[i];
                acc += (uint)v.x + (uint)v.y + (uint)v.z + (uint)v.w;
            }
            atomic_fetch_add_explicit(sink, acc, memory_order_relaxed);
        }

        kernel void bench_fault_read(
            device const uchar* src [[buffer(0)]],
            device atomic_uint* sink [[buffer(1)]],
            constant uint32_t& nPages [[buffer(2)]],
            constant uint32_t& pageSize [[buffer(3)]],
            uint tid [[thread_position_in_grid]]
        ) {
            if (tid >= nPages) return;
            uchar v = src[tid * pageSize];
            atomic_fetch_add_explicit(sink, (uint)v, memory_order_relaxed);
        }

        kernel void bench_fp8_moe_gate_up_fused(
            device const uchar* staging [[buffer(0)]],
            device const float* inputVector [[buffer(1)]],
            device float* intermediateOutput [[buffer(2)]],
            device const uint32_t* expertIds [[buffer(3)]],
            constant uint64_t& expertSize [[buffer(4)]],
            constant uint32_t& hiddenDim [[buffer(5)]],
            constant uint32_t& intermediateDim [[buffer(6)]],
            constant uint32_t& topK [[buffer(7)]],
            uint2 tgPos [[threadgroup_position_in_grid]],
            uint laneId [[thread_index_in_simdgroup]]
        ) {
            uint r = tgPos.x;
            uint slot = tgPos.y;
            if (r >= intermediateDim || slot >= topK) return;
            uint64_t base = (uint64_t)slot * expertSize;
            float gs = bf16_to_fp32(((device const ushort*)(staging + base + 1048576))[r]);
            float us = bf16_to_fp32(((device const ushort*)(staging + base + 2098176))[r]);
            device const uchar* gRow = staging + base + (uint64_t)r * hiddenDim;
            device const uchar* uRow = staging + base + 1049600 + (uint64_t)r * hiddenDim;
            float gate_dot = 0.0f;
            float up_dot = 0.0f;
            for (uint32_t baseD = laneId * 8; baseD < hiddenDim; baseD += 32 * 8) {
                uchar4 g0 = *(device const uchar4*)(gRow + baseD);
                uchar4 g1 = *(device const uchar4*)(gRow + baseD + 4);
                uchar4 u0 = *(device const uchar4*)(uRow + baseD);
                uchar4 u1 = *(device const uchar4*)(uRow + baseD + 4);
                float4 i0 = *(device const float4*)(inputVector + baseD);
                float4 i1 = *(device const float4*)(inputVector + baseD + 4);
                gate_dot += (unpack_e4m3(g0.x)*i0.x)+(unpack_e4m3(g0.y)*i0.y)+(unpack_e4m3(g1.x)*i1.x)+(unpack_e4m3(g1.y)*i1.y)
                          + (unpack_e4m3(g0.z)*i0.z)+(unpack_e4m3(g0.w)*i0.w)+(unpack_e4m3(g1.z)*i1.z)+(unpack_e4m3(g1.w)*i1.w);
                up_dot   += (unpack_e4m3(u0.x)*i0.x)+(unpack_e4m3(u0.y)*i0.y)+(unpack_e4m3(u1.x)*i1.x)+(unpack_e4m3(u1.y)*i1.y)
                          + (unpack_e4m3(u0.z)*i0.z)+(unpack_e4m3(u0.w)*i0.w)+(unpack_e4m3(u1.z)*i1.z)+(unpack_e4m3(u1.w)*i1.w);
            }
            gate_dot = simd_sum(gate_dot);
            up_dot = simd_sum(up_dot);
            if (laneId == 0) {
                float fg = gate_dot * gs;
                float fu = up_dot * us;
                intermediateOutput[((uint64_t)slot * intermediateDim) + r] = (fg / (1.0f + exp(-fg))) * fu;
            }
        }

        kernel void bench_fp8_moe_down_fused(
            device const uchar* staging [[buffer(0)]],
            device const float* intermediateVector [[buffer(1)]],
            device float* outputAccumulator [[buffer(2)]],
            device const uint32_t* expertIds [[buffer(3)]],
            device const float* routingWeights [[buffer(4)]],
            constant uint64_t& expertSize [[buffer(5)]],
            constant uint32_t& intermediateDim [[buffer(6)]],
            constant uint32_t& hiddenDim [[buffer(7)]],
            constant uint32_t& topK [[buffer(8)]],
            uint2 tgPos [[threadgroup_position_in_grid]],
            uint laneId [[thread_index_in_simdgroup]]
        ) {
            uint d = tgPos.x;
            uint slot = tgPos.y;
            if (d >= hiddenDim || slot >= topK) return;
            uint64_t base = (uint64_t)slot * expertSize;
            float ds = bf16_to_fp32(((device const ushort*)(staging + base + 3147776))[d]);
            device const uchar* dRow = staging + base + 2099200 + (uint64_t)d * intermediateDim;
            device const float* interPtr = intermediateVector + ((uint64_t)slot * intermediateDim);
            float dot = 0.0f;
            for (uint32_t baseI = laneId * 8; baseI < intermediateDim; baseI += 32 * 8) {
                uchar4 d0 = *(device const uchar4*)(dRow + baseI);
                uchar4 d1 = *(device const uchar4*)(dRow + baseI + 4);
                float4 i0 = *(device const float4*)(interPtr + baseI);
                float4 i1 = *(device const float4*)(interPtr + baseI + 4);
                dot += (unpack_e4m3(d0.x)*i0.x)+(unpack_e4m3(d0.y)*i0.y)+(unpack_e4m3(d1.x)*i1.x)+(unpack_e4m3(d1.y)*i1.y)
                     + (unpack_e4m3(d0.z)*i0.z)+(unpack_e4m3(d0.w)*i0.w)+(unpack_e4m3(d1.z)*i1.z)+(unpack_e4m3(d1.w)*i1.w);
            }
            dot = simd_sum(dot);
            if (laneId == 0) {
                outputAccumulator[d] += routingWeights[slot] * (dot * ds);
            }
        }
        """
        let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("bench_fused.metal")
        try? (src + fused).write(to: tmp, atomically: true, encoding: .utf8)
        gpuMetalPath = tmp.path
    }

    guard let gpu = GPU(metalPath: gpuMetalPath) else {
        print("Metal init failed")
        exit(1)
    }
    print("Metal OK: \(gpu.device.name)")
    print("")

    // Buffers (same sizes as app: staging = max(64MB, 16*expert_size))
    let stagingSize = max(16 * 4194304, 16 * gExpertSize)
    guard let stagingA = gpu.device.makeBuffer(length: stagingSize, options: .storageModeShared),
          let stagingB = gpu.device.makeBuffer(length: stagingSize, options: .storageModeShared),
          let hBuf = gpu.device.makeBuffer(length: HIDDEN * 4, options: .storageModeShared),
          let xNorm = gpu.device.makeBuffer(length: HIDDEN * 4, options: .storageModeShared),
          let inter = gpu.device.makeBuffer(length: TOPK * INTER * 4, options: .storageModeShared),
          let hMlp = gpu.device.makeBuffer(length: HIDDEN * 4, options: .storageModeShared),
          let gamma = gpu.device.makeBuffer(length: HIDDEN * 2, options: .storageModeShared),
          let routerW = gpu.device.makeBuffer(length: gNumExperts * HIDDEN * 2, options: .storageModeShared),
          let rIdx = gpu.device.makeBuffer(length: 16 * 4, options: .storageModeShared),
          let rW = gpu.device.makeBuffer(length: 16 * 4, options: .storageModeShared),
          let idsBuf = gpu.device.makeBuffer(length: 16 * 4, options: .storageModeShared),
          let weightsBuf = gpu.device.makeBuffer(length: 16 * 4, options: .storageModeShared) else {
        print("buffer alloc failed"); exit(1)
    }
    for b in [stagingA, stagingB, hBuf, xNorm, inter, hMlp, gamma, routerW] {
        let p = b.contents().assumingMemoryBound(to: UInt8.self)
        var x: UInt32 = 0x1234567
        for i in stride(from: 0, to: b.length, by: 4096) {
            x = x &* 1664525 &+ 1013904223
            p[i] = UInt8(truncatingIfNeeded: x)
        }
    }

    let bytesPerToken = Double(gNumLayers * TOPK * gExpertSize)
    let comp = MoEComponents(
        gateW: findComponent(layout, "gate_proj", false), gateS: findComponent(layout, "gate_proj", true),
        upW: findComponent(layout, "up_proj", false), upS: findComponent(layout, "up_proj", true),
        downW: findComponent(layout, "down_proj", false), downS: findComponent(layout, "down_proj", true))
    print("component offsets: gateW=\(comp.gateW) gateS=\(comp.gateS) upW=\(comp.upW) upS=\(comp.upS) downW=\(comp.downW) downS=\(comp.downS)")
    print("")

    // ============================ T1: sequential cold ============================
    print("--- T1: single-stream sequential read, one layer file (\(String(format: "%.0f", Double(gExpertSize * gNumExperts) / 1048576.0)) MB), 256KB chunks, F_NOCACHE ---")
    do {
        let fd = openLayerFD(packedDir, layer: 7, noCache: true)
        let len = Int64(gExpertSize) * Int64(gNumExperts)
        let buf = UnsafeMutableRawPointer.allocate(byteCount: 262144, alignment: 4096)
        defer { buf.deallocate(); close(fd) }
        _ = preadFull(fd, buf, 0, 262144) // stabilize
        var off: off_t = 0
        var total: Int64 = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        while off < len {
            let n = pread(fd, buf, 262144, off)
            if n <= 0 { print("   ⚠️ pread returned \(n) at offset \(off), errno=\(errno)"); break }
            off += off_t(n); total += Int64(n)
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print("   \(String(format: "%.2f", Double(total) / GB)) GB in \(String(format: "%.3f", dt))s -> \(String(format: "%.2f", Double(total) / GB / dt)) GB/s")
    }

    // ============================ T2: 8-way parallel sequential ============================
    print("--- T2: 8-thread parallel sequential read of same file (F_NOCACHE) ---")
    do {
        let fd = openLayerFD(packedDir, layer: 7, noCache: true)
        let len = Int64(gExpertSize) * Int64(gNumExperts)
        let bufs = (0..<8).map { _ in UnsafeMutableRawPointer.allocate(byteCount: 262144, alignment: 4096) }
        defer { bufs.forEach { $0.deallocate() }; close(fd) }
        let chunkBytes = Int(len / 8)
        var totalRead: Int64 = 0
        let acc = NSLock()
        let t0 = CFAbsoluteTimeGetCurrent()
        DispatchQueue.concurrentPerform(iterations: 8) { i in
            var off = off_t(i) * off_t(chunkBytes)
            let end = off_t(i + 1) * off_t(chunkBytes)
            var local: Int64 = 0
            while off < end {
                let n = pread(fd, bufs[i], 262144, off)
                if n <= 0 { print("   ⚠️ worker \(i) pread returned \(n) at offset \(off), errno=\(errno)"); break }
                off += off_t(n); local += Int64(n)
            }
            acc.lock(); totalRead += local; acc.unlock()
        }
        let dt = CFAbsoluteTimeGetCurrent() - t0
        print("   \(String(format: "%.2f", Double(totalRead) / GB)) GB in \(String(format: "%.3f", dt))s -> \(String(format: "%.2f", Double(totalRead) / GB / dt)) GB/s")
    }

    // ============================ T3: app decode pread pattern ============================
    print("--- T3: app's exact decode pread pattern: \(gNumLayers) layers x \(TOPK) experts x \(String(format: "%.1f", Double(gExpertSize) / 1048576.0)) MB ---")
    print("    (page-cache fds like the app, concurrentPerform from Task.detached .userInitiated)")
    do {
        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }
        let rng = Rand(seed: 0xDA7A4A)
        var idSeq: [[Int]] = []
        for _ in 0..<gTokens { for _ in 0..<gNumLayers { idSeq.append(rng.tokenExperts()) } }

        func runPass(label: String, concurrent: Bool) {
            var layerMs: [Double] = []
            var bytesRead: Int64 = 0
            let t0 = CFAbsoluteTimeGetCurrent()
            let sem = DispatchSemaphore(value: 0)
            let lock = NSLock()
            Task.detached(priority: .userInitiated) {
                for (idx, ids) in idSeq.enumerated() {
                    let layerIdx = idx % gNumLayers
                    let tl = CFAbsoluteTimeGetCurrent()
                    let raw = stagingA.contents()
                    var tasks: [ExpertTask] = []
                    for (slot, e) in ids.enumerated() {
                        tasks.append(ExpertTask(fd: fds[layerIdx], dst: raw.advanced(by: slot * gExpertSize),
                                                offset: off_t(e * gExpertSize), size: gExpertSize))
                    }
                    if concurrent {
                        dispatchPreads(&tasks)
                    } else {
                        for i in tasks.indices {
                            tasks[i].result = preadFull(tasks[i].fd, tasks[i].dst, tasks[i].offset, tasks[i].size)
                        }
                    }
                    lock.lock()
                    layerMs.append((CFAbsoluteTimeGetCurrent() - tl) * 1000)
                    for t in tasks {
                        if t.result != t.size {
                            print("   ⚠️ short pread layer \(layerIdx): got \(t.result)/\(t.size) bytes")
                        }
                        bytesRead += Int64(t.result)
                    }
                    lock.unlock()
                }
                sem.signal()
            }
            sem.wait()
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let mean = layerMs.reduce(0, +) / Double(layerMs.count)
            let actualGB = Double(bytesRead) / GB
            print("   [\(label)] total \(String(format: "%.3f", dt))s, \(String(format: "%.3f", actualGB)) GB actually read -> \(String(format: "%.2f", actualGB / dt)) GB/s | layer IO mean \(String(format: "%.2f", mean))ms p50 \(String(format: "%.2f", pct(layerMs, 50)))ms p95 \(String(format: "%.2f", pct(layerMs, 95)))ms")
        }
        runPass(label: "pass1 (cold-ish)  ", concurrent: true)
        runPass(label: "pass2 (warm)      ", concurrent: true)
        runPass(label: "serial (cold-ish) ", concurrent: false)
    }

    // ============================ T4: GPU MoE phase (real kernels) ============================
    print("--- T4: GPU cost of decode MoE phase (\(TOPK) experts, 16 dispatches + barriers, app kernels) ---")
    do {
        let rng = Rand(seed: 0xBEEF)
        var samples: [Double] = []
        for _ in 0..<40 {
            let ids = rng.tokenExperts()
            let cmd = gpu.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            encodeMoEPhase(enc: enc, gpu: gpu, staging: stagingA, inter: inter, hMlp: hMlp,
                           xNorm: xNorm, ids: ids, expertSize: gExpertSize, comp: comp,
                           idsBuf: idsBuf, weightsBuf: weightsBuf, fused: false)
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cmd.commit(); cmd.waitUntilCompleted()
            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        let mean = samples.reduce(0, +) / Double(samples.count)
        print("   per-layer MoE GPU (incl commit+wait): mean \(String(format: "%.2f", mean))ms p50 \(String(format: "%.2f", pct(samples, 50)))ms -> per-token est \(String(format: "%.1f", mean * Double(gNumLayers)))ms")
    }

    // ============================ T4b: fused variant ============================
    print("--- T4b: fused variant (all \(TOPK) experts in 2 dispatches) ---")
    do {
        if gpu.pipe("bench_fp8_moe_gate_up_fused") != nil {
            let rng = Rand(seed: 0xBEEF)
            var samples: [Double] = []
            for _ in 0..<40 {
                let ids = rng.tokenExperts()
                let cmd = gpu.queue.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                encodeMoEPhase(enc: enc, gpu: gpu, staging: stagingA, inter: inter, hMlp: hMlp,
                               xNorm: xNorm, ids: ids, expertSize: gExpertSize, comp: comp,
                               idsBuf: idsBuf, weightsBuf: weightsBuf, fused: true)
                enc.endEncoding()
                let t0 = CFAbsoluteTimeGetCurrent()
                cmd.commit(); cmd.waitUntilCompleted()
                samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            let mean = samples.reduce(0, +) / Double(samples.count)
            print("   fused per-layer GPU: mean \(String(format: "%.2f", mean))ms -> per-token est \(String(format: "%.1f", mean * Double(gNumLayers)))ms")
        } else {
            print("   (fused kernels unavailable - skipped)")
        }
    }

    // ============================ T5: router phase ============================
    print("--- T5: router phase GPU (rmsnorm_bf16 + moe_router_topk_bf16, \(gNumExperts) experts) ---")
    do {
        var samples: [Double] = []
        for _ in 0..<100 {
            let cmd = gpu.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            encodeRouter(enc: enc, gpu: gpu, h: hBuf, xNorm: xNorm, gamma: gamma, routerW: routerW,
                         indices: rIdx, weights: rW, numExperts: gNumExperts, topK: TOPK)
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cmd.commit(); cmd.waitUntilCompleted()
            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        let mean = samples.reduce(0, +) / Double(samples.count)
        print("   router phase mean \(String(format: "%.3f", mean))ms -> per-token est \(String(format: "%.1f", mean * Double(gNumLayers)))ms")
    }

    // ============================ T6: serialized full token (CURRENT app behavior) ============================
    print("--- T6: FULL serialized token sim (current app pipeline: router -> IO -> MoE per layer) ---")
    do {
        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }
        let rng = Rand(seed: 0xDA7A4A)
        var tokenMs: [Double] = []
        var routerMs: [Double] = []
        var ioMs: [Double] = []
        var moeMs: [Double] = []

        let sem = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            for _ in 0..<gTokens {
                let tTok = CFAbsoluteTimeGetCurrent()
                for l in 0..<gNumLayers {
                    let ids = rng.tokenExperts()

                    // Phase A: router GPU + blocking sync
                    let tA = CFAbsoluteTimeGetCurrent()
                    let cmdA = gpu.queue.makeCommandBuffer()!
                    let encA = cmdA.makeComputeCommandEncoder()!
                    encodeRouter(enc: encA, gpu: gpu, h: hBuf, xNorm: xNorm, gamma: gamma, routerW: routerW,
                                 indices: rIdx, weights: rW, numExperts: gNumExperts, topK: TOPK)
                    encA.endEncoding()
                    cmdA.commit(); cmdA.waitUntilCompleted()
                    routerMs.append((CFAbsoluteTimeGetCurrent() - tA) * 1000)

                    // IO: 8 preads exactly like ExpertIOThreadPool.dispatchSync
                    let tIO = CFAbsoluteTimeGetCurrent()
                    let raw = stagingA.contents()
                    var tasks: [ExpertTask] = []
                    for (slot, e) in ids.enumerated() {
                        tasks.append(ExpertTask(fd: fds[l], dst: raw.advanced(by: slot * gExpertSize),
                                                offset: off_t(e * gExpertSize), size: gExpertSize))
                    }
                    dispatchPreads(&tasks)
                    ioMs.append((CFAbsoluteTimeGetCurrent() - tIO) * 1000)

                    // Phase B: MoE GPU + blocking sync
                    let tB = CFAbsoluteTimeGetCurrent()
                    let cmdB = gpu.queue.makeCommandBuffer()!
                    let encB = cmdB.makeComputeCommandEncoder()!
                    encodeMoEPhase(enc: encB, gpu: gpu, staging: stagingA, inter: inter, hMlp: hMlp,
                                   xNorm: xNorm, ids: ids, expertSize: gExpertSize, comp: comp,
                                   idsBuf: idsBuf, weightsBuf: weightsBuf, fused: false)
                    encB.endEncoding()
                    cmdB.commit(); cmdB.waitUntilCompleted()
                    moeMs.append((CFAbsoluteTimeGetCurrent() - tB) * 1000)
                }
                tokenMs.append((CFAbsoluteTimeGetCurrent() - tTok) * 1000)
            }
            sem.signal()
        }
        sem.wait()
        let meanTok = tokenMs.reduce(0, +) / Double(tokenMs.count)
        let meanRouter = routerMs.reduce(0, +) / Double(routerMs.count)
        let meanIO = ioMs.reduce(0, +) / Double(ioMs.count)
        let meanMoe = moeMs.reduce(0, +) / Double(moeMs.count)
        let perLayer = meanTok / Double(gNumLayers)
        print("   per-token: \(String(format: "%.0f", meanTok))ms -> \(String(format: "%.2f", 1000.0 / meanTok)) tok/s -> effective SSD bandwidth \(String(format: "%.2f", bytesPerToken / GB / (meanTok / 1000))) GB/s")
        print("   per-layer breakdown: router \(String(format: "%.2f", meanRouter))ms | IO \(String(format: "%.2f", meanIO))ms | MoE GPU \(String(format: "%.2f", meanMoe))ms | sync/CPU ~\(String(format: "%.2f", perLayer - meanRouter - meanIO - meanMoe))ms")
        print("   GPU busy: ~\(String(format: "%.0f", (meanRouter + meanMoe) / perLayer * 100))% | IO busy: ~\(String(format: "%.0f", meanIO / perLayer * 100))%")
    }

    // ============================ T7: pipelined upper bound ============================
    print("--- T7: pipelined upper bound (oracle prefetch of layer l+1 during layer l GPU work) ---")
    do {
        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }
        let rng = Rand(seed: 0xDA7A4A)
        var allIds: [[Int]] = []
        for _ in 0..<gTokens { for _ in 0..<gNumLayers { allIds.append(rng.tokenExperts()) } }

        var tokenMs: [Double] = []
        let sem = DispatchSemaphore(value: 0)
        let prefetchQ = DispatchQueue(label: "bench.prefetch", qos: .userInitiated, attributes: .concurrent)
        let stateLock = NSLock()
        var prefetchedIdx = -1

        Task.detached(priority: .userInitiated) {
            for tok in 0..<gTokens {
                let tTok = CFAbsoluteTimeGetCurrent()
                for l in 0..<gNumLayers {
                    let idx = tok * gNumLayers + l
                    let ids = allIds[idx]

                    // wait for prefetch of this layer (l==0 within a token is never prefetched)
                    if l > 0 {
                        while true {
                            stateLock.lock()
                            let ok = prefetchedIdx >= idx
                            stateLock.unlock()
                            if ok { break }
                            usleep(50)
                        }
                    }

                    // router GPU
                    let cmdA = gpu.queue.makeCommandBuffer()!
                    let encA = cmdA.makeComputeCommandEncoder()!
                    encodeRouter(enc: encA, gpu: gpu, h: hBuf, xNorm: xNorm, gamma: gamma, routerW: routerW,
                                 indices: rIdx, weights: rW, numExperts: gNumExperts, topK: TOPK)
                    encA.endEncoding()
                    cmdA.commit(); cmdA.waitUntilCompleted()

                    // MoE GPU: commit first, then kick off prefetch for layer l+1, then wait (overlap)
                    let cmdB = gpu.queue.makeCommandBuffer()!
                    let encB = cmdB.makeComputeCommandEncoder()!
                    encodeMoEPhase(enc: encB, gpu: gpu, staging: (l % 2 == 0) ? stagingA : stagingB,
                                   inter: inter, hMlp: hMlp, xNorm: xNorm, ids: ids,
                                   expertSize: gExpertSize, comp: comp, idsBuf: idsBuf,
                                   weightsBuf: weightsBuf, fused: false)
                    encB.endEncoding()
                    cmdB.commit()

                    if l + 1 < gNumLayers {
                        let nextIds = allIds[idx + 1]
                        let dstBuf = ((l + 1) % 2 == 0) ? stagingA : stagingB
                        let targetLayer = l + 1
                        let targetIdx = idx + 1
                        prefetchQ.async {
                            let raw = dstBuf.contents()
                            var tasks: [ExpertTask] = []
                            for (slot, e) in nextIds.enumerated() {
                                tasks.append(ExpertTask(fd: fds[targetLayer], dst: raw.advanced(by: slot * gExpertSize),
                                                        offset: off_t(e * gExpertSize), size: gExpertSize))
                            }
                            dispatchPreads(&tasks)
                            stateLock.lock(); prefetchedIdx = max(prefetchedIdx, targetIdx); stateLock.unlock()
                        }
                    }
                    cmdB.waitUntilCompleted()
                }
                tokenMs.append((CFAbsoluteTimeGetCurrent() - tTok) * 1000)
            }
            sem.signal()
        }
        sem.wait()
        let meanTok = tokenMs.reduce(0, +) / Double(tokenMs.count)
        print("   per-token: \(String(format: "%.0f", meanTok))ms -> \(String(format: "%.2f", 1000.0 / meanTok)) tok/s -> effective SSD bandwidth \(String(format: "%.2f", bytesPerToken / GB / (meanTok / 1000))) GB/s")
    }

    // ============================ T13: PINNED backbone (the fix) ============================
    // Implements the app fix: pread model_weights.bin once into an ANONYMOUS
    // .storageModeShared MTLBuffer, then run the same app-shaped per-layer loop as T10
    // reading the backbone from the pinned buffer instead of the mmap. Expected: the
    // ~945 ms/layer re-fault collapses to sub-ms DRAM reads.
    print("--- T13: full token sim with PINNED backbone (pread model_weights.bin -> anon buffer) ---")
    do {
        let pinPath = dir.appendingPathComponent("model_weights.bin")
        let pinSize = Int(((try? FileManager.default.attributesOfItem(atPath: pinPath.path))?[.size] as? NSNumber)?.uint64Value ?? 0)
        if pinSize <= 0 || pinSize > Int(gpu.device.maxBufferLength) {
            print("   model_weights.bin missing or too large for single buffer (\(pinSize) bytes, max \(gpu.device.maxBufferLength)) - skipped")
        } else {
            guard let pinned = gpu.device.makeBuffer(length: pinSize, options: .storageModeShared) else {
                print("   pin alloc failed"); return
            }
            let fdPin = open(pinPath.path, O_RDONLY | O_CLOEXEC)
            guard fdPin >= 0 else { print("   open failed errno=\(errno)"); return }
            defer { close(fdPin) }
            let tPin = CFAbsoluteTimeGetCurrent()
            // 8-way parallel pread of disjoint ranges (same mechanism as the app fix)
            let workers = 8
            let rangeLen = pinSize / workers
            DispatchQueue.concurrentPerform(iterations: workers) { i in
                var done = 0
                let end = (i == workers - 1) ? pinSize - (workers - 1) * rangeLen : rangeLen
                let base = pinned.contents().advanced(by: i * rangeLen)
                var fileOff = off_t(i) * off_t(rangeLen)
                while done < end {
                    let toRead = min(262144, end - done)
                    let n = pread(fdPin, base.advanced(by: done), toRead, fileOff)
                    if n <= 0 { break }
                    done += n
                    fileOff += off_t(n)
                }
            }
            print("   pinned \(String(format: "%.2f", Double(pinSize) / GB)) GB in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - tPin))s")

            // every-byte reader over the pinned (anonymous) buffer
            let readAll = gpu.pipe("bench_read_all")
            let pageSize = Int(vm_page_size)

            func pinnedBackboneRead(_ offset: Int, _ length: Int) -> Double {
                guard let ra = readAll else { return 0 }
                let cmd = gpu.queue.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                let chunks = length / 256
                enc.setComputePipelineState(ra)
                enc.setBuffer(pinned, offset: offset, index: 0)
                enc.setBuffer(rIdx, offset: 0, index: 1)
                var nVal: UInt32 = UInt32(chunks)
                enc.setBytes(&nVal, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: Int(nVal), height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: min(Int(nVal), 512), height: 1, depth: 1))
                enc.endEncoding()
                let t0 = CFAbsoluteTimeGetCurrent()
                cmd.commit(); cmd.waitUntilCompleted()
                return (CFAbsoluteTimeGetCurrent() - t0) * 1000
            }

            let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
            defer { fds.forEach { close($0) } }
            let rng = Rand(seed: 0xDA7A4A)
            var tokenMs: [Double] = []
            var perTokenBackboneMs: [Double] = []
            let sem = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                for tok in 0..<gTokens {
                    let tTok = CFAbsoluteTimeGetCurrent()
                    var tokBackbone: Double = 0
                    for l in 0..<gNumLayers {
                        let ids = rng.tokenExperts()
                        // 1. Backbone read from PINNED anonymous memory (spans vary per layer)
                        let maxOff = pinSize - 61612064 - 4096
                        let off = (l * 61537001) % max(1, maxOff)
                        tokBackbone += pinnedBackboneRead(off, 61612064)

                        // 2. Router GPU + sync
                        let cmdA = gpu.queue.makeCommandBuffer()!
                        let encA = cmdA.makeComputeCommandEncoder()!
                        encodeRouter(enc: encA, gpu: gpu, h: hBuf, xNorm: xNorm, gamma: gamma, routerW: routerW,
                                     indices: rIdx, weights: rW, numExperts: gNumExperts, topK: TOPK)
                        encA.endEncoding()
                        cmdA.commit(); cmdA.waitUntilCompleted()

                        // 3. Pread 8 experts
                        let raw = stagingA.contents()
                        var tasks: [ExpertTask] = []
                        for (slot, e) in ids.enumerated() {
                            tasks.append(ExpertTask(fd: fds[l], dst: raw.advanced(by: slot * gExpertSize),
                                                    offset: off_t(e * gExpertSize), size: gExpertSize))
                        }
                        dispatchPreads(&tasks)

                        // 4. MoE GPU + sync
                        let cmdB = gpu.queue.makeCommandBuffer()!
                        let encB = cmdB.makeComputeCommandEncoder()!
                        encodeMoEPhase(enc: encB, gpu: gpu, staging: stagingA, inter: inter, hMlp: hMlp,
                                       xNorm: xNorm, ids: ids, expertSize: gExpertSize, comp: comp,
                                       idsBuf: idsBuf, weightsBuf: weightsBuf, fused: false)
                        encB.endEncoding()
                        cmdB.commit(); cmdB.waitUntilCompleted()
                    }
                    tokenMs.append((CFAbsoluteTimeGetCurrent() - tTok) * 1000)
                    perTokenBackboneMs.append(tokBackbone)
                }
                sem.signal()
            }
            sem.wait()
            for (tok, ms) in tokenMs.enumerated() {
                print("   token \(tok): \(String(format: "%.0f", ms))ms (\(String(format: "%.2f", 1000.0 / ms)) tok/s) | pinned backbone reads \(String(format: "%.1f", perTokenBackboneMs[tok]))ms total")
            }
        }
    }

    // ============================ T14: pipelined expert decode (the fix #2) ============================
    // App-shaped pipelined loop: wait(prefetch l) -> consume hits, sync pread misses ->
    // kick prefetch(l+1, Markov-predicted) -> MoE GPU. Routing is generated ONCE with
    // realistic cross-layer locality (sticky hot sets shared between adjacent layers) and
    // replayed identically in both variants so the only difference is pipelining.
    print("--- T14: pipelined expert decode w/ Markov predictor (sticky routing) ---")
    do {
        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }

        // Sticky expert routing with cross-layer correlation
        var rg = Rand(seed: 0x5EED)
        var hot: [[Int]] = []
        for l in 0..<gNumLayers {
            var s = Set<Int>()
            if l > 0 { for e in hot[l - 1].prefix(20) { s.insert(e) } }
            while s.count < 40 { s.insert(rg.next(gNumExperts)) }
            hot.append(Array(s))
        }
        var routing: [[[Int]]] = []
        for _ in 0..<gTokens {
            var tokEx: [[Int]] = []
            for l in 0..<gNumLayers {
                var s = Set<Int>()
                let pool = Set(hot[l]).union(l + 1 < gNumLayers ? Set(hot[l + 1]) : [])
                let poolArr = Array(pool)
                for _ in 0..<5 { s.insert(poolArr[rg.next(poolArr.count)]) }
                while s.count < TOPK { s.insert(rg.next(gNumExperts)) }
                tokEx.append(Array(s))
            }
            routing.append(tokEx)
        }

        // Markov transition predictor (mirrors ExpertTransitionTracker)
        var transitions: [Int: [Int: [Int: Int]]] = [:]
        func recordTrans(_ l: Int, _ src: [Int], _ dst: [Int]) {
            var layerMap = transitions[l] ?? [:]
            for s in src {
                var dstMap = layerMap[s] ?? [:]
                for d in dst { dstMap[d, default: 0] += 1 }
                layerMap[s] = dstMap
            }
            transitions[l] = layerMap
        }
        func markovPredict(_ l: Int, _ ids: [Int], _ topN: Int) -> [Int] {
            guard let layerMap = transitions[l], !ids.isEmpty else { return [] }
            var score: [Int: Int] = [:]
            for id in ids {
                if let dstMap = layerMap[id] { for (d, c) in dstMap { score[d, default: 0] += c } }
            }
            return score.sorted(by: { $0.value > $1.value }).prefix(topN).map { $0.key }
        }

        for pipelined in [false, true] {
            var resident: [[Int: Int]] = [[:], [:]]       // [bufferIdx] expertId -> slot
            var residentLayer: [Int] = [-1, -1]
            var nextSlot: [Int] = [0, 0]
            var pending: (layer: Int, sem: DispatchSemaphore)? = nil
            let lock = NSLock()
            let pfQueue = DispatchQueue(label: "t14.pf", qos: .userInitiated, attributes: .concurrent)
            let capacity = stagingSize / gExpertSize

            var tokenMs: [Double] = []
            var ioMsTotal: [Double] = []
            var hitRates: [Double] = []
            let sem = DispatchSemaphore(value: 0)
            Task.detached(priority: .userInitiated) {
                for tok in 0..<gTokens {
                    let tTok = CFAbsoluteTimeGetCurrent()
                    var tokIo: Double = 0
                    var tokHits = 0
                    var tokTotal = 0
                    for l in 0..<gNumLayers {
                        let actual = routing[tok][l]
                        // 1. wait for prefetch(l)
                        lock.lock()
                        if let p = pending, p.layer == l {
                            let s = p.sem
                            pending = nil
                            lock.unlock()
                            _ = s.wait(timeout: .now() + 5.0)
                        } else {
                            lock.unlock()
                        }
                        // 2. consume: hits from prefetched map, misses sync-pread
                let bufIdx = l & 1
                let stagingBuf = (bufIdx == 0) ? stagingA : stagingB
                var slotOf: [Int: Int] = [:]
                lock.lock()
                if residentLayer[bufIdx] == l {
                    slotOf = resident[bufIdx]
                } else {
                    nextSlot[bufIdx] = 0
                }
                lock.unlock()
                var hits = 0
                for e in actual where slotOf[e] != nil { hits += 1 }
                var missTasks: [ExpertTask] = []
                for e in actual where slotOf[e] == nil {
                    lock.lock()
                    let slot = nextSlot[bufIdx]
                    nextSlot[bufIdx] = slot + 1
                    lock.unlock()
                    slotOf[e] = slot
                    missTasks.append(ExpertTask(fd: fds[l], dst: stagingBuf.contents().advanced(by: slot * gExpertSize),
                                                offset: off_t(e * gExpertSize), size: gExpertSize))
                }
                let tIo = CFAbsoluteTimeGetCurrent()
                if !pipelined {
                    // serialized baseline: ALL experts synchronously (ignore prefetched slots)
                    slotOf = [:]
                    for (idx, e) in actual.enumerated() {
                        slotOf[e] = idx
                        missTasks.append(ExpertTask(fd: fds[l], dst: stagingBuf.contents().advanced(by: idx * gExpertSize),
                                                    offset: off_t(e * gExpertSize), size: gExpertSize))
                    }
                }
                if !missTasks.isEmpty {
                    dispatchPreads(&missTasks)
                }
                        let ioMs = (CFAbsoluteTimeGetCurrent() - tIo) * 1000
                        tokIo += ioMs
                        tokHits += hits
                        tokTotal += actual.count

                        // 3. kick prefetch(l+1) — temporal signal first (previous token's set at
                        // layer l+1), Markov fills remaining budget (mirrors the app)
                        if pipelined && l + 1 < gNumLayers {
                            let temporal = (tok > 0) ? routing[tok - 1][l + 1] : []
                            var predicted = temporal
                            let markov = markovPredict(l, actual, 10)
                            for id in markov where !predicted.contains(id) && predicted.count < 13 {
                                predicted.append(id)
                            }
                            if !predicted.isEmpty {
                                let tBuf = (l + 1) & 1
                                let tStaging = (tBuf == 0) ? stagingA : stagingB
                                let s2 = DispatchSemaphore(value: 0)
                                lock.lock()
                                resident[tBuf] = [:]
                                residentLayer[tBuf] = -1
                                nextSlot[tBuf] = min(predicted.count, tStaging.length / gExpertSize - TOPK)
                                pending = (l + 1, s2)
                                lock.unlock()
                                pfQueue.async {
                                    var map: [Int: Int] = [:]
                                    var tasks2: [ExpertTask] = []
                                    for (i, e) in predicted.prefix(min(predicted.count, tStaging.length / gExpertSize - TOPK)).enumerated() {
                                        map[e] = i
                                        tasks2.append(ExpertTask(fd: fds[l + 1], dst: tStaging.contents().advanced(by: i * gExpertSize),
                                                                 offset: off_t(e * gExpertSize), size: gExpertSize))
                                    }
                                    if !tasks2.isEmpty { dispatchPreads(&tasks2) }
                                    lock.lock()
                                    resident[tBuf] = map
                                    residentLayer[tBuf] = l + 1
                                    lock.unlock()
                                    s2.signal()
                                }
                            }
                        }
                        if !pipelined && l > 0 { recordTrans(l - 1, routing[tok][l - 1], actual) }
                        if pipelined && l > 0 { recordTrans(l - 1, routing[tok][l - 1], actual) }

                        // 4. MoE GPU (app kernels, same as T6)
                        let cmdB = gpu.queue.makeCommandBuffer()!
                        let encB = cmdB.makeComputeCommandEncoder()!
                        encodeMoEPhase(enc: encB, gpu: gpu, staging: stagingBuf, inter: inter, hMlp: hMlp,
                                       xNorm: xNorm, ids: actual, expertSize: gExpertSize, comp: comp,
                                       idsBuf: idsBuf, weightsBuf: weightsBuf, fused: false)
                        encB.endEncoding()
                        cmdB.commit(); cmdB.waitUntilCompleted()
                    }
                    tokenMs.append((CFAbsoluteTimeGetCurrent() - tTok) * 1000)
                    ioMsTotal.append(tokIo)
                    hitRates.append(Double(tokHits) / Double(max(tokTotal, 1)))
                }
                sem.signal()
            }
            sem.wait()
            let meanTok = tokenMs.reduce(0, +) / Double(tokenMs.count)
            let meanIo = ioMsTotal.reduce(0, +) / Double(ioMsTotal.count)
            let meanHit = hitRates.reduce(0, +) / Double(hitRates.count)
            // skip token 0 (predictor cold) in the mean
            let warmTok = tokenMs.dropFirst().reduce(0, +) / Double(max(1, tokenMs.count - 1))
            print("   [\(pipelined ? "pipelined" : "serialized")] all-tokens \(String(format: "%.0f", meanTok))ms avg (\(String(format: "%.2f", 1000.0 / meanTok)) tok/s) | warm tokens (2+) \(String(format: "%.0f", warmTok))ms (\(String(format: "%.2f", 1000.0 / warmTok)) tok/s) | sync IO \(String(format: "%.1f", meanIo))ms/token | hit rate \(String(format: "%.0f", meanHit * 100))%")
        }
    }

    // ============================ T8: mmap cold page-fault path ============================
    // The backbone (attention/GDN weights) and lm_head are read by the GPU straight
    // from memmap'd .safetensors MTLBuffers -> 16KB VM page faults on cold pages.
    print("--- T8: mmap cold page-fault throughput (file-backed pages) ---")
    do {
        var pageFaultSink: Int64 = 0
        let stFiles = (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?
            .filter { $0.hasSuffix(".safetensors") }.sorted() ?? []
        if stFiles.isEmpty {
            print("   (no safetensors found - skipped)")
        } else {
            let firstST = stFiles[0]
            var stURL = dir.appendingPathComponent(firstST)
            // resolve HF symlink (blobs)
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: stURL.path) {
                let resolved = stURL.deletingLastPathComponent().appendingPathComponent(target).standardized
                if FileManager.default.fileExists(atPath: resolved.path) { stURL = resolved }
            }
            let attr = (try? FileManager.default.attributesOfItem(atPath: stURL.path)) ?? [:]
            let fileSize = (attr[.size] as? NSNumber)?.uint64Value ?? 0
            if fileSize < 32 * 1024 * 1024 {
                print("   safetensors too small - skipped (size=\(fileSize))")
            } else {
            let mapLen = Int(min(UInt64(1024 * 1024 * 1024), fileSize - 16 * 1024 * 1024))
            let fd = open(stURL.path, O_RDONLY)
            if fd < 0 {
                print("   open failed errno=\(errno)")
            } else {
                let map = mmap(nil, mapLen, PROT_READ, MAP_SHARED, fd, 16 * 1024 * 1024)
                if map == MAP_FAILED {
                    print("   mmap failed errno=\(errno)")
                } else {
                    defer { munmap(map!, mapLen); close(fd) }
                    if let mtl = gpu.device.makeBuffer(bytesNoCopy: map!, length: mapLen, options: .storageModeShared, deallocator: nil) {
                        // GPU fault path: strided reader kernel over the file-backed buffer
                        if let reader = gpu.pipe("bench_fault_read") {
                            let pageSize = Int(vm_page_size)
                            let nPages = mapLen / pageSize
                            var samples: [Double] = []
                            for rep in 0..<3 {
                                posix_madvise(map!, mapLen, POSIX_MADV_DONTNEED)
                                let cmd = gpu.queue.makeCommandBuffer()!
                                let enc = cmd.makeComputeCommandEncoder()!
                                enc.setComputePipelineState(reader)
                                enc.setBuffer(mtl, offset: 0, index: 0)
                                enc.setBuffer(rIdx, offset: 0, index: 1) // sink
                                var nVal: UInt32 = UInt32(nPages)
                                var psVal: UInt32 = UInt32(pageSize)
                                enc.setBytes(&nVal, length: 4, index: 2)
                                enc.setBytes(&psVal, length: 4, index: 3)
                                enc.dispatchThreads(MTLSize(width: Int(nVal), height: 1, depth: 1),
                                                    threadsPerThreadgroup: MTLSize(width: min(Int(nVal), 512), height: 1, depth: 1))
                                enc.endEncoding()
                                let t0 = CFAbsoluteTimeGetCurrent()
                                cmd.commit(); cmd.waitUntilCompleted()
                                let dt = CFAbsoluteTimeGetCurrent() - t0
                                samples.append(dt)
                                print("   GPU rep\(rep): \(String(format: "%.2f", Double(mapLen) / GB)) GB faulted in \(String(format: "%.3f", dt))s -> \(String(format: "%.2f", Double(mapLen) / GB / dt)) GB/s (\(nVal / 1000)K page faults)")
                            }
                        }
                        // CPU fault path (what WorkingSetManager.primeSlices fallback does)
                        let pageSize2 = Int(vm_page_size)
                        for rep in 0..<3 {
                            posix_madvise(map!, mapLen, POSIX_MADV_DONTNEED)
                            let t0 = CFAbsoluteTimeGetCurrent()
                            let bytePtr = map!.assumingMemoryBound(to: UInt8.self)
                            var dummy: UInt64 = 0
                            for off in stride(from: 0, to: mapLen, by: pageSize2) {
                                dummy &+= UInt64(bytePtr[off])
                            }
                            OSAtomicAdd64(Int64(bitPattern: dummy), &pageFaultSink)
                            let dt = CFAbsoluteTimeGetCurrent() - t0
                            print("   CPU rep\(rep): \(String(format: "%.2f", Double(mapLen) / GB)) GB touched in \(String(format: "%.3f", dt))s -> \(String(format: "%.2f", Double(mapLen) / GB / dt)) GB/s")
                        }
                    }
                }
            }
        }
    }
    }

    // ============================ T9: lm_head logits GEMV ============================
    print("--- T9: lm_head GEMV GPU cost (248K vocab x 2048, BF16, resident buffer) ---")
    do {
        let vocabSize = 248320
        let lmBytes = vocabSize * HIDDEN * 2
        // allocate in chunks to survive 16GB RAM
        guard let lmBuf = gpu.device.makeBuffer(length: lmBytes, options: .storageModeShared),
              let logits = gpu.device.makeBuffer(length: vocabSize * 4, options: .storageModeShared) else {
            print("   lm_head alloc failed (\(Double(lmBytes)/GB) GB)"); return
        }
        let p = lmBuf.contents().assumingMemoryBound(to: UInt16.self)
        for i in stride(from: 0, to: vocabSize * HIDDEN, by: 4096) { p[i] = 0x3F80 }
        if let gemv = gpu.pipe("bf16_gemv_simd") {
            var samples: [Double] = []
            for _ in 0..<20 {
                let cmd = gpu.queue.makeCommandBuffer()!
                let enc = cmd.makeComputeCommandEncoder()!
                enc.setComputePipelineState(gemv)
                enc.setBuffer(lmBuf, offset: 0, index: 0)
                enc.setBuffer(xNorm, offset: 0, index: 1)
                enc.setBuffer(logits, offset: 0, index: 2)
                var wOff: UInt64 = 0
                var inD: UInt32 = UInt32(HIDDEN)
                var outD: UInt32 = UInt32(vocabSize)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: vocabSize, height: 1, depth: 1),
                                         threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
                let t0 = CFAbsoluteTimeGetCurrent()
                cmd.commit(); cmd.waitUntilCompleted()
                samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            let mean = samples.reduce(0, +) / Double(samples.count)
            print("   lm_head GEMV (warm, resident): mean \(String(format: "%.2f", mean))ms -> \(String(format: "%.2f", Double(lmBytes)/GB/(mean/1000))) GB/s effective read")
        }
    }

    // ============================ T10b: fault dynamics isolation ============================
    // A. Same 62MB window, 6 consecutive GPU reads, NO eviction in between:
    //    does it stay fast (fault-once) or re-fault every dispatch?
    // B. Same but with a ~1.2GB "churn" of expert preads between reads (mimic 1 token of
    //    app traffic) -> does the window survive the churn?
    // C. CPU pre-touch (primeSlices-style) then GPU read -> does pre-touch rescue it?
    print("--- T10b: mmap fault dynamics (62 MB window, GPU reader) ---")
    do {
        var shardPaths: [URL] = []
        let stNames = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        for n in stNames {
            var u = dir.appendingPathComponent(n)
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: u.path) {
                let resolved = u.deletingLastPathComponent().appendingPathComponent(target).standardized
                if FileManager.default.fileExists(atPath: resolved.path) { u = resolved }
            }
            shardPaths.append(u)
        }
        guard !shardPaths.isEmpty else { print("   no shards"); return }
        var maps: [(UnsafeMutableRawPointer, Int, Int32)] = []
        for sp in shardPaths {
            let fd = open(sp.path, O_RDONLY)
            guard fd >= 0 else { continue }
            let size = Int(((try? FileManager.default.attributesOfItem(atPath: sp.path))?[.size] as? NSNumber)?.uint64Value ?? 0)
            guard size > 0 else { close(fd); continue }
            let map = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
            guard map != MAP_FAILED else { close(fd); continue }
            maps.append((map!, size, fd))
        }
        defer { for m in maps { munmap(m.0, m.1); close(m.2) } }
        var mtlMaps: [MTLBuffer] = []
        for (ptr, len, _) in maps {
            if let b = gpu.device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                mtlMaps.append(b)
            }
        }
        guard let rd = gpu.pipe("bench_fault_read") else { print("   no reader kernel"); return }
        let backboneBytes = 61612064
        let pageSize = Int(vm_page_size)

        func gpuFaultRead(_ buf: MTLBuffer, _ offset: Int, _ length: Int) -> Double {
            let cmd = gpu.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            let pages = length / pageSize
            enc.setComputePipelineState(rd)
            enc.setBuffer(buf, offset: offset, index: 0)
            enc.setBuffer(rIdx, offset: 0, index: 1)
            var nVal: UInt32 = UInt32(pages)
            var psVal: UInt32 = UInt32(pageSize)
            enc.setBytes(&nVal, length: 4, index: 2)
            enc.setBytes(&psVal, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: Int(nVal), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(Int(nVal), 512), height: 1, depth: 1))
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cmd.commit(); cmd.waitUntilCompleted()
            return (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        // A: consecutive reads, no eviction
        print("   [A] consecutive GPU reads of same 62MB window (no eviction):")
        for rep in 0..<5 {
            let ms = gpuFaultRead(mtlMaps[0], 16 * 1024 * 1024, backboneBytes)
            print("      rep\(rep): \(String(format: "%.1f", ms))ms (\(String(format: "%.2f", Double(backboneBytes) / 1e9 / (ms / 1000))) GB/s)")
        }

        // B: churn with ~1.2GB of expert preads between reads (mimics ~0.5 token of traffic)
        print("   [B] GPU read -> 1.2GB pread churn -> GPU read (does the window survive?):")
        let fdC = openLayerFD(packedDir, layer: 5, noCache: false)
        defer { close(fdC) }
        var rngC = Rand(seed: 0xC9C9)
        let churnBuf = UnsafeMutableRawPointer.allocate(byteCount: 64 * 1024 * 1024, alignment: 4096)
        defer { churnBuf.deallocate() }
        var seq: [Double] = []
        for rep in 0..<6 {
            let ms1 = gpuFaultRead(mtlMaps[0], 16 * 1024 * 1024, backboneBytes)
            var tasks: [ExpertTask] = []
            for i in 0..<8 {
                tasks.append(ExpertTask(fd: fdC, dst: churnBuf.advanced(by: i * 4 * 1024 * 1024),
                                        offset: off_t(rngC.next(300) * gExpertSize), size: 4 * 1024 * 1024))
            }
            for _ in 0..<36 { dispatchPreads(&tasks) } // 36 rounds x 32MB = 1.2GB
            let ms2 = gpuFaultRead(mtlMaps[0], 16 * 1024 * 1024, backboneBytes)
            seq.append(ms2)
            print("      rep\(rep): before-churn \(String(format: "%.1f", ms1))ms -> churn 1.2GB -> after-churn \(String(format: "%.1f", ms2))ms (\(String(format: "%.2f", Double(backboneBytes) / 1e9 / (ms2 / 1000))) GB/s)")
        }

        // C: CPU pre-touch then GPU read, after forcing eviction via churn
        print("   [C] after churn: CPU stride-touch of window -> GPU read:")
        for rep in 0..<2 {
            let mapPtr = maps[0].0
            posix_madvise(mapPtr.advanced(by: 16 * 1024 * 1024), backboneBytes, POSIX_MADV_WILLNEED)
            let bp = mapPtr.advanced(by: 16 * 1024 * 1024).assumingMemoryBound(to: UInt8.self)
            var dummy: UInt64 = 0
            let tc0 = CFAbsoluteTimeGetCurrent()
            for po in stride(from: 0, to: backboneBytes, by: pageSize) {
                dummy &+= UInt64(bp[po])
            }
            OSAtomicAdd64(Int64(bitPattern: dummy), &sinkGlobal)
            let cpuMs = (CFAbsoluteTimeGetCurrent() - tc0) * 1000
            let ms = gpuFaultRead(mtlMaps[0], 16 * 1024 * 1024, backboneBytes)
            print("      rep\(rep): CPU touch \(String(format: "%.1f", cpuMs))ms -> GPU read \(String(format: "%.1f", ms))ms")
        }
    }
    // Replicates the real decode loop data movement:
    //   per layer: GPU fault-reads ~62MB backbone from mmap'd safetensors -> router GPU ->
    //   pread 8 experts -> MoE GPU. Then watches whether the backbone stays page-cached
    //   across tokens (period = 1 token, ~4.2 GB of intervening traffic through the cache).
    print("--- T10: full token sim incl. mmap backbone fault-in (~62 MB/layer GPU reads) ---")
    do {
        // resolve shard symlinks once
        var shardPaths: [URL] = []
        let stNames = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? [])
            .filter { $0.hasSuffix(".safetensors") }.sorted()
        for n in stNames {
            var u = dir.appendingPathComponent(n)
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: u.path) {
                let resolved = u.deletingLastPathComponent().appendingPathComponent(target).standardized
                if FileManager.default.fileExists(atPath: resolved.path) { u = resolved }
            }
            shardPaths.append(u)
        }
        guard !shardPaths.isEmpty else { print("   no shards"); return }

        // map every shard (like the app maps all 16 shards)
        var maps: [(UnsafeMutableRawPointer, Int, Int32)] = []
        for sp in shardPaths {
            let fd = open(sp.path, O_RDONLY)
            guard fd >= 0 else { continue }
            let size = Int(((try? FileManager.default.attributesOfItem(atPath: sp.path))?[.size] as? NSNumber)?.uint64Value ?? 0)
            guard size > 0 else { close(fd); continue }
            let map = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0)
            guard map != MAP_FAILED else { close(fd); continue }
            maps.append((map!, size, fd))
        }
        guard !maps.isEmpty else { print("   mmap failed"); return }
        defer { for m in maps { munmap(m.0, m.1); close(m.2) } }

        var mtlMaps: [MTLBuffer] = []
        for (ptr, len, _) in maps {
            if let b = gpu.device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                mtlMaps.append(b)
            }
        }
        print("   mapped \(mtlMaps.count)/\(maps.count) shards into zero-copy MTLBuffers")

        let backboneBytes = 61612064 // measured mean per-layer backbone from model_weights.json
        let pageSize = Int(vm_page_size)
        let reader = gpu.pipe("bench_fault_read")

        func gpuFaultRead(_ buf: MTLBuffer, _ offset: Int, _ length: Int) -> Double {
            guard let rd = reader else { return 0 }
            let cmd = gpu.queue.makeCommandBuffer()!
            let enc = cmd.makeComputeCommandEncoder()!
            let pages = length / pageSize
            enc.setComputePipelineState(rd)
            enc.setBuffer(buf, offset: offset, index: 0)
            enc.setBuffer(rIdx, offset: 0, index: 1)
            var nVal: UInt32 = UInt32(pages)
            var psVal: UInt32 = UInt32(pageSize)
            enc.setBytes(&nVal, length: 4, index: 2)
            enc.setBytes(&psVal, length: 4, index: 3)
            enc.dispatchThreads(MTLSize(width: Int(nVal), height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(Int(nVal), 512), height: 1, depth: 1))
            enc.endEncoding()
            let t0 = CFAbsoluteTimeGetCurrent()
            cmd.commit(); cmd.waitUntilCompleted()
            return (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }
        let rng = Rand(seed: 0xDA7A4A)
        var tokenMs: [Double] = []
        var perTokenFaultMs: [Double] = []
        let sem = DispatchSemaphore(value: 0)

        Task.detached(priority: .userInitiated) {
            for tok in 0..<gTokens {
                let tTok = CFAbsoluteTimeGetCurrent()
                var tokFault: Double = 0
                for l in 0..<gNumLayers {
                    let ids = rng.tokenExperts()
                    // 1. Backbone fault-in via GPU (mmap'd safetensors), like real Phase A reads
                    let shard = l % maps.count
                    let maxOff = maps[shard].1 - backboneBytes - 16 * 1024 * 1024
                    let off = 16 * 1024 * 1024 + (l * 7919) % max(1, maxOff)
                    tokFault += gpuFaultRead(mtlMaps[shard], off, backboneBytes)

                    // 2. Router GPU + sync
                    let cmdA = gpu.queue.makeCommandBuffer()!
                    let encA = cmdA.makeComputeCommandEncoder()!
                    encodeRouter(enc: encA, gpu: gpu, h: hBuf, xNorm: xNorm, gamma: gamma, routerW: routerW,
                                 indices: rIdx, weights: rW, numExperts: gNumExperts, topK: TOPK)
                    encA.endEncoding()
                    cmdA.commit(); cmdA.waitUntilCompleted()

                    // 3. Pread 8 experts
                    let raw = stagingA.contents()
                    var tasks: [ExpertTask] = []
                    for (slot, e) in ids.enumerated() {
                        tasks.append(ExpertTask(fd: fds[l], dst: raw.advanced(by: slot * gExpertSize),
                                                offset: off_t(e * gExpertSize), size: gExpertSize))
                    }
                    dispatchPreads(&tasks)

                    // 4. MoE GPU + sync
                    let cmdB = gpu.queue.makeCommandBuffer()!
                    let encB = cmdB.makeComputeCommandEncoder()!
                    encodeMoEPhase(enc: encB, gpu: gpu, staging: stagingA, inter: inter, hMlp: hMlp,
                                   xNorm: xNorm, ids: ids, expertSize: gExpertSize, comp: comp,
                                   idsBuf: idsBuf, weightsBuf: weightsBuf, fused: false)
                    encB.endEncoding()
                    cmdB.commit(); cmdB.waitUntilCompleted()
                }
                tokenMs.append((CFAbsoluteTimeGetCurrent() - tTok) * 1000)
                perTokenFaultMs.append(tokFault)
            }
            sem.signal()
        }
        sem.wait()
        let totalTraffic = Double(gNumLayers * TOPK * gExpertSize) / GB + Double(gNumLayers) * Double(backboneBytes) / GB + 0.948
        for (tok, ms) in tokenMs.enumerated() {
            print("   token \(tok): \(String(format: "%.0f", ms))ms (\(String(format: "%.2f", 1000.0 / ms)) tok/s) | backbone fault time \(String(format: "%.0f", perTokenFaultMs[tok]))ms | total movement \(String(format: "%.2f", totalTraffic)) GB -> eff \(String(format: "%.2f", totalTraffic / (ms / 1000))) GB/s")
        }
    }

    // ============================ T11: prefill read pattern ============================
    // runLayerWisePrefill reads ALL unique active experts per layer (up to 256 for prompts
    // >32 tokens) via concurrentPerform into an 807 MB staging buffer.
    print("--- T11: prefill read pattern: K-way concurrent expert preads from one layer file ---")
    do {
        for k in [32, 64, 128, 256] {
            let fd = openLayerFD(packedDir, layer: 3, noCache: false)
            var rng2 = Rand(seed: 0xF111)
            var experts: [Int] = []
            var seen2 = Set<Int>()
            while experts.count < k {
                let e = rng2.next(gNumExperts)
                if seen2.insert(e).inserted { experts.append(e) }
            }
            let raw = UnsafeMutableRawPointer.allocate(byteCount: k * gExpertSize, alignment: 4096)
            defer { raw.deallocate(); close(fd) }
            var tasks: [ExpertTask] = []
            for (slot, e) in experts.enumerated() {
                tasks.append(ExpertTask(fd: fd, dst: raw.advanced(by: slot * gExpertSize),
                                        offset: off_t(e * gExpertSize), size: gExpertSize))
            }
            // warm once (these files were touched in T3/T6)
            var warm = tasks
            dispatchPreads(&warm)
            let t0 = CFAbsoluteTimeGetCurrent()
            dispatchPreads(&tasks)
            let dt = CFAbsoluteTimeGetCurrent() - t0
            let gb = Double(k * gExpertSize) / GB
            print("   K=\(String(format: "%3d", k)): \(String(format: "%.2f", gb)) GB in \(String(format: "%.3f", dt))s -> \(String(format: "%.2f", gb / dt)) GB/s (warm)")
        }
        print("   (for a >32-token prompt this runs 40x - one full 807 MB layer per layer)")
    }

    // ============================ T12: staging buffer alloc cost ============================
    print("--- T12: 807 MB prefill staging MTLBuffer alloc + first-touch zero cost ---")
    do {
        let bytes = 256 * gExpertSize
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let buf = gpu.device.makeBuffer(length: bytes, options: .storageModeShared) else {
            print("   alloc failed"); return
        }
        let t1 = CFAbsoluteTimeGetCurrent()
        let p = buf.contents().assumingMemoryBound(to: UInt8.self)
        var x: UInt8 = 0
        for off in stride(from: 0, to: buf.length, by: 4096) { x = x &+ p[off] }
        OSAtomicAdd64(Int64(x), &sinkGlobal)
        let t2 = CFAbsoluteTimeGetCurrent()
        print("   alloc \(String(format: "%.0f", (t1 - t0) * 1000))ms + touch \(String(format: "%.0f", (t2 - t1) * 1000))ms -> total fixed prefill overhead \(String(format: "%.0f", (t2 - t0) * 1000))ms")
    }

    // ============================ T15: PIPELINE DATA CORRECTNESS ============================
    // Replicates the app's EXACT fixed pipelined consume/kick shape (slot maps, per-kick
    // semaphore, parity buffers, stale-map guard + consumer slot-space ownership) and
    // verifies that every expert the GPU consumes has the CORRECT bytes in its slot
    // (compared against a direct pread of the layer file).
    print("--- T15: pipelined consume/kick DATA CORRECTNESS (app-shaped, fixed) ---")
    do {
        let fds = (0..<gNumLayers).map { openLayerFD(packedDir, layer: $0, noCache: false) }
        defer { fds.forEach { close($0) } }

        var rg = Rand(seed: 0x5EED)
        var hot: [[Int]] = []
        for l in 0..<gNumLayers {
            var s = Set<Int>()
            if l > 0 { for e in hot[l - 1].prefix(20) { s.insert(e) } }
            while s.count < 40 { s.insert(rg.next(gNumExperts)) }
            hot.append(Array(s))
        }
        var routing: [[[Int]]] = []
        for _ in 0..<gTokens {
            var tokEx: [[Int]] = []
            for l in 0..<gNumLayers {
                var s = Set<Int>()
                let pool = Set(hot[l]).union(l + 1 < gNumLayers ? Set(hot[l + 1]) : [])
                let poolArr = Array(pool)
                for _ in 0..<5 { s.insert(poolArr[rg.next(poolArr.count)]) }
                while s.count < TOPK { s.insert(rg.next(gNumExperts)) }
                tokEx.append(Array(s))
            }
            routing.append(tokEx)
        }

        let verifyBuf = UnsafeMutableRawPointer.allocate(byteCount: gExpertSize, alignment: 4096)
        defer { verifyBuf.deallocate() }
        var mismatches = 0
        var checks = 0

        var resident: [[Int: Int]] = [[:], [:]]
        var residentLayer: [Int] = [-1, -1]
        var nextSlot: [Int] = [0, 0]
        var slotWriter: [[String]] = [[], []]
        var pending: (layer: Int, sem: DispatchSemaphore)? = nil
        let lock = NSLock()
        let pfQueue = DispatchQueue(label: "t15.pf", qos: .userInitiated, attributes: .concurrent)
        let capacity = stagingSize / gExpertSize

        for tok in 0..<gTokens {
            for l in 0..<gNumLayers {
                let actual = routing[tok][l]
                lock.lock()
                if let p = pending, p.layer == l {
                    let s = p.sem
                    pending = nil
                    lock.unlock()
                    _ = s.wait(timeout: .now() + 5.0)
                } else {
                    lock.unlock()
                }
                let bufIdx = l & 1
                let stagingBuf = (bufIdx == 0) ? stagingA : stagingB
                var slotOf: [Int: Int] = [:]
                lock.lock()
                if residentLayer[bufIdx] == l {
                    slotOf = resident[bufIdx]
                } else {
                    nextSlot[bufIdx] = 0
                    slotWriter[bufIdx] = [String](repeating: "reset", count: capacity)
                }
                lock.unlock()
                var hits = 0
                for e in actual where slotOf[e] != nil { hits += 1 }
                var missTasks: [ExpertTask] = []
                for e in actual where slotOf[e] == nil {
                    lock.lock()
                    let slot = nextSlot[bufIdx]
                    nextSlot[bufIdx] = slot + 1
                    lock.unlock()
                    guard slot < capacity else { mismatches += 1; continue }
                    slotOf[e] = slot
                    missTasks.append(ExpertTask(fd: fds[l], dst: stagingBuf.contents().advanced(by: slot * gExpertSize),
                                                offset: off_t(e * gExpertSize), size: gExpertSize))
                }
                if !missTasks.isEmpty {
                    dispatchPreads(&missTasks)
                    lock.lock()
                    for t in missTasks {
                        let slot = Int((t.dst - stagingBuf.contents()) / gExpertSize)
                        while slotWriter[bufIdx].count <= slot { slotWriter[bufIdx].append("?") }
                        slotWriter[bufIdx][slot] = "miss L\(l) e\(Int(t.offset / off_t(gExpertSize))) tok\(tok)"
                    }
                    lock.unlock()
                }

                // kick prefetch(l+1) with the CURRENT layer's set as the prediction
                if l + 1 < gNumLayers {
                    let target = l + 1
                    let tokStamp = tok
                    let tBuf = target & 1
                    let tStaging = (tBuf == 0) ? stagingA : stagingB
                    let predicted = actual
                    let s2 = DispatchSemaphore(value: 0)
                    let tSlots = min(predicted.count, tStaging.length / gExpertSize - TOPK)
                    lock.lock()
                    resident[tBuf] = [:]
                    residentLayer[tBuf] = -1
                    nextSlot[tBuf] = tSlots
                    pending = (target, s2)
                    lock.unlock()
                    pfQueue.async {
                        var t2: [ExpertTask] = []
                        for (i, e) in predicted.prefix(tSlots).enumerated() {
                            t2.append(ExpertTask(fd: fds[target], dst: tStaging.contents().advanced(by: i * gExpertSize),
                                                 offset: off_t(e * gExpertSize), size: gExpertSize))
                        }
                        var map: [Int: Int] = [:]
                        if !t2.isEmpty {
                            dispatchPreads(&t2)
                            lock.lock()
                            for (i, t) in t2.enumerated() where t.result == t.size {
                                map[predicted[i]] = i
                                let slot = Int((t.dst - tStaging.contents()) / gExpertSize)
                                while slotWriter[tBuf].count <= slot { slotWriter[tBuf].append("?") }
                                slotWriter[tBuf][slot] = "kick L\(target) e\(Int(t.offset / off_t(gExpertSize))) tok\(tokStamp)"
                            }
                            lock.unlock()
                        }
                        lock.lock()
                        resident[tBuf] = map
                        residentLayer[tBuf] = target
                        lock.unlock()
                        s2.signal()
                    }
                }

                // VERIFY: each consumed expert's slot bytes == direct pread of that expert
                let checkBase = stagingBuf.contents()
                for e in actual {
                    guard let slot = slotOf[e] else { mismatches += 1; continue }
                    checks += 1
                    let fdv = openLayerFD(packedDir, layer: l, noCache: true)
                    _ = preadFull(fdv, verifyBuf, off_t(e * gExpertSize), gExpertSize)
                    close(fdv)
                    let staged = checkBase.advanced(by: slot * gExpertSize)
                    if memcmp(staged, verifyBuf, gExpertSize) != 0 {
                        mismatches += 1
                        if mismatches < 5 {
                            var diffOff = -1
                            for b in 0..<gExpertSize where staged.load(fromByteOffset: b, as: UInt8.self) != verifyBuf.load(fromByteOffset: b, as: UInt8.self) {
                                diffOff = b; break
                            }
                            print("   ⚠️ MISMATCH tok\(tok) layer\(l) expert\(e) slot\(slot): first diff at byte \(diffOff) | slot last written by: '\(slot < slotWriter[bufIdx].count ? slotWriter[bufIdx][slot] : "?")'")
                        }
                    }
                }
            }
        }
        print("   checked \(checks) consumed experts, mismatches: \(mismatches)")
    }

    print("")
    print("=== done ===")
}

var sinkGlobal: Int64 = 0

main()
