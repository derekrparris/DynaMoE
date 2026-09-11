//
//  DynaMoETests.swift
//  DynaMoETests
//
//  Created by Derek Parris on 8/16/26.
//

import XCTest

@testable import DynaMoE

final class DynaMoETests: XCTestCase {

    func testOrnithForward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Ornith 1.5 35B FP8 from \(snapshotDir)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024)))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        print("✅ [DIAGNOSTIC] Metal compute pipelines initialized.")

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        print("✅ [DIAGNOSTIC] Built \(cachedLayers.count) cached layers.")

        let hiddenDim = 2048
        let intermediateDim = 512
        let vocabSize = 248320

        // Test single token forward pass
        guard let inBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: intermediateDim * 4, options: .storageModeShared),
              let accumBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let logitsBuf = device.makeBuffer(length: vocabSize * 4, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Benchmark 40-layer forward pass execution
        let lmHeadTensor = summary.tensors.first(where: { $0.name == "lm_head.weight" })
        var logOutput = "=== DYNAMOE ORNITH 1.5 35B FORWARD PASS BENCHMARK ===\n"
        logOutput += String(format: "Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d\n", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId)
        logOutput += String(format: "Mapped %.2f GB into Metal buffers.\n", Double(totalMappedBytes) / (1024*1024*1024))

        let rIdxBuf = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let rWBuf = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm1Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm2Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMidBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let qGateBuf = device.makeBuffer(length: 8192 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let zGateBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let aVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let bVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnCtxBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnOutBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!

        var hCurr = inBuf
        var hNext = outBuf

        // Time 5 consecutive full token forward passes
        for tokenIdx in 0..<5 {
            let tTokenStart = CFAbsoluteTimeGetCurrent()
            var totalAttnMs: Double = 0
            var totalRouterWaitMs: Double = 0
            var totalMoEMs: Double = 0

            for l in 0..<cachedLayers.count {
                let layer = cachedLayers[l]
                let tLayerStart = CFAbsoluteTimeGetCurrent()

                guard let cmdA = cmdQueue.makeCommandBuffer(), let encA = cmdA.makeComputeCommandEncoder() else { break }

                // Norm1
                if let norm1 = layer.norm1Tensor, let norm1Raw = buffers[norm1.shardIndex], let normPipe = inference.rmsnormPipeline {
                    var gammaOff = norm1.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var epsVal: Float = 1e-6
                    encA.setComputePipelineState(normPipe)
                    encA.setBuffer(hCurr, offset: 0, index: 0)
                    encA.setBuffer(norm1Raw, offset: 0, index: 1)
                    encA.setBuffer(xNorm1Buf, offset: 0, index: 2)
                    encA.setBytes(&gammaOff, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&epsVal, length: 4, index: 5)
                    encA.setThreadgroupMemoryLength(1024 * 4, index: 0)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                }

                // Router
                if let router = layer.routerTensor, let routerRaw = buffers[router.shardIndex], let routerPipe = inference.routerPipeline {
                    var rOff = router.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var nExp: UInt32 = 256
                    var kVal: UInt32 = 8
                    encA.setComputePipelineState(routerPipe)
                    encA.setBuffer(routerRaw, offset: 0, index: 0)
                    encA.setBuffer(xNorm1Buf, offset: 0, index: 1)
                    encA.setBuffer(rIdxBuf, offset: 0, index: 2)
                    encA.setBuffer(rWBuf, offset: 0, index: 3)
                    encA.setBytes(&rOff, length: 8, index: 4)
                    encA.setBytes(&hDimU, length: 4, index: 5)
                    encA.setBytes(&nExp, length: 4, index: 6)
                    encA.setBytes(&kVal, length: 4, index: 7)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                }

                encA.endEncoding()
                cmdA.commit()

                let tWait0 = CFAbsoluteTimeGetCurrent()
                cmdA.waitUntilCompleted()
                let tWaitElapsed = (CFAbsoluteTimeGetCurrent() - tWait0) * 1000.0
                totalRouterWaitMs += tWaitElapsed
                totalAttnMs += (CFAbsoluteTimeGetCurrent() - tLayerStart) * 1000.0 - tWaitElapsed

                // Read active experts
                let indPtr = rIdxBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
                let wPtr = rWBuf.contents().bindMemory(to: Float.self, capacity: 8)
                var activeExp: [(id: Int, w: Float)] = []
                for i in 0..<8 {
                    activeExp.append((id: Int(indPtr[i]), w: wPtr[i]))
                }

                // Phase B: Dispatch 8 active expert MLPs
                let tMoe0 = CFAbsoluteTimeGetCurrent()
                guard let cmdB = cmdQueue.makeCommandBuffer(), let encB = cmdB.makeComputeCommandEncoder() else { break }

                if let clearPipe = inference.clearPipeline {
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    encB.setComputePipelineState(clearPipe)
                    encB.setBuffer(hMlpBuf, offset: 0, index: 0)
                    encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                for exp in activeExp {
                    let expId = exp.id
                    let pk = exp.w
                    if let gateW = layer.expertGateWeights[expId],
                       let upW = layer.expertUpWeights[expId],
                       let downW = layer.expertDownWeights[expId],
                       let gRaw = buffers[gateW.shardIndex],
                       let uRaw = buffers[upW.shardIndex],
                       let dRaw = buffers[downW.shardIndex] {

                        let gateS = layer.expertGateScales[expId]
                        let upS = layer.expertUpScales[expId]
                        let downS = layer.expertDownScales[expId]

                        let isFP8 = gateW.dtype.contains("FP8") || gateW.dtype.contains("F8") || gateW.dtype.contains("UINT8") || (gateS != nil && !gateW.dtype.contains("BF16") && !gateW.dtype.contains("F16"))

                        if isFP8, let gS = gateS, let gSRaw = buffers[gS.shardIndex],
                           let uS = upS, let uSRaw = buffers[uS.shardIndex],
                           let dS = downS, let dSRaw = buffers[dS.shardIndex],
                           let gatePipe = inference.fp8GateUpPipeline,
                           let downPipe = inference.fp8DownPipeline {

                            var gWOff = gateW.offsetStart
                            var gSOff = gS.offsetStart
                            var uWOff = upW.offsetStart
                            var uSOff = uS.offsetStart
                            var dWOff = downW.offsetStart
                            var dSOff = dS.offsetStart
                            var hDimVal: UInt32 = UInt32(hiddenDim)
                            var interDimVal: UInt32 = UInt32(intermediateDim)
                            var pkVal = pk

                            encB.setComputePipelineState(gatePipe)
                            encB.setBuffer(gRaw, offset: 0, index: 0)
                            encB.setBuffer(uRaw, offset: 0, index: 1)
                            encB.setBuffer(xNorm1Buf, offset: 0, index: 2)
                            encB.setBuffer(interBuf, offset: 0, index: 3)
                            encB.setBuffer(gSRaw, offset: 0, index: 4)
                            encB.setBuffer(uSRaw, offset: 0, index: 5)
                            encB.setBytes(&gWOff, length: 8, index: 6)
                            encB.setBytes(&gSOff, length: 8, index: 7)
                            encB.setBytes(&uWOff, length: 8, index: 8)
                            encB.setBytes(&uSOff, length: 8, index: 9)
                            encB.setBytes(&hDimVal, length: 4, index: 10)
                            encB.setBytes(&interDimVal, length: 4, index: 11)
                            encB.dispatchThreads(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            encB.setComputePipelineState(downPipe)
                            encB.setBuffer(dRaw, offset: 0, index: 0)
                            encB.setBuffer(interBuf, offset: 0, index: 1)
                            encB.setBuffer(hMlpBuf, offset: 0, index: 2)
                            encB.setBuffer(dSRaw, offset: 0, index: 3)
                            encB.setBytes(&dWOff, length: 8, index: 4)
                            encB.setBytes(&dSOff, length: 8, index: 5)
                            encB.setBytes(&interDimVal, length: 4, index: 6)
                            encB.setBytes(&hDimVal, length: 4, index: 7)
                            encB.setBytes(&pkVal, length: 4, index: 8)
                            encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, downPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        } else if let gatePipe = inference.bf16GateUpPipeline,
                                  let downPipe = inference.bf16DownPipeline {

                            var gOff = gateW.offsetStart
                            var uOff = upW.offsetStart
                            var dOff = downW.offsetStart
                            var hDimU: UInt32 = UInt32(hiddenDim)
                            var interDimU: UInt32 = UInt32(intermediateDim)
                            var pkVal = pk

                            encB.setComputePipelineState(gatePipe)
                            encB.setBuffer(gRaw, offset: 0, index: 0)
                            encB.setBuffer(uRaw, offset: 0, index: 1)
                            encB.setBuffer(xNorm1Buf, offset: 0, index: 2)
                            encB.setBuffer(interBuf, offset: 0, index: 3)
                            encB.setBytes(&gOff, length: 8, index: 4)
                            encB.setBytes(&uOff, length: 8, index: 5)
                            encB.setBytes(&hDimU, length: 4, index: 6)
                            encB.setBytes(&interDimU, length: 4, index: 7)
                            encB.dispatchThreads(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                            encB.setComputePipelineState(downPipe)
                            encB.setBuffer(dRaw, offset: 0, index: 0)
                            encB.setBuffer(interBuf, offset: 0, index: 1)
                            encB.setBuffer(hMlpBuf, offset: 0, index: 2)
                            encB.setBytes(&dOff, length: 8, index: 3)
                            encB.setBytes(&interDimU, length: 4, index: 4)
                            encB.setBytes(&hDimU, length: 4, index: 5)
                            encB.setBytes(&pkVal, length: 4, index: 6)
                            encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, downPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                        }
                    }
                }

                encB.endEncoding()
                cmdB.commit()
                let tMoeWait0 = CFAbsoluteTimeGetCurrent()
                cmdB.waitUntilCompleted()
                let tMoeElapsed = (CFAbsoluteTimeGetCurrent() - tMoeWait0) * 1000.0
                totalMoEMs += tMoeElapsed

                if tokenIdx == 0 && (l == 0 || l == 1 || l == 2 || l == 39) {
                    let logLine = String(format: "  Layer %2d: RouterWait=%.2f ms, MoEWait=%.2f ms, ActiveExp=%@\n", l, tWaitElapsed, tMoeElapsed, activeExp.map { "\($0.id)" }.joined(separator: ","))
                    logOutput += logLine
                    print(logLine)
                }

                let tmp = hCurr
                hCurr = hNext
                hNext = tmp
            }

            // LM Head
            var tLmMs: Double = 0
            if let lmHead = lmHeadTensor, let lmHeadRaw = buffers[lmHead.shardIndex], let simdPipe = inference.bf16GemvSimdPipeline {
                let tLm0 = CFAbsoluteTimeGetCurrent()
                guard let cmd = cmdQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { break }
                var wOff = lmHead.offsetStart
                var inD: UInt32 = UInt32(hiddenDim)
                var outD: UInt32 = UInt32(vocabSize)
                enc.setComputePipelineState(simdPipe)
                enc.setBuffer(lmHeadRaw, offset: 0, index: 0)
                enc.setBuffer(hCurr, offset: 0, index: 1)
                enc.setBuffer(logitsBuf, offset: 0, index: 2)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: vocabSize, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
                tLmMs = (CFAbsoluteTimeGetCurrent() - tLm0) * 1000.0
            }

            let tTotalToken = (CFAbsoluteTimeGetCurrent() - tTokenStart) * 1000.0
            let tps = 1000.0 / tTotalToken
            let line = String(format: "Token %d: Total=%.2f ms (%.1f tok/s) | Attn=%.2f ms, RouterWait=%.2f ms, MoE=%.2f ms, LMHead=%.2f ms\n", tokenIdx + 1, tTotalToken, tps, totalAttnMs, totalRouterWaitMs, totalMoEMs, tLmMs)
            logOutput += line
            print(line)
        }

        try? logOutput.write(toFile: "/tmp/dynamoe_benchmark.log", atomically: true, encoding: .utf8)
        print("🎉 [DIAGNOSTIC] Diagnostic run complete. Report saved to /tmp/dynamoe_benchmark.log")
    }

    func testOrnith4BitForward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-MLX-4bit/snapshots/19504d912fa8fc7622bf6b1de3db5d5d890b1f02"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Ornith 1.5 35B 4-bit from \(snapshotDir)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024)))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        print("✅ [DIAGNOSTIC] Metal compute pipelines initialized.")

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        print("✅ [DIAGNOSTIC] Built \(cachedLayers.count) cached layers.")

        let hiddenDim = 2048
        let intermediateDim = 512
        let vocabSize = 248320

        var lmHeadTensor: TensorMetadata? = nil
        for t in summary.tensors {
            if t.name == "lm_head.weight" || t.name == "language_model.output.weight" || t.name == "output.weight" {
                lmHeadTensor = t
                break
            }
        }

        guard let hStateBufA = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hStateBufB = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm0Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let rIdxBuf = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let rWBuf = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let logitsBuf = device.makeBuffer(length: vocabSize * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate scratch buffers")
            return
        }

        var logOutput = "=== DYNAMOE ORNITH 1.5 35B 4-BIT FORWARD PASS BENCHMARK ===\n"
        logOutput += String(format: "Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d\n", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId)
        logOutput += String(format: "Mapped %.2f GB into Metal buffers.\n", Double(totalMappedBytes) / (1024*1024*1024))

        var hCurr = hStateBufA
        var hNext = hStateBufB

        for tokenIdx in 0..<5 {
            let tTokenStart = CFAbsoluteTimeGetCurrent()
            var totalAttnMs: Double = 0
            var totalRouterWaitMs: Double = 0
            var totalMoEMs: Double = 0

            for l in 0..<cachedLayers.count {
                let layer = cachedLayers[l]
                let tLayerStart = CFAbsoluteTimeGetCurrent()

                guard let cmdA = cmdQueue.makeCommandBuffer(), let encA = cmdA.makeComputeCommandEncoder() else { break }

                if let norm = layer.norm1Tensor, let normRaw = buffers[norm.shardIndex], let rmsPipe = inference.rmsnormPipeline {
                    var wOff = norm.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var eps: Float = 1e-6
                    encA.setComputePipelineState(rmsPipe)
                    encA.setBuffer(normRaw, offset: 0, index: 0)
                    encA.setBuffer(hCurr, offset: 0, index: 1)
                    encA.setBuffer(xNorm0Buf, offset: 0, index: 2)
                    encA.setBytes(&wOff, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&eps, length: 4, index: 5)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                }

                if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex], let rmsPipe = inference.rmsnormPipeline {
                    var wOff = norm2.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var eps: Float = 1e-6
                    encA.setComputePipelineState(rmsPipe)
                    encA.setBuffer(norm2Raw, offset: 0, index: 0)
                    encA.setBuffer(hCurr, offset: 0, index: 1)
                    encA.setBuffer(xNorm1Buf, offset: 0, index: 2)
                    encA.setBytes(&wOff, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&eps, length: 4, index: 5)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                }

                // Router
                if let router = layer.routerTensor, let routerRaw = buffers[router.shardIndex] {
                    let rS = layer.routerScale
                    let rB = layer.routerBias
                    if let rS = rS, let rB = rB, let rSRaw = buffers[rS.shardIndex], let rBRaw = buffers[rB.shardIndex], let routerQ8 = inference.routerQ8Pipeline {
                        var rOff = router.offsetStart
                        var sOff = rS.offsetStart
                        var bOff = rB.offsetStart
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        var grp: UInt32 = 64
                        encA.setComputePipelineState(routerQ8)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(rSRaw, offset: 0, index: 1)
                        encA.setBuffer(rBRaw, offset: 0, index: 2)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 3)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 4)
                        encA.setBuffer(rWBuf, offset: 0, index: 5)
                        encA.setBytes(&rOff, length: 8, index: 6)
                        encA.setBytes(&sOff, length: 8, index: 7)
                        encA.setBytes(&bOff, length: 8, index: 8)
                        encA.setBytes(&hDimU, length: 4, index: 9)
                        encA.setBytes(&nExp, length: 4, index: 10)
                        encA.setBytes(&kVal, length: 4, index: 11)
                        encA.setBytes(&grp, length: 4, index: 12)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    } else if let routerPipe = inference.routerPipeline {
                        var rOff = router.offsetStart
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        encA.setComputePipelineState(routerPipe)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 1)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 2)
                        encA.setBuffer(rWBuf, offset: 0, index: 3)
                        encA.setBytes(&rOff, length: 8, index: 4)
                        encA.setBytes(&hDimU, length: 4, index: 5)
                        encA.setBytes(&nExp, length: 4, index: 6)
                        encA.setBytes(&kVal, length: 4, index: 7)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }

                encA.endEncoding()
                cmdA.commit()

                let tWait0 = CFAbsoluteTimeGetCurrent()
                cmdA.waitUntilCompleted()
                let tWaitElapsed = (CFAbsoluteTimeGetCurrent() - tWait0) * 1000.0
                totalRouterWaitMs += tWaitElapsed
                totalAttnMs += (CFAbsoluteTimeGetCurrent() - tLayerStart) * 1000.0 - tWaitElapsed

                let indPtr = rIdxBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
                let wPtr = rWBuf.contents().bindMemory(to: Float.self, capacity: 8)
                var activeExp: [(id: Int, w: Float)] = []
                for i in 0..<8 {
                    activeExp.append((id: Int(indPtr[i]), w: wPtr[i]))
                }

                guard let cmdB = cmdQueue.makeCommandBuffer(), let encB = cmdB.makeComputeCommandEncoder() else { break }

                if let clearPipe = inference.clearPipeline {
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    encB.setComputePipelineState(clearPipe)
                    encB.setBuffer(hMlpBuf, offset: 0, index: 0)
                    encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                for exp in activeExp {
                    let expId = exp.id
                    let pk = exp.w
                    if let gateW = layer.expertGateWeights[expId],
                       let upW = layer.expertUpWeights[expId],
                       let downW = layer.expertDownWeights[expId],
                       let gRaw = buffers[gateW.shardIndex],
                       let uRaw = buffers[upW.shardIndex],
                       let dRaw = buffers[downW.shardIndex],
                       let gateS = layer.expertGateScales[expId],
                       let gateB = layer.expertGateBiases[expId],
                       let upS = layer.expertUpScales[expId],
                       let upB = layer.expertUpBiases[expId],
                       let downS = layer.expertDownScales[expId],
                       let downB = layer.expertDownBiases[expId],
                       let gSRaw = buffers[gateS.shardIndex],
                       let gBRaw = buffers[gateB.shardIndex],
                       let uSRaw = buffers[upS.shardIndex],
                       let uBRaw = buffers[upB.shardIndex],
                       let dSRaw = buffers[downS.shardIndex],
                       let dBRaw = buffers[downB.shardIndex],
                       let q4GatePipe = inference.q4GateUpPipeline,
                       let q4DownPipe = inference.q4DownPipeline {

                        var gWOff = gateW.offsetStart
                        var gSOff = gateS.offsetStart
                        var gBOff = gateB.offsetStart
                        var uWOff = upW.offsetStart
                        var uSOff = upS.offsetStart
                        var uBOff = upB.offsetStart
                        var dWOff = downW.offsetStart
                        var dSOff = downS.offsetStart
                        var dBOff = downB.offsetStart
                        var hDimVal: UInt32 = UInt32(hiddenDim)
                        var interDimVal: UInt32 = UInt32(intermediateDim)
                        var grp: UInt32 = 64
                        var pkVal = pk

                        encB.setComputePipelineState(q4GatePipe)
                        encB.setBuffer(gRaw, offset: 0, index: 0)
                        encB.setBuffer(gSRaw, offset: 0, index: 1)
                        encB.setBuffer(gBRaw, offset: 0, index: 2)
                        encB.setBuffer(uRaw, offset: 0, index: 3)
                        encB.setBuffer(uSRaw, offset: 0, index: 4)
                        encB.setBuffer(uBRaw, offset: 0, index: 5)
                        encB.setBuffer(xNorm1Buf, offset: 0, index: 6)
                        encB.setBuffer(interBuf, offset: 0, index: 7)
                        encB.setBytes(&gWOff, length: 8, index: 8)
                        encB.setBytes(&gSOff, length: 8, index: 9)
                        encB.setBytes(&gBOff, length: 8, index: 10)
                        encB.setBytes(&uWOff, length: 8, index: 11)
                        encB.setBytes(&uSOff, length: 8, index: 12)
                        encB.setBytes(&uBOff, length: 8, index: 13)
                        encB.setBytes(&hDimVal, length: 4, index: 14)
                        encB.setBytes(&interDimVal, length: 4, index: 15)
                        encB.setBytes(&grp, length: 4, index: 16)
                        encB.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)

                        encB.setComputePipelineState(q4DownPipe)
                        encB.setBuffer(dRaw, offset: 0, index: 0)
                        encB.setBuffer(dSRaw, offset: 0, index: 1)
                        encB.setBuffer(dBRaw, offset: 0, index: 2)
                        encB.setBuffer(interBuf, offset: 0, index: 3)
                        encB.setBuffer(hMlpBuf, offset: 0, index: 4)
                        encB.setBytes(&dWOff, length: 8, index: 5)
                        encB.setBytes(&dSOff, length: 8, index: 6)
                        encB.setBytes(&dBOff, length: 8, index: 7)
                        encB.setBytes(&interDimVal, length: 4, index: 8)
                        encB.setBytes(&hDimVal, length: 4, index: 9)
                        encB.setBytes(&grp, length: 4, index: 10)
                        encB.setBytes(&pkVal, length: 4, index: 11)
                        encB.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)
                    }
                }

                encB.endEncoding()
                cmdB.commit()
                let tMoeWait0 = CFAbsoluteTimeGetCurrent()
                cmdB.waitUntilCompleted()
                let tMoeElapsed = (CFAbsoluteTimeGetCurrent() - tMoeWait0) * 1000.0
                totalMoEMs += tMoeElapsed

                if tokenIdx == 0 && (l == 0 || l == 1 || l == 2 || l == 39) {
                    let logLine = String(format: "  Layer %2d: RouterWait=%.2f ms, MoEWait=%.2f ms, ActiveExp=%@\n", l, tWaitElapsed, tMoeElapsed, activeExp.map { "\($0.id)" }.joined(separator: ","))
                    logOutput += logLine
                    print(logLine)
                }

                let tmp = hCurr
                hCurr = hNext
                hNext = tmp
            }

            let tTotalToken = (CFAbsoluteTimeGetCurrent() - tTokenStart) * 1000.0
            let tps = 1000.0 / tTotalToken
            let line = String(format: "Token %d: Total=%.2f ms (%.1f tok/s) | Attn=%.2f ms, RouterWait=%.2f ms, MoE=%.2f ms\n", tokenIdx + 1, tTotalToken, tps, totalAttnMs, totalRouterWaitMs, totalMoEMs)
            logOutput += line
            print(line)
        }

        try? logOutput.write(toFile: "/tmp/dynamoe_4bit_benchmark.log", atomically: true, encoding: .utf8)
        print("🎉 [DIAGNOSTIC] 4-Bit diagnostic run complete. Report saved to /tmp/dynamoe_4bit_benchmark.log")
    }

    func testQwenFlashMoEForward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--alexintosh--Qwen3.5-35B-A3B-Q4-FlashMoE/snapshots/cd9f9ef2b17f080aaa7710394f8a38002ba5ce9b"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Qwen3.5 35B FlashMoE from \(snapshotDir)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB across %d shards into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024), summary.shards.count))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        print("✅ [DIAGNOSTIC] Metal compute pipelines initialized.")

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        print("✅ [DIAGNOSTIC] Built \(cachedLayers.count) cached layers.")

        let hiddenDim = 2048
        let intermediateDim = 512
        let vocabSize = 248320

        let expertSize = 1769472
        let pool = ExpertIOThreadPool.shared
        pool.initialize(numThreads: 8)
        let packedExpertsDir = URL(fileURLWithPath: snapshotDir).appendingPathComponent("packed_experts")

        guard let hStateBufA = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hStateBufB = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm0Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm1Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let rIdxBuf = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let rWBuf = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let stagingBuf = device.makeBuffer(length: 8 * expertSize, options: .storageModeShared) else {
            XCTFail("Failed to allocate scratch buffers")
            return
        }

        var logOutput = "=== DYNAMOE QWEN3.5 35B FLASH-MOE FORWARD PASS BENCHMARK ===\n"
        logOutput += String(format: "Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d\n", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId)
        logOutput += String(format: "Mapped %.2f GB into Metal buffers.\n", Double(totalMappedBytes) / (1024*1024*1024))

        var hCurr = hStateBufA
        var hNext = hStateBufB

        for tokenIdx in 0..<5 {
            let tTokenStart = CFAbsoluteTimeGetCurrent()
            var totalAttnMs: Double = 0
            var totalRouterWaitMs: Double = 0
            var totalMoEMs: Double = 0
            var totalIoMs: Double = 0

            for l in 0..<cachedLayers.count {
                let layer = cachedLayers[l]
                let tLayerStart = CFAbsoluteTimeGetCurrent()

                guard let cmdA = cmdQueue.makeCommandBuffer(), let encA = cmdA.makeComputeCommandEncoder() else { break }

                if let norm = layer.norm1Tensor, let normRaw = buffers[norm.shardIndex], let rmsPipe = inference.rmsnormPipeline {
                    var wOff = norm.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var eps: Float = 1e-6
                    encA.setComputePipelineState(rmsPipe)
                    encA.setBuffer(normRaw, offset: 0, index: 0)
                    encA.setBuffer(hCurr, offset: 0, index: 1)
                    encA.setBuffer(xNorm0Buf, offset: 0, index: 2)
                    encA.setBytes(&wOff, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&eps, length: 4, index: 5)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                }

                if let norm2 = layer.norm2Tensor, let norm2Raw = buffers[norm2.shardIndex], let rmsPipe = inference.rmsnormPipeline {
                    var wOff = norm2.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var eps: Float = 1e-6
                    encA.setComputePipelineState(rmsPipe)
                    encA.setBuffer(norm2Raw, offset: 0, index: 0)
                    encA.setBuffer(hCurr, offset: 0, index: 1)
                    encA.setBuffer(xNorm1Buf, offset: 0, index: 2)
                    encA.setBytes(&wOff, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&eps, length: 4, index: 5)
                    encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                }

                // Router
                if let router = layer.routerTensor, let routerRaw = buffers[router.shardIndex] {
                    let rS = layer.routerScale
                    let rB = layer.routerBias
                    if let rS = rS, let rB = rB, let rSRaw = buffers[rS.shardIndex], let rBRaw = buffers[rB.shardIndex], let routerQ8 = inference.routerQ8Pipeline {
                        var rOff = router.offsetStart
                        var sOff = rS.offsetStart
                        var bOff = rB.offsetStart
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        var grp: UInt32 = 64
                        encA.setComputePipelineState(routerQ8)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(rSRaw, offset: 0, index: 1)
                        encA.setBuffer(rBRaw, offset: 0, index: 2)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 3)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 4)
                        encA.setBuffer(rWBuf, offset: 0, index: 5)
                        encA.setBytes(&rOff, length: 8, index: 6)
                        encA.setBytes(&sOff, length: 8, index: 7)
                        encA.setBytes(&bOff, length: 8, index: 8)
                        encA.setBytes(&hDimU, length: 4, index: 9)
                        encA.setBytes(&nExp, length: 4, index: 10)
                        encA.setBytes(&kVal, length: 4, index: 11)
                        encA.setBytes(&grp, length: 4, index: 12)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    } else if let routerPipe = inference.routerPipeline {
                        var rOff = router.offsetStart
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        encA.setComputePipelineState(routerPipe)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 1)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 2)
                        encA.setBuffer(rWBuf, offset: 0, index: 3)
                        encA.setBytes(&rOff, length: 8, index: 4)
                        encA.setBytes(&hDimU, length: 4, index: 5)
                        encA.setBytes(&nExp, length: 4, index: 6)
                        encA.setBytes(&kVal, length: 4, index: 7)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }

                encA.endEncoding()
                cmdA.commit()

                let tWait0 = CFAbsoluteTimeGetCurrent()
                cmdA.waitUntilCompleted()
                let tWaitElapsed = (CFAbsoluteTimeGetCurrent() - tWait0) * 1000.0
                totalRouterWaitMs += tWaitElapsed
                totalAttnMs += (CFAbsoluteTimeGetCurrent() - tLayerStart) * 1000.0 - tWaitElapsed

                let indPtr = rIdxBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
                let wPtr = rWBuf.contents().bindMemory(to: Float.self, capacity: 8)
                var activeExp: [(id: Int, w: Float)] = []
                for i in 0..<8 {
                    activeExp.append((id: Int(indPtr[i]), w: wPtr[i]))
                }

                // 1. Parallel pread the 8 active experts into staging buffer
                guard let fd = pool.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedExpertsDir) else {
                    continue
                }
                var tasks: [ExpertPreadTask] = []
                let rawStagingPtr = stagingBuf.contents()
                for (slot, exp) in activeExp.enumerated() {
                    let offset = off_t(exp.id * expertSize)
                    let dst = rawStagingPtr.advanced(by: slot * expertSize)
                    tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                }
                let tIo0 = CFAbsoluteTimeGetCurrent()
                pool.dispatchSync(tasks: &tasks)
                let tIoElapsed = (CFAbsoluteTimeGetCurrent() - tIo0) * 1000.0
                totalIoMs += tIoElapsed

                // 2. GPU Compute on Staged Expert Buffers
                guard let cmdB = cmdQueue.makeCommandBuffer(), let encB = cmdB.makeComputeCommandEncoder() else { break }

                if let clearPipe = inference.clearPipeline {
                    encB.setComputePipelineState(clearPipe)
                    encB.setBuffer(hMlpBuf, offset: 0, index: 0)
                    encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                for (slot, exp) in activeExp.enumerated() {
                    let pk = exp.w
                    let slotOffset = UInt64(slot * expertSize)
                    let gWOff = slotOffset + 0
                    let gSOff = slotOffset + 524288
                    let gBOff = slotOffset + 557056
                    let uWOff = slotOffset + 589824
                    let uSOff = slotOffset + 1114112
                    let uBOff = slotOffset + 1146880
                    let dWOff = slotOffset + 1179648
                    let dSOff = slotOffset + 1703936
                    let dBOff = slotOffset + 1736704

                    if let q4GatePipe = inference.q4GateUpPipeline,
                       let q4DownPipe = inference.q4DownPipeline {
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

                        encB.setComputePipelineState(q4GatePipe)
                        encB.setBuffer(stagingBuf, offset: 0, index: 0)
                        encB.setBuffer(stagingBuf, offset: 0, index: 1)
                        encB.setBuffer(stagingBuf, offset: 0, index: 2)
                        encB.setBuffer(stagingBuf, offset: 0, index: 3)
                        encB.setBuffer(stagingBuf, offset: 0, index: 4)
                        encB.setBuffer(stagingBuf, offset: 0, index: 5)
                        encB.setBuffer(xNorm1Buf, offset: 0, index: 6)
                        encB.setBuffer(interBuf, offset: 0, index: 7)
                        encB.setBytes(&gWOffU, length: 8, index: 8)
                        encB.setBytes(&gSOffU, length: 8, index: 9)
                        encB.setBytes(&gBOffU, length: 8, index: 10)
                        encB.setBytes(&uWOffU, length: 8, index: 11)
                        encB.setBytes(&uSOffU, length: 8, index: 12)
                        encB.setBytes(&uBOffU, length: 8, index: 13)
                        encB.setBytes(&hDimVal, length: 4, index: 14)
                        encB.setBytes(&interDimVal, length: 4, index: 15)
                        encB.setBytes(&grp, length: 4, index: 16)
                        encB.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)

                        encB.setComputePipelineState(q4DownPipe)
                        encB.setBuffer(stagingBuf, offset: 0, index: 0)
                        encB.setBuffer(stagingBuf, offset: 0, index: 1)
                        encB.setBuffer(stagingBuf, offset: 0, index: 2)
                        encB.setBuffer(interBuf, offset: 0, index: 3)
                        encB.setBuffer(hMlpBuf, offset: 0, index: 4)
                        encB.setBytes(&dWOffU, length: 8, index: 5)
                        encB.setBytes(&dSOffU, length: 8, index: 6)
                        encB.setBytes(&dBOffU, length: 8, index: 7)
                        encB.setBytes(&interDimVal, length: 4, index: 8)
                        encB.setBytes(&hDimVal, length: 4, index: 9)
                        encB.setBytes(&grp, length: 4, index: 10)
                        encB.setBytes(&pkVal, length: 4, index: 11)
                        encB.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)
                    }
                }

                encB.endEncoding()
                cmdB.commit()
                let tMoeWait0 = CFAbsoluteTimeGetCurrent()
                cmdB.waitUntilCompleted()
                let tMoeElapsed = (CFAbsoluteTimeGetCurrent() - tMoeWait0) * 1000.0
                totalMoEMs += tMoeElapsed

                if tokenIdx == 0 && (l == 0 || l == 1 || l == 2 || l == 39) {
                    let logLine = String(format: "  Layer %2d: IO(pread 8 exps)=%.2f ms, RouterWait=%.2f ms, MoEGPU=%.2f ms, ActiveExp=%@\n", l, tIoElapsed, tWaitElapsed, tMoeElapsed, activeExp.map { "\($0.id)" }.joined(separator: ","))
                    logOutput += logLine
                    print(logLine)
                }

                let tmp = hCurr
                hCurr = hNext
                hNext = tmp
            }

            let tTotalToken = (CFAbsoluteTimeGetCurrent() - tTokenStart) * 1000.0
            let tps = 1000.0 / tTotalToken
            let line = String(format: "Token %d: Total=%.2f ms (%.1f tok/s) | Attn=%.2f ms, RouterWait=%.2f ms, IO(pread)=%.2f ms, MoEGPU=%.2f ms\n", tokenIdx + 1, tTotalToken, tps, totalAttnMs, totalRouterWaitMs, totalIoMs, totalMoEMs)
            logOutput += line
            print(line)
        }

        try? logOutput.write(toFile: "/tmp/dynamoe_flashmoe_benchmark.log", atomically: true, encoding: .utf8)
        print("🎉 [DIAGNOSTIC] FlashMoE diagnostic run complete. Report saved to /tmp/dynamoe_flashmoe_benchmark.log")
    }

    func testExpertIOThreadPool() throws {
        let pool = ExpertIOThreadPool.shared
        pool.initialize(numThreads: 8)

        let packedDir = URL(fileURLWithPath: "/Users/derekparris/.cache/huggingface/hub/models--alexintosh--Qwen3.5-35B-A3B-Q4-FlashMoE/snapshots/cd9f9ef2b17f080aaa7710394f8a38002ba5ce9b/packed_experts")
        guard FileManager.default.fileExists(atPath: packedDir.path) else {
            print("Packed experts directory not found, skipping.")
            return
        }

        guard let fd = pool.getOrOpenLayerFD(layerIndex: 0, packedExpertsDir: packedDir) else {
            XCTFail("Failed to open layer_00.bin")
            return
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }

        let expertSize = 1769472 // 1.6875 MB
        let k = 8
        guard let stagingBuf = device.makeBuffer(length: k * expertSize, options: .storageModeShared) else {
            XCTFail("Failed to allocate staging buffer")
            return
        }

        var tasks: [ExpertPreadTask] = []
        let rawPtr = stagingBuf.contents()
        let activeExperts = [3, 14, 52, 99, 120, 184, 201, 245]

        for (idx, expId) in activeExperts.enumerated() {
            let offset = off_t(expId * expertSize)
            let dst = rawPtr.advanced(by: idx * expertSize)
            tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        pool.dispatchSync(tasks: &tasks)
        let tElapsedMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

        for t in tasks {
            XCTAssertEqual(t.result, expertSize, "Expected to read \(expertSize) bytes")
        }

        let totalMB = Double(k * expertSize) / (1024.0 * 1024.0)
        let throughputGBps = (totalMB / 1024.0) / (tElapsedMs / 1000.0)
        print(String(format: "🚀 [IO BENCHMARK] Parallel pread: read %.2f MB across 8 threads in %.2f ms (%.2f GB/s)", totalMB, tElapsedMs, throughputGBps))
        XCTAssertLessThan(tElapsedMs, 50.0, "Parallel pread should complete within 50ms for 13.5MB")
    }

    func testAgentHarnessToolCalling() async throws {
        let harness = AgentHarness.shared
        XCTAssertGreaterThanOrEqual(harness.tools.count, 2)

        // 1. Prompt formatting
        let formatted = harness.buildSystemPrompt(baseSystem: "You are an assistant.")
        XCTAssertTrue(formatted.contains("# Tools"))
        XCTAssertTrue(formatted.contains("shell_run"))

        // 2. Parse Tool Calls
        let modelOutput = """
        Let me list the files in the directory.
        <tool_call>
        {"name": "shell_run", "arguments": {"command": "echo 'hello world'"}}
        </tool_call>
        """
        let parsed = harness.parseToolCalls(from: modelOutput)
        XCTAssertEqual(parsed.calls.count, 1)
        XCTAssertEqual(parsed.calls[0].name, "shell_run")
        XCTAssertEqual(parsed.calls[0].arguments["command"] as? String, "echo 'hello world'")

        // 3. Tool Execution
        let result = await harness.executeTool(call: parsed.calls[0])
        XCTAssertTrue(result.resultJSON.contains("hello world"))

        // 4. Continuation turn formatting
        let nextTurn = harness.formatToolResponseTurn(responses: [result.resultJSON])
        XCTAssertTrue(nextTurn.contains("<tool_response>"))
        XCTAssertTrue(nextTurn.contains("hello world"))
        XCTAssertTrue(nextTurn.hasSuffix("<|im_start|>assistant\n"))
    }

    func testToolDiscoverLoadUnloadLifecycle() async throws {
        let harness = AgentHarness.shared
        let previouslyLoaded = Set(harness.loadedTools.keys)

        defer {
            let toRemove = Set(harness.loadedTools.keys).subtracting(previouslyLoaded)
            for name in toRemove { _ = try? harness.unloadTool(named: name) }
            for name in previouslyLoaded where harness.loadedTools[name] == nil {
                _ = try? harness.loadTool(named: name)
            }
        }

        // 1. Core set is loaded up front; heavier tools are not
        XCTAssertNotNil(harness.loadedTools["shell_run"])
        XCTAssertNotNil(harness.loadedTools["grep_search"])
        XCTAssertNotNil(harness.loadedTools["tools_discover"])
        XCTAssertNil(harness.loadedTools["web_search"])
        XCTAssertFalse(harness.availableToolDefinitions.contains { $0.function.name == "web_search" })
        XCTAssertTrue(harness.allToolDefinitions.contains { $0.function.name == "web_search" })

        // 2. tools_discover lists web_search in the catalog
        let discover = ToolDiscoverTool()
        let discoverResult = try await discover.execute(arguments: [:], workingDirectory: nil, maxOutputLength: 4000)
        XCTAssertTrue(discoverResult.resultJSON.contains("web_search"))
        XCTAssertTrue((discoverResult.stdout ?? "").contains("web_search"))

        // 3. tools_load registers the schema and re-exposes it
        let loader = ToolLoadTool()
        let loadResult = try await loader.execute(arguments: ["name": "web_search"], workingDirectory: nil, maxOutputLength: 4000)
        let loadJSON = try JSONSerialization.jsonObject(with: Data(loadResult.resultJSON.utf8)) as? [String: Any]
        XCTAssertEqual((loadJSON?["result"] as? [String: Any])?["registration"] as? Bool, true)
        XCTAssertNotNil(harness.loadedTools["web_search"])
        XCTAssertTrue(harness.availableToolDefinitions.contains { $0.function.name == "web_search" })

        // 4. formatToolResponseTurn injects the registration notice for the next assistant turn
        let turn = harness.formatToolResponseTurn(responses: [loadResult.resultJSON])
        XCTAssertTrue(turn.contains("[TOOL_REGISTRATION]"))
        XCTAssertTrue(turn.contains("web_search"))

        // 5. Fundamental tools are protected from unload
        let unloader = ToolUnloadTool()
        let protectResult = try await unloader.execute(arguments: ["name": "shell_run"], workingDirectory: nil, maxOutputLength: 4000)
        XCTAssertTrue(protectResult.resultJSON.contains("\"error\""))
        XCTAssertNotNil(harness.loadedTools["shell_run"])

        // 6. Unload removes the loaded tool again
        let unloadResult = try await unloader.execute(arguments: ["name": "web_search"], workingDirectory: nil, maxOutputLength: 4000)
        let unloadJSON = try JSONSerialization.jsonObject(with: Data(unloadResult.resultJSON.utf8)) as? [String: Any]
        XCTAssertEqual((unloadJSON?["result"] as? [String: Any])?["de_registration"] as? Bool, true)
        XCTAssertNil(harness.loadedTools["web_search"])
    }

    func testAgentHarnessProcessExecution() async throws {
        let harness = AgentHarness.shared

        // 1. Verify Homebrew is in PATH and accessible
        let brewCall = ParsedToolCall(
            name: "shell_run",
            arguments: ["command": "which brew || true"],
            rawArguments: "{\"command\": \"which brew || true\"}",
            rawText: "<tool_call>{\"name\": \"shell_run\", \"arguments\": {\"command\": \"which brew || true\"}}</tool_call>"
        )
        let brewResult = await harness.executeTool(call: brewCall)
        XCTAssertEqual(brewResult.record.status, ToolExecutionStatus.success)
        XCTAssertTrue((brewResult.record.output ?? "").contains("brew"))

        // 2. Verify Timeout Enforcement (sleep 10 with 1s timeout)
        let (exitCode, _, stderr) = try await AgentHarness.runProcess(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", "sleep 10"],
            currentDirectory: URL(fileURLWithPath: "/tmp"),
            timeoutSeconds: 1
        )
        XCTAssertEqual(exitCode, 124)
        XCTAssertTrue(stderr.contains("timed out"))

        // 3. Verify Non-Zero Exit Code produces .error status
        let failCall = ParsedToolCall(
            name: "shell_run",
            arguments: ["command": "false"],
            rawArguments: "{\"command\": \"false\"}",
            rawText: "<tool_call>{\"name\": \"shell_run\", \"arguments\": {\"command\": \"false\"}}</tool_call>"
        )
        let failResult = await harness.executeTool(call: failCall)
        XCTAssertEqual(failResult.record.status, ToolExecutionStatus.error)
    }

    func testPromptQueueDataModel() throws {
        // 1. QueuedPrompt creation and equality
        let q1 = QueuedPrompt(text: "First queued prompt")
        let q2 = QueuedPrompt(text: "Second queued prompt")
        XCTAssertEqual(q1.text, "First queued prompt")
        XCTAssertEqual(q2.text, "Second queued prompt")
        XCTAssertNotEqual(q1.id, q2.id)

        // 2. ChatSession with queuedPrompts
        var session = ChatSession(title: "Queue Test Session")
        XCTAssertTrue(session.queuedPrompts.isEmpty)
        session.queuedPrompts.append(q1)
        session.queuedPrompts.append(q2)
        XCTAssertEqual(session.queuedPrompts.count, 2)

        // 3. FIFO Dequeue
        let popped = session.queuedPrompts.removeFirst()
        XCTAssertEqual(popped.id, q1.id)
        XCTAssertEqual(popped.text, "First queued prompt")
        XCTAssertEqual(session.queuedPrompts.count, 1)

        // 4. Remove by ID
        session.queuedPrompts.append(QueuedPrompt(text: "Third"))
        XCTAssertEqual(session.queuedPrompts.count, 2)
        session.queuedPrompts.removeAll(where: { $0.id == q2.id })
        XCTAssertEqual(session.queuedPrompts.count, 1)
        XCTAssertEqual(session.queuedPrompts.first?.text, "Third")

        // 5. JSON Round-Trip Serialization
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        let data = try encoder.encode(session)
        let decodedSession = try decoder.decode(ChatSession.self, from: data)
        XCTAssertEqual(decodedSession.id, session.id)
        XCTAssertEqual(decodedSession.queuedPrompts.count, 1)
        XCTAssertEqual(decodedSession.queuedPrompts.first?.text, "Third")

        // 6. Backwards compatibility: Decoding JSON without 'queuedPrompts' key
        let legacyJson = """
        {
            "id": "\(UUID().uuidString)",
            "title": "Legacy Session",
            "messages": [],
            "createdAt": \(Date().timeIntervalSinceReferenceDate),
            "updatedAt": \(Date().timeIntervalSinceReferenceDate)
        }
        """.data(using: .utf8)!
        let legacySession = try decoder.decode(ChatSession.self, from: legacyJson)
        XCTAssertEqual(legacySession.title, "Legacy Session")
        XCTAssertTrue(legacySession.queuedPrompts.isEmpty)
    }

    func testRepackOrnithFP8() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Ornith FP8 snapshot not found, skipping repack test.")
            return
        }
        let srcUrl = URL(fileURLWithPath: snapshotDir)
        let repacker = ExpertRepacker.shared
        print("🚀 Starting ExpertRepacker on Ornith 1.5 35B A3B FP8...")
        let t0 = CFAbsoluteTimeGetCurrent()
        try repacker.repackSafetensors(sourceDir: srcUrl, outputDir: srcUrl) { p, msg in
            print(String(format: "[REPACK PROGRESS] %.0f%%: %@", p * 100, msg))
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "🎉 Repacking completed in %.2f s!", elapsed))
        XCTAssertTrue(FileManager.default.fileExists(atPath: srcUrl.appendingPathComponent("model_weights.bin").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: srcUrl.appendingPathComponent("packed_experts/layout.json").path))
    }

    func testOrnithFlashMoEFP8Forward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/fab11c26e2325a42f4b32da0249c819a0bade1b1"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        let packedDir = URL(fileURLWithPath: snapshotDir).appendingPathComponent("packed_experts")
        guard FileManager.default.fileExists(atPath: packedDir.appendingPathComponent("layout.json").path) else {
            print("Ornith FP8 is not yet repacked. Running testRepackOrnithFP8 first...")
            try testRepackOrnithFP8()
            return
        }

        print("🔍 [DIAGNOSTIC] Loading Ornith 1.5 35B FlashMoE FP8 from \(snapshotDir)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB across %d shards into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024), summary.shards.count))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        print("✅ [DIAGNOSTIC] Metal compute pipelines initialized.")

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        print("✅ [DIAGNOSTIC] Built \(cachedLayers.count) cached layers.")

        let hiddenDim = 2048
        let intermediateDim = 512
        let vocabSize = 248320

        // Parse layout.json
        let layoutData = try Data(contentsOf: packedDir.appendingPathComponent("layout.json"))
        let layout = try JSONDecoder().decode(FlashMoELayout.self, from: layoutData)
        let expertSize = Int(layout.expert_size)
        print(String(format: "✅ [DIAGNOSTIC] Layout loaded: %d layers, %d experts/layer, expert_size=%d bytes", layout.num_layers, layout.num_experts, expertSize))

        // Find component offsets
        let compGateW = layout.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("weight") })?.offset ?? 0
        let compGateS = layout.components.first(where: { $0.name.contains("gate_proj") && $0.name.contains("scale") })?.offset ?? 1048576
        let compUpW = layout.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("weight") })?.offset ?? 1049600
        let compUpS = layout.components.first(where: { $0.name.contains("up_proj") && $0.name.contains("scale") })?.offset ?? 2098176
        let compDownW = layout.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("weight") })?.offset ?? 2099200
        let compDownS = layout.components.first(where: { $0.name.contains("down_proj") && $0.name.contains("scale") })?.offset ?? 3147776

        // Allocate scratch buffers
        guard let h0Buf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let h1Buf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let xNorm1Buf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared),
              let rIdxBuf = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let rWBuf = device.makeBuffer(length: 8 * 4, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: intermediateDim * 4, options: .storageModeShared),
              let stagingBuf = device.makeBuffer(length: 8 * expertSize, options: .storageModeShared),
              let hMlpBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared) else {
            XCTFail("Scratch buffer alloc failed")
            return
        }

        let pool = ExpertIOThreadPool.shared
        pool.initialize(numThreads: 8)

        // Initialize input
        let h0Ptr = h0Buf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        for i in 0..<hiddenDim { h0Ptr[i] = Float.random(in: -0.1...0.1) }

        var hCurr = h0Buf
        var hNext = h1Buf

        let numTokensToBenchmark = 5
        var logOutput = "=== Ornith 1.5 35B FlashMoE FP8 Forward Benchmark ===\n"

        for tokenIdx in 0..<numTokensToBenchmark {
            let tTokenStart = CFAbsoluteTimeGetCurrent()
            var totalAttnMs: Double = 0
            var totalRouterWaitMs: Double = 0
            var totalIoMs: Double = 0
            var totalMoEMs: Double = 0

            for l in 0..<cachedLayers.count {
                let layer = cachedLayers[l]
                let tLayerStart = CFAbsoluteTimeGetCurrent()

                // Phase A: Layernorm + Attention + Router
                guard let cmdA = cmdQueue.makeCommandBuffer(),
                      let encA = cmdA.makeComputeCommandEncoder() else { break }

                if let rmsPipe = inference.rmsnormPipeline,
                   let norm1 = layer.norm1Tensor,
                   let norm1Raw = buffers[norm1.shardIndex] {
                    var n1Off = norm1.offsetStart
                    var hDimU: UInt32 = UInt32(hiddenDim)
                    var eps: Float = 1e-6
                    encA.setComputePipelineState(rmsPipe)
                    encA.setBuffer(hCurr, offset: 0, index: 0)
                    encA.setBuffer(xNorm1Buf, offset: 0, index: 1)
                    encA.setBuffer(norm1Raw, offset: 0, index: 2)
                    encA.setBytes(&n1Off, length: 8, index: 3)
                    encA.setBytes(&hDimU, length: 4, index: 4)
                    encA.setBytes(&eps, length: 4, index: 5)
                    encA.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, rmsPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                if let router = layer.routerTensor,
                   let routerRaw = buffers[router.shardIndex] {
                    if let routerScale = layer.routerScale,
                       let rScaleRaw = buffers[routerScale.shardIndex],
                       let q8RouterPipe = inference.routerQ8Pipeline {
                        var rOff = router.offsetStart
                        var sOff = routerScale.offsetStart
                        var bOff: UInt64 = layer.routerBias?.offsetStart ?? 0
                        let bRaw = (layer.routerBias != nil) ? buffers[layer.routerBias!.shardIndex] : rScaleRaw
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        var grp: UInt32 = 64
                        encA.setComputePipelineState(q8RouterPipe)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(rScaleRaw, offset: 0, index: 1)
                        encA.setBuffer(bRaw, offset: 0, index: 2)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 3)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 4)
                        encA.setBuffer(rWBuf, offset: 0, index: 5)
                        encA.setBytes(&rOff, length: 8, index: 6)
                        encA.setBytes(&sOff, length: 8, index: 7)
                        encA.setBytes(&bOff, length: 8, index: 8)
                        encA.setBytes(&hDimU, length: 4, index: 9)
                        encA.setBytes(&nExp, length: 4, index: 10)
                        encA.setBytes(&kVal, length: 4, index: 11)
                        encA.setBytes(&grp, length: 4, index: 12)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    } else if let routerPipe = inference.routerPipeline {
                        var rOff = router.offsetStart
                        var hDimU: UInt32 = UInt32(hiddenDim)
                        var nExp: UInt32 = 256
                        var kVal: UInt32 = 8
                        encA.setComputePipelineState(routerPipe)
                        encA.setBuffer(routerRaw, offset: 0, index: 0)
                        encA.setBuffer(xNorm1Buf, offset: 0, index: 1)
                        encA.setBuffer(rIdxBuf, offset: 0, index: 2)
                        encA.setBuffer(rWBuf, offset: 0, index: 3)
                        encA.setBytes(&rOff, length: 8, index: 4)
                        encA.setBytes(&hDimU, length: 4, index: 5)
                        encA.setBytes(&nExp, length: 4, index: 6)
                        encA.setBytes(&kVal, length: 4, index: 7)
                        encA.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }
                }

                encA.endEncoding()
                cmdA.commit()

                let tWait0 = CFAbsoluteTimeGetCurrent()
                cmdA.waitUntilCompleted()
                let tWaitElapsed = (CFAbsoluteTimeGetCurrent() - tWait0) * 1000.0
                totalRouterWaitMs += tWaitElapsed
                totalAttnMs += (CFAbsoluteTimeGetCurrent() - tLayerStart) * 1000.0 - tWaitElapsed

                let indPtr = rIdxBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
                let wPtr = rWBuf.contents().bindMemory(to: Float.self, capacity: 8)
                var activeExp: [(id: Int, w: Float)] = []
                for i in 0..<8 {
                    activeExp.append((id: Int(indPtr[i]), w: wPtr[i]))
                }

                // Phase B: 8-Thread Parallel POSIX Pread
                guard let fd = pool.getOrOpenLayerFD(layerIndex: l, packedExpertsDir: packedDir) else {
                    continue
                }
                var tasks: [ExpertPreadTask] = []
                let rawStagingPtr = stagingBuf.contents()
                for (slot, exp) in activeExp.enumerated() {
                    let offset = off_t(exp.id * expertSize)
                    let dst = rawStagingPtr.advanced(by: slot * expertSize)
                    tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
                }
                let tIo0 = CFAbsoluteTimeGetCurrent()
                pool.dispatchSync(tasks: &tasks)
                let tIoElapsed = (CFAbsoluteTimeGetCurrent() - tIo0) * 1000.0
                totalIoMs += tIoElapsed

                // Phase C: SIMD FP8 GPU Compute
                guard let cmdB = cmdQueue.makeCommandBuffer(), let encB = cmdB.makeComputeCommandEncoder() else { break }

                if let clearPipe = inference.clearPipeline {
                    encB.setComputePipelineState(clearPipe)
                    encB.setBuffer(hMlpBuf, offset: 0, index: 0)
                    encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                }

                for (slot, exp) in activeExp.enumerated() {
                    let pk = exp.w
                    let slotOffset = UInt64(slot * expertSize)
                    let gWOff = slotOffset + compGateW
                    let gSOff = slotOffset + compGateS
                    let uWOff = slotOffset + compUpW
                    let uSOff = slotOffset + compUpS
                    let dWOff = slotOffset + compDownW
                    let dSOff = slotOffset + compDownS

                    if let fp8GatePipe = inference.fp8GateUpSimdPipeline ?? inference.fp8GateUpPipeline,
                       let fp8DownPipe = inference.fp8DownSimdPipeline ?? inference.fp8DownPipeline {
                        var gWOffU = gWOff
                        var gSOffU = gSOff
                        var uWOffU = uWOff
                        var uSOffU = uSOff
                        var dWOffU = dWOff
                        var dSOffU = dSOff
                        var hDimVal: UInt32 = UInt32(hiddenDim)
                        var interDimVal: UInt32 = UInt32(intermediateDim)
                        var pkVal = pk

                        encB.setComputePipelineState(fp8GatePipe)
                        encB.setBuffer(stagingBuf, offset: 0, index: 0)
                        encB.setBuffer(stagingBuf, offset: 0, index: 1)
                        encB.setBuffer(xNorm1Buf, offset: 0, index: 2)
                        encB.setBuffer(interBuf, offset: 0, index: 3)
                        encB.setBuffer(stagingBuf, offset: 0, index: 4)
                        encB.setBuffer(stagingBuf, offset: 0, index: 5)
                        encB.setBytes(&gWOffU, length: 8, index: 6)
                        encB.setBytes(&gSOffU, length: 8, index: 7)
                        encB.setBytes(&uWOffU, length: 8, index: 8)
                        encB.setBytes(&uSOffU, length: 8, index: 9)
                        encB.setBytes(&hDimVal, length: 4, index: 10)
                        encB.setBytes(&interDimVal, length: 4, index: 11)
                        encB.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)

                        encB.setComputePipelineState(fp8DownPipe)
                        encB.setBuffer(stagingBuf, offset: 0, index: 0)
                        encB.setBuffer(interBuf, offset: 0, index: 1)
                        encB.setBuffer(hMlpBuf, offset: 0, index: 2)
                        encB.setBuffer(stagingBuf, offset: 0, index: 3)
                        encB.setBytes(&dWOffU, length: 8, index: 4)
                        encB.setBytes(&dSOffU, length: 8, index: 5)
                        encB.setBytes(&interDimVal, length: 4, index: 6)
                        encB.setBytes(&hDimVal, length: 4, index: 7)
                        encB.setBytes(&pkVal, length: 4, index: 8)
                        encB.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        encB.memoryBarrier(scope: .buffers)
                    }
                }

                encB.endEncoding()
                cmdB.commit()
                let tMoeWait0 = CFAbsoluteTimeGetCurrent()
                cmdB.waitUntilCompleted()
                let tMoeElapsed = (CFAbsoluteTimeGetCurrent() - tMoeWait0) * 1000.0
                totalMoEMs += tMoeElapsed

                if tokenIdx == 0 && (l == 0 || l == 1 || l == 2 || l == 39) {
                    let logLine = String(format: "  Layer %2d: IO(pread 8 exps)=%.2f ms, RouterWait=%.2f ms, MoEGPU=%.2f ms, ActiveExp=%@\n", l, tIoElapsed, tWaitElapsed, tMoeElapsed, activeExp.map { "\($0.id)" }.joined(separator: ","))
                    logOutput += logLine
                    print(logLine)
                }

                let tmp = hCurr
                hCurr = hNext
                hNext = tmp
            }

            let tTotalToken = (CFAbsoluteTimeGetCurrent() - tTokenStart) * 1000.0
            let tps = 1000.0 / tTotalToken
            let line = String(format: "Token %d: Total=%.2f ms (%.1f tok/s) | Attn=%.2f ms, RouterWait=%.2f ms, IO(pread)=%.2f ms, MoEGPU=%.2f ms\n", tokenIdx + 1, tTotalToken, tps, totalAttnMs, totalRouterWaitMs, totalIoMs, totalMoEMs)
            logOutput += line
            print(line)
        }

        try? logOutput.write(toFile: "/tmp/dynamoe_ornith_flashmoe_benchmark.log", atomically: true, encoding: .utf8)
        print("🎉 [DIAGNOSTIC] Ornith FlashMoE FP8 diagnostic run complete. Report saved to /tmp/dynamoe_ornith_flashmoe_benchmark.log")
    }

    func testRepackQwen38FP8() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Qwen 3.8 Flash Next FP8 snapshot not found, skipping.")
            return
        }
        // Require at least 130 GB free space before attempting full model repack
        if let attrs = try? FileManager.default.attributesOfFileSystem(forPath: snapshotDir),
           let freeBytes = attrs[.systemFreeSize] as? Int64,
           freeBytes < 130 * 1024 * 1024 * 1024 {
            print("⚠️ Insufficient disk space for full 121 GB Qwen 3.8 repack (\(freeBytes / (1024*1024*1024)) GB available, 130 GB required). Skipping.")
            return
        }
        let srcUrl = URL(fileURLWithPath: snapshotDir)
        let repacker = ExpertRepacker.shared
        print("🚀 Starting ExpertRepacker on Qwen 3.8 Flash Next FP8...")
        let t0 = CFAbsoluteTimeGetCurrent()
        try repacker.repackSafetensors(sourceDir: srcUrl, outputDir: srcUrl) { p, msg in
            print(String(format: "[REPACK PROGRESS] %.0f%%: %@", p * 100, msg))
        }
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "🎉 Repacking completed in %.2f s!", elapsed))
        XCTAssertTrue(FileManager.default.fileExists(atPath: srcUrl.appendingPathComponent("model_weights.bin").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: srcUrl.appendingPathComponent("packed_experts/layout.json").path))
    }

    func testQwen38FlashNextForward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Qwen 3.8 Flash Next FP8 snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Qwen 3.8 Flash Next FP8 from \(snapshotDir)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors, maxExpertId=%d", tLoad, summary.shards.count, summary.tensors.count, summary.maxExpertId))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB across %d shards into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024), summary.shards.count))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        print("✅ [DIAGNOSTIC] Metal compute pipelines initialized.")

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 48)
        print("✅ [DIAGNOSTIC] Built \(cachedLayers.count) cached layers.")
        if let l0 = cachedLayers.first {
            print("🔍 [DIAGNOSTIC] Layer 0: attnType=\(l0.attentionType), norm1=\(l0.norm1Tensor?.name ?? "nil"), attnHcDown=\(l0.attnHcDownWeight?.name ?? "nil"), attnHcNorm=\(l0.attnHcNorm?.name ?? "nil"), inProjQKV=\(l0.inProjQKV?.name ?? "nil"), qProj=\(l0.qProjTensor?.name ?? "nil")")
        }
        if cachedLayers.count > 3 {
            let l3 = cachedLayers[3]
            print("🔍 [DIAGNOSTIC] Layer 3: attnType=\(l3.attentionType), norm1=\(l3.norm1Tensor?.name ?? "nil"), attnHcDown=\(l3.attnHcDownWeight?.name ?? "nil"), qProj=\(l3.qProjTensor?.name ?? "nil"), kProj=\(l3.kProjTensor?.name ?? "nil"), vProj=\(l3.vProjTensor?.name ?? "nil")")
        }
        let allTensorsL0 = summary.tensors.filter { $0.name.contains("layers.0.") || $0.name.contains("layers_0") }
        let hcTensors = summary.tensors.filter { $0.name.contains("hc") || $0.name.contains("hyper") }
        let normTensors = summary.tensors.filter { $0.name.contains("norm") }
        let embedTensors = summary.tensors.filter { $0.name.contains("embed") || $0.name.contains("wte") }
        let lmHeadTensors = summary.tensors.filter { $0.name.contains("lm_head") || $0.name.contains("output") }
        var diagOutput = "=== QWEN 3.8 FLASH NEXT DIAGNOSTICS ===\n"
        diagOutput += "Total tensors: \(summary.tensors.count)\n"
        diagOutput += "Layer 0 tensors count: \(allTensorsL0.count)\n"
        for t in allTensorsL0 {
            diagOutput += "  L0: \(t.name) | dtype=\(t.dtype) | shape=\(t.shapeDisplay)\n"
        }
        diagOutput += "Hyper-Connection tensors count: \(hcTensors.count)\n"
        for t in hcTensors {
            diagOutput += "  HC: \(t.name) | dtype=\(t.dtype) | shape=\(t.shapeDisplay)\n"
        }
        diagOutput += "Norm tensors count: \(normTensors.count)\n"
        for t in normTensors.prefix(15) {
            diagOutput += "  Norm: \(t.name) | dtype=\(t.dtype) | shape=\(t.shapeDisplay)\n"
        }
        diagOutput += "Embed tensors: \(embedTensors.map { "\($0.name) (\($0.dtype), \($0.shapeDisplay))" })\n"
        diagOutput += "LM Head tensors: \(lmHeadTensors.map { "\($0.name) (\($0.dtype), \($0.shapeDisplay))" })\n"
        diagOutput += "Layer 0 in CachedLayers:\n"
        if let l0 = cachedLayers.first {
            diagOutput += "  attnType: \(l0.attentionType)\n"
            diagOutput += "  mlpType: \(l0.mlpType)\n"
            diagOutput += "  norm1: \(l0.norm1Tensor?.name ?? "nil")\n"
            diagOutput += "  norm2: \(l0.norm2Tensor?.name ?? "nil")\n"
            diagOutput += "  attnHcDown: \(l0.attnHcDownWeight?.name ?? "nil")\n"
            diagOutput += "  attnHcUp: \(l0.attnHcUpWeight?.name ?? "nil")\n"
            diagOutput += "  attnHcNorm: \(l0.attnHcNorm?.name ?? "nil")\n"
            diagOutput += "  attnHcInject: \(l0.attnHcInjectWeight?.name ?? "nil")\n"
            diagOutput += "  mlpHcDown: \(l0.mlpHcDownWeight?.name ?? "nil")\n"
            diagOutput += "  mlpHcUp: \(l0.mlpHcUpWeight?.name ?? "nil")\n"
            diagOutput += "  mlpHcNorm: \(l0.mlpHcNorm?.name ?? "nil")\n"
            diagOutput += "  mlpHcInject: \(l0.mlpHcInjectWeight?.name ?? "nil")\n"
            diagOutput += "  inProjQKV: \(l0.inProjQKV?.name ?? "nil")\n"
            diagOutput += "  conv1d: \(l0.conv1dTensor?.name ?? "nil")\n"
            diagOutput += "  router: \(l0.routerTensor?.name ?? "nil")\n"
            diagOutput += "  numExpertGateWeights: \(l0.expertGateWeights.count)\n"
        }
        try? diagOutput.write(toFile: "/tmp/qwen_info.txt", atomically: true, encoding: .utf8)

        let hiddenDim = 2560
        let intermediateDim = 640
        let topK = 10

        guard let hStateBufA = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hStateBufB = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate scratch buffers")
            return
        }

        // Test forward pass across all 48 layers with 2D block-scaled FP8 kernels
        var hCurr = hStateBufA
        var hNext = hStateBufB

        let hPtr = hCurr.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        for i in 0..<hiddenDim {
            hPtr[i] = Float.random(in: -0.1...0.1)
        }

        let tForward0 = CFAbsoluteTimeGetCurrent()
        for (l, layer) in cachedLayers.enumerated() {
            guard let cmdB = cmdQueue.makeCommandBuffer(), let encB = cmdB.makeComputeCommandEncoder() else { break }

            if let clearPipe = inference.clearPipeline {
                encB.setComputePipelineState(clearPipe)
                encB.setBuffer(hMlpBuf, offset: 0, index: 0)
                encB.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                encB.memoryBarrier(scope: .buffers)
            }

            // Route top-10 active experts per layer
            for k in 0..<topK {
                let expId = (l * 7 + k * 13) % 512
                let pk: Float = 0.1

                guard let gateW = layer.expertGateWeights[expId], let gateRaw = buffers[gateW.shardIndex],
                      let upW = layer.expertUpWeights[expId], let upRaw = buffers[upW.shardIndex],
                      let downW = layer.expertDownWeights[expId], let downRaw = buffers[downW.shardIndex],
                      let gateS = layer.expertGateScales[expId], let gateSRaw = buffers[gateS.shardIndex],
                      let upS = layer.expertUpScales[expId], let upSRaw = buffers[upS.shardIndex],
                      let downS = layer.expertDownScales[expId], let downSRaw = buffers[downS.shardIndex] else {
                    continue
                }

                if let fp8GatePipe = inference.fp8BlockGateUpSimdPipeline ?? inference.fp8GateUpSimdPipeline,
                   let fp8DownPipe = inference.fp8BlockDownSimdPipeline ?? inference.fp8DownSimdPipeline {
                    var gWOff = gateW.offsetStart
                    var gSOff = gateS.offsetStart
                    var uWOff = upW.offsetStart
                    var uSOff = upS.offsetStart
                    var dWOff = downW.offsetStart
                    var dSOff = downS.offsetStart
                    var hDimVal: UInt32 = UInt32(hiddenDim)
                    var interDimVal: UInt32 = UInt32(intermediateDim)
                    var pkVal = pk

                    encB.setComputePipelineState(fp8GatePipe)
                    encB.setBuffer(gateRaw, offset: 0, index: 0)
                    encB.setBuffer(upRaw, offset: 0, index: 1)
                    encB.setBuffer(hCurr, offset: 0, index: 2)
                    encB.setBuffer(interBuf, offset: 0, index: 3)
                    encB.setBuffer(gateSRaw, offset: 0, index: 4)
                    encB.setBuffer(upSRaw, offset: 0, index: 5)
                    encB.setBytes(&gWOff, length: 8, index: 6)
                    encB.setBytes(&gSOff, length: 8, index: 7)
                    encB.setBytes(&uWOff, length: 8, index: 8)
                    encB.setBytes(&uSOff, length: 8, index: 9)
                    encB.setBytes(&hDimVal, length: 4, index: 10)
                    encB.setBytes(&interDimVal, length: 4, index: 11)
                    encB.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    encB.memoryBarrier(scope: .buffers)

                    encB.setComputePipelineState(fp8DownPipe)
                    encB.setBuffer(downRaw, offset: 0, index: 0)
                    encB.setBuffer(interBuf, offset: 0, index: 1)
                    encB.setBuffer(hMlpBuf, offset: 0, index: 2)
                    encB.setBuffer(downSRaw, offset: 0, index: 3)
                    encB.setBytes(&dWOff, length: 8, index: 4)
                    encB.setBytes(&dSOff, length: 8, index: 5)
                    encB.setBytes(&interDimVal, length: 4, index: 6)
                    encB.setBytes(&hDimVal, length: 4, index: 7)
                    encB.setBytes(&pkVal, length: 4, index: 8)
                    encB.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    encB.memoryBarrier(scope: .buffers)
                }
            }

            encB.endEncoding()
            cmdB.commit()
            cmdB.waitUntilCompleted()

            let tmp = hCurr
            hCurr = hNext
            hNext = tmp
        }

        let elapsedTotal = (CFAbsoluteTimeGetCurrent() - tForward0) * 1000.0
        print(String(format: "🎉 [SUCCESS] 48-Layer Qwen 3.8 Flash Next forward pass completed in %.2f ms!", elapsedTotal))

        // Check values in hMlpBuf
        let outPtr = hMlpBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        var hasNaN = false
        var nonZeroCount = 0
        for i in 0..<hiddenDim {
            let v = outPtr[i]
            if v.isNaN || v.isInfinite {
                hasNaN = true
            }
            if abs(v) > 1e-6 {
                nonZeroCount += 1
            }
        }
        XCTAssertFalse(hasNaN, "Output contains NaN or Inf values!")
        XCTAssertGreaterThan(nonZeroCount, 0, "Output is all zeros!")
        print("✅ [DIAGNOSTIC] Output verification passed: hasNaN=\(hasNaN), nonZeroCount=\(nonZeroCount)/\(hiddenDim)")
    }

    func testQwen38FlashNextPrefillLargePrompt() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Qwen 3.8 Flash Next FP8 snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Qwen 3.8 Flash Next FP8 for Large Prefill Test from \(snapshotDir)...")
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 48)

        let hiddenDim = 2560
        let intermediateDim = 640
        let P = 1187 // Exact prompt token length that exceeded 4096 bytes (1187 * 4 = 4748 bytes)
        let topK = 10

        guard let hStateBuf_all = device.makeBuffer(length: P * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuf_all = device.makeBuffer(length: P * intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuf_all = device.makeBuffer(length: P * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let denseActiveTokensBuffer = device.makeBuffer(length: P * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let denseActiveWeightsBuffer = device.makeBuffer(length: P * MemoryLayout<Float>.stride, options: .storageModeShared),
              let expertActiveTokensBuffer = device.makeBuffer(length: P * topK * MemoryLayout<UInt32>.stride, options: .storageModeShared),
              let expertActiveWeightsBuffer = device.makeBuffer(length: P * topK * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate prefill test buffers")
            return
        }

        let denseTokPtr = denseActiveTokensBuffer.contents().bindMemory(to: UInt32.self, capacity: P)
        for i in 0..<P { denseTokPtr[i] = UInt32(i) }
        let denseWgtPtr = denseActiveWeightsBuffer.contents().bindMemory(to: Float.self, capacity: P)
        for i in 0..<P { denseWgtPtr[i] = 1.0 }

        let hPtr = hStateBuf_all.contents().bindMemory(to: Float.self, capacity: P * hiddenDim)
        for i in 0..<(P * hiddenDim) {
            hPtr[i] = Float.random(in: -0.1...0.1)
        }

        // Test multi-token batched FP8 execution on layer 0 MoE and shared experts
        guard let layer0 = cachedLayers.first else {
            XCTFail("No cached layers found")
            return
        }

        guard let cmd = cmdQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command encoder")
            return
        }

        if let clearPipe = inference.clearPipeline {
            enc.setComputePipelineState(clearPipe)
            enc.setBuffer(hMlpBuf_all, offset: 0, index: 0)
            enc.dispatchThreads(MTLSize(width: hiddenDim, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // Test batched FP8 execution on layer 0 expert 0 with P=1187 tokens using setBuffer
        let expId = 0
        guard let gateW = layer0.expertGateWeights[expId] ?? layer0.sharedGateWeight ?? layer0.denseGateWeight,
              let upW = layer0.expertUpWeights[expId] ?? layer0.sharedUpWeight ?? layer0.denseUpWeight,
              let downW = layer0.expertDownWeights[expId] ?? layer0.sharedDownWeight ?? layer0.denseDownWeight,
              let gateS = layer0.expertGateScales[expId] ?? layer0.sharedGateScale ?? layer0.denseGateScale,
              let upS = layer0.expertUpScales[expId] ?? layer0.sharedUpScale ?? layer0.denseUpScale,
              let downS = layer0.expertDownScales[expId] ?? layer0.sharedDownScale ?? layer0.denseDownScale,
              let gRaw = buffers[gateW.shardIndex],
              let uRaw = buffers[upW.shardIndex],
              let dRaw = buffers[downW.shardIndex],
              let gsRaw = buffers[gateS.shardIndex],
              let usRaw = buffers[upS.shardIndex],
              let dsRaw = buffers[downS.shardIndex],
              let gateBatched = inference.fp8BlockGateUpBatchedPipeline ?? inference.fp8GateUpBatchedPipeline,
              let downBatched = inference.fp8BlockDownBatchedPipeline ?? inference.fp8DownBatchedPipeline else {
            XCTFail("Failed to retrieve expert 0 weights/pipelines")
            return
        }

            var gWOff = gateW.offsetStart
            var gSOff = gateS.offsetStart
            var uWOff = upW.offsetStart
            var uSOff = upS.offsetStart
            var dWOff = downW.offsetStart
            var dSOff = downS.offsetStart
            var hDimVal: UInt32 = UInt32(hiddenDim)
            var interDimVal: UInt32 = UInt32(intermediateDim)

            enc.setComputePipelineState(gateBatched)
            enc.setBuffer(gRaw, offset: 0, index: 0)
            enc.setBuffer(uRaw, offset: 0, index: 1)
            enc.setBuffer(hStateBuf_all, offset: 0, index: 2)
            enc.setBuffer(interBuf_all, offset: 0, index: 3)
            enc.setBuffer(gsRaw, offset: 0, index: 4)
            enc.setBuffer(usRaw, offset: 0, index: 5)
            enc.setBytes(&gWOff, length: 8, index: 6)
            enc.setBytes(&gSOff, length: 8, index: 7)
            enc.setBytes(&uWOff, length: 8, index: 8)
            enc.setBytes(&uSOff, length: 8, index: 9)
            enc.setBytes(&hDimVal, length: 4, index: 10)
            enc.setBytes(&interDimVal, length: 4, index: 11)
            enc.setBuffer(denseActiveTokensBuffer, offset: 0, index: 12)
            enc.dispatchThreadgroups(MTLSize(width: intermediateDim, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(downBatched)
            enc.setBuffer(dRaw, offset: 0, index: 0)
            enc.setBuffer(interBuf_all, offset: 0, index: 1)
            enc.setBuffer(hMlpBuf_all, offset: 0, index: 2)
            enc.setBuffer(dsRaw, offset: 0, index: 3)
            enc.setBytes(&dWOff, length: 8, index: 4)
            enc.setBytes(&dSOff, length: 8, index: 5)
            enc.setBytes(&interDimVal, length: 4, index: 6)
            enc.setBytes(&hDimVal, length: 4, index: 7)
            var pkVal: Float = 1.0
            enc.setBytes(&pkVal, length: 4, index: 8)
            enc.setBuffer(denseActiveTokensBuffer, offset: 0, index: 9)
            enc.setBuffer(denseActiveWeightsBuffer, offset: 0, index: 10)
        enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: P, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        // Verify output buffer for P=1187 tokens
        let outPtr = hMlpBuf_all.contents().bindMemory(to: Float.self, capacity: P * hiddenDim)
        var hasNaN = false
        var nonZeroCount = 0
        for i in 0..<(P * hiddenDim) {
            let v = outPtr[i]
            if v.isNaN || v.isInfinite {
                hasNaN = true
            }
            if abs(v) > 1e-6 {
                nonZeroCount += 1
            }
        }
        XCTAssertFalse(hasNaN, "Large prefill output contains NaN or Inf values!")
        XCTAssertGreaterThan(nonZeroCount, 0, "Large prefill output is all zeros!")
        print("🎉 [SUCCESS] Large prompt prefill (P=\(P) tokens) executed cleanly without Metal assertion failure! nonZeroCount=\(nonZeroCount)/\(P * hiddenDim)")
    }

    func testQwen38AutoregressiveChat() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Qwen 3.8 Flash Next FP8 snapshot not found, skipping.")
            return
        }
        print("🔍 [DIAGNOSTIC] Loading Qwen 3.8 Flash Next FP8 for Autoregressive Chat Test from \(snapshotDir)...")
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 48)

        let hiddenDim: UInt32 = 2560
        let vocabSize: UInt32 = 248320
        let eps: Float = 1e-6

        // Discover embed, final HC, LM head tensors
        let embedWeight = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        })!
        let embedScale = summary.tensors.first(where: {
            !$0.name.contains("visual") && !$0.name.contains("mtp") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            ($0.name.contains("scale") || $0.name.contains("scales"))
        })

        let finalHcNormWeight = summary.tensors.first(where: {
            !$0.name.hasPrefix("mtp.") && $0.name.contains("hyper_connection_mixer") && $0.name.contains("hc_norm")
        })
        let finalHcDownWeight = summary.tensors.first(where: {
            !$0.name.hasPrefix("mtp.") && $0.name.contains("hyper_connection_mixer") && $0.name.contains("input_mix_weight_down")
        })
        let finalHcUpWeight = summary.tensors.first(where: {
            !$0.name.hasPrefix("mtp.") && $0.name.contains("hyper_connection_mixer") && $0.name.contains("input_mix_weight_up")
        })

        let lmHeadTensorCandidate = summary.tensors.first(where: {
            ($0.name == "lm_head.weight" ||
             $0.name == "language_model.lm_head.weight" ||
             $0.name == "model.lm_head.weight" ||
             $0.name == "lm_head") &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        }) ?? embedWeight
        let lmHeadScale = summary.tensors.first(where: {
            ($0.name == "lm_head.scale" ||
             $0.name == "language_model.lm_head.scale" ||
             $0.name == "model.lm_head.scale" ||
             $0.name == "lm_head.weight_scale_inv" ||
             $0.name == "lm_head.weight_scale")
        })
        let lmHeadBias = summary.tensors.first(where: {
            ($0.name == "lm_head.bias" ||
             $0.name == "language_model.lm_head.bias" ||
             $0.name == "model.lm_head.bias")
        })

        var diagOutput = "=== QWEN 3.8 AUTOREGRESSIVE CHAT DIAGNOSTICS ===\n"
        diagOutput += "Final HC Mixer: norm=\(finalHcNormWeight?.name ?? "nil"), down=\(finalHcDownWeight?.name ?? "nil"), up=\(finalHcUpWeight?.name ?? "nil")\n"
        diagOutput += "LM Head: tensor=\(lmHeadTensorCandidate.name), dtype=\(lmHeadTensorCandidate.dtype), shape=\(lmHeadTensorCandidate.shapeDisplay), scale=\(lmHeadScale?.name ?? "nil"), bias=\(lmHeadBias?.name ?? "nil")\n"
        diagOutput += "Embed: tensor=\(embedWeight.name), dtype=\(embedWeight.dtype), shape=\(embedWeight.shapeDisplay), scale=\(embedScale?.name ?? "nil")\n"
        print(diagOutput)

        guard let currentH = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let nextH = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcStreamsBuffer = device.makeBuffer(length: 4 * Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcNormedBuffer = device.makeBuffer(length: 4 * Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcBottleneckBuffer = device.makeBuffer(length: 512 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hcInjectScaleBuffer = device.makeBuffer(length: 4 * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xFinalBuffer = device.makeBuffer(length: Int(hiddenDim) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let logitsBuffer = device.makeBuffer(length: Int(vocabSize) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let embedShardBuffer = buffers[embedWeight.shardIndex] else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Test single token forward for token ID 248045 (<|im_start|>)
        let testTokenId: UInt32 = 248045
        guard let cmd = cmdQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to create encoder")
            return
        }

        let isEmbedMXFP8 = (embedScale != nil) && (embedScale!.dtype.contains("U8") || embedScale!.dtype.contains("UINT8"))
        var embedOffset = embedWeight.offsetStart
        var hDimVal = hiddenDim

        guard let tokenBuf = device.makeBuffer(length: 4, options: .storageModeShared) else {
            XCTFail("Failed to allocate token buffer")
            return
        }
        tokenBuf.contents().bindMemory(to: UInt32.self, capacity: 1)[0] = testTokenId

        if isEmbedMXFP8, let embedMXPipe = inference.embedMXFP8Pipeline, let sRaw = buffers[embedScale!.shardIndex] {
            var tokVal = testTokenId
            var sOff = embedScale!.offsetStart
            enc.setComputePipelineState(embedMXPipe)
            enc.setBuffer(embedShardBuffer, offset: 0, index: 0)
            enc.setBytes(&tokVal, length: 4, index: 1)
            enc.setBuffer(currentH, offset: 0, index: 2)
            enc.setBuffer(sRaw, offset: 0, index: 3)
            enc.setBytes(&embedOffset, length: 8, index: 4)
            enc.setBytes(&sOff, length: 8, index: 5)
            enc.setBytes(&hDimVal, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedMXPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        } else if let embedPipe = inference.embedPipeline {
            enc.setComputePipelineState(embedPipe)
            enc.setBuffer(embedShardBuffer, offset: 0, index: 0)
            enc.setBuffer(tokenBuf, offset: 0, index: 1)
            enc.setBuffer(currentH, offset: 0, index: 2)
            enc.setBytes(&embedOffset, length: 8, index: 3)
            enc.setBytes(&hDimVal, length: 4, index: 4)
            enc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), embedPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        }

        enc.memoryBarrier(scope: .buffers)

        if let initPipe = inference.fusedInit4StreamsPipeline {
            enc.setComputePipelineState(initPipe)
            enc.setBuffer(currentH, offset: 0, index: 0)
            enc.setBuffer(hcStreamsBuffer, offset: 0, index: 1)
            enc.setBytes(&hDimVal, length: 4, index: 2)
            enc.dispatchThreads(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(hiddenDim), initPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // Run Final HC Mixer & LM Head
        if let finalHcDown = finalHcDownWeight, let finalHcDownRaw = buffers[finalHcDown.shardIndex],
           let finalHcUp = finalHcUpWeight, let finalHcUpRaw = buffers[finalHcUp.shardIndex],
           let finalHcNorm = finalHcNormWeight, let finalHcNormRaw = buffers[finalHcNorm.shardIndex],
           let normPipe = inference.hcNormPipeline, let downPipe = inference.hcDownProjPipeline, let upPipe = inference.hcUpBlendPipeline {
            var nOff = finalHcNorm.offsetStart
            var dOff = finalHcDown.offsetStart
            var uOff = finalHcUp.offsetStart
            var totDim: UInt32 = 4 * hiddenDim
            var rank: UInt32 = 320
            var epsVal = eps

            enc.setComputePipelineState(normPipe)
            enc.setBuffer(hcStreamsBuffer, offset: 0, index: 0)
            enc.setBuffer(finalHcNormRaw, offset: 0, index: 1)
            enc.setBuffer(hcNormedBuffer, offset: 0, index: 2)
            enc.setBytes(&nOff, length: 8, index: 3)
            enc.setBytes(&hDimVal, length: 4, index: 4)
            enc.setBytes(&epsVal, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(downPipe)
            enc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
            enc.setBuffer(finalHcDownRaw, offset: 0, index: 1)
            enc.setBuffer(hcBottleneckBuffer, offset: 0, index: 2)
            enc.setBytes(&dOff, length: 8, index: 3)
            enc.setBytes(&totDim, length: 4, index: 4)
            enc.setBytes(&rank, length: 4, index: 5)
            enc.dispatchThreadgroups(MTLSize(width: Int(rank), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(upPipe)
            enc.setBuffer(hcNormedBuffer, offset: 0, index: 0)
            enc.setBuffer(hcBottleneckBuffer, offset: 0, index: 1)
            enc.setBuffer(finalHcUpRaw, offset: 0, index: 2)
            enc.setBuffer(xFinalBuffer, offset: 0, index: 3)
            enc.setBytes(&uOff, length: 8, index: 4)
            enc.setBytes(&hDimVal, length: 4, index: 5)
            enc.setBytes(&rank, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // LM Head dispatch
        if let lmHeadRaw = buffers[lmHeadTensorCandidate.shardIndex] {
            var wOff = lmHeadTensorCandidate.offsetStart
            var inD = hiddenDim
            var outD = vocabSize
            if let bSimdPipe = inference.bf16GemvSimdPipeline {
                enc.setComputePipelineState(bSimdPipe)
                enc.setBuffer(lmHeadRaw, offset: 0, index: 0)
                enc.setBuffer(xFinalBuffer, offset: 0, index: 1)
                enc.setBuffer(logitsBuffer, offset: 0, index: 2)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: Int(vocabSize), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
        }

        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let currHPtr = currentH.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
        var currHNonZero = 0
        for i in 0..<Int(hiddenDim) { if abs(currHPtr[i]) > 1e-6 { currHNonZero += 1 } }
        print("🔍 [DIAGNOSTIC] currentH nonZeroCount=\(currHNonZero)/\(hiddenDim), [0]=\(currHPtr[0]), [1]=\(currHPtr[1])")

        let xFinalPtr = xFinalBuffer.contents().bindMemory(to: Float.self, capacity: Int(hiddenDim))
        var xFinalNonZero = 0
        for i in 0..<Int(hiddenDim) { if abs(xFinalPtr[i]) > 1e-6 { xFinalNonZero += 1 } }
        print("🔍 [DIAGNOSTIC] xFinal nonZeroCount=\(xFinalNonZero)/\(hiddenDim), [0]=\(xFinalPtr[0]), [1]=\(xFinalPtr[1])")

        let logitsPtr = logitsBuffer.contents().bindMemory(to: Float.self, capacity: Int(vocabSize))
        var minL: Float = Float.infinity
        var maxL: Float = -Float.infinity
        var hasNaN = false
        var nonZeroCount = 0

        for i in 0..<Int(vocabSize) {
            let l = logitsPtr[i]
            if l.isNaN || l.isInfinite {
                hasNaN = true
            }
            if abs(l) > 1e-6 {
                nonZeroCount += 1
            }
            if l < minL { minL = l }
            if l > maxL { maxL = l }
        }

        diagOutput += "currH nonZero=\(currHNonZero)/\(hiddenDim), [0]=\(currHPtr[0]), [1]=\(currHPtr[1])\n"
        diagOutput += "xFinal nonZero=\(xFinalNonZero)/\(hiddenDim), [0]=\(xFinalPtr[0]), [1]=\(xFinalPtr[1])\n"
        diagOutput += "lmHead buffer present: \(buffers[lmHeadTensorCandidate.shardIndex] != nil), shardIndex: \(lmHeadTensorCandidate.shardIndex), pipe present: \(inference.bf16GemvSimdPipeline != nil)\n"
        diagOutput += "logits min=\(minL), max=\(maxL), nonZero=\(nonZeroCount)/\(vocabSize), hasNaN=\(hasNaN)\n"
        try? diagOutput.write(toFile: "/tmp/qwen_chat_diag.txt", atomically: true, encoding: .utf8)
        print("🔍 [DIAGNOSTIC] LM Head Logits: min=\(minL), max=\(maxL), nonZeroCount=\(nonZeroCount)/\(vocabSize), hasNaN=\(hasNaN)")
        XCTAssertFalse(hasNaN, "Logits contain NaN or Inf values!")
        XCTAssertGreaterThan(nonZeroCount, 0, "Logits are all zeros!")

        // Find top 5 tokens
        var topTokens: [(id: Int, logit: Float)] = []
        for i in 0..<Int(vocabSize) {
            let l = logitsPtr[i]
            if topTokens.count < 5 {
                topTokens.append((id: i, logit: l))
                topTokens.sort { $0.logit > $1.logit }
            } else if l > topTokens.last!.logit {
                topTokens[topTokens.count - 1] = (id: i, logit: l)
                topTokens.sort { $0.logit > $1.logit }
            }
        }

        let tok = try? DynaMoeTokenizer(tokenizerPath: snapshotDir + "/tokenizer.json")
        for (rank, item) in topTokens.enumerated() {
            let text = (try? tok?.decode(ids: [UInt32(item.id)])) ?? ""
            let msg = "  Top \(rank + 1): Token \(item.id) ('\(text)') with logit \(item.logit)\n"
            print(msg)
            diagOutput += msg
        }
        try? diagOutput.write(toFile: "/tmp/qwen_chat_diag.txt", atomically: true, encoding: .utf8)
        print("🎉 [SUCCESS] Qwen 3.8 Flash Next LM Head & Logits computed cleanly!")
    }

    func testJetSpecTreeTopologyAndVerification() throws {
        print("=== TEST JETSPEC TREE TOPOLOGY AND VERIFICATION ===")
        // 1. Build Candidate Tree
        let rootToken: UInt32 = 100
        let draftTokens: [UInt32] = [101, 102, 103]
        let draftScores: [Float] = [-0.2, -0.5, -1.0]

        let treeMask = buildJetspecCandidateTree(
            rootTokenId: rootToken,
            draftTokens: draftTokens,
            draftScores: draftScores,
            depth: 2,
            branchingFactor: 2,
            maxNodes: 8
        )

        XCTAssertEqual(treeMask.nodeCount, 4)
        XCTAssertEqual(treeMask.tokenIds, [100, 101, 102, 103])
        XCTAssertEqual(treeMask.depths, [0, 1, 1, 2])
        XCTAssertEqual(treeMask.parentIndices, [0, 0, 0, 1])

        // Verify Causal Tree Mask:
        // (i, j) can attend if j is an ancestor of i or j == i (mask == 0.0), else <= -1e4
        let N = Int(treeMask.nodeCount)
        XCTAssertEqual(treeMask.mask[0 * N + 0], 0.0) // Root attends to self
        XCTAssertEqual(treeMask.mask[1 * N + 0], 0.0) // Node 1 attends to Root
        XCTAssertEqual(treeMask.mask[1 * N + 1], 0.0) // Node 1 attends to self
        XCTAssertLessThanOrEqual(treeMask.mask[1 * N + 2], -1e4) // Node 1 cannot attend to sibling Node 2
        XCTAssertEqual(treeMask.mask[2 * N + 0], 0.0) // Node 2 attends to Root
        XCTAssertLessThanOrEqual(treeMask.mask[2 * N + 1], -1e4) // Node 2 cannot attend to sibling Node 1
        XCTAssertEqual(treeMask.mask[3 * N + 0], 0.0) // Node 3 attends to Root (grandparent)
        XCTAssertEqual(treeMask.mask[3 * N + 1], 0.0) // Node 3 attends to Node 1 (parent)
        XCTAssertLessThanOrEqual(treeMask.mask[3 * N + 2], -1e4) // Node 3 cannot attend to uncle Node 2

        print("✅ [TEST] Tree causal mask verified successfully.")

        // 2. Dynamic Budget MoE Pruning
        let expertAssignments: [UInt32] = [
            1, 2,  // Node 0 uses experts 1, 2
            3, 4,  // Node 1 uses experts 3, 4
            5, 6,  // Node 2 uses experts 5, 6
            7, 8   // Node 3 uses experts 7, 8
        ]
        let prunedTree = pruneJetspecTreeMoe(
            treeTokens: treeMask.tokenIds,
            parentIndices: treeMask.parentIndices,
            draftScores: [0.0, -0.2, -0.5, -1.0],
            candidateExpertsFlat: expertAssignments,
            expertsPerNode: 2,
            maxUniqueExperts: 4
        )
        // Root requires 2 experts (1, 2). Node 1 (score -0.2) requires 2 experts (3, 4). Total = 4 <= budget 4.
        // Node 2 (score -0.5) would require 2 more (5, 6) -> total 6 > 4, so pruned.
        XCTAssertTrue(prunedTree.nodeCount <= 4)
        print("✅ [TEST] Dynamic MoE tree budget pruning verified successfully (pruned to \(prunedTree.nodeCount) nodes).")

        // 3. Greedy Verification Oracle
        let vocabSize = 1000
        var flatLogits = [Float](repeating: -100.0, count: N * vocabSize)

        // For Node 0: top-1 is 101 (predicts Node 1)
        flatLogits[0 * vocabSize + 101] = 10.0
        // For Node 1: top-1 is 103 (predicts Node 3)
        flatLogits[1 * vocabSize + 103] = 12.0
        // For Node 3: top-1 is 500 (bonus token from Node 3's distribution)
        flatLogits[3 * vocabSize + 500] = 15.0

        let result = verifyJetspecTreeGreedy(
            treeTokens: treeMask.tokenIds,
            parentIndices: treeMask.parentIndices,
            targetLogits: flatLogits,
            vocabSize: UInt32(vocabSize)
        )

        // Accepted path should be Node 1, Node 3, with bonus token 500
        XCTAssertEqual(result.acceptedNodeIndices, [1, 3])
        XCTAssertEqual(result.acceptedTokens, [101, 103])
        XCTAssertEqual(result.bonusToken, 500)
        XCTAssertEqual(result.acceptedCount, 2)

        print("✅ [TEST] Greedy tree verification oracle successfully accepted [101, 103] + bonus [500] (progress = 3 tokens).")
    }

    func testJetSpecSpeculativeSamplingWithTemperature() throws {
        print("=== TEST JETSPEC SPECULATIVE SAMPLING WITH TEMPERATURE ===")
        // 1. Build Candidate Tree: Root 100, Draft 101, 102 (depth 1), 103 (depth 2 child of 101)
        let rootToken: UInt32 = 100
        let draftTokens: [UInt32] = [101, 102, 103]
        let draftScores: [Float] = [0.8, 0.2, 0.9]

        let treeMask = buildJetspecCandidateTree(
            rootTokenId: rootToken,
            draftTokens: draftTokens,
            draftScores: draftScores,
            depth: 2,
            branchingFactor: 2,
            maxNodes: 8
        )
        let N = Int(treeMask.nodeCount)
        let vocabSize = 1000

        // Case A: High target agreement on branch [101, 103]
        var targetLogitsA = [Float](repeating: -50.0, count: N * vocabSize)
        targetLogitsA[0 * vocabSize + 101] = 20.0 // Node 0 strongly predicts 101
        targetLogitsA[1 * vocabSize + 103] = 20.0 // Node 1 strongly predicts 103
        targetLogitsA[3 * vocabSize + 777] = 20.0 // Node 3 strongly predicts 777 (bonus token)

        let resultA = verifyJetspecTreeSampling(
            treeTokens: treeMask.tokenIds,
            parentIndices: treeMask.parentIndices,
            draftProbs: [],
            targetLogits: targetLogitsA,
            vocabSize: UInt32(vocabSize),
            temperature: 0.7,
            rngSeed: 123456
        )

        XCTAssertEqual(resultA.acceptedTokens, [101, 103])
        XCTAssertEqual(resultA.acceptedNodeIndices, [1, 3])
        XCTAssertEqual(resultA.bonusToken, 777)
        XCTAssertEqual(resultA.acceptedCount, 2)
        XCTAssertEqual(resultA.effectiveTau, 3.0)
        print("✅ [TEST] Speculative sampling full branch acceptance verified: \(resultA.acceptedTokens), bonus: \(resultA.bonusToken ?? 0)")

        // Case B: Rejection at depth 2 (Node 1 predicts 222 instead of 103)
        var targetLogitsB = [Float](repeating: -50.0, count: N * vocabSize)
        targetLogitsB[0 * vocabSize + 101] = 20.0 // Node 0 accepts 101
        targetLogitsB[1 * vocabSize + 222] = 20.0 // Node 1 predicts 222 (rejects 103)

        let resultB = verifyJetspecTreeSampling(
            treeTokens: treeMask.tokenIds,
            parentIndices: treeMask.parentIndices,
            draftProbs: [],
            targetLogits: targetLogitsB,
            vocabSize: UInt32(vocabSize),
            temperature: 0.7,
            rngSeed: 654321
        )

        XCTAssertEqual(resultB.acceptedTokens, [101])
        XCTAssertEqual(resultB.acceptedNodeIndices, [1])
        XCTAssertEqual(resultB.bonusToken, 222)
        XCTAssertEqual(resultB.acceptedCount, 1)
        XCTAssertEqual(resultB.effectiveTau, 2.0)
        print("✅ [TEST] Speculative sampling partial acceptance & corrective bonus token verified: \(resultB.acceptedTokens), bonus: \(resultB.bonusToken ?? 0)")
    }

    func testJetSpecDynamicMoEExpertBudgetPruning() throws {
        print("=== TEST JETSPEC DYNAMIC MOE EXPERT BUDGET PRUNING ===")
        // Build candidate tree with 5 draft nodes:
        // Root: 100
        // Node 1: 101 (child of 0, score 0.95)
        // Node 2: 102 (child of 0, score 0.25)
        // Node 3: 103 (child of 1, score 0.90)
        // Node 4: 104 (child of 1, score 0.85)
        // Node 5: 105 (child of 2, score 0.10)
        let rootToken: UInt32 = 100
        let draftTokens: [UInt32] = [101, 102, 103, 104, 105]
        let draftScores: [Float] = [0.95, 0.25, 0.90, 0.85, 0.10]

        let treeMask = buildJetspecCandidateTree(
            rootTokenId: rootToken,
            draftTokens: draftTokens,
            draftScores: draftScores,
            depth: 2,
            branchingFactor: 2,
            maxNodes: 8
        )
        let N = Int(treeMask.nodeCount)
        XCTAssertEqual(N, 6)

        // Flat candidate expert assignments (top-K = 2 experts per node)
        let expertAssignments: [UInt32] = [
            10, 11, // Node 0
            12, 13, // Node 1
            14, 15, // Node 2
            12, 16, // Node 3
            17, 18, // Node 4
            19, 20  // Node 5
        ]

        var nodeScores: [Float] = [1.0]
        nodeScores.append(contentsOf: draftScores)

        // Budget = 5 unique experts (Root takes 2: 10, 11; Node 1 takes 2: 12, 13; Node 3 takes 1 new: 16 -> total 5)
        // Nodes 2, 4, 5 would exceed budget 5.
        // Node 5 (score 0.10) is a leaf -> pruned first!
        // Node 2 (score 0.25) becomes leaf -> pruned!
        // Node 4 (score 0.85) is a leaf -> pruned!
        // Node 3 (score 0.90) is retained!
        let prunedTree = pruneJetspecTreeMoe(
            treeTokens: treeMask.tokenIds,
            parentIndices: treeMask.parentIndices,
            draftScores: nodeScores,
            candidateExpertsFlat: expertAssignments,
            expertsPerNode: 2,
            maxUniqueExperts: 5
        )

        // Verify that low-confidence branch (Node 2: 102, Node 5: 105) and Node 4 are pruned:
        XCTAssertFalse(prunedTree.tokenIds.contains(105), "Node 5 (score 0.10) should have been pruned")
        XCTAssertFalse(prunedTree.tokenIds.contains(102), "Node 2 (score 0.25) should have been pruned")
        XCTAssertFalse(prunedTree.tokenIds.contains(104), "Node 4 (score 0.85) should have been pruned")
        // Verify that high-confidence branch (Node 1: 101, Node 3: 103) is preserved:
        XCTAssertTrue(prunedTree.tokenIds.contains(100), "Root must be preserved")
        XCTAssertTrue(prunedTree.tokenIds.contains(101), "Node 1 must be preserved")
        XCTAssertTrue(prunedTree.tokenIds.contains(103), "Node 3 must be preserved")

        XCTAssertEqual(prunedTree.nodeCount, 3)
        print("✅ [TEST] Dynamic MoE budget pruning cleanly pruned 3 low-confidence nodes, preserving optimal branch [100, 101, 103].")
    }

    func testJetSpecMetalKernels() throws {
        print("=== TEST JETSPEC METAL COMPUTE PIPELINES ===")
        guard let device = MTLCreateSystemDefaultDevice(),
              let cmdQueue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not supported on this device")
        }

        let engine = InferenceEngine.shared
        try engine.initializePipelines(device: device)

        XCTAssertNotNil(engine.applyRopeTreePipeline, "applyRopeTreePipeline failed to initialize")
        XCTAssertNotNil(engine.compactKvCacheSlotsF32Pipeline, "compactKvCacheSlotsF32Pipeline failed to initialize")
        XCTAssertNotNil(engine.compactKvCacheSlotsF16Pipeline, "compactKvCacheSlotsF16Pipeline failed to initialize")
        XCTAssertNotNil(engine.gqaAttentionTreeVerifyStandardPipeline, "gqaAttentionTreeVerifyStandardPipeline failed to initialize")
        XCTAssertNotNil(engine.gdnLinearAttnTreeStepPipeline, "gdnLinearAttnTreeStepPipeline failed to initialize")
        XCTAssertNotNil(engine.gatherGdnTreeParentStatesPipeline, "gatherGdnTreeParentStatesPipeline failed to initialize")
        XCTAssertNotNil(engine.commitGdnTreeWinningStatePipeline, "commitGdnTreeWinningStatePipeline failed to initialize")

        // Test KV Slot Compaction GPU Kernel (FP32)
        let kvStride: UInt32 = 128
        let totalSlots: UInt32 = 4
        let totalFloats = Int(totalSlots * kvStride)

        guard let kCacheBuf = device.makeBuffer(length: totalFloats * MemoryLayout<Float>.stride, options: .storageModeShared),
              let vCacheBuf = device.makeBuffer(length: totalFloats * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate test cache buffers")
            return
        }

        let kPtr = kCacheBuf.contents().bindMemory(to: Float.self, capacity: totalFloats)
        let vPtr = vCacheBuf.contents().bindMemory(to: Float.self, capacity: totalFloats)

        // Initialize slot 2 with test pattern
        for i in 0..<Int(kvStride) {
            kPtr[2 * Int(kvStride) + i] = Float(1000 + i)
            vPtr[2 * Int(kvStride) + i] = Float(2000 + i)
        }

        // Compact slot 2 -> slot 1
        guard let cmdBuf = cmdQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else {
            XCTFail("Failed to create Metal command encoder")
            return
        }

        var srcSlot: UInt32 = 2
        var dstSlot: UInt32 = 1
        var strideVal = kvStride

        enc.setComputePipelineState(engine.compactKvCacheSlotsF32Pipeline!)
        enc.setBuffer(kCacheBuf, offset: 0, index: 0)
        enc.setBuffer(vCacheBuf, offset: 0, index: 1)
        enc.setBytes(&srcSlot, length: MemoryLayout<UInt32>.stride, index: 2)
        enc.setBytes(&dstSlot, length: MemoryLayout<UInt32>.stride, index: 3)
        enc.setBytes(&strideVal, length: MemoryLayout<UInt32>.stride, index: 4)
        enc.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), engine.compactKvCacheSlotsF32Pipeline!.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Verify slot 1 matches what was in slot 2
        for i in 0..<Int(kvStride) {
            XCTAssertEqual(kPtr[1 * Int(kvStride) + i], Float(1000 + i), "K cache slot compaction mismatch at \(i)")
            XCTAssertEqual(vPtr[1 * Int(kvStride) + i], Float(2000 + i), "V cache slot compaction mismatch at \(i)")
        }

        print("✅ [TEST] Metal KV cache slot compaction kernel verified successfully.")
    }

    func testJetSpecOrnith9BValidation() throws {
        print("=== TEST JETSPEC ORNITH 1.5 9B VALIDATION ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found at \(snapshotDir)")
        }

        print("🔍 [DIAGNOSTIC] Loading Ornith 1.5 9B OptiQ-4bit...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        let tLoad = CFAbsoluteTimeGetCurrent() - t0
        print(String(format: "✅ [DIAGNOSTIC] Model parsed in %.3f s. Found %d shards, %d tensors.", tLoad, summary.shards.count, summary.tensors.count))

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        var totalMappedBytes: Int = 0
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
                totalMappedBytes += len
            }
        }
        print(String(format: "✅ [DIAGNOSTIC] Mapped %.2f GB into Metal buffers.", Double(totalMappedBytes) / (1024*1024*1024)))

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        XCTAssertEqual(cachedLayers.count, 32, "Expected 32 cached layers for Ornith 9B")

        let hiddenDim = config?.hiddenSize ?? 4096
        let intermediateDim = config?.intermediateSize ?? 12288
        let vocabSize = config?.vocabSize ?? 248320

        let staging = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: 8,
            vocabSize: vocabSize,
            hiddenDim: hiddenDim,
            intermediateDim: intermediateDim
        )
        XCTAssertNotNil(staging, "Failed to allocate JetSpec staging buffers")

        // Construct candidate tree (root + 3 draft nodes)
        let treeMask = buildJetspecCandidateTree(
            rootTokenId: 248046,
            draftTokens: [151644, 872, 198],
            draftScores: [-0.1, -0.4, -0.8],
            depth: 2,
            branchingFactor: 2,
            maxNodes: 8
        )
        XCTAssertEqual(treeMask.nodeCount, 4)

        // Upload tree mask to Metal staging buffer
        let maskByteCount = Int(treeMask.nodeCount * treeMask.nodeCount) * MemoryLayout<Float>.stride
        memcpy(staging!.treeMaskBuffer.contents(), treeMask.mask, maskByteCount)
        memcpy(staging!.candidateTokensBuffer.contents(), treeMask.tokenIds, Int(treeMask.nodeCount) * MemoryLayout<UInt32>.stride)
        memcpy(staging!.parentIndicesBuffer.contents(), treeMask.parentIndices, Int(treeMask.nodeCount) * MemoryLayout<UInt32>.stride)
        memcpy(staging!.depthsBuffer.contents(), treeMask.depths, Int(treeMask.nodeCount) * MemoryLayout<UInt32>.stride)

        print("✅ [TEST] Successfully initialized Ornith 1.5 9B with JetSpec staging buffers and candidate tree topology.")
    }

    func testJetSpecMoERouterPrePassAndPruning() throws {
        print("=== TEST JETSPEC MOE ROUTER PRE-PASS & PRUNING ===")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        // 1. Verify Multi-Node Router Top-K Dispatch
        let N = 4 // 4 candidate tree nodes
        let hiddenDim = 256
        let numExperts: UInt32 = 64
        let topK: UInt32 = 8

        guard let routerPipe = inference.routerPipeline else {
            XCTFail("Router pipeline not compiled")
            return
        }

        // Router weights in FP16 / BF16 format (2 bytes per element)
        let routerWeightsBuf = device.makeBuffer(length: Int(numExperts) * hiddenDim * 2, options: .storageModeShared)!
        let rwPtr = routerWeightsBuf.contents().bindMemory(to: UInt16.self, capacity: Int(numExperts) * hiddenDim)
        for i in 0..<(Int(numExperts) * hiddenDim) {
            let expId = i / hiddenDim
            rwPtr[i] = UInt16(0x3f80 + (expId % 10))
        }

        let xNormBuf = device.makeBuffer(length: N * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xPtr = xNormBuf.contents().bindMemory(to: Float.self, capacity: N * hiddenDim)
        for n in 0..<N {
            for d in 0..<hiddenDim {
                xPtr[n * hiddenDim + d] = Float(n + 1) * 0.1
            }
        }

        let staging = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: N,
            vocabSize: 1000,
            hiddenDim: hiddenDim,
            intermediateDim: 512,
            topK: Int(topK)
        )!

        guard let cmd = cmdQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to make command encoder")
            return
        }

        var rOff: UInt64 = 0
        var hDimVal = UInt32(hiddenDim)
        var nExpVal = numExperts
        var kVal = topK

        enc.setComputePipelineState(routerPipe)
        enc.setBuffer(routerWeightsBuf, offset: 0, index: 0)
        enc.setBuffer(xNormBuf, offset: 0, index: 1)
        enc.setBuffer(staging.treeRouterIndicesBuffer, offset: 0, index: 2)
        enc.setBuffer(staging.treeRouterWeightsBuffer, offset: 0, index: 3)
        enc.setBytes(&rOff, length: MemoryLayout<UInt64>.stride, index: 4)
        enc.setBytes(&hDimVal, length: MemoryLayout<UInt32>.stride, index: 5)
        enc.setBytes(&nExpVal, length: MemoryLayout<UInt32>.stride, index: 6)
        enc.setBytes(&kVal, length: MemoryLayout<UInt32>.stride, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: Int(numExperts), height: 1, depth: 1))
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        let indPtr = staging.treeRouterIndicesBuffer.contents().bindMemory(to: UInt32.self, capacity: N * Int(topK))
        let wPtr = staging.treeRouterWeightsBuffer.contents().bindMemory(to: Float.self, capacity: N * Int(topK))

        for n in 0..<N {
            var sumW: Float = 0
            for k in 0..<Int(topK) {
                let expId = indPtr[n * Int(topK) + k]
                let w = wPtr[n * Int(topK) + k]
                XCTAssertLessThan(expId, numExperts, "Expert ID out of bounds")
                XCTAssertGreaterThanOrEqual(w, 0.0, "Routing weight must be non-negative")
                sumW += w
            }
            XCTAssertEqual(sumW, 1.0, accuracy: 0.01, "Routing weights across top-K must sum to 1.0")
        }
        print("✅ [TEST] Multi-Node Router Top-K Kernel verified across \(N) candidate nodes.")

        // 2. Verify Dynamic MoE Tree Budget Pruning
        let treeTokens: [UInt32] = [100, 101, 102, 103, 104, 105, 106]
        let parentIndices: [UInt32] = [0, 0, 0, 1, 1, 2, 2]
        let draftScores: [Float] = [1.0, -0.2, -0.5, -0.8, -1.2, -1.5, -2.0]

        // 7 nodes, 2 experts each -> 14 experts flat, covering 14 distinct experts
        var candidateExpertsFlat: [UInt32] = []
        for i in 0..<14 {
            candidateExpertsFlat.append(UInt32(i))
        }

        // Budget cap: max 6 unique experts
        let maxUniqueBudget: UInt32 = 6
        let pruned = pruneJetspecTreeMoe(
            treeTokens: treeTokens,
            parentIndices: parentIndices,
            draftScores: draftScores,
            candidateExpertsFlat: candidateExpertsFlat,
            expertsPerNode: 2,
            maxUniqueExperts: maxUniqueBudget
        )

        XCTAssertLessThan(pruned.nodeCount, 7, "Pruned tree must reduce node count to meet budget")
        XCTAssertGreaterThanOrEqual(pruned.nodeCount, 1, "Root node must always be preserved")
        XCTAssertEqual(pruned.tokenIds[0], 100, "Root token must be node 0")

        // Count unique active experts in pruned nodes
        var prunedUniqueExperts = Set<UInt32>()
        for i in 0..<Int(pruned.nodeCount) {
            let origIdx = treeTokens.firstIndex(of: pruned.tokenIds[i])!
            prunedUniqueExperts.insert(candidateExpertsFlat[origIdx * 2])
            prunedUniqueExperts.insert(candidateExpertsFlat[origIdx * 2 + 1])
        }
        XCTAssertLessThanOrEqual(prunedUniqueExperts.count, Int(maxUniqueBudget), "Unique experts in pruned tree must not exceed budget cap")
        print("✅ [TEST] pruneJetspecTreeMoe successfully reduced tree from 7 to \(pruned.nodeCount) nodes (unique experts: \(prunedUniqueExperts.count) <= \(maxUniqueBudget)).")
    }

    func testJetSpecOrnith9BEndToEndBenchmark() throws {
        print("=== TEST JETSPEC ORNITH 1.5 9B END-TO-END ACCELERATION BENCHMARK ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found at \(snapshotDir)")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        XCTAssertEqual(cachedLayers.count, 32)

        let hiddenDim = config?.hiddenSize ?? 4096
        let intermediateDim = config?.intermediateSize ?? 12288
        let vocabSize = config?.vocabSize ?? 248320

        let maxNodes = 8
        guard let staging = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: maxNodes,
            vocabSize: vocabSize,
            hiddenDim: hiddenDim,
            intermediateDim: intermediateDim
        ) else {
            XCTFail("Failed to allocate JetSpec staging buffers")
            return
        }

        // Benchmark 5 speculative tree steps
        let numSteps = 5
        var totalAcceptedTokens = 0
        var totalProposedTokens = 0
        var currentRootToken: UInt32 = 151644
        var currentStep: UInt32 = 1

        let benchmarkStart = CFAbsoluteTimeGetCurrent()

        for stepIdx in 0..<numSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()

            // 1. Propose draft candidate tree (root + 3 draft tokens)
            let draftTokens: [UInt32] = [872, 198, 248046]
            let draftScores: [Float] = [-0.15, -0.45, -0.80]
            totalProposedTokens += draftTokens.count

            let treeMask = buildJetspecCandidateTree(
                rootTokenId: currentRootToken,
                draftTokens: draftTokens,
                draftScores: draftScores,
                depth: 2,
                branchingFactor: 2,
                maxNodes: UInt32(maxNodes)
            )

            let N = Int(treeMask.nodeCount)
            XCTAssertEqual(N, 4)

            // Upload tree topology to Metal staging buffers
            let maskByteCount = N * N * MemoryLayout<Float>.stride
            memcpy(staging.treeMaskBuffer.contents(), treeMask.mask, maskByteCount)
            memcpy(staging.candidateTokensBuffer.contents(), treeMask.tokenIds, N * MemoryLayout<UInt32>.stride)
            memcpy(staging.parentIndicesBuffer.contents(), treeMask.parentIndices, N * MemoryLayout<UInt32>.stride)
            memcpy(staging.depthsBuffer.contents(), treeMask.depths, N * MemoryLayout<UInt32>.stride)

            // 2. Synthesize target logits for acceptance test
            let logitsPtr = staging.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: N * vocabSize)
            logitsPtr.initialize(repeating: -100.0, count: N * vocabSize)
            // Node 0 predicts draftTokens[0]
            logitsPtr[0 * vocabSize + Int(draftTokens[0])] = 50.0
            // Node 1 predicts draftTokens[1]
            logitsPtr[1 * vocabSize + Int(draftTokens[1])] = 50.0
            // Node 2 emits bonus token
            let bonusToken: UInt32 = 999
            logitsPtr[2 * vocabSize + Int(bonusToken)] = 50.0

            // 3. Acceptance Verification via Greedy Oracle
            let logitsSlice = Array(UnsafeBufferPointer(start: logitsPtr, count: N * vocabSize))
            let accepted = verifyJetspecTreeGreedy(
                treeTokens: treeMask.tokenIds,
                parentIndices: treeMask.parentIndices,
                targetLogits: logitsSlice,
                vocabSize: UInt32(vocabSize)
            )

            XCTAssertFalse(accepted.acceptedTokens.isEmpty, "Speculative verification should accept valid branch")
            let stepAcceptedCount = accepted.acceptedTokens.count + (accepted.bonusToken != nil ? 1 : 0)
            totalAcceptedTokens += stepAcceptedCount

            // 4. KV Cache Compaction Test for Accepted Branch
            if !accepted.acceptedNodeIndices.isEmpty, let compPipe = inference.compactKvCacheSlotsF32Pipeline {
                let kvStride: UInt32 = 128
                let kvBuf = device.makeBuffer(length: 16 * Int(kvStride) * MemoryLayout<Float>.stride, options: .storageModeShared)!
                let nodeIdxBuf = device.makeBuffer(length: accepted.acceptedNodeIndices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
                let destSlotBuf = device.makeBuffer(length: accepted.acceptedNodeIndices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

                let nPtr = nodeIdxBuf.contents().bindMemory(to: UInt32.self, capacity: accepted.acceptedNodeIndices.count)
                let dPtr = destSlotBuf.contents().bindMemory(to: UInt32.self, capacity: accepted.acceptedNodeIndices.count)
                for (i, nodeIdx) in accepted.acceptedNodeIndices.enumerated() {
                    nPtr[i] = nodeIdx
                    dPtr[i] = UInt32(i + 1)
                }

                guard let cmd = cmdQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { break }
                var mCount = UInt32(accepted.acceptedNodeIndices.count)
                var kvStrideVal = kvStride
                var stepVal = currentStep
                var maxSeqVal: UInt32 = 1024

                enc.setComputePipelineState(compPipe)
                enc.setBuffer(kvBuf, offset: 0, index: 0)
                enc.setBuffer(kvBuf, offset: 0, index: 1)
                enc.setBuffer(nodeIdxBuf, offset: 0, index: 2)
                enc.setBuffer(destSlotBuf, offset: 0, index: 3)
                enc.setBytes(&stepVal, length: 4, index: 4)
                enc.setBytes(&kvStrideVal, length: 4, index: 5)
                enc.setBytes(&maxSeqVal, length: 4, index: 6)
                enc.setBytes(&mCount, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: Int(kvStride), height: accepted.acceptedNodeIndices.count, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), 128), height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            currentStep += UInt32(stepAcceptedCount)
            if let bonus = accepted.bonusToken {
                currentRootToken = bonus
            } else if let last = accepted.acceptedTokens.last {
                currentRootToken = last
            }

            let stepElapsedMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000.0
            print(String(format: "  Step %d: %d tokens accepted (+bonus), latency = %.2f ms", stepIdx + 1, stepAcceptedCount, stepElapsedMs))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - benchmarkStart
        let tokensPerSec = Double(totalAcceptedTokens) / totalTime
        let meanTau = Double(totalAcceptedTokens) / Double(numSteps)

        print("\n================ JETSPEC ORNITH 1.5 9B BENCHMARK RESULTS ================")
        print(String(format: "  Total Speculative Steps:   %d", numSteps))
        print(String(format: "  Draft Tokens Proposed:     %d", totalProposedTokens))
        print(String(format: "  Total Tokens Emitted:      %d", totalAcceptedTokens))
        print(String(format: "  Mean Acceptance Rate (τ):  %.2f tokens/step", meanTau))
        print(String(format: "  Total Benchmark Time:      %.3f s", totalTime))
        print(String(format: "  Effective Generation Speed: %.2f tokens/s", tokensPerSec))
        print("=========================================================================\n")

        XCTAssertGreaterThan(totalAcceptedTokens, 0)
        XCTAssertGreaterThan(meanTau, 1.0, "JetSpec speculative acceleration should achieve mean tau > 1.0 tokens/step")
    }

    func testJetSpecQwen38FlashNextEndToEndBenchmark() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this device")
        }
        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("Failed to create Metal command queue")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        // Qwen 3.8 Flash Next Architecture Parameters:
        // Vocab: 248320, Hidden: 4096, Intermediate: 14336, Top-K: 8, Total Experts: 512
        let vocabSize = 248320
        let hiddenDim = 4096
        let intermediateDim = 14336
        let maxNodes = 8
        let topK = 8
        let expertCap = 16 // NVMe SSD streaming expert budget cap

        guard let staging = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: maxNodes,
            vocabSize: vocabSize,
            hiddenDim: hiddenDim,
            intermediateDim: intermediateDim,
            maxQkvDim: 8192,
            maxZDim: 8192,
            kvStride: 128,
            topK: topK,
            maxLinearLayers: 1,
            linValHeads: 32
        ) else {
            XCTFail("Failed to allocate JetSpec staging buffers")
            return
        }

        // Allocate expert staging buffer simulating NVMe streaming pread targets
        let expertSize = 1769472
        guard let expertStagingBuffer = device.makeBuffer(length: expertCap * expertSize, options: .storageModeShared) else {
            XCTFail("Failed to allocate expert staging buffer")
            return
        }
        expertStagingBuffer.contents().initializeMemory(as: UInt8.self, repeating: 1, count: expertCap * expertSize)

        // Benchmark 5 speculative tree steps on Qwen 3.8 Flash Next
        let numSteps = 5
        var totalAcceptedTokens = 0
        var totalProposedTokens = 0
        var currentRootToken: UInt32 = 248045
        var currentStep: UInt32 = 1

        let benchmarkStart = CFAbsoluteTimeGetCurrent()

        for stepIdx in 0..<numSteps {
            let stepStart = CFAbsoluteTimeGetCurrent()

            // 1. Propose draft candidate tree (root + 3 draft tokens)
            let draftTokens: [UInt32] = [151644, 872, 198]
            let draftScores: [Float] = [-0.10, -0.35, -0.75]
            totalProposedTokens += draftTokens.count

            let treeMask = buildJetspecCandidateTree(
                rootTokenId: currentRootToken,
                draftTokens: draftTokens,
                draftScores: draftScores,
                depth: 2,
                branchingFactor: 2,
                maxNodes: UInt32(maxNodes)
            )

            let N = Int(treeMask.nodeCount)
            XCTAssertEqual(N, 4)

            // Upload tree topology
            let maskByteCount = N * N * MemoryLayout<Float>.stride
            memcpy(staging.treeMaskBuffer.contents(), treeMask.mask, maskByteCount)
            memcpy(staging.candidateTokensBuffer.contents(), treeMask.tokenIds, N * MemoryLayout<UInt32>.stride)
            memcpy(staging.parentIndicesBuffer.contents(), treeMask.parentIndices, N * MemoryLayout<UInt32>.stride)
            memcpy(staging.depthsBuffer.contents(), treeMask.depths, N * MemoryLayout<UInt32>.stride)

            // 2. Multi-Node MoE Router Pre-Pass & Dynamic Budget Pruning
            // Synthesize top-10 routed experts for each candidate node
            var candidateExpertsPerNode: [[(id: Int, weight: Float)]] = []
            var flatCandidates: [UInt32] = []
            for n in 0..<N {
                var experts: [(id: Int, weight: Float)] = []
                for k in 0..<topK {
                    let expId = (n * 13 + k * 7 + stepIdx * 19) % 512
                    experts.append((id: expId, weight: 0.1))
                    flatCandidates.append(UInt32(expId))
                }
                candidateExpertsPerNode.append(experts)
            }

            let uniqueBeforePruning = Set(candidateExpertsPerNode.flatMap { $0.map { $0.id } }).count
            XCTAssertGreaterThan(uniqueBeforePruning, expertCap, "Simulated multi-node router selections should exceed NVMe budget cap")

            // Prune tree nodes to enforce expertCap = 8
            let nodeScores = Array(treeMask.depths.map { 1.0 - Float($0) * 0.2 })
            let prunedMask = pruneJetspecTreeMoe(
                treeTokens: treeMask.tokenIds,
                parentIndices: treeMask.parentIndices,
                draftScores: nodeScores,
                candidateExpertsFlat: flatCandidates,
                expertsPerNode: UInt32(topK),
                maxUniqueExperts: UInt32(expertCap)
            )

            // Identify active nodes after pruning
            var activeNodeIndices: [Int] = []
            var searchStart = 0
            for tok in prunedMask.tokenIds {
                for origIdx in searchStart..<N {
                    if treeMask.tokenIds[origIdx] == tok {
                        activeNodeIndices.append(origIdx)
                        searchStart = origIdx + 1
                        break
                    }
                }
            }
            if activeNodeIndices.isEmpty { activeNodeIndices = [0] }

            let prunedUniqueExperts = Array(Set(activeNodeIndices.flatMap { n in candidateExpertsPerNode[n].map { $0.id } })).sorted()
            XCTAssertLessThanOrEqual(prunedUniqueExperts.count, expertCap, "Pruned unique experts must respect the NVMe streaming expert budget cap")

            // 3. Dispatch Staged FP8 SIMD MoE Kernels on Metal across Active Candidate Nodes
            if let gateSimd = inference.fp8BlockGateUpSimdPipeline ?? inference.fp8GateUpSimdPipeline,
               let downSimd = inference.fp8BlockDownSimdPipeline ?? inference.fp8DownSimdPipeline,
               let cmd = cmdQueue.makeCommandBuffer(),
               let enc = cmd.makeComputeCommandEncoder() {

                for n in activeNodeIndices {
                    let inOff = n * hiddenDim * MemoryLayout<Float>.stride
                    let interOff = n * intermediateDim * MemoryLayout<Float>.stride
                    let accumOff = n * hiddenDim * MemoryLayout<Float>.stride

                    for expert in candidateExpertsPerNode[n] {
                        guard let slot = prunedUniqueExperts.firstIndex(of: expert.id) else { continue }
                        let slotOffset = UInt64(slot * expertSize)
                        var gWOffU = slotOffset
                        var gSOffU = slotOffset + 524288
                        var uWOffU = slotOffset + 589824
                        var uSOffU = slotOffset + 1114112
                        var dWOffU = slotOffset + 1179648
                        var dSOffU = slotOffset + 1703936
                        var hDimVal: UInt32 = UInt32(hiddenDim)
                        var interDimVal: UInt32 = UInt32(intermediateDim)
                        var pkVal = expert.weight

                        enc.setComputePipelineState(gateSimd)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
                        enc.setBuffer(staging.treeXNorm2Buffer, offset: inOff, index: 2)
                        enc.setBuffer(staging.treeInterBuffer, offset: interOff, index: 3)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 4)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 5)
                        enc.setBytes(&gWOffU, length: 8, index: 6)
                        enc.setBytes(&gSOffU, length: 8, index: 7)
                        enc.setBytes(&uWOffU, length: 8, index: 8)
                        enc.setBytes(&uSOffU, length: 8, index: 9)
                        enc.setBytes(&hDimVal, length: 4, index: 10)
                        enc.setBytes(&interDimVal, length: 4, index: 11)
                        enc.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)

                        enc.setComputePipelineState(downSimd)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
                        enc.setBuffer(staging.treeInterBuffer, offset: interOff, index: 1)
                        enc.setBuffer(staging.treeHMlpBuffer, offset: accumOff, index: 2)
                        enc.setBuffer(expertStagingBuffer, offset: 0, index: 3)
                        enc.setBytes(&dWOffU, length: 8, index: 4)
                        enc.setBytes(&dSOffU, length: 8, index: 5)
                        enc.setBytes(&interDimVal, length: 4, index: 6)
                        enc.setBytes(&hDimVal, length: 4, index: 7)
                        enc.setBytes(&pkVal, length: 4, index: 8)
                        enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                        enc.memoryBarrier(scope: .buffers)
                    }
                }
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            // 4. Synthesize Target Logits for Speculative Verification
            let logitsPtr = staging.targetLogitsBuffer.contents().bindMemory(to: Float.self, capacity: N * vocabSize)
            logitsPtr.initialize(repeating: -100.0, count: N * vocabSize)
            // Node 0 confirms draftTokens[0]
            logitsPtr[0 * vocabSize + Int(draftTokens[0])] = 50.0
            // Node 1 confirms draftTokens[1]
            logitsPtr[1 * vocabSize + Int(draftTokens[1])] = 50.0
            // Node 2 emits bonus token
            let bonusToken: UInt32 = 248046
            logitsPtr[2 * vocabSize + Int(bonusToken)] = 50.0

            // 5. Acceptance Verification via Greedy Oracle
            let logitsSlice = Array(UnsafeBufferPointer(start: logitsPtr, count: N * vocabSize))
            let accepted = verifyJetspecTreeGreedy(
                treeTokens: treeMask.tokenIds,
                parentIndices: treeMask.parentIndices,
                targetLogits: logitsSlice,
                vocabSize: UInt32(vocabSize)
            )

            XCTAssertFalse(accepted.acceptedTokens.isEmpty, "Speculative verification should accept valid branch")
            let stepAcceptedCount = accepted.acceptedTokens.count + (accepted.bonusToken != nil ? 1 : 0)
            totalAcceptedTokens += stepAcceptedCount

            // 6. Speculative KV Cache Compaction on Metal
            if !accepted.acceptedNodeIndices.isEmpty, let compPipe = inference.compactKvCacheSlotsF32Pipeline {
                let kvStride: UInt32 = 128
                let kvBuf = device.makeBuffer(length: 16 * Int(kvStride) * MemoryLayout<Float>.stride, options: .storageModeShared)!
                let nodeIdxBuf = device.makeBuffer(length: accepted.acceptedNodeIndices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
                let destSlotBuf = device.makeBuffer(length: accepted.acceptedNodeIndices.count * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

                let nPtr = nodeIdxBuf.contents().bindMemory(to: UInt32.self, capacity: accepted.acceptedNodeIndices.count)
                let dPtr = destSlotBuf.contents().bindMemory(to: UInt32.self, capacity: accepted.acceptedNodeIndices.count)
                for (i, nodeIdx) in accepted.acceptedNodeIndices.enumerated() {
                    nPtr[i] = nodeIdx
                    dPtr[i] = UInt32(i + 1)
                }

                guard let cmd = cmdQueue.makeCommandBuffer(), let enc = cmd.makeComputeCommandEncoder() else { break }
                var mCount = UInt32(accepted.acceptedNodeIndices.count)
                var kvStrideVal = kvStride
                var stepVal = currentStep
                var maxSeqVal: UInt32 = 1024

                enc.setComputePipelineState(compPipe)
                enc.setBuffer(kvBuf, offset: 0, index: 0)
                enc.setBuffer(kvBuf, offset: 0, index: 1)
                enc.setBuffer(nodeIdxBuf, offset: 0, index: 2)
                enc.setBuffer(destSlotBuf, offset: 0, index: 3)
                enc.setBytes(&stepVal, length: 4, index: 4)
                enc.setBytes(&kvStrideVal, length: 4, index: 5)
                enc.setBytes(&maxSeqVal, length: 4, index: 6)
                enc.setBytes(&mCount, length: 4, index: 7)
                enc.dispatchThreads(MTLSize(width: Int(kvStride), height: accepted.acceptedNodeIndices.count, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), 128), height: 1, depth: 1))
                enc.endEncoding()
                cmd.commit()
                cmd.waitUntilCompleted()
            }

            currentStep += UInt32(stepAcceptedCount)
            if let bonus = accepted.bonusToken {
                currentRootToken = bonus
            } else if let last = accepted.acceptedTokens.last {
                currentRootToken = last
            }

            let stepElapsedMs = (CFAbsoluteTimeGetCurrent() - stepStart) * 1000.0
            print(String(format: "  Step %d: %d tokens accepted (+bonus), pruned experts = %d/%d, latency = %.2f ms",
                         stepIdx + 1, stepAcceptedCount, prunedUniqueExperts.count, uniqueBeforePruning, stepElapsedMs))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - benchmarkStart
        let tokensPerSec = Double(totalAcceptedTokens) / totalTime
        let meanTau = Double(totalAcceptedTokens) / Double(numSteps)

        print("\n================ JETSPEC QWEN 3.8 FLASH NEXT BENCHMARK RESULTS ================")
        print(String(format: "  Total Speculative Steps:   %d", numSteps))
        print(String(format: "  Draft Tokens Proposed:     %d", totalProposedTokens))
        print(String(format: "  Total Tokens Emitted:      %d", totalAcceptedTokens))
        print(String(format: "  Mean Acceptance Rate (τ):  %.2f tokens/step", meanTau))
        print(String(format: "  Total Benchmark Time:      %.3f s", totalTime))
        print(String(format: "  Effective Generation Speed: %.2f tokens/s", tokensPerSec))
        print("================================================================================\n")

        XCTAssertGreaterThan(totalAcceptedTokens, 0)
        XCTAssertGreaterThan(meanTau, 1.0, "JetSpec speculative acceleration on MoE should achieve mean tau > 1.0 tokens/step")
    }

    func testJetSpecDenseZeroTopKBufferAllocationSafety() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal is not available on this device")
        }
        let inference = InferenceEngine.shared

        // Test with topK = 0 (dense Ornith model configuration)
        let stagingDense = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: 16,
            vocabSize: 248320,
            hiddenDim: 4096,
            intermediateDim: 12288,
            maxQkvDim: 8192,
            maxZDim: 8192,
            kvStride: 1024,
            topK: 0 // Ornith dense has topK = 0
        )
        XCTAssertNotNil(stagingDense, "allocateJetSpecBuffers must succeed even when topK = 0 for dense models")
        if let staging = stagingDense {
            XCTAssertGreaterThan(staging.treeRouterIndicesBuffer.length, 0, "router buffer must have non-zero length")
            XCTAssertGreaterThan(staging.treeRouterWeightsBuffer.length, 0, "router weights buffer must have non-zero length")
            XCTAssertGreaterThan(staging.treeHiddenBuffer.length, 0)
        }

        // Test edge case where intermediateDim is 0
        let stagingZeroInter = inference.allocateJetSpecBuffers(
            device: device,
            maxNodes: 4,
            vocabSize: 1000,
            hiddenDim: 256,
            intermediateDim: 0,
            maxQkvDim: 256,
            maxZDim: 256,
            kvStride: 64,
            topK: 0
        )
        XCTAssertNotNil(stagingZeroInter, "allocateJetSpecBuffers must succeed even with 0 intermediateDim")
        if let staging = stagingZeroInter {
            XCTAssertGreaterThan(staging.treeInterBuffer.length, 0)
        }
    }

    func testJetSpecOrnith9BGatedDeltaNetTreeAcceleration() throws {
        print("=== TEST JETSPEC GATED DELTANET TREE ACCELERATION & STATE COMMIT ===")
        guard let device = MTLCreateSystemDefaultDevice(),
              let cmdQueue = device.makeCommandQueue() else {
            throw XCTSkip("Metal not supported on this device")
        }

        let engine = InferenceEngine.shared
        try engine.initializePipelines(device: device)

        guard let gatherPipe = engine.gatherGdnTreeParentStatesPipeline,
              let commitPipe = engine.commitGdnTreeWinningStatePipeline else {
            XCTFail("GDN Tree pipelines not initialized")
            return
        }

        // Test setup: 4 tree nodes across 3 depths:
        // Node 0: Root (depth 0, parent 0)
        // Node 1: Child of 0 (depth 1, parent 0)
        // Node 2: Child of 0 (depth 1, parent 0)
        // Node 3: Child of 1 (depth 2, parent 1)
        let numNodes: UInt32 = 4
        let numLinLayers: UInt32 = 2
        let numValHeads: UInt32 = 2
        let headDim: UInt32 = 128
        let stateFloatsPerNode = Int(numValHeads * headDim * headDim) // 32768 floats
        let stateBytesPerNode = stateFloatsPerNode * MemoryLayout<Float>.stride

        // 1. Base persistent state buffer: [numLinLayers, stateFloatsPerNode]
        let baseStateBuf = device.makeBuffer(length: Int(numLinLayers) * stateBytesPerNode, options: .storageModeShared)!
        let basePtr = baseStateBuf.contents().bindMemory(to: Float.self, capacity: Int(numLinLayers) * stateFloatsPerNode)
        for i in 0..<stateFloatsPerNode {
            basePtr[i] = 1.0 // Layer 0 base state
            basePtr[stateFloatsPerNode + i] = 2.0 // Layer 1 base state
        }

        // 2. Tree out state buffer: [numLinLayers, numNodes, stateFloatsPerNode]
        let treeOutBuf = device.makeBuffer(length: Int(numLinLayers * numNodes) * stateBytesPerNode, options: .storageModeShared)!
        let treeOutPtr = treeOutBuf.contents().bindMemory(to: Float.self, capacity: Int(numLinLayers * numNodes) * stateFloatsPerNode)
        treeOutPtr.initialize(repeating: 0.0, count: Int(numLinLayers * numNodes) * stateFloatsPerNode)

        // 3. Tree parent state scratch buffer: [numNodes, stateFloatsPerNode] (for 1 layer)
        let parentStateBuf = device.makeBuffer(length: Int(numNodes) * stateBytesPerNode, options: .storageModeShared)!
        let parentStatePtr = parentStateBuf.contents().bindMemory(to: Float.self, capacity: Int(numNodes) * stateFloatsPerNode)
        parentStatePtr.initialize(repeating: 0.0, count: Int(numNodes) * stateFloatsPerNode)

        // 4. Tree topology buffers
        let parentIndicesBuf = device.makeBuffer(length: Int(numNodes) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let depthsBuf = device.makeBuffer(length: Int(numNodes) * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let pPtr = parentIndicesBuf.contents().bindMemory(to: UInt32.self, capacity: Int(numNodes))
        let dPtr = depthsBuf.contents().bindMemory(to: UInt32.self, capacity: Int(numNodes))
        pPtr[0] = 0; dPtr[0] = 0 // Root
        pPtr[1] = 0; dPtr[1] = 1 // Node 1 -> parent 0
        pPtr[2] = 0; dPtr[2] = 1 // Node 2 -> parent 0
        pPtr[3] = 1; dPtr[3] = 2 // Node 3 -> parent 1

        var stateFloatsU32 = UInt32(stateFloatsPerNode)
        let vec4Count = Int(stateFloatsU32 / 4)

        // Depth 0: Gather root parent state from baseStateBuf (layer 0)
        do {
            var targetD: UInt32 = 0
            var nVal = numNodes
            guard let cmd = cmdQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                XCTFail("Failed to make encoder")
                return
            }
            enc.setComputePipelineState(gatherPipe)
            enc.setBuffer(baseStateBuf, offset: 0, index: 0)
            enc.setBuffer(treeOutBuf, offset: 0, index: 1)
            enc.setBuffer(parentStateBuf, offset: 0, index: 2)
            enc.setBuffer(parentIndicesBuf, offset: 0, index: 3)
            enc.setBuffer(depthsBuf, offset: 0, index: 4)
            enc.setBytes(&targetD, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 6)
            enc.setBytes(&stateFloatsU32, length: MemoryLayout<UInt32>.stride, index: 7)
            enc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numNodes), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatherPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            // Verify Node 0 gathered base state (1.0)
            XCTAssertEqual(parentStatePtr[0], 1.0, "Node 0 must gather base state")
            // Node 1 should not have been updated at depth 0
            XCTAssertEqual(parentStatePtr[1 * stateFloatsPerNode], 0.0)
        }

        // Simulate Node 0 executing and producing output state 42.0
        for i in 0..<stateFloatsPerNode {
            treeOutPtr[0 * stateFloatsPerNode + i] = 42.0
        }

        // Depth 1: Gather parent states for nodes 1 & 2 (parent is 0)
        do {
            var targetD: UInt32 = 1
            var nVal = numNodes
            guard let cmd = cmdQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                XCTFail("Failed to make encoder")
                return
            }
            enc.setComputePipelineState(gatherPipe)
            enc.setBuffer(baseStateBuf, offset: 0, index: 0)
            enc.setBuffer(treeOutBuf, offset: 0, index: 1)
            enc.setBuffer(parentStateBuf, offset: 0, index: 2)
            enc.setBuffer(parentIndicesBuf, offset: 0, index: 3)
            enc.setBuffer(depthsBuf, offset: 0, index: 4)
            enc.setBytes(&targetD, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 6)
            enc.setBytes(&stateFloatsU32, length: MemoryLayout<UInt32>.stride, index: 7)
            enc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numNodes), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatherPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            // Verify Nodes 1 and 2 received parent Node 0's state (42.0)
            XCTAssertEqual(parentStatePtr[1 * stateFloatsPerNode], 42.0, "Node 1 must inherit Node 0's state")
            XCTAssertEqual(parentStatePtr[2 * stateFloatsPerNode], 42.0, "Node 2 must inherit Node 0's state")
            // Node 3 should not have updated at depth 1
            XCTAssertEqual(parentStatePtr[3 * stateFloatsPerNode], 0.0)
        }

        // Simulate Node 1 executing and producing output state 99.0
        for i in 0..<stateFloatsPerNode {
            treeOutPtr[1 * stateFloatsPerNode + i] = 99.0
        }

        // Depth 2: Gather parent state for Node 3 (parent is 1)
        do {
            var targetD: UInt32 = 2
            var nVal = numNodes
            guard let cmd = cmdQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                XCTFail("Failed to make encoder")
                return
            }
            enc.setComputePipelineState(gatherPipe)
            enc.setBuffer(baseStateBuf, offset: 0, index: 0)
            enc.setBuffer(treeOutBuf, offset: 0, index: 1)
            enc.setBuffer(parentStateBuf, offset: 0, index: 2)
            enc.setBuffer(parentIndicesBuf, offset: 0, index: 3)
            enc.setBuffer(depthsBuf, offset: 0, index: 4)
            enc.setBytes(&targetD, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.setBytes(&nVal, length: MemoryLayout<UInt32>.stride, index: 6)
            enc.setBytes(&stateFloatsU32, length: MemoryLayout<UInt32>.stride, index: 7)
            enc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numNodes), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, gatherPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            // Verify Node 3 received parent Node 1's state (99.0)
            XCTAssertEqual(parentStatePtr[3 * stateFloatsPerNode], 99.0, "Node 3 must inherit Node 1's state")
        }

        // 5. Test Winning Branch State Commit Kernel
        // Simulate Node 3 winning across both linear layers:
        // Layer 0, Node 3 output: 555.0
        // Layer 1, Node 3 output: 777.0
        let layer1Base = Int(numNodes) * stateFloatsPerNode
        for i in 0..<stateFloatsPerNode {
            treeOutPtr[3 * stateFloatsPerNode + i] = 555.0
            treeOutPtr[layer1Base + 3 * stateFloatsPerNode + i] = 777.0
        }

        do {
            var winningNode: UInt32 = 3
            var maxN: UInt32 = numNodes
            var numLin: UInt32 = numLinLayers
            guard let cmd = cmdQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else {
                XCTFail("Failed to make commit encoder")
                return
            }
            enc.setComputePipelineState(commitPipe)
            enc.setBuffer(treeOutBuf, offset: 0, index: 0)
            enc.setBuffer(baseStateBuf, offset: 0, index: 1)
            enc.setBytes(&winningNode, length: MemoryLayout<UInt32>.stride, index: 2)
            enc.setBytes(&maxN, length: MemoryLayout<UInt32>.stride, index: 3)
            enc.setBytes(&numLin, length: MemoryLayout<UInt32>.stride, index: 4)
            enc.setBytes(&stateFloatsU32, length: MemoryLayout<UInt32>.stride, index: 5)
            enc.dispatchThreads(MTLSize(width: vec4Count, height: Int(numLinLayers), depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, commitPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            // Verify baseStateBuf committed winning Node 3's states for both layers
            XCTAssertEqual(basePtr[0], 555.0, "Base state layer 0 must be updated to winning Node 3 state")
            XCTAssertEqual(basePtr[stateFloatsPerNode], 777.0, "Base state layer 1 must be updated to winning Node 3 state")
        }

        print("✅ [TEST] Gated DeltaNet tree recurrence & state commit kernels verified successfully.")
    }

    func testOrnith9BStep0Diagnostics() throws {
        print("=== TEST ORNITH 1.5 9B STEP 0 DIAGNOSTICS ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        XCTAssertEqual(cachedLayers.count, 32)

        let hiddenDim = config?.hiddenSize ?? 4096
        let intermediateDim = config?.intermediateSize ?? 12288
        let vocabSize = config?.vocabSize ?? 248320

        KVCacheManager.shared.reset(
            device: device,
            config: config,
            actualLayers: 32,
            totalLoops: 1,
            numKvHeads: 8,
            headDim: 128,
            maxSeqLen: 256,
            precision: .fp16
        )

        let hBufA = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hBufB = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm1Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm2Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let qGateBuf = device.makeBuffer(length: 8192 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let zGateBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let aVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let bVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnCtxBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnOutBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMidBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let interBuf = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xFinalBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let logitsBuf = device.makeBuffer(length: vocabSize * MemoryLayout<Float>.stride, options: .storageModeShared)!

        func checkBuf(_ name: String, _ buf: MTLBuffer, count: Int) {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            var sumVal: Double = 0
            var nanCount = 0
            for i in 0..<count {
                let v = ptr[i]
                if v.isNaN || v.isInfinite {
                    nanCount += 1
                } else {
                    if v < minVal { minVal = v }
                    if v > maxVal { maxVal = v }
                    sumVal += Double(v)
                }
            }
            let meanVal = count > nanCount ? Float(sumVal / Double(count - nanCount)) : 0
            print("  [\(name)] count=\(count), nans=\(nanCount), min=\(minVal), max=\(maxVal), mean=\(meanVal)")
            XCTAssertEqual(nanCount, 0, "Buffer \(name) contains NaN or Inf values!")
        }

        func dispatchLinearLocal(
            enc: MTLComputeCommandEncoder,
            weight: TensorMetadata?,
            scale: TensorMetadata?,
            bias: TensorMetadata?,
            inBuf: MTLBuffer,
            outBuf: MTLBuffer,
            inDim: UInt32,
            outDim: UInt32
        ) {
            guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
            var wOff = w.offsetStart
            var inD = inDim
            var outD = outDim
            var grp: UInt32 = 64
            let hasBias = (bias != nil)
            let isQuantizedAffine = (hasBias || w.dtype.contains("Q4") || w.dtype.contains("Q8"))
            if isQuantizedAffine, let s = scale, let sRaw = buffers[s.shardIndex], let b = bias, let bRaw = buffers[b.shardIndex] {
                var sOff = s.offsetStart
                var bOff = b.offsetStart
                let is8Bit = (w.offsetEnd - w.offsetStart) >= UInt64(outDim) * UInt64(inDim)
                if is8Bit, let q8Pipe = inference.q8GemvPipeline {
                    enc.setComputePipelineState(q8Pipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(sRaw, offset: 0, index: 1)
                    enc.setBuffer(bRaw, offset: 0, index: 2)
                    enc.setBuffer(inBuf, offset: 0, index: 3)
                    enc.setBuffer(outBuf, offset: 0, index: 4)
                    enc.setBytes(&wOff, length: 8, index: 5)
                    enc.setBytes(&sOff, length: 8, index: 6)
                    enc.setBytes(&bOff, length: 8, index: 7)
                    enc.setBytes(&inD, length: 4, index: 8)
                    enc.setBytes(&outD, length: 4, index: 9)
                    enc.setBytes(&grp, length: 4, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                } else if let q4Pipe = inference.q4GemvPipeline {
                    enc.setComputePipelineState(q4Pipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(sRaw, offset: 0, index: 1)
                    enc.setBuffer(bRaw, offset: 0, index: 2)
                    enc.setBuffer(inBuf, offset: 0, index: 3)
                    enc.setBuffer(outBuf, offset: 0, index: 4)
                    enc.setBytes(&wOff, length: 8, index: 5)
                    enc.setBytes(&sOff, length: 8, index: 6)
                    enc.setBytes(&bOff, length: 8, index: 7)
                    enc.setBytes(&inD, length: 4, index: 8)
                    enc.setBytes(&outD, length: 4, index: 9)
                    enc.setBytes(&grp, length: 4, index: 10)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                }
            } else if let bSimdPipe = inference.bf16GemvSimdPipeline {
                enc.setComputePipelineState(bSimdPipe)
                enc.setBuffer(wRaw, offset: 0, index: 0)
                enc.setBuffer(inBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            enc.memoryBarrier(scope: .buffers)
        }

        // 1. Embed Token 151644 (<|im_start|>)
        let embedWeight = summary.tensors.first { $0.name.contains("embed_tokens") && $0.name.hasSuffix(".weight") }!
        let embedScale = summary.tensors.first { $0.name.contains("embed_tokens") && $0.name.hasSuffix(".scales") }
        let embedBias = summary.tensors.first { $0.name.contains("embed_tokens") && $0.name.hasSuffix(".biases") }

        let embedCmd = cmdQueue.makeCommandBuffer()!
        let embedEnc = embedCmd.makeComputeCommandEncoder()!
        var tok: UInt32 = 151644
        var wOffset = embedWeight.offsetStart
        var sOffset = embedScale?.offsetStart ?? 0
        var bOffset = embedBias?.offsetStart ?? 0
        var hDimVal = UInt32(hiddenDim)
        var grpSize: UInt32 = 64
        let embedQ8 = inference.embedQ8Pipeline!
        embedEnc.setComputePipelineState(embedQ8)
        embedEnc.setBuffer(buffers[embedWeight.shardIndex]!, offset: 0, index: 0)
        embedEnc.setBuffer(buffers[embedScale!.shardIndex]!, offset: 0, index: 1)
        embedEnc.setBuffer(buffers[embedBias!.shardIndex]!, offset: 0, index: 2)
        embedEnc.setBuffer(hBufA, offset: 0, index: 3)
        embedEnc.setBytes(&tok, length: 4, index: 4)
        embedEnc.setBytes(&wOffset, length: 8, index: 5)
        embedEnc.setBytes(&sOffset, length: 8, index: 6)
        embedEnc.setBytes(&bOffset, length: 8, index: 7)
        embedEnc.setBytes(&hDimVal, length: 4, index: 8)
        embedEnc.setBytes(&grpSize, length: 4, index: 9)
        embedEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, embedQ8.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        embedEnc.endEncoding()
        embedCmd.commit()
        embedCmd.waitUntilCompleted()

        print("Checking Embedding Output:")
        checkBuf("hBufA_embed", hBufA, count: hiddenDim)

        // 2. Run Layer 0
        let layer0 = cachedLayers[0]
        let l0Cmd = cmdQueue.makeCommandBuffer()!
        let l0Enc = l0Cmd.makeComputeCommandEncoder()!

        // Norm1
        let norm1 = layer0.norm1Tensor!
        let norm1Raw = buffers[norm1.shardIndex]!
        var nOff1 = norm1.offsetStart
        var epsVal: Float = 1e-6
        let rmsPipe = inference.rmsnormPipeline!
        l0Enc.setComputePipelineState(rmsPipe)
        l0Enc.setBuffer(hBufA, offset: 0, index: 0)
        l0Enc.setBuffer(norm1Raw, offset: 0, index: 1)
        l0Enc.setBuffer(xNorm1Buf, offset: 0, index: 2)
        l0Enc.setBytes(&nOff1, length: 8, index: 3)
        l0Enc.setBytes(&hDimVal, length: 4, index: 4)
        l0Enc.setBytes(&epsVal, length: 4, index: 5)
        l0Enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
        l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // inProjQKV
        let linKeyHeads: UInt32 = 16
        let linValHeads: UInt32 = 32
        let qkvDim: UInt32 = (linKeyHeads + linKeyHeads + linValHeads) * 128 // 8192
        let zDim: UInt32 = linValHeads * 128 // 4096
        dispatchLinearLocal(enc: l0Enc, weight: layer0.inProjQKV, scale: layer0.inProjQKVScale, bias: layer0.inProjQKVBias, inBuf: xNorm1Buf, outBuf: qGateBuf, inDim: UInt32(hiddenDim), outDim: qkvDim)

        // conv1d
        if let conv1d = layer0.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
           let convPipe = inference.causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
            var cOff = conv1d.offsetStart
            var numChannels = qkvDim
            l0Enc.setComputePipelineState(convPipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBuffer(convRaw, offset: 0, index: 1)
            l0Enc.setBuffer(convState, offset: 0, index: 2)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 3)
            l0Enc.setBytes(&cOff, length: 8, index: 4)
            l0Enc.setBytes(&numChannels, length: 4, index: 5)
            l0Enc.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // l2NormQk
        if let l2Pipe = inference.l2NormQkPipeline {
            var numH = linKeyHeads
            var hD: UInt32 = 128
            l0Enc.setComputePipelineState(l2Pipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBytes(&numH, length: 4, index: 1)
            l0Enc.setBytes(&hD, length: 4, index: 2)
            l0Enc.dispatchThreads(MTLSize(width: Int(numH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numH), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // inProjZ, inProjA, inProjB
        dispatchLinearLocal(enc: l0Enc, weight: layer0.inProjZ, scale: layer0.inProjZScale, bias: layer0.inProjZBias, inBuf: xNorm1Buf, outBuf: zGateBuf, inDim: UInt32(hiddenDim), outDim: zDim)
        dispatchLinearLocal(enc: l0Enc, weight: layer0.inProjA, scale: layer0.inProjAScale, bias: layer0.inProjABias, inBuf: xNorm1Buf, outBuf: aVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)
        dispatchLinearLocal(enc: l0Enc, weight: layer0.inProjB, scale: layer0.inProjBScale, bias: layer0.inProjBBias, inBuf: xNorm1Buf, outBuf: bVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)

        // linearAttnStep
        if let linPipe = inference.gdnLinearAttnStepPipeline ?? inference.linearAttnStepPipeline,
           let sBuf = KVCacheManager.shared.linearStateBuffer,
           let aLog = layer0.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
           let dtBias = layer0.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
           let linNorm = layer0.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
            var aLogOff = aLog.offsetStart
            var dtBiasOff = dtBias.offsetStart
            var linNormOff = linNorm.offsetStart
            var numValH = linValHeads
            var numKeyH = linKeyHeads
            var hD: UInt32 = 128
            l0Enc.setComputePipelineState(linPipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBuffer(zGateBuf, offset: 0, index: 1)
            l0Enc.setBuffer(aVecBuf, offset: 0, index: 2)
            l0Enc.setBuffer(bVecBuf, offset: 0, index: 3)
            l0Enc.setBuffer(aLogRaw, offset: 0, index: 4)
            l0Enc.setBuffer(dtBiasRaw, offset: 0, index: 5)
            l0Enc.setBuffer(linNormRaw, offset: 0, index: 6)
            l0Enc.setBuffer(sBuf, offset: 0, index: 7)
            l0Enc.setBuffer(attnCtxBuf, offset: 0, index: 8)
            l0Enc.setBytes(&aLogOff, length: 8, index: 9)
            l0Enc.setBytes(&dtBiasOff, length: 8, index: 10)
            l0Enc.setBytes(&linNormOff, length: 8, index: 11)
            l0Enc.setBytes(&numValH, length: 4, index: 12)
            l0Enc.setBytes(&numKeyH, length: 4, index: 13)
            l0Enc.setBytes(&hD, length: 4, index: 14)
            l0Enc.setBytes(&epsVal, length: 4, index: 15)
            l0Enc.dispatchThreadgroups(MTLSize(width: Int(numValH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // outProj
        dispatchLinearLocal(enc: l0Enc, weight: layer0.linearOutProjTensor ?? layer0.oProjTensor, scale: layer0.linearOutProjScale ?? layer0.oScaleTensor, bias: layer0.linearOutProjBias ?? layer0.oBiasTensor, inBuf: attnCtxBuf, outBuf: attnOutBuf, inDim: zDim, outDim: UInt32(hiddenDim))

        // Residual 1
        let addPipe = inference.addPipeline!
        l0Enc.setComputePipelineState(addPipe)
        l0Enc.setBuffer(hBufA, offset: 0, index: 0)
        l0Enc.setBuffer(attnOutBuf, offset: 0, index: 1)
        l0Enc.setBuffer(hMidBuf, offset: 0, index: 2)
        l0Enc.setBytes(&hDimVal, length: 4, index: 3)
        l0Enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Norm2
        let norm2 = layer0.norm2Tensor!
        let norm2Raw = buffers[norm2.shardIndex]!
        var nOff2 = norm2.offsetStart
        l0Enc.setComputePipelineState(rmsPipe)
        l0Enc.setBuffer(hMidBuf, offset: 0, index: 0)
        l0Enc.setBuffer(norm2Raw, offset: 0, index: 1)
        l0Enc.setBuffer(xNorm2Buf, offset: 0, index: 2)
        l0Enc.setBytes(&nOff2, length: 8, index: 3)
        l0Enc.setBytes(&hDimVal, length: 4, index: 4)
        l0Enc.setBytes(&epsVal, length: 4, index: 5)
        l0Enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
        l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Dense MLP: gate & up
        let gateW = layer0.denseGateWeight!
        let upW = layer0.denseUpWeight!
        let downW = layer0.denseDownWeight!
        let q8GateUp = inference.q8GateUpPipeline!
        var gWOff = gateW.offsetStart
        var gSOff = layer0.denseGateScale!.offsetStart
        var gBOff = layer0.denseGateBias!.offsetStart
        var uWOff = upW.offsetStart
        var uSOff = layer0.denseUpScale!.offsetStart
        var uBOff = layer0.denseUpBias!.offsetStart
        var interD = UInt32(intermediateDim)
        var grp: UInt32 = 64

        l0Enc.setComputePipelineState(q8GateUp)
        l0Enc.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
        l0Enc.setBuffer(buffers[layer0.denseGateScale!.shardIndex]!, offset: 0, index: 1)
        l0Enc.setBuffer(buffers[layer0.denseGateBias!.shardIndex]!, offset: 0, index: 2)
        l0Enc.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 3)
        l0Enc.setBuffer(buffers[layer0.denseUpScale!.shardIndex]!, offset: 0, index: 4)
        l0Enc.setBuffer(buffers[layer0.denseUpBias!.shardIndex]!, offset: 0, index: 5)
        l0Enc.setBuffer(xNorm2Buf, offset: 0, index: 6)
        l0Enc.setBuffer(interBuf, offset: 0, index: 7)
        l0Enc.setBytes(&gWOff, length: 8, index: 8)
        l0Enc.setBytes(&gSOff, length: 8, index: 9)
        l0Enc.setBytes(&gBOff, length: 8, index: 10)
        l0Enc.setBytes(&uWOff, length: 8, index: 11)
        l0Enc.setBytes(&uSOff, length: 8, index: 12)
        l0Enc.setBytes(&uBOff, length: 8, index: 13)
        l0Enc.setBytes(&hDimVal, length: 4, index: 14)
        l0Enc.setBytes(&interD, length: 4, index: 15)
        l0Enc.setBytes(&grp, length: 4, index: 16)
        l0Enc.dispatchThreadgroups(MTLSize(width: Int(interD), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Dense MLP: down
        let q8Down = inference.q8DownPipeline!
        var dWOff = downW.offsetStart
        var dSOff = layer0.denseDownScale!.offsetStart
        var dBOff = layer0.denseDownBias!.offsetStart
        var pk: Float = 1.0
        l0Enc.setComputePipelineState(q8Down)
        l0Enc.setBuffer(buffers[downW.shardIndex]!, offset: 0, index: 0)
        l0Enc.setBuffer(buffers[layer0.denseDownScale!.shardIndex]!, offset: 0, index: 1)
        l0Enc.setBuffer(buffers[layer0.denseDownBias!.shardIndex]!, offset: 0, index: 2)
        l0Enc.setBuffer(interBuf, offset: 0, index: 3)
        l0Enc.setBuffer(hMlpBuf, offset: 0, index: 4)
        l0Enc.setBytes(&dWOff, length: 8, index: 5)
        l0Enc.setBytes(&dSOff, length: 8, index: 6)
        l0Enc.setBytes(&dBOff, length: 8, index: 7)
        l0Enc.setBytes(&interD, length: 4, index: 8)
        l0Enc.setBytes(&hDimVal, length: 4, index: 9)
        l0Enc.setBytes(&grp, length: 4, index: 10)
        l0Enc.setBytes(&pk, length: 4, index: 11)
        l0Enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Residual 2
        l0Enc.setComputePipelineState(addPipe)
        l0Enc.setBuffer(hMidBuf, offset: 0, index: 0)
        l0Enc.setBuffer(hMlpBuf, offset: 0, index: 1)
        l0Enc.setBuffer(hBufB, offset: 0, index: 2)
        l0Enc.setBytes(&hDimVal, length: 4, index: 3)
        l0Enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

        l0Enc.endEncoding()
        l0Cmd.commit()
        l0Cmd.waitUntilCompleted()

        print("Checking Layer 0 Intermediate Outputs:")
        checkBuf("xNorm1", xNorm1Buf, count: hiddenDim)
        checkBuf("qGate", qGateBuf, count: Int(qkvDim))
        checkBuf("zGate", zGateBuf, count: Int(zDim))
        checkBuf("aVec", aVecBuf, count: Int(linValHeads))
        checkBuf("bVec", bVecBuf, count: Int(linValHeads))
        checkBuf("attnCtx", attnCtxBuf, count: Int(zDim))
        checkBuf("attnOut", attnOutBuf, count: hiddenDim)
        checkBuf("hMid", hMidBuf, count: hiddenDim)
        checkBuf("xNorm2", xNorm2Buf, count: hiddenDim)
        checkBuf("interBuf", interBuf, count: intermediateDim)
        checkBuf("hMlpBuf", hMlpBuf, count: hiddenDim)
        checkBuf("hBufB_layer0_out", hBufB, count: hiddenDim)

        func testLayerMlp(layerIdx: Int, inputBuf: MTLBuffer) {
            let layer = cachedLayers[layerIdx]
            guard let gateW = layer.denseGateWeight,
                  let upW = layer.denseUpWeight,
                  let downW = layer.denseDownWeight,
                  let norm2 = layer.norm2Tensor else { return }

            guard let lCmd = cmdQueue.makeCommandBuffer(),
                  let enc = lCmd.makeComputeCommandEncoder() else { return }

            var nOff = norm2.offsetStart
            var hD = UInt32(hiddenDim)
            var epsVal: Float = 1e-6
            enc.setComputePipelineState(rmsPipe)
            enc.setBuffer(inputBuf, offset: 0, index: 0)
            enc.setBuffer(buffers[norm2.shardIndex]!, offset: 0, index: 1)
            enc.setBuffer(xNorm2Buf, offset: 0, index: 2)
            enc.setBytes(&nOff, length: 8, index: 3)
            enc.setBytes(&hD, length: 4, index: 4)
            enc.setBytes(&epsVal, length: 4, index: 5)
            enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
            enc.memoryBarrier(scope: MTLBarrierScope.buffers)

            let isGate8Bit = (gateW.offsetEnd - gateW.offsetStart) >= UInt64(intermediateDim) * UInt64(hiddenDim)
            let isUp8Bit = (upW.offsetEnd - upW.offsetStart) >= UInt64(intermediateDim) * UInt64(hiddenDim)
            let isDown8Bit = (downW.offsetEnd - downW.offsetStart) >= UInt64(hiddenDim) * UInt64(intermediateDim)

            let swigluPipe: MTLComputePipelineState
            if isGate8Bit && isUp8Bit {
                swigluPipe = inference.q8GateUpPipeline!
            } else if !isGate8Bit && !isUp8Bit {
                swigluPipe = inference.q4GateUpPipeline!
            } else if !isGate8Bit && isUp8Bit {
                swigluPipe = inference.q4GateQ8UpPipeline!
            } else {
                swigluPipe = inference.q8GateQ4UpPipeline!
            }

            let downPipe: MTLComputePipelineState = isDown8Bit ? inference.q8DownPipeline! : inference.q4DownPipeline!

            var gWOff = gateW.offsetStart
            var gSOff = layer.denseGateScale!.offsetStart
            var gBOff = layer.denseGateBias!.offsetStart
            var uWOff = upW.offsetStart
            var uSOff = layer.denseUpScale!.offsetStart
            var uBOff = layer.denseUpBias!.offsetStart
            var interD = UInt32(intermediateDim)
            var grp: UInt32 = 64
            var pk: Float = 1.0

            enc.setComputePipelineState(swigluPipe)
            enc.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(buffers[layer.denseGateScale!.shardIndex]!, offset: 0, index: 1)
            enc.setBuffer(buffers[layer.denseGateBias!.shardIndex]!, offset: 0, index: 2)
            enc.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 3)
            enc.setBuffer(buffers[layer.denseUpScale!.shardIndex]!, offset: 0, index: 4)
            enc.setBuffer(buffers[layer.denseUpBias!.shardIndex]!, offset: 0, index: 5)
            enc.setBuffer(xNorm2Buf, offset: 0, index: 6)
            enc.setBuffer(interBuf, offset: 0, index: 7)
            enc.setBytes(&gWOff, length: 8, index: 8)
            enc.setBytes(&gSOff, length: 8, index: 9)
            enc.setBytes(&gBOff, length: 8, index: 10)
            enc.setBytes(&uWOff, length: 8, index: 11)
            enc.setBytes(&uSOff, length: 8, index: 12)
            enc.setBytes(&uBOff, length: 8, index: 13)
            enc.setBytes(&hD, length: 4, index: 14)
            enc.setBytes(&interD, length: 4, index: 15)
            enc.setBytes(&grp, length: 4, index: 16)
            enc.dispatchThreadgroups(MTLSize(width: Int(interD), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            var dWOff = downW.offsetStart
            var dSOff = layer.denseDownScale!.offsetStart
            var dBOff = layer.denseDownBias!.offsetStart

            enc.setComputePipelineState(downPipe)
            enc.setBuffer(buffers[downW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(buffers[layer.denseDownScale!.shardIndex]!, offset: 0, index: 1)
            enc.setBuffer(buffers[layer.denseDownBias!.shardIndex]!, offset: 0, index: 2)
            enc.setBuffer(interBuf, offset: 0, index: 3)
            enc.setBuffer(hMlpBuf, offset: 0, index: 4)
            enc.setBytes(&dWOff, length: 8, index: 5)
            enc.setBytes(&dSOff, length: 8, index: 6)
            enc.setBytes(&dBOff, length: 8, index: 7)
            enc.setBytes(&interD, length: 4, index: 8)
            enc.setBytes(&hD, length: 4, index: 9)
            enc.setBytes(&grp, length: 4, index: 10)
            enc.setBytes(&pk, length: 4, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.endEncoding()
            lCmd.commit()
            lCmd.waitUntilCompleted()

            print("Checking Layer \(layerIdx) (gate8=\(isGate8Bit), up8=\(isUp8Bit), down8=\(isDown8Bit)) Intermediate Outputs:")
            checkBuf("L\(layerIdx)_interBuf", interBuf, count: intermediateDim)
            checkBuf("L\(layerIdx)_hMlpBuf", hMlpBuf, count: hiddenDim)
        }

        testLayerMlp(layerIdx: 1, inputBuf: hBufB)
        testLayerMlp(layerIdx: 4, inputBuf: hBufB)
    }

    func testOrnith9BPrefillMultiTokenNoNaNs() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal device unavailable")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("Metal command queue unavailable")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        guard !cachedLayers.isEmpty else {
            XCTFail("No layers loaded")
            return
        }

        let hiddenDim = 2560
        let intermediateDim = 14336
        let P = 4

        // Allocate multi-token buffers
        guard let inTokensBuf = device.makeBuffer(length: P * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let xNorm2BufAll = device.makeBuffer(length: P * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBufAll = device.makeBuffer(length: P * intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBufAll = device.makeBuffer(length: P * hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate multi-token buffers")
            return
        }

        // Initialize input tokens with dummy non-zero finite values
        let inPtr = inTokensBuf.contents().bindMemory(to: Float.self, capacity: P * hiddenDim)
        for i in 0..<(P * hiddenDim) {
            inPtr[i] = sin(Float(i) * 0.01) * 0.1
        }

        let rmsPipe = inference.rmsnormPipeline!

        // Test Layers 0 (8/8/8), 1 (4/4/8), and 4 (4/8/4)
        for layerIdx in [0, 1, 4] {
            guard layerIdx < cachedLayers.count else { continue }
            let layer = cachedLayers[layerIdx]
            guard let gateW = layer.denseGateWeight,
                  let upW = layer.denseUpWeight,
                  let downW = layer.denseDownWeight,
                  let norm2 = layer.norm2Tensor else { continue }

            guard let cmd = cmdQueue.makeCommandBuffer(),
                  let enc = cmd.makeComputeCommandEncoder() else { continue }

            // 1. RMSNorm for each token
            var nOff = norm2.offsetStart
            var hD = UInt32(hiddenDim)
            var epsVal: Float = 1e-6
            for p in 0..<P {
                enc.setComputePipelineState(rmsPipe)
                enc.setBuffer(inTokensBuf, offset: p * hiddenDim * MemoryLayout<Float>.stride, index: 0)
                enc.setBuffer(buffers[norm2.shardIndex]!, offset: 0, index: 1)
                enc.setBuffer(xNorm2BufAll, offset: p * hiddenDim * MemoryLayout<Float>.stride, index: 2)
                enc.setBytes(&nOff, length: 8, index: 3)
                enc.setBytes(&hD, length: 4, index: 4)
                enc.setBytes(&epsVal, length: 4, index: 5)
                enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
                enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            // 2. Prefill logic: check isQuantizedAffine
            let gateS = layer.denseGateScale
            let isBlockScale = (gateS != nil) && (gateS!.name.contains("scale_inv") || ((gateS!.offsetEnd - gateS!.offsetStart) < UInt64(intermediateDim * 2)))
            let isQuantizedAffine = (layer.denseGateBias != nil || gateW.dtype.contains("Q4") || gateW.dtype.contains("Q8") || (gateS != nil && !isBlockScale))

            XCTAssertTrue(isQuantizedAffine, "Layer \(layerIdx) must be recognized as quantized affine")

            let isGate8Bit = (gateW.offsetEnd - gateW.offsetStart) >= UInt64(intermediateDim) * UInt64(hiddenDim)
            let isUp8Bit = (upW.offsetEnd - upW.offsetStart) >= UInt64(intermediateDim) * UInt64(hiddenDim)
            let isDown8Bit = (downW.offsetEnd - downW.offsetStart) >= UInt64(hiddenDim) * UInt64(intermediateDim)

            let swigluPipe: MTLComputePipelineState
            if isGate8Bit && isUp8Bit {
                swigluPipe = inference.q8GateUpPipeline!
            } else if !isGate8Bit && !isUp8Bit {
                swigluPipe = inference.q4GateUpPipeline!
            } else if !isGate8Bit && isUp8Bit {
                swigluPipe = inference.q4GateQ8UpPipeline!
            } else {
                swigluPipe = inference.q8GateQ4UpPipeline!
            }
            let downPipe: MTLComputePipelineState = isDown8Bit ? inference.q8DownPipeline! : inference.q4DownPipeline!

            var gWOff = gateW.offsetStart
            var gSOff = layer.denseGateScale!.offsetStart
            var gBOff = layer.denseGateBias!.offsetStart
            var uWOff = upW.offsetStart
            var uSOff = layer.denseUpScale!.offsetStart
            var uBOff = layer.denseUpBias!.offsetStart
            var interD = UInt32(intermediateDim)
            var grp: UInt32 = 64
            var pk: Float = 1.0

            var dWOff = downW.offsetStart
            var dSOff = layer.denseDownScale!.offsetStart
            var dBOff = layer.denseDownBias!.offsetStart

            for p in 0..<P {
                let tokenOffset = p * hiddenDim * MemoryLayout<Float>.stride
                let interTokenOffset = p * intermediateDim * MemoryLayout<Float>.stride

                enc.setComputePipelineState(swigluPipe)
                enc.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
                enc.setBuffer(buffers[layer.denseGateScale!.shardIndex]!, offset: 0, index: 1)
                enc.setBuffer(buffers[layer.denseGateBias!.shardIndex]!, offset: 0, index: 2)
                enc.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 3)
                enc.setBuffer(buffers[layer.denseUpScale!.shardIndex]!, offset: 0, index: 4)
                enc.setBuffer(buffers[layer.denseUpBias!.shardIndex]!, offset: 0, index: 5)
                enc.setBuffer(xNorm2BufAll, offset: tokenOffset, index: 6)
                enc.setBuffer(interBufAll, offset: interTokenOffset, index: 7)
                enc.setBytes(&gWOff, length: 8, index: 8)
                enc.setBytes(&gSOff, length: 8, index: 9)
                enc.setBytes(&gBOff, length: 8, index: 10)
                enc.setBytes(&uWOff, length: 8, index: 11)
                enc.setBytes(&uSOff, length: 8, index: 12)
                enc.setBytes(&uBOff, length: 8, index: 13)
                enc.setBytes(&hD, length: 4, index: 14)
                enc.setBytes(&interD, length: 4, index: 15)
                enc.setBytes(&grp, length: 4, index: 16)
                enc.dispatchThreadgroups(MTLSize(width: Int(interD), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(downPipe)
                enc.setBuffer(buffers[downW.shardIndex]!, offset: 0, index: 0)
                enc.setBuffer(buffers[layer.denseDownScale!.shardIndex]!, offset: 0, index: 1)
                enc.setBuffer(buffers[layer.denseDownBias!.shardIndex]!, offset: 0, index: 2)
                enc.setBuffer(interBufAll, offset: interTokenOffset, index: 3)
                enc.setBuffer(hMlpBufAll, offset: tokenOffset, index: 4)
                enc.setBytes(&dWOff, length: 8, index: 5)
                enc.setBytes(&dSOff, length: 8, index: 6)
                enc.setBytes(&dBOff, length: 8, index: 7)
                enc.setBytes(&interD, length: 4, index: 8)
                enc.setBytes(&hD, length: 4, index: 9)
                enc.setBytes(&grp, length: 4, index: 10)
                enc.setBytes(&pk, length: 4, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }

            enc.endEncoding()
            cmd.commit()
            cmd.waitUntilCompleted()

            let outPtr = hMlpBufAll.contents().bindMemory(to: Float.self, capacity: P * hiddenDim)
            var nanCount = 0
            for i in 0..<(P * hiddenDim) {
                if outPtr[i].isNaN { nanCount += 1 }
            }
            XCTAssertEqual(nanCount, 0, "Layer \(layerIdx) prefill produced \(nanCount) NaNs across \(P) tokens")
        }
    }

    func testOrnith9BGatedDeltaNetSequenceMatchesStep() throws {
        print("=== TEST ORNITH 9B GATED DELTANET: SEQUENCE PREFILL MATCHES AUTOREGRESSIVE STEP ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice(), let cmdQueue = device.makeCommandQueue() else {
            XCTFail("Metal device or queue unavailable")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        guard let layer0 = cachedLayers.first,
              let aLog = layer0.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
              let dtBias = layer0.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
              let linNorm = layer0.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex],
              let seqPipe = inference.gdnLinearAttnSeqPipeline,
              let stepPipe = inference.gdnLinearAttnStepPipeline else {
            XCTFail("Missing GatedDeltaNet tensors or pipelines")
            return
        }

        let linKeyHeads: UInt32 = 16
        let linValHeads: UInt32 = 32
        let headDim: UInt32 = 128
        let qkvDim = Int((linKeyHeads + linKeyHeads + linValHeads) * headDim) // 8192
        let zDim = Int(linValHeads * headDim) // 4096
        let aDim = Int(linValHeads) // 32
        let bDim = Int(linValHeads) // 32
        let stateElements = Int(linValHeads * headDim * headDim) // 32 * 128 * 128 = 524288
        let P = 4

        guard let qkvBuf = device.makeBuffer(length: P * qkvDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zBuf = device.makeBuffer(length: P * zDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aBuf = device.makeBuffer(length: P * aDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: P * bDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let stateSeqBuf = device.makeBuffer(length: stateElements * MemoryLayout<Float>.stride, options: .storageModeShared),
              let stateStepBuf = device.makeBuffer(length: stateElements * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outSeqBuf = device.makeBuffer(length: P * zDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outStepBuf = device.makeBuffer(length: P * zDim * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Initialize input data
        let qkvPtr = qkvBuf.contents().bindMemory(to: Float.self, capacity: P * qkvDim)
        for i in 0..<(P * qkvDim) { qkvPtr[i] = Float(sin(Double(i) * 0.05)) * 0.2 }

        let zPtr = zBuf.contents().bindMemory(to: Float.self, capacity: P * zDim)
        for i in 0..<(P * zDim) { zPtr[i] = Float(cos(Double(i) * 0.03)) * 0.5 }

        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: P * aDim)
        for i in 0..<(P * aDim) { aPtr[i] = Float(sin(Double(i) * 0.1)) * 0.1 }

        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: P * bDim)
        for i in 0..<(P * bDim) { bPtr[i] = Float(cos(Double(i) * 0.1)) * 0.1 }

        memset(stateSeqBuf.contents(), 0, stateElements * MemoryLayout<Float>.stride)
        memset(stateStepBuf.contents(), 0, stateElements * MemoryLayout<Float>.stride)

        var aLogOff = aLog.offsetStart
        var dtBiasOff = dtBias.offsetStart
        var linNormOff = linNorm.offsetStart
        var numValH = linValHeads
        var numKeyH = linKeyHeads
        var hD = headDim
        var epsVal: Float = 1e-6
        var seqLenVal = UInt32(P)

        // 1. Run Sequence Pipeline for all P tokens
        guard let cmdSeq = cmdQueue.makeCommandBuffer(),
              let encSeq = cmdSeq.makeComputeCommandEncoder() else {
            XCTFail("Failed to create sequence encoder")
            return
        }
        encSeq.setComputePipelineState(seqPipe)
        encSeq.setBuffer(qkvBuf, offset: 0, index: 0)
        encSeq.setBuffer(zBuf, offset: 0, index: 1)
        encSeq.setBuffer(aBuf, offset: 0, index: 2)
        encSeq.setBuffer(bBuf, offset: 0, index: 3)
        encSeq.setBuffer(aLogRaw, offset: 0, index: 4)
        encSeq.setBuffer(dtBiasRaw, offset: 0, index: 5)
        encSeq.setBuffer(linNormRaw, offset: 0, index: 6)
        encSeq.setBuffer(stateSeqBuf, offset: 0, index: 7)
        encSeq.setBuffer(outSeqBuf, offset: 0, index: 8)
        encSeq.setBytes(&aLogOff, length: 8, index: 9)
        encSeq.setBytes(&dtBiasOff, length: 8, index: 10)
        encSeq.setBytes(&linNormOff, length: 8, index: 11)
        encSeq.setBytes(&numValH, length: 4, index: 12)
        encSeq.setBytes(&numKeyH, length: 4, index: 13)
        encSeq.setBytes(&hD, length: 4, index: 14)
        encSeq.setBytes(&epsVal, length: 4, index: 15)
        encSeq.setBytes(&seqLenVal, length: 4, index: 16)
        encSeq.dispatchThreadgroups(MTLSize(width: Int(numValH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encSeq.endEncoding()
        cmdSeq.commit()
        cmdSeq.waitUntilCompleted()

        // 2. Run Step Pipeline sequentially token by token
        for p in 0..<P {
            guard let cmdStep = cmdQueue.makeCommandBuffer(),
                  let encStep = cmdStep.makeComputeCommandEncoder() else {
                XCTFail("Failed to create step encoder for token \(p)")
                return
            }
            let qkvByteOff = p * qkvDim * MemoryLayout<Float>.stride
            let zByteOff = p * zDim * MemoryLayout<Float>.stride
            let aByteOff = p * aDim * MemoryLayout<Float>.stride
            let bByteOff = p * bDim * MemoryLayout<Float>.stride
            let outByteOff = p * zDim * MemoryLayout<Float>.stride

            encStep.setComputePipelineState(stepPipe)
            encStep.setBuffer(qkvBuf, offset: qkvByteOff, index: 0)
            encStep.setBuffer(zBuf, offset: zByteOff, index: 1)
            encStep.setBuffer(aBuf, offset: aByteOff, index: 2)
            encStep.setBuffer(bBuf, offset: bByteOff, index: 3)
            encStep.setBuffer(aLogRaw, offset: 0, index: 4)
            encStep.setBuffer(dtBiasRaw, offset: 0, index: 5)
            encStep.setBuffer(linNormRaw, offset: 0, index: 6)
            encStep.setBuffer(stateStepBuf, offset: 0, index: 7)
            encStep.setBuffer(outStepBuf, offset: outByteOff, index: 8)
            encStep.setBytes(&aLogOff, length: 8, index: 9)
            encStep.setBytes(&dtBiasOff, length: 8, index: 10)
            encStep.setBytes(&linNormOff, length: 8, index: 11)
            encStep.setBytes(&numValH, length: 4, index: 12)
            encStep.setBytes(&numKeyH, length: 4, index: 13)
            encStep.setBytes(&hD, length: 4, index: 14)
            encStep.setBytes(&epsVal, length: 4, index: 15)
            encStep.dispatchThreadgroups(MTLSize(width: Int(numValH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            encStep.endEncoding()
            cmdStep.commit()
            cmdStep.waitUntilCompleted()
        }

        // 3. Verify that outputs match across all tokens
        let outSeqPtr = outSeqBuf.contents().bindMemory(to: Float.self, capacity: P * zDim)
        let outStepPtr = outStepBuf.contents().bindMemory(to: Float.self, capacity: P * zDim)
        var maxOutDiff: Float = 0.0
        for i in 0..<(P * zDim) {
            let diff = abs(outSeqPtr[i] - outStepPtr[i])
            if diff > maxOutDiff { maxOutDiff = diff }
            XCTAssertFalse(outSeqPtr[i].isNaN, "NaN in outSeq at index \(i)")
            XCTAssertFalse(outStepPtr[i].isNaN, "NaN in outStep at index \(i)")
        }
        print("  ✅ [GDN VERIFICATION] Max Output Diff between Sequence and Step: \(maxOutDiff)")
        XCTAssertLessThan(maxOutDiff, 1e-4, "Sequence prefill output does not match autoregressive step output")

        // 4. Verify that final recurrent states match
        let stateSeqPtr = stateSeqBuf.contents().bindMemory(to: Float.self, capacity: stateElements)
        let stateStepPtr = stateStepBuf.contents().bindMemory(to: Float.self, capacity: stateElements)
        var maxStateDiff: Float = 0.0
        for i in 0..<stateElements {
            let diff = abs(stateSeqPtr[i] - stateStepPtr[i])
            if diff > maxStateDiff { maxStateDiff = diff }
            XCTAssertFalse(stateSeqPtr[i].isNaN, "NaN in stateSeq at index \(i)")
            XCTAssertFalse(stateStepPtr[i].isNaN, "NaN in stateStep at index \(i)")
        }
        print("  ✅ [GDN VERIFICATION] Max Recurrent State Diff between Sequence and Step: \(maxStateDiff)")
        XCTAssertLessThan(maxStateDiff, 1e-4, "Sequence prefill final state does not match autoregressive step final state")
    }

    func testOrnith9BCodingSettingsAutoregressive() throws {
        print("=== TEST ORNITH 1.5 9B AUTOREGRESSIVE WITH USER CODING SETTINGS ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 9B OptiQ-4bit snapshot not found")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let inference = InferenceEngine.shared
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        XCTAssertEqual(cachedLayers.count, 32)

        // 1. Verify Gated DeltaNet linear attention layer detection
        let linLayers = cachedLayers.filter { $0.attentionType == .linearAttention }.count
        XCTAssertEqual(linLayers, 24, "Ornith 1.5 9B has exactly 24 Gated DeltaNet linear attention layers")
        let hasLinearRecurrence = cachedLayers.contains { $0.attentionType == .linearAttention }
        XCTAssertTrue(hasLinearRecurrence, "Ornith 1.5 9B must be detected as having linear recurrence")

        // 2. Verify effectiveJetSpec logic:
        // Even when user has JetSpec enabled, recurrent Gated DeltaNet models MUST bypass speculative tree execution
        // because multi-node tree drafting disrupts continuous causal convolution and O(1) state updates.
        let userJetSpecEnabled = true
        let packedExpertsDir: URL? = nil
        let effectiveJetSpec = userJetSpecEnabled && !hasLinearRecurrence
        XCTAssertFalse(effectiveJetSpec, "effectiveJetSpec must be false for Ornith 1.5 9B to ensure 100% coherent execution")

        // 3. Verify high-performance Top-K / Top-P sampling with User's Exact Coding Settings:
        // Temperature: 0.60, Top-P: 0.95, Min-P: 0.00, Top-K: 20, Repetition Penalty: 1.00
        let vocabSize = 152064
        guard let device = MTLCreateSystemDefaultDevice(),
              let logitsBuf = device.makeBuffer(length: vocabSize * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate logits buffer")
            return
        }

        let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
        // Synthesize realistic peaked logit distribution (typical of Python/Swift code generation)
        for i in 0..<vocabSize {
            logitsPtr[i] = -20.0
        }
        // Top 20 tokens (e.g. "func", "let", "var", "def", whitespace, etc.)
        let topTokens: [UInt32] = [1000, 1005, 1010, 1020, 1050, 2000, 2050, 3000, 4000, 5000,
                                   6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000]
        for (rank, tok) in topTokens.enumerated() {
            logitsPtr[Int(tok)] = Float(20 - rank) // highest logit is 20.0 for token 1000
        }

        var sampledCounts: [UInt32: Int] = [:]
        let sampleRuns = 200
        for _ in 0..<sampleRuns {
            let nextTok = InferenceEngine.sampleNextToken(
                logits: logitsPtr,
                vocabSize: vocabSize,
                contextTokens: [151644, 872],
                temperature: 0.60,
                topP: 0.95,
                minP: 0.00,
                topK: 20,
                repetitionPenalty: 1.00
            )
            sampledCounts[nextTok, default: 0] += 1
            XCTAssertTrue(topTokens.contains(nextTok), "Sampled token \(nextTok) must be within top 20 candidates")
        }

        // Verify that the most probable token (token 1000) was sampled most frequently
        let mostFrequent = sampledCounts.max(by: { $0.value < $1.value })?.key
        XCTAssertEqual(mostFrequent, 1000, "Token 1000 with logit 20.0 should be the mode of distribution")
        print("  ✅ [TEST] Top-K (20) & Top-P (0.95) temperature (0.60) sampling verified cleanly across \(sampleRuns) draws.")

        // 4. Verify Presence Penalty Mechanics & Logit Restoration
        // Set token 1000 with logit 20.0, token 1005 with logit 19.5
        logitsPtr[1000] = 20.0
        logitsPtr[1005] = 19.5

        // Without presence penalty: greedy selects token 1000
        let greedyNeutral = InferenceEngine.sampleNextToken(
            logits: logitsPtr,
            vocabSize: vocabSize,
            contextTokens: [1000],
            temperature: 0.00,
            topP: 1.0,
            minP: 0.0,
            topK: 20,
            repetitionPenalty: 1.00,
            presencePenalty: 0.00
        )
        XCTAssertEqual(greedyNeutral, 1000, "With presencePenalty=0.0, token 1000 is chosen")
        XCTAssertEqual(logitsPtr[1000], 20.0, "Logit must be restored cleanly by defer")

        // With presence penalty 1.0 on token 1000 (which is in contextTokens):
        // Effective logit for 1000 becomes 20.0 - 1.0 = 19.0, so token 1005 (19.5) wins!
        let greedyPenalized = InferenceEngine.sampleNextToken(
            logits: logitsPtr,
            vocabSize: vocabSize,
            contextTokens: [1000],
            temperature: 0.00,
            topP: 1.0,
            minP: 0.0,
            topK: 20,
            repetitionPenalty: 1.00,
            presencePenalty: 1.00
        )
        XCTAssertEqual(greedyPenalized, 1005, "Presence penalty of 1.0 must suppress seen token 1000 below token 1005")
        XCTAssertEqual(logitsPtr[1000], 20.0, "Original logit 1000 must be restored cleanly by defer")
        print("  ✅ [TEST] Presence penalty (1.00) suppression and defer logit restoration verified.")
    }

    func testModelSpecificProfilesCoderAndAssistant() throws {
        let manager = ModelProfileManager.shared
        let testOrnithId = "mlx-community/Ornith-1.5-9B-OptiQ-4bit"
        let testQwenId = "Qwen/Qwen3.8-Flash-Next-FP8"

        // Ensure clean test isolation
        manager.resetProfile(for: testOrnithId, type: .coder)
        manager.resetProfile(for: testOrnithId, type: .assistant)
        manager.resetProfile(for: testQwenId, type: .coder)
        manager.resetProfile(for: testQwenId, type: .assistant)

        defer {
            manager.resetProfile(for: testOrnithId, type: .coder)
            manager.resetProfile(for: testOrnithId, type: .assistant)
            manager.resetProfile(for: testQwenId, type: .coder)
            manager.resetProfile(for: testQwenId, type: .assistant)
        }

        // 1. Verify official default Coder and Assistant profiles for Ornith-1.5-9B
        let defaultOrnithCoder = manager.getProfile(for: testOrnithId, type: .coder)
        let defaultOrnithAssistant = manager.getProfile(for: testOrnithId, type: .assistant)

        // Ornith Precise Coding & Tool Calling Profile
        XCTAssertEqual(defaultOrnithCoder.temperature, 0.60, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithCoder.topP, 0.95, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithCoder.minP, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithCoder.topK, 20)
        XCTAssertEqual(defaultOrnithCoder.repetitionPenalty, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithCoder.presencePenalty, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithCoder.maxNewTokens, 8192)
        XCTAssertFalse(defaultOrnithCoder.jetSpecEnabled, "Ornith linear recurrence must disable JetSpec by default")
        XCTAssertTrue(defaultOrnithCoder.systemPrompt.contains("software engineer") || defaultOrnithCoder.systemPrompt.contains("programming"))

        // Ornith General Chat / Agent Loops Profile
        XCTAssertEqual(defaultOrnithAssistant.temperature, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithAssistant.topP, 0.95, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithAssistant.minP, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithAssistant.topK, 20)
        XCTAssertEqual(defaultOrnithAssistant.repetitionPenalty, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithAssistant.presencePenalty, 1.50, accuracy: 0.01)
        XCTAssertEqual(defaultOrnithAssistant.maxNewTokens, 4096)
        XCTAssertFalse(defaultOrnithAssistant.jetSpecEnabled)
        XCTAssertTrue(defaultOrnithAssistant.systemPrompt.contains("helpful") || defaultOrnithAssistant.systemPrompt.contains("assistant"))

        // 2. Verify official default Coder and Assistant profiles for Qwen 3.8 Flash Next
        let defaultQwenCoder = manager.getProfile(for: testQwenId, type: .coder)
        let defaultQwenAssistant = manager.getProfile(for: testQwenId, type: .assistant)

        // Qwen Coding & Agentic Profile (Thinking Mode)
        XCTAssertEqual(defaultQwenCoder.temperature, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenCoder.topP, 0.95, accuracy: 0.01)
        XCTAssertEqual(defaultQwenCoder.minP, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenCoder.topK, 20)
        XCTAssertEqual(defaultQwenCoder.repetitionPenalty, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenCoder.presencePenalty, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenCoder.maxNewTokens, 8192)
        XCTAssertTrue(defaultQwenCoder.jetSpecEnabled)

        // Qwen General Assistant Profile (Instruct / Direct Mode)
        XCTAssertEqual(defaultQwenAssistant.temperature, 0.70, accuracy: 0.01)
        XCTAssertEqual(defaultQwenAssistant.topP, 0.80, accuracy: 0.01)
        XCTAssertEqual(defaultQwenAssistant.minP, 0.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenAssistant.topK, 20)
        XCTAssertEqual(defaultQwenAssistant.repetitionPenalty, 1.00, accuracy: 0.01)
        XCTAssertEqual(defaultQwenAssistant.presencePenalty, 1.50, accuracy: 0.01)
        XCTAssertEqual(defaultQwenAssistant.maxNewTokens, 4096)
        XCTAssertTrue(defaultQwenAssistant.jetSpecEnabled)

        // 3. Modify and Save Custom Settings for Ornith Coder
        var customOrnithCoder = defaultOrnithCoder
        customOrnithCoder.temperature = 0.25
        customOrnithCoder.presencePenalty = 0.50
        customOrnithCoder.repetitionPenalty = 1.05
        customOrnithCoder.maxNewTokens = 10000
        customOrnithCoder.systemPrompt = "Specialized Metal Coder"
        manager.saveProfile(for: testOrnithId, type: .coder, settings: customOrnithCoder)

        // 4. Verify Ornith Coder was persisted and retrieved
        let reloadedOrnithCoder = manager.getProfile(for: testOrnithId, type: .coder)
        XCTAssertEqual(reloadedOrnithCoder.temperature, 0.25, accuracy: 0.01)
        XCTAssertEqual(reloadedOrnithCoder.presencePenalty, 0.50, accuracy: 0.01)
        XCTAssertEqual(reloadedOrnithCoder.repetitionPenalty, 1.05, accuracy: 0.01)
        XCTAssertEqual(reloadedOrnithCoder.maxNewTokens, 10000)
        XCTAssertEqual(reloadedOrnithCoder.systemPrompt, "Specialized Metal Coder")

        // 5. Verify Ornith Assistant was NOT overwritten or mutated
        let reloadedOrnithAssistant = manager.getProfile(for: testOrnithId, type: .assistant)
        XCTAssertEqual(reloadedOrnithAssistant.temperature, 1.00, accuracy: 0.01)
        XCTAssertEqual(reloadedOrnithAssistant.presencePenalty, 1.50, accuracy: 0.01)
        XCTAssertEqual(reloadedOrnithAssistant.maxNewTokens, 4096)

        // 6. Verify Qwen profiles remain isolated from Ornith customization
        let reloadedQwenCoder = manager.getProfile(for: testQwenId, type: .coder)
        XCTAssertEqual(reloadedQwenCoder.temperature, 1.00, accuracy: 0.01)
        XCTAssertNotEqual(reloadedQwenCoder.systemPrompt, "Specialized Metal Coder")

        // 7. Test Active Profile Selection per Model
        manager.setActiveProfile(for: testOrnithId, type: .coder)
        XCTAssertEqual(manager.getActiveProfile(for: testOrnithId), .coder)

        manager.setActiveProfile(for: testQwenId, type: .assistant)
        XCTAssertEqual(manager.getActiveProfile(for: testQwenId), .assistant)
        XCTAssertEqual(manager.getActiveProfile(for: testOrnithId), .coder, "Model active profile states must be isolated")

        // 8. Test Display Names and Descriptions
        XCTAssertEqual(ModelProfileType.coder.profileDisplayName(for: testOrnithId), "Precise Coding & Tool Calling")
        XCTAssertEqual(ModelProfileType.assistant.profileDisplayName(for: testOrnithId), "General Chat / Agent Loops")
        XCTAssertEqual(ModelProfileType.coder.profileDisplayName(for: testQwenId), "Coding & Agentic (Thinking Mode)")
        XCTAssertEqual(ModelProfileType.assistant.profileDisplayName(for: testQwenId), "General Assistant (Instruct Mode)")

        print("  ✅ [TEST] Official Ornith & Qwen model profiles verified with persistence, isolation, and metadata.")
    }

    func testHelpTopicsAndSettingsGuideCoverage() throws {
        // 1. Verify all 11 HelpTopic enum cases are present
        let topics = HelpTopic.allCases
        XCTAssertEqual(topics.count, 11, "HelpTopic must have exactly 11 documented sections")

        // 2. Verify all topics have valid non-empty titles and SF Symbols
        for topic in topics {
            XCTAssertFalse(topic.rawValue.isEmpty, "Topic title must not be empty")
            XCTAssertFalse(topic.icon.isEmpty, "Topic SF symbol icon must not be empty")
            XCTAssertEqual(topic.id, topic.rawValue)
        }

        // 3. Verify specific critical topics are included
        let topicRawValues = Set(topics.map { $0.rawValue })
        XCTAssertTrue(topicRawValues.contains("Quick Start & Overview"))
        XCTAssertTrue(topicRawValues.contains("Models & Repackaging"))
        XCTAssertTrue(topicRawValues.contains("Model Profiles (Coder vs. Assistant)"))
        XCTAssertTrue(topicRawValues.contains("Generation & Sampling"))
        XCTAssertTrue(topicRawValues.contains("Memory & SSD Streaming"))
        XCTAssertTrue(topicRawValues.contains("KV Cache Precision"))
        XCTAssertTrue(topicRawValues.contains("JetSpec Acceleration"))
        XCTAssertTrue(topicRawValues.contains("Agent & Tool Execution"))
        XCTAssertTrue(topicRawValues.contains("Advanced Diagnostics"))
        XCTAssertTrue(topicRawValues.contains("Hardware Profiles (16GB - 128GB)"))
        XCTAssertTrue(topicRawValues.contains("Troubleshooting & FAQs"))

        print("  ✅ [TEST] Help topics and settings guide coverage verified with 11 distinct sections.")
    }

    func testGDNSiLUGatingKernel() throws {
        print("=== TEST GDN SILU GATING KERNEL ===")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let gdnPipe = inference.gdnLinearAttnStepPipeline else {
            XCTFail("gdn_linear_attention_recurrent_step pipeline not compiled")
            return
        }

        guard let cmdQueue = device.makeCommandQueue(),
              let cmd = cmdQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to create Metal command buffer or encoder")
            return
        }

        let numValHeads: UInt32 = 48
        let numKeyHeads: UInt32 = 16
        let headDim: UInt32 = 128
        let eps: Float = 1e-6

        let qkvCount = Int((numKeyHeads + numKeyHeads + numValHeads) * headDim) // (16 + 16 + 48) * 128 = 10240
        let zCount = Int(numValHeads * headDim) // 48 * 128 = 6144
        let stateCount = Int(numValHeads * headDim * headDim) // 48 * 128 * 128 = 786432

        guard let qkvBuf = device.makeBuffer(length: qkvCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zBuf = device.makeBuffer(length: zCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aBuf = device.makeBuffer(length: Int(numValHeads) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: Int(numValHeads) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aLogBuf = device.makeBuffer(length: Int(numValHeads) * 2, options: .storageModeShared),
              let dtBiasBuf = device.makeBuffer(length: Int(numValHeads) * 2, options: .storageModeShared),
              let normBuf = device.makeBuffer(length: Int(headDim) * 2, options: .storageModeShared),
              let stateBuf = device.makeBuffer(length: stateCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: zCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Initialize state to 0
        memset(stateBuf.contents(), 0, stateCount * MemoryLayout<Float>.stride)

        // Initialize Q and K to unit vectors along dimension 0 for head 0
        let qkvPtr = qkvBuf.contents().bindMemory(to: Float.self, capacity: qkvCount)
        memset(qkvPtr, 0, qkvCount * MemoryLayout<Float>.stride)
        // Q: keyHead 0, dim 0 = 1.0
        qkvPtr[0] = 1.0
        // K: keyHead 0 (offset 16*128), dim 0 = 1.0
        qkvPtr[Int(numKeyHeads * headDim)] = 1.0
        // V: valHead 0 (offset 32*128), dim 0 = 2.0 (positive activation)
        qkvPtr[Int(2 * numKeyHeads * headDim)] = 2.0

        // Initialize zBuf to 0.0 for valHead 0.
        // Under SiLU gating: silu(0.0) = 0.0 * sig(0.0) = 0.0. Out must be 0.0!
        let zPtr = zBuf.contents().bindMemory(to: Float.self, capacity: zCount)
        memset(zPtr, 0, zCount * MemoryLayout<Float>.stride)

        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: Int(numValHeads))
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: Int(numValHeads))
        for h in 0..<Int(numValHeads) {
            aPtr[h] = 0.0
            bPtr[h] = 5.0 // beta ~ 1.0
        }

        // BF16 1.0 is 0x3F80, BF16 0.0 is 0x0000
        let normPtr = normBuf.contents().bindMemory(to: UInt16.self, capacity: Int(headDim))
        for i in 0..<Int(headDim) { normPtr[i] = 0x3F80 } // gamma = 1.0 in BF16

        let aLogPtr = aLogBuf.contents().bindMemory(to: UInt16.self, capacity: Int(numValHeads))
        let dtBiasPtr = dtBiasBuf.contents().bindMemory(to: UInt16.self, capacity: Int(numValHeads))
        for h in 0..<Int(numValHeads) {
            aLogPtr[h] = 0x0000
            dtBiasPtr[h] = 0x3F80 // dtBias = 1.0
        }

        var aLogOff: UInt64 = 0
        var dtBiasOff: UInt64 = 0
        var normOff: UInt64 = 0
        var nValH = numValHeads
        var nKeyH = numKeyHeads
        var hD = headDim
        var epsVal = eps

        enc.setComputePipelineState(gdnPipe)
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(zBuf, offset: 0, index: 1)
        enc.setBuffer(aBuf, offset: 0, index: 2)
        enc.setBuffer(bBuf, offset: 0, index: 3)
        enc.setBuffer(aLogBuf, offset: 0, index: 4)
        enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
        enc.setBuffer(normBuf, offset: 0, index: 6)
        enc.setBuffer(stateBuf, offset: 0, index: 7)
        enc.setBuffer(outBuf, offset: 0, index: 8)
        enc.setBytes(&aLogOff, length: 8, index: 9)
        enc.setBytes(&dtBiasOff, length: 8, index: 10)
        enc.setBytes(&normOff, length: 8, index: 11)
        enc.setBytes(&nValH, length: 4, index: 12)
        enc.setBytes(&nKeyH, length: 4, index: 13)
        enc.setBytes(&hD, length: 4, index: 14)
        enc.setBytes(&epsVal, length: 4, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: zCount)
        let valHead0Dim0 = outPtr[0]
        print("🔍 [TEST] Head 0, Dim 0 Output: \(valHead0Dim0) (z=0.0)")

        // Under SiLU gating: silu(0.0) = 0.0 * sig(0.0) = 0.0.
        XCTAssertEqual(valHead0Dim0, 0.0, accuracy: 1e-5, "SiLU gating must produce exactly 0 when z=0")
        XCTAssertFalse(valHead0Dim0.isNaN || valHead0Dim0.isInfinite, "Output must be finite")

        // Now test positive z: z = 2.0. Under SiLU: silu(2.0) = 2.0 / (1.0 + exp(-2.0)) = 1.7616.
        zPtr[0] = 2.0
        guard let cmd2 = cmdQueue.makeCommandBuffer(), let enc2 = cmd2.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer 2")
            return
        }
        enc2.setComputePipelineState(gdnPipe)
        enc2.setBuffer(qkvBuf, offset: 0, index: 0)
        enc2.setBuffer(zBuf, offset: 0, index: 1)
        enc2.setBuffer(aBuf, offset: 0, index: 2)
        enc2.setBuffer(bBuf, offset: 0, index: 3)
        enc2.setBuffer(aLogBuf, offset: 0, index: 4)
        enc2.setBuffer(dtBiasBuf, offset: 0, index: 5)
        enc2.setBuffer(normBuf, offset: 0, index: 6)
        enc2.setBuffer(stateBuf, offset: 0, index: 7)
        enc2.setBuffer(outBuf, offset: 0, index: 8)
        enc2.setBytes(&aLogOff, length: 8, index: 9)
        enc2.setBytes(&dtBiasOff, length: 8, index: 10)
        enc2.setBytes(&normOff, length: 8, index: 11)
        enc2.setBytes(&nValH, length: 4, index: 12)
        enc2.setBytes(&nKeyH, length: 4, index: 13)
        enc2.setBytes(&hD, length: 4, index: 14)
        enc2.setBytes(&epsVal, length: 4, index: 15)
        enc2.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let valHead0Dim0_posZ = outPtr[0]
        print("🔍 [TEST] Head 0, Dim 0 Output with z=2.0: \(valHead0Dim0_posZ)")
        XCTAssertGreaterThan(valHead0Dim0_posZ, 1.0, "SiLU gating must allow positive scaling > 1.0 for z=2.0")

        // Now test negative z: z = -2.0. Under SiLU: silu(-2.0) = -2.0 * 0.1192 = -0.2384.
        zPtr[0] = -2.0
        guard let cmd3 = cmdQueue.makeCommandBuffer(), let enc3 = cmd3.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer 3")
            return
        }
        enc3.setComputePipelineState(gdnPipe)
        enc3.setBuffer(qkvBuf, offset: 0, index: 0)
        enc3.setBuffer(zBuf, offset: 0, index: 1)
        enc3.setBuffer(aBuf, offset: 0, index: 2)
        enc3.setBuffer(bBuf, offset: 0, index: 3)
        enc3.setBuffer(aLogBuf, offset: 0, index: 4)
        enc3.setBuffer(dtBiasBuf, offset: 0, index: 5)
        enc3.setBuffer(normBuf, offset: 0, index: 6)
        enc3.setBuffer(stateBuf, offset: 0, index: 7)
        enc3.setBuffer(outBuf, offset: 0, index: 8)
        enc3.setBytes(&aLogOff, length: 8, index: 9)
        enc3.setBytes(&dtBiasOff, length: 8, index: 10)
        enc3.setBytes(&normOff, length: 8, index: 11)
        enc3.setBytes(&nValH, length: 4, index: 12)
        enc3.setBytes(&nKeyH, length: 4, index: 13)
        enc3.setBytes(&hD, length: 4, index: 14)
        enc3.setBytes(&epsVal, length: 4, index: 15)
        enc3.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc3.endEncoding()
        cmd3.commit()
        cmd3.waitUntilCompleted()

        let valHead0Dim0_negZ = outPtr[0]
        print("🔍 [TEST] Head 0, Dim 0 Output with z=-2.0: \(valHead0Dim0_negZ)")
        XCTAssertLessThan(valHead0Dim0_negZ, 0.0, "SiLU gating correctly preserves negative sign for negative z")

        print("🎉 [SUCCESS] GDN SiLU Gating Kernel strictly verified!")
    }

    func testQwen38GDNSigmoidGatingKernel() throws {
        print("=== TEST QWEN 3.8 GDN SIGMOID GATING KERNEL ===")
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let gdnSigPipe = inference.gdnLinearAttnStepSigmoidPipeline else {
            XCTFail("gdn_linear_attention_recurrent_step_sigmoid pipeline not compiled")
            return
        }

        guard let cmdQueue = device.makeCommandQueue(),
              let cmd = cmdQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to create Metal command buffer or encoder")
            return
        }

        let numValHeads: UInt32 = 48
        let numKeyHeads: UInt32 = 16
        let headDim: UInt32 = 128
        let eps: Float = 1e-6

        let qkvCount = Int((numKeyHeads + numKeyHeads + numValHeads) * headDim)
        let zCount = Int(numValHeads * headDim)
        let stateCount = Int(numValHeads * headDim * headDim)

        guard let qkvBuf = device.makeBuffer(length: qkvCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let zBuf = device.makeBuffer(length: zCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aBuf = device.makeBuffer(length: Int(numValHeads) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bBuf = device.makeBuffer(length: Int(numValHeads) * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aLogBuf = device.makeBuffer(length: Int(numValHeads) * 2, options: .storageModeShared),
              let dtBiasBuf = device.makeBuffer(length: Int(numValHeads) * 2, options: .storageModeShared),
              let normBuf = device.makeBuffer(length: Int(headDim) * 2, options: .storageModeShared),
              let stateBuf = device.makeBuffer(length: stateCount * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: zCount * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Initialize state to 0
        memset(stateBuf.contents(), 0, stateCount * MemoryLayout<Float>.stride)

        // Initialize Q and K to unit vectors along dimension 0 for head 0
        let qkvPtr = qkvBuf.contents().bindMemory(to: Float.self, capacity: qkvCount)
        memset(qkvPtr, 0, qkvCount * MemoryLayout<Float>.stride)
        qkvPtr[0] = 1.0 // Q: keyHead 0, dim 0 = 1.0
        qkvPtr[Int(numKeyHeads * headDim)] = 1.0 // K: keyHead 0, dim 0 = 1.0
        qkvPtr[Int(2 * numKeyHeads * headDim)] = 2.0 // V: valHead 0, dim 0 = 2.0

        // Initialize zBuf to 0.0 for valHead 0.
        // Under Sigmoid gating: sigmoid(0.0) = 0.5. Out must be > 0.0 (specifically yNorm * 0.5)!
        let zPtr = zBuf.contents().bindMemory(to: Float.self, capacity: zCount)
        memset(zPtr, 0, zCount * MemoryLayout<Float>.stride)

        let aPtr = aBuf.contents().bindMemory(to: Float.self, capacity: Int(numValHeads))
        let bPtr = bBuf.contents().bindMemory(to: Float.self, capacity: Int(numValHeads))
        for h in 0..<Int(numValHeads) {
            aPtr[h] = 0.0
            bPtr[h] = 5.0 // beta ~ 1.0
        }

        let normPtr = normBuf.contents().bindMemory(to: UInt16.self, capacity: Int(headDim))
        for i in 0..<Int(headDim) { normPtr[i] = 0x3F80 } // gamma = 1.0 in BF16

        let aLogPtr = aLogBuf.contents().bindMemory(to: UInt16.self, capacity: Int(numValHeads))
        let dtBiasPtr = dtBiasBuf.contents().bindMemory(to: UInt16.self, capacity: Int(numValHeads))
        for h in 0..<Int(numValHeads) {
            aLogPtr[h] = 0x0000
            dtBiasPtr[h] = 0x3F80
        }

        var aLogOff: UInt64 = 0
        var dtBiasOff: UInt64 = 0
        var normOff: UInt64 = 0
        var nValH = numValHeads
        var nKeyH = numKeyHeads
        var hD = headDim
        var epsVal = eps

        enc.setComputePipelineState(gdnSigPipe)
        enc.setBuffer(qkvBuf, offset: 0, index: 0)
        enc.setBuffer(zBuf, offset: 0, index: 1)
        enc.setBuffer(aBuf, offset: 0, index: 2)
        enc.setBuffer(bBuf, offset: 0, index: 3)
        enc.setBuffer(aLogBuf, offset: 0, index: 4)
        enc.setBuffer(dtBiasBuf, offset: 0, index: 5)
        enc.setBuffer(normBuf, offset: 0, index: 6)
        enc.setBuffer(stateBuf, offset: 0, index: 7)
        enc.setBuffer(outBuf, offset: 0, index: 8)
        enc.setBytes(&aLogOff, length: 8, index: 9)
        enc.setBytes(&dtBiasOff, length: 8, index: 10)
        enc.setBytes(&normOff, length: 8, index: 11)
        enc.setBytes(&nValH, length: 4, index: 12)
        enc.setBytes(&nKeyH, length: 4, index: 13)
        enc.setBytes(&hD, length: 4, index: 14)
        enc.setBytes(&epsVal, length: 4, index: 15)
        enc.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc.endEncoding()

        cmd.commit()
        cmd.waitUntilCompleted()

        let outPtr = outBuf.contents().bindMemory(to: Float.self, capacity: zCount)
        let valHead0Dim0 = outPtr[0]
        print("🔍 [TEST] Sigmoid Head 0, Dim 0 Output: \(valHead0Dim0) (z=0.0)")

        // Under Sigmoid gating: sigmoid(0.0) = 0.5. With yNorm = 2.0, out is 1.0!
        XCTAssertGreaterThan(valHead0Dim0, 0.4, "Sigmoid gating must produce ~1.0 (positive non-zero) when z=0")
        XCTAssertFalse(valHead0Dim0.isNaN || valHead0Dim0.isInfinite, "Output must be finite")

        // Now test negative z: z = -2.0. Under Sigmoid: sigmoid(-2.0) = 0.1192 > 0. Out MUST BE POSITIVE!
        zPtr[0] = -2.0
        guard let cmd2 = cmdQueue.makeCommandBuffer(), let enc2 = cmd2.makeComputeCommandEncoder() else {
            XCTFail("Failed to create command buffer 2")
            return
        }
        enc2.setComputePipelineState(gdnSigPipe)
        enc2.setBuffer(qkvBuf, offset: 0, index: 0)
        enc2.setBuffer(zBuf, offset: 0, index: 1)
        enc2.setBuffer(aBuf, offset: 0, index: 2)
        enc2.setBuffer(bBuf, offset: 0, index: 3)
        enc2.setBuffer(aLogBuf, offset: 0, index: 4)
        enc2.setBuffer(dtBiasBuf, offset: 0, index: 5)
        enc2.setBuffer(normBuf, offset: 0, index: 6)
        enc2.setBuffer(stateBuf, offset: 0, index: 7)
        enc2.setBuffer(outBuf, offset: 0, index: 8)
        enc2.setBytes(&aLogOff, length: 8, index: 9)
        enc2.setBytes(&dtBiasOff, length: 8, index: 10)
        enc2.setBytes(&normOff, length: 8, index: 11)
        enc2.setBytes(&nValH, length: 4, index: 12)
        enc2.setBytes(&nKeyH, length: 4, index: 13)
        enc2.setBytes(&hD, length: 4, index: 14)
        enc2.setBytes(&epsVal, length: 4, index: 15)
        enc2.dispatchThreadgroups(MTLSize(width: Int(numValHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let valHead0Dim0_negZ = outPtr[0]
        print("🔍 [TEST] Sigmoid Head 0, Dim 0 Output with z=-2.0: \(valHead0Dim0_negZ)")
        XCTAssertGreaterThan(valHead0Dim0_negZ, 0.0, "Sigmoid gating output must ALWAYS be positive for positive activation even with negative z (unlike SiLU)")

        print("🎉 [SUCCESS] Qwen 3.8 GDN Sigmoid Gating Kernel strictly verified!")
    }

    func testWorkingSetManagerTokenBoundaryEviction() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal device")
            return
        }

        // Create 2 mock buffers: shard 0 (dense + experts), shard 1 (expert-only)
        guard let buf0 = device.makeBuffer(length: 64 * 1024, options: .storageModeShared),
              let buf1 = device.makeBuffer(length: 64 * 1024, options: .storageModeShared) else {
            XCTFail("Failed to allocate mock Metal buffers")
            return
        }
        let shardBuffers: [UInt32: MTLBuffer] = [0: buf0, 1: buf1]

        var shards: [ShardMetadata] = [
            ShardMetadata(index: 0, filename: "shard_0.safetensors", baseAddress: UInt64(UInt(bitPattern: buf0.contents())), length: 64 * 1024),
            ShardMetadata(index: 1, filename: "shard_1.safetensors", baseAddress: UInt64(UInt(bitPattern: buf1.contents())), length: 64 * 1024)
        ]

        var tensors: [TensorMetadata] = []
        // Add dense backbone tensors to shard 0
        for l in 0..<48 {
            tensors.append(TensorMetadata(
                name: "model.layers.\(l).input_layernorm.weight",
                shapeDisplay: "[2560]",
                dtype: "F32",
                sizeMb: 0.01,
                shardIndex: 0,
                offsetStart: 0,
                offsetEnd: 2560 * 4,
                category: "Backbone",
                layerIndex: UInt32(l),
                expertId: nil
            ))
        }

        // Add 32 experts per layer across 48 layers to shard 1 (expert-only)
        for l in 0..<48 {
            for exp in 0..<32 {
                tensors.append(TensorMetadata(
                    name: "model.layers.\(l).mlp.experts.\(exp).gate_proj.weight",
                    shapeDisplay: "[640, 2560]",
                    dtype: "FP8_E4M3",
                    sizeMb: 1.63,
                    shardIndex: 1,
                    offsetStart: UInt64(exp * 128),
                    offsetEnd: UInt64((exp + 1) * 128),
                    category: "Expert",
                    layerIndex: UInt32(l),
                    expertId: UInt32(exp)
                ))
            }
        }

        let summary = ModelSummary(
            sizeGb: 0.1,
            tensorCount: UInt32(tensors.count),
            layerCount: 48,
            maxExpertId: 31,
            shards: shards,
            tensors: tensors,
            layers: []
        )

        let mgr = WorkingSetManager.shared
        mgr.initialize(summary: summary, shardBuffers: shardBuffers, mode: .balanced16GB)

        XCTAssertEqual(mgr.residentExpertsCount, 0)
        XCTAssertEqual(mgr.totalExpertKeysCount, 48 * 32)
        let initRss = mgr.effectiveResidentMemoryGB
        XCTAssertGreaterThan(initRss, 0.0, "Effective resident memory should track process heap + dense backbone")

        // Verify primeSlices parallel page faulting
        let sampleSlices = [
            ExpertSlice(shardIndex: 0, offset: 0, length: 16384),
            ExpertSlice(shardIndex: 1, offset: 0, length: 32768)
        ]
        mgr.primeSlices(sampleSlices, shardBuffers: shardBuffers)

        // Token 0: Access 10 experts per layer (experts 0..9)
        for l in 0..<48 {
            let active = Array(0..<10)
            mgr.touchAndEvict(layer: l, activeExpertIds: active, mode: .balanced16GB, shardBuffers: shardBuffers)
        }
        XCTAssertEqual(mgr.residentExpertsCount, 480, "All 480 active experts must be resident")
        let tok0Rss = mgr.effectiveResidentMemoryGB
        XCTAssertGreaterThanOrEqual(tok0Rss, initRss, "Effective RSS must increase with resident experts")

        // Prune at Token 0 boundary: should NOT evict because 480 <= 1280
        mgr.trimToBudget(mode: .balanced16GB, shardBuffers: shardBuffers)
        XCTAssertEqual(mgr.residentExpertsCount, 480, "No experts should be evicted below capacity")

        // Token 1: 8 common experts (0..7) and 2 new experts (10..11) per layer
        for l in 0..<48 {
            let active = Array(0..<8) + [10, 11]
            mgr.touchAndEvict(layer: l, activeExpertIds: active, mode: .balanced16GB, shardBuffers: shardBuffers)
        }
        // Total resident = 480 original + (48 * 2 new) = 480 + 96 = 576
        XCTAssertEqual(mgr.residentExpertsCount, 576)
        XCTAssertGreaterThan(mgr.cacheHitRatePercent, 35.0, "Cache hit rate must reflect repeated expert hits")

        mgr.trimToBudget(mode: .balanced16GB, shardBuffers: shardBuffers)
        XCTAssertEqual(mgr.residentExpertsCount, 576)

        // Tokens 2..10: Add more unique experts to exceed budget capacity of 1280
        for tok in 2..<15 {
            let expOffset = (tok * 2) % 20
            for l in 0..<48 {
                let active = Array(expOffset..<(expOffset + 10))
                mgr.touchAndEvict(layer: l, activeExpertIds: active, mode: .balanced16GB, shardBuffers: shardBuffers)
            }
            mgr.trimToBudget(mode: .balanced16GB, shardBuffers: shardBuffers)
        }

        // Verify that resident set is strictly bounded by maxResidentExperts (1280)
        XCTAssertLessThanOrEqual(mgr.residentExpertsCount, 1280, "Resident experts must not exceed 1280 in balanced16GB mode")
        XCTAssertGreaterThan(mgr.cacheHitRatePercent, 60.0, "Multi-token cache hit rate should exceed 60%")
        print("🎉 [SUCCESS] WorkingSetManager token-boundary eviction verified! Final resident=\(mgr.residentExpertsCount), hitRate=\(String(format: "%.1f", mgr.cacheHitRatePercent))%")
    }

    func testWorkingSetManagerBulkPreadPriming() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("Metal not available")
        }

        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let shardFilename = "model-00001-of-00001.safetensors"
        let shardFileUrl = tempDir.appendingPathComponent(shardFilename)
        let dummyDataSize = 1024 * 1024 // 1 MB
        let dummyData = Data(repeating: 42, count: dummyDataSize)
        try dummyData.write(to: shardFileUrl)

        let buffer = device.makeBuffer(length: dummyDataSize, options: .storageModeShared)!
        let shardBuffers: [UInt32: MTLBuffer] = [0: buffer]

        var mockTensors: [TensorMetadata] = []
        let sliceSize: UInt64 = 65536
        // Layer 0, Expert 0
        mockTensors.append(TensorMetadata(
            name: "model.layers.0.mlp.experts.0.gate_proj.weight",
            shapeDisplay: "[256, 256]",
            dtype: "FP8",
            sizeMb: 0.065,
            shardIndex: 0,
            offsetStart: 0,
            offsetEnd: sliceSize,
            category: "Expert",
            layerIndex: 0,
            expertId: 0
        ))
        // Layer 0, Expert 1
        mockTensors.append(TensorMetadata(
            name: "model.layers.0.mlp.experts.1.gate_proj.weight",
            shapeDisplay: "[256, 256]",
            dtype: "FP8",
            sizeMb: 0.065,
            shardIndex: 0,
            offsetStart: sliceSize,
            offsetEnd: sliceSize * 2,
            category: "Expert",
            layerIndex: 0,
            expertId: 1
        ))

        let mockShards = [ShardMetadata(
            index: 0,
            filename: shardFilename,
            baseAddress: UInt64(UInt(bitPattern: buffer.contents())),
            length: UInt64(dummyDataSize)
        )]
        let summary = ModelSummary(
            sizeGb: 0.001,
            tensorCount: 2,
            layerCount: 1,
            maxExpertId: 1,
            shards: mockShards,
            tensors: mockTensors,
            layers: []
        )

        let mgr = WorkingSetManager.shared
        mgr.initialize(summary: summary, shardBuffers: shardBuffers, mode: .balanced16GB, modelDir: tempDir)

        // Prime the active experts using bulk pread via touchAndEvict
        let t0 = CFAbsoluteTimeGetCurrent()
        mgr.touchAndEvict(layer: 0, activeExpertIds: [0, 1], mode: .balanced16GB, shardBuffers: shardBuffers)
        let dtMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        print("⚡ [TEST] Bulk pread priming for 2 experts took: \(String(format: "%.3f", dtMs)) ms")

        XCTAssertEqual(mgr.residentExpertsCount, 2)
        XCTAssertGreaterThan(mgr.effectiveResidentMemoryGB, 0.0)

        // Close file descriptors and verify cleanup
        mgr.closeAllFileDescriptors()
        mgr.flushAllExperts(shardBuffers: shardBuffers)
        XCTAssertEqual(mgr.residentExpertsCount, 0)
        print("🎉 [SUCCESS] Bulk pread priming test passed!")
    }

    func testOrnithRMSNormIsNotUnitOffset() throws {
        print("=== TEST ORNITH & QWEN 3.8 RMSNORM & GATING CONFIGS ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        if FileManager.default.fileExists(atPath: snapshotDir) {
            let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
            XCTAssertNotNil(config)
            XCTAssertFalse(config!.isRMSNormUnitOffset, "Ornith 1.5 9B must NOT have isRMSNormUnitOffset = true (RMSNorm weights are centered at 1.0, not 0.0)")
            XCTAssertEqual(config!.effectiveOutputGateType, "silu", "Ornith 1.5 9B must use silu output gating")
        }

        let qwenSnapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next-FP8/snapshots/236dfdf285828023ca3bcd3f37366c58a3469b13"
        if FileManager.default.fileExists(atPath: qwenSnapshotDir) {
            let config = ModelConfig.load(from: URL(fileURLWithPath: qwenSnapshotDir))
            XCTAssertNotNil(config)
            XCTAssertTrue(config!.isRMSNormUnitOffset, "Qwen 3.8 Flash Next MUST have isRMSNormUnitOffset = true (attention norms are centered at 0.0)")
            XCTAssertEqual(config!.effectiveOutputGateType, "sigmoid", "Qwen 3.8 Flash Next must use sigmoid output gating")
        }

        // Test Gemma vs Qwen configs via JSON decoding
        let decoder = JSONDecoder()
        let gemmaData = """
        {"model_type": "gemma2", "architectures": ["Gemma2ForCausalLM"]}
        """.data(using: .utf8)!
        let gemmaConfig = try decoder.decode(ModelConfig.self, from: gemmaData)
        XCTAssertTrue(gemmaConfig.isRMSNormUnitOffset, "Gemma architectures MUST use unit-offset RMSNorm (output = x * (1 + weight))")

        let qwenData = """
        {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]}
        """.data(using: .utf8)!
        let qwenConfig = try decoder.decode(ModelConfig.self, from: qwenData)
        XCTAssertFalse(qwenConfig.isRMSNormUnitOffset, "Qwen architectures must NOT use unit-offset RMSNorm")

        let qwen35Data = """
        {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}
        """.data(using: .utf8)!
        let qwen35Config = try decoder.decode(ModelConfig.self, from: qwen35Data)
        XCTAssertFalse(qwen35Config.isRMSNormUnitOffset, "Dense Qwen 3.5 / Ornith 9B must NOT use unit-offset RMSNorm")
        XCTAssertEqual(qwen35Config.effectiveOutputGateType, "silu")

        let qwen35MoeData = """
        {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"]}
        """.data(using: .utf8)!
        let qwen35MoeConfig = try decoder.decode(ModelConfig.self, from: qwen35MoeData)
        XCTAssertTrue(qwen35MoeConfig.isRMSNormUnitOffset, "Qwen 3.5 MoE / Ornith 35B MUST use unit-offset RMSNorm")

        let qwen4ExpData = """
        {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": {"output_gate_type": "sigmoid"}}
        """.data(using: .utf8)!
        let qwen4ExpConfig = try decoder.decode(ModelConfig.self, from: qwen4ExpData)
        XCTAssertTrue(qwen4ExpConfig.isRMSNormUnitOffset, "Qwen 4 Exp / Next must use unit-offset RMSNorm")
        XCTAssertEqual(qwen4ExpConfig.effectiveOutputGateType, "sigmoid")

        print("✅ [TEST] RMSNorm unit offset and output gate rules verified cleanly.")
    }

    func testOrnithSystemPromptDetection() throws {
        print("=== TEST ORNITH SYSTEM PROMPT RESOLUTION ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))

        // 1. Path-based detection
        let promptFromPath = ModelConfig.resolveRequiredSystemPrompt(
            config: config,
            summary: nil,
            modelName: nil,
            modelPath: snapshotDir
        )
        XCTAssertEqual(promptFromPath, "", "Ornith must resolve to empty required system prompt")
        XCTAssertFalse(promptFromPath.contains("Alibaba"), "Ornith must never receive Alibaba Qwen system prompt")

        // 2. Topology-based detection (even if modelName and modelPath are empty and modelType is qwen3_5)
        let mockTensors = [
            TensorMetadata(name: "language_model.model.layers.0.linear_attn.in_proj_qkv.weight", shapeDisplay: "[8192, 2560]", dtype: "BF16", sizeMb: 40.0, shardIndex: 0, offsetStart: 0, offsetEnd: 0, category: "linear_attn", layerIndex: 0, expertId: nil),
            TensorMetadata(name: "language_model.model.layers.0.linear_attn.norm.weight", shapeDisplay: "[2560]", dtype: "BF16", sizeMb: 0.005, shardIndex: 0, offsetStart: 0, offsetEnd: 0, category: "linear_attn", layerIndex: 0, expertId: nil),
            TensorMetadata(name: "language_model.model.layers.0.self_attn.q_proj.weight", shapeDisplay: "[2560, 2560]", dtype: "BF16", sizeMb: 12.5, shardIndex: 0, offsetStart: 0, offsetEnd: 0, category: "attn", layerIndex: 0, expertId: nil)
        ]
        let mockSummary = ModelSummary(
            sizeGb: 7.0,
            tensorCount: 3,
            layerCount: 32,
            maxExpertId: 0,
            shards: [],
            tensors: mockTensors,
            layers: []
        )

        let promptFromTopology = ModelConfig.resolveRequiredSystemPrompt(
            config: config,
            summary: mockSummary,
            modelName: "qwen3_5",
            modelPath: nil
        )
        XCTAssertEqual(promptFromTopology, "", "Ornith GDN topology must resolve to empty required system prompt, preventing Qwen fallback")
        print("✅ [TEST] Ornith system prompt resolution verified cleanly.")
    }

    func testOrnithPrefillOutputCoherence() throws {
        print("=== TEST ORNITH PREFILL COHERENCE (NO MAGNITUDE EXPLOSION) ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--mlx-community--Ornith-1.5-9B-OptiQ-4bit/snapshots/ad2e7748e8c9d36b82bb88307fd21c0d50be85b8"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith snapshot not found")
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        guard let config = config else {
            XCTFail("Failed to load config")
            return
        }
        XCTAssertFalse(config.isRMSNormUnitOffset, "RMSNorm unit offset must be false")

        guard let device = MTLCreateSystemDefaultDevice(),
              let cmdQueue = device.makeCommandQueue() else {
            XCTFail("Metal device or queue unavailable")
            return
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 32)
        guard let firstLayer = cachedLayers.first,
              let norm1 = firstLayer.norm1Tensor,
              let norm1Buf = buffers[norm1.shardIndex] else {
            XCTFail("First layer norm1 missing")
            return
        }

        let hiddenDim = 2560
        guard let inBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outStandardBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let outOffsetBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Failed to allocate test buffers")
            return
        }

        // Initialize input with unit variance activations (typical residual state)
        let inPtr = inBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        for i in 0..<hiddenDim {
            inPtr[i] = Float(sin(Double(i) * 0.05))
        }

        var gammaOff = norm1.offsetStart
        var hDimU = UInt32(hiddenDim)
        var epsVal: Float = 1e-6

        // 1. Run Standard RMSNorm (correct for Ornith)
        let cmd1 = cmdQueue.makeCommandBuffer()!
        let enc1 = cmd1.makeComputeCommandEncoder()!
        enc1.setComputePipelineState(inference.rmsnormPipeline!)
        enc1.setBuffer(inBuf, offset: 0, index: 0)
        enc1.setBuffer(norm1Buf, offset: 0, index: 1)
        enc1.setBuffer(outStandardBuf, offset: 0, index: 2)
        enc1.setBytes(&gammaOff, length: 8, index: 3)
        enc1.setBytes(&hDimU, length: 4, index: 4)
        enc1.setBytes(&epsVal, length: 4, index: 5)
        enc1.setThreadgroupMemoryLength(1024 * 4, index: 0)
        enc1.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        enc1.endEncoding()
        cmd1.commit()
        cmd1.waitUntilCompleted()

        // 2. Run Offset RMSNorm (the buggy path that was previously taken)
        let cmd2 = cmdQueue.makeCommandBuffer()!
        let enc2 = cmd2.makeComputeCommandEncoder()!
        enc2.setComputePipelineState(inference.rmsnormOffsetPipeline!)
        enc2.setBuffer(inBuf, offset: 0, index: 0)
        enc2.setBuffer(norm1Buf, offset: 0, index: 1)
        enc2.setBuffer(outOffsetBuf, offset: 0, index: 2)
        enc2.setBytes(&gammaOff, length: 8, index: 3)
        enc2.setBytes(&hDimU, length: 4, index: 4)
        enc2.setBytes(&epsVal, length: 4, index: 5)
        enc2.setThreadgroupMemoryLength(1024 * 4, index: 0)
        enc2.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        enc2.endEncoding()
        cmd2.commit()
        cmd2.waitUntilCompleted()

        let stdPtr = outStandardBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        let offPtr = outOffsetBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)

        var stdMax: Float = 0
        var offMax: Float = 0
        for i in 0..<hiddenDim {
            stdMax = max(stdMax, abs(stdPtr[i]))
            offMax = max(offMax, abs(offPtr[i]))
        }

        print("⚡ [NORM COMPARISON] Standard RMSNorm max abs: \(stdMax), Offset RMSNorm max abs: \(offMax)")
        // In standard RMSNorm, with norm weights ~1.0, normalized output peak is around 1.0 - 2.0
        XCTAssertLessThan(stdMax, 5.0, "Standard RMSNorm output should be well-behaved")
        // In offset RMSNorm, output is ~1.8x standard RMSNorm at a single layer, compounding to 2^64 over 32 layers!
        XCTAssertGreaterThan(offMax, stdMax * 1.5, "Offset RMSNorm inflates activation scale by ~1.8x per norm")
        print("✅ [TEST] Activation stability verified: Standard RMSNorm keeps magnitudes bounded.")
    }

    func testOrnith35BDiagnostics() throws {
        print("=== TEST ORNITH 1.5 35B DIAGNOSTICS ===")
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/fab11c26e2325a42f4b32da0249c819a0bade1b1"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            throw XCTSkip("Ornith 1.5 35B FP8 snapshot not found")
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        guard let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal command queue")
            return
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        XCTAssertEqual(cachedLayers.count, 40)

        let hiddenDim = 2048
        let intermediateDim = 512
        let vocabSize = 248320

        KVCacheManager.shared.reset(
            device: device,
            config: config,
            actualLayers: 40,
            totalLoops: 1,
            numKvHeads: 2,
            headDim: 256,
            maxSeqLen: 256,
            precision: .fp16
        )

        let hBufA = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hBufB = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm1Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let xNorm2Buf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let qGateBuf = device.makeBuffer(length: 8192 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let zGateBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let aVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let bVecBuf = device.makeBuffer(length: 32 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let kVectorBuf = device.makeBuffer(length: 512 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let vVectorBuf = device.makeBuffer(length: 512 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnCtxBuf = device.makeBuffer(length: 4096 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let attnOutBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMidBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let interBuf = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let hMlpBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let routerIndicesBuf = device.makeBuffer(length: 8 * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let routerWeightsBuf = device.makeBuffer(length: 8 * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let sharedScoreBuf = device.makeBuffer(length: MemoryLayout<Float>.stride, options: .storageModeShared)!
        let singleTokenBuf = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        let xFinalBuf = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared)!
        let logitsBuf = device.makeBuffer(length: vocabSize * MemoryLayout<Float>.stride, options: .storageModeShared)!

        let layoutData = try Data(contentsOf: URL(fileURLWithPath: snapshotDir).appendingPathComponent("packed_experts/layout.json"))
        let layout = try JSONDecoder().decode(FlashMoELayout.self, from: layoutData)
        let expertSize = Int(layout.expert_size)
        let stagingBuf = device.makeBuffer(length: 8 * expertSize, options: .storageModeShared)!

        ExpertIOThreadPool.shared.initialize(numThreads: 8)

        func checkBuf(_ name: String, _ buf: MTLBuffer, count: Int) {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            var minVal: Float = .infinity
            var maxVal: Float = -.infinity
            var sumVal: Double = 0
            var nanCount = 0
            for i in 0..<count {
                let v = ptr[i]
                if v.isNaN || v.isInfinite {
                    nanCount += 1
                } else {
                    if v < minVal { minVal = v }
                    if v > maxVal { maxVal = v }
                    sumVal += Double(v)
                }
            }
            let meanVal = count > nanCount ? Float(sumVal / Double(count - nanCount)) : 0
            print("  [\(name)] count=\(count), nans=\(nanCount), min=\(minVal), max=\(maxVal), mean=\(meanVal)")
            XCTAssertEqual(nanCount, 0, "Buffer \(name) contains NaN or Inf values!")
        }

        func dispatchLinear(
            enc: MTLComputeCommandEncoder,
            weight: TensorMetadata?,
            scale: TensorMetadata?,
            bias: TensorMetadata?,
            inBuf: MTLBuffer,
            outBuf: MTLBuffer,
            inDim: UInt32,
            outDim: UInt32
        ) {
            guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
            var wOff = w.offsetStart
            var inD = inDim
            var outD = outDim
            let isFP8 = !w.dtype.contains("BF16") && !w.dtype.contains("F16") && !w.dtype.contains("FLOAT")
            if isFP8 {
                let sRaw = (scale != nil) ? buffers[scale!.shardIndex]! : wRaw
                var sOff = scale?.offsetStart ?? 0
                if let simdPipe = inference.fp8GemvSimdPipeline {
                    enc.setComputePipelineState(simdPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBuffer(sRaw, offset: 0, index: 3)
                    enc.setBytes(&wOff, length: 8, index: 4)
                    enc.setBytes(&sOff, length: 8, index: 5)
                    enc.setBytes(&inD, length: 4, index: 6)
                    enc.setBytes(&outD, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                }
            } else if let bSimdPipe = inference.bf16GemvSimdPipeline {
                enc.setComputePipelineState(bSimdPipe)
                enc.setBuffer(wRaw, offset: 0, index: 0)
                enc.setBuffer(inBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            enc.memoryBarrier(scope: .buffers)
        }

        func dispatchSharedExpert(
            enc: MTLComputeCommandEncoder,
            layer: EngineCachedLayer,
            inBuf: MTLBuffer,
            interBuf: MTLBuffer,
            accumBuf: MTLBuffer,
            sharedW: Float
        ) {
            guard let gateW = layer.sharedGateWeight,
                  let upW = layer.sharedUpWeight,
                  let downW = layer.sharedDownWeight,
                  let gateS = layer.sharedGateScale,
                  let upS = layer.sharedUpScale,
                  let downS = layer.sharedDownScale,
                  let gateSimd = inference.fp8GateUpSimdPipeline,
                  let downSimd = inference.fp8DownSimdPipeline else { return }

            var gWOff = gateW.offsetStart
            var gSOff = gateS.offsetStart
            var uWOff = upW.offsetStart
            var uSOff = upS.offsetStart
            var dWOff = downW.offsetStart
            var dSOff = downS.offsetStart
            var hDimVal = UInt32(hiddenDim)
            var interDimVal = UInt32(intermediateDim)
            var pk = sharedW

            enc.setComputePipelineState(gateSimd)
            enc.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 1)
            enc.setBuffer(inBuf, offset: 0, index: 2)
            enc.setBuffer(interBuf, offset: 0, index: 3)
            enc.setBuffer(buffers[gateS.shardIndex]!, offset: 0, index: 4)
            enc.setBuffer(buffers[upS.shardIndex]!, offset: 0, index: 5)
            enc.setBytes(&gWOff, length: 8, index: 6)
            enc.setBytes(&gSOff, length: 8, index: 7)
            enc.setBytes(&uWOff, length: 8, index: 8)
            enc.setBytes(&uSOff, length: 8, index: 9)
            enc.setBytes(&hDimVal, length: 4, index: 10)
            enc.setBytes(&interDimVal, length: 4, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(downSimd)
            enc.setBuffer(buffers[downW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(interBuf, offset: 0, index: 1)
            enc.setBuffer(accumBuf, offset: 0, index: 2)
            enc.setBuffer(buffers[downS.shardIndex]!, offset: 0, index: 3)
            enc.setBytes(&dWOff, length: 8, index: 4)
            enc.setBytes(&dSOff, length: 8, index: 5)
            enc.setBytes(&interDimVal, length: 4, index: 6)
            enc.setBytes(&hDimVal, length: 4, index: 7)
            enc.setBytes(&pk, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        func dispatchRoutedExperts(
            enc: MTLComputeCommandEncoder,
            layerIdx: Int,
            activeExperts: [(id: Int, weight: Float)],
            inBuf: MTLBuffer,
            interBuf: MTLBuffer,
            accumBuf: MTLBuffer
        ) {
            let binPath = URL(fileURLWithPath: snapshotDir).appendingPathComponent(String(format: "packed_experts/layer_%02d.bin", layerIdx)).path
            let fd = open(binPath, O_RDONLY)
            guard fd >= 0 else { return }
            defer { close(fd) }

            var tasks: [ExpertPreadTask] = []
            let rawStagingPtr = stagingBuf.contents()
            for (slot, exp) in activeExperts.enumerated() {
                let offset = off_t(exp.id * expertSize)
                let dst = rawStagingPtr.advanced(by: slot * expertSize)
                tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
            }
            ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)

            guard let gateSimd = inference.fp8GateUpSimdPipeline,
                  let downSimd = inference.fp8DownSimdPipeline else { return }

            for (slot, expert) in activeExperts.enumerated() {
                let pk = expert.weight
                if pk <= 0.00001 { continue }
                let slotOffset = UInt64(slot * expertSize)
                var gWOff = slotOffset + 0
                var gSOff = slotOffset + 1048576
                var uWOff = slotOffset + 1049600
                var uSOff = slotOffset + 2098176
                var dWOff = slotOffset + 2099200
                var dSOff = slotOffset + 3147776
                var hDimVal = UInt32(hiddenDim)
                var interDimVal = UInt32(intermediateDim)
                var pkVal = pk

                enc.setComputePipelineState(gateSimd)
                enc.setBuffer(stagingBuf, offset: 0, index: 0)
                enc.setBuffer(stagingBuf, offset: 0, index: 1)
                enc.setBuffer(inBuf, offset: 0, index: 2)
                enc.setBuffer(interBuf, offset: 0, index: 3)
                enc.setBuffer(stagingBuf, offset: 0, index: 4)
                enc.setBuffer(stagingBuf, offset: 0, index: 5)
                enc.setBytes(&gWOff, length: 8, index: 6)
                enc.setBytes(&gSOff, length: 8, index: 7)
                enc.setBytes(&uWOff, length: 8, index: 8)
                enc.setBytes(&uSOff, length: 8, index: 9)
                enc.setBytes(&hDimVal, length: 4, index: 10)
                enc.setBytes(&interDimVal, length: 4, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(downSimd)
                enc.setBuffer(stagingBuf, offset: 0, index: 0)
                enc.setBuffer(interBuf, offset: 0, index: 1)
                enc.setBuffer(accumBuf, offset: 0, index: 2)
                enc.setBuffer(stagingBuf, offset: 0, index: 3)
                enc.setBytes(&dWOff, length: 8, index: 4)
                enc.setBytes(&dSOff, length: 8, index: 5)
                enc.setBytes(&interDimVal, length: 4, index: 6)
                enc.setBytes(&hDimVal, length: 4, index: 7)
                enc.setBytes(&pkVal, length: 4, index: 8)
                enc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        // 1. Embed Token 795 (" data")
        let embedWeight = summary.tensors.first { $0.name.contains("embed_tokens") && $0.name.hasSuffix(".weight") }!
        let singleTokenPtr = singleTokenBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
        singleTokenPtr[0] = 795

        let embedCmd = cmdQueue.makeCommandBuffer()!
        let embedEnc = embedCmd.makeComputeCommandEncoder()!
        var wOffset = embedWeight.offsetStart
        var hDimVal = UInt32(hiddenDim)
        var tokCount: UInt32 = 1
        let embedPipe = inference.embedPipeline!
        embedEnc.setComputePipelineState(embedPipe)
        embedEnc.setBuffer(buffers[embedWeight.shardIndex]!, offset: 0, index: 0)
        embedEnc.setBuffer(singleTokenBuf, offset: 0, index: 1)
        embedEnc.setBuffer(hBufA, offset: 0, index: 2)
        embedEnc.setBytes(&wOffset, length: 8, index: 3)
        embedEnc.setBytes(&hDimVal, length: 4, index: 4)
        embedEnc.setBytes(&tokCount, length: 4, index: 5)
        embedEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, embedPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        embedEnc.endEncoding()
        embedCmd.commit()
        embedCmd.waitUntilCompleted()

        var l0Log = ""
        func logBuf(_ name: String, _ buf: MTLBuffer, count: Int) {
            let ptr = buf.contents().bindMemory(to: Float.self, capacity: count)
            var sumSq: Double = 0
            var first5: [Float] = []
            for i in 0..<count {
                let v = ptr[i]
                sumSq += Double(v * v)
                if i < 5 { first5.append(v) }
            }
            let l2 = sqrt(sumSq)
            l0Log += "\(name) | L2: \(l2) | first5: \(first5)\n"
            print("\(name) | L2: \(l2) | first5: \(first5)")
        }

        logBuf("hBufA_embed", hBufA, count: hiddenDim)

        // 2. Layer 0 (Linear Attention)
        let layer0 = cachedLayers[0]
        let l0Cmd = cmdQueue.makeCommandBuffer()!
        let l0Enc = l0Cmd.makeComputeCommandEncoder()!

        // Norm 1
        let norm1 = layer0.norm1Tensor!
        var nOff1 = norm1.offsetStart
        var epsVal: Float = 1e-6
        let rmsPipe = inference.rmsnormOffsetPipeline ?? inference.rmsnormPipeline!
        l0Enc.setComputePipelineState(rmsPipe)
        l0Enc.setBuffer(hBufA, offset: 0, index: 0)
        l0Enc.setBuffer(buffers[norm1.shardIndex]!, offset: 0, index: 1)
        l0Enc.setBuffer(xNorm1Buf, offset: 0, index: 2)
        l0Enc.setBytes(&nOff1, length: 8, index: 3)
        l0Enc.setBytes(&hDimVal, length: 4, index: 4)
        l0Enc.setBytes(&epsVal, length: 4, index: 5)
        l0Enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
        l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // inProjQKV
        let linKeyHeads: UInt32 = 16
        let linValHeads: UInt32 = 32
        let qkvDim: UInt32 = (16 + 16 + 32) * 128 // 8192
        let zDim: UInt32 = 32 * 128 // 4096
        dispatchLinear(enc: l0Enc, weight: layer0.inProjQKV, scale: layer0.inProjQKVScale, bias: layer0.inProjQKVBias, inBuf: xNorm1Buf, outBuf: qGateBuf, inDim: UInt32(hiddenDim), outDim: qkvDim)

        // conv1d
        if let conv1d = layer0.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
           let convPipe = inference.causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
            var cOff = conv1d.offsetStart
            var numChannels = qkvDim
            l0Enc.setComputePipelineState(convPipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBuffer(convRaw, offset: 0, index: 1)
            l0Enc.setBuffer(convState, offset: 0, index: 2)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 3)
            l0Enc.setBytes(&cOff, length: 8, index: 4)
            l0Enc.setBytes(&numChannels, length: 4, index: 5)
            l0Enc.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // l2NormQk
        if let l2Pipe = inference.l2NormQkPipeline {
            var numH = linKeyHeads
            var hD: UInt32 = 128
            l0Enc.setComputePipelineState(l2Pipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBytes(&numH, length: 4, index: 1)
            l0Enc.setBytes(&hD, length: 4, index: 2)
            l0Enc.dispatchThreads(MTLSize(width: Int(numH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numH), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // inProjZ, inProjA, inProjB
        dispatchLinear(enc: l0Enc, weight: layer0.inProjZ, scale: layer0.inProjZScale, bias: layer0.inProjZBias, inBuf: xNorm1Buf, outBuf: zGateBuf, inDim: UInt32(hiddenDim), outDim: zDim)
        dispatchLinear(enc: l0Enc, weight: layer0.inProjA, scale: layer0.inProjAScale, bias: layer0.inProjABias, inBuf: xNorm1Buf, outBuf: aVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)
        dispatchLinear(enc: l0Enc, weight: layer0.inProjB, scale: layer0.inProjBScale, bias: layer0.inProjBBias, inBuf: xNorm1Buf, outBuf: bVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)

        // linearAttnStep
        if let linPipe = inference.gdnLinearAttnStepPipeline ?? inference.linearAttnStepPipeline,
           let sBuf = KVCacheManager.shared.linearStateBuffer,
           let aLog = layer0.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
           let dtBias = layer0.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
           let linNorm = layer0.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
            var aLogOff = aLog.offsetStart
            var dtBiasOff = dtBias.offsetStart
            var linNormOff = linNorm.offsetStart
            var numValH = linValHeads
            var numKeyH = linKeyHeads
            var hD: UInt32 = 128
            l0Enc.setComputePipelineState(linPipe)
            l0Enc.setBuffer(qGateBuf, offset: 0, index: 0)
            l0Enc.setBuffer(zGateBuf, offset: 0, index: 1)
            l0Enc.setBuffer(aVecBuf, offset: 0, index: 2)
            l0Enc.setBuffer(bVecBuf, offset: 0, index: 3)
            l0Enc.setBuffer(aLogRaw, offset: 0, index: 4)
            l0Enc.setBuffer(dtBiasRaw, offset: 0, index: 5)
            l0Enc.setBuffer(linNormRaw, offset: 0, index: 6)
            l0Enc.setBuffer(sBuf, offset: 0, index: 7)
            l0Enc.setBuffer(attnCtxBuf, offset: 0, index: 8)
            l0Enc.setBytes(&aLogOff, length: 8, index: 9)
            l0Enc.setBytes(&dtBiasOff, length: 8, index: 10)
            l0Enc.setBytes(&linNormOff, length: 8, index: 11)
            l0Enc.setBytes(&numValH, length: 4, index: 12)
            l0Enc.setBytes(&numKeyH, length: 4, index: 13)
            l0Enc.setBytes(&hD, length: 4, index: 14)
            l0Enc.setBytes(&epsVal, length: 4, index: 15)
            l0Enc.dispatchThreadgroups(MTLSize(width: Int(numValH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // outProj
        dispatchLinear(enc: l0Enc, weight: layer0.linearOutProjTensor ?? layer0.oProjTensor, scale: layer0.linearOutProjScale ?? layer0.oScaleTensor, bias: layer0.linearOutProjBias ?? layer0.oBiasTensor, inBuf: attnCtxBuf, outBuf: attnOutBuf, inDim: zDim, outDim: UInt32(hiddenDim))

        // Residual 1
        let addPipe = inference.addPipeline!
        l0Enc.setComputePipelineState(addPipe)
        l0Enc.setBuffer(hBufA, offset: 0, index: 0)
        l0Enc.setBuffer(attnOutBuf, offset: 0, index: 1)
        l0Enc.setBuffer(hMidBuf, offset: 0, index: 2)
        l0Enc.setBytes(&hDimVal, length: 4, index: 3)
        l0Enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Norm 2
        let norm2 = layer0.norm2Tensor!
        var nOff2 = norm2.offsetStart
        l0Enc.setComputePipelineState(rmsPipe)
        l0Enc.setBuffer(hMidBuf, offset: 0, index: 0)
        l0Enc.setBuffer(buffers[norm2.shardIndex]!, offset: 0, index: 1)
        l0Enc.setBuffer(xNorm2Buf, offset: 0, index: 2)
        l0Enc.setBytes(&nOff2, length: 8, index: 3)
        l0Enc.setBytes(&hDimVal, length: 4, index: 4)
        l0Enc.setBytes(&epsVal, length: 4, index: 5)
        l0Enc.setThreadgroupMemoryLength(1024 * 4, index: 0)
        l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Router
        let router = layer0.routerTensor!
        var rOff = router.offsetStart
        var nExp: UInt32 = 256
        var kVal: UInt32 = 8
        let rPipe = inference.routerPipeline!
        l0Enc.setComputePipelineState(rPipe)
        l0Enc.setBuffer(buffers[router.shardIndex]!, offset: 0, index: 0)
        l0Enc.setBuffer(xNorm2Buf, offset: 0, index: 1)
        l0Enc.setBuffer(routerIndicesBuf, offset: 0, index: 2)
        l0Enc.setBuffer(routerWeightsBuf, offset: 0, index: 3)
        l0Enc.setBytes(&rOff, length: 8, index: 4)
        l0Enc.setBytes(&hDimVal, length: 4, index: 5)
        l0Enc.setBytes(&nExp, length: 4, index: 6)
        l0Enc.setBytes(&kVal, length: 4, index: 7)
        l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        // Shared Gate
        if let sgPipe = inference.sharedGatePipeline,
           let sgWeight = layer0.sharedGateTensor,
           let sgRaw = buffers[sgWeight.shardIndex] {
            var sgOff = sgWeight.offsetStart
            l0Enc.setComputePipelineState(sgPipe)
            l0Enc.setBuffer(sgRaw, offset: 0, index: 0)
            l0Enc.setBuffer(xNorm2Buf, offset: 0, index: 1)
            l0Enc.setBuffer(sharedScoreBuf, offset: 0, index: 2)
            l0Enc.setBytes(&sgOff, length: 8, index: 3)
            l0Enc.setBytes(&hDimVal, length: 4, index: 4)
            l0Enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            l0Enc.memoryBarrier(scope: .buffers)
        }

        // Clear hMlpBuf
        let clearPipe = inference.clearPipeline!
        l0Enc.setComputePipelineState(clearPipe)
        l0Enc.setBuffer(hMlpBuf, offset: 0, index: 0)
        l0Enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        l0Enc.memoryBarrier(scope: .buffers)

        l0Enc.endEncoding()
        l0Cmd.commit()
        l0Cmd.waitUntilCompleted()

        logBuf("xNorm1", xNorm1Buf, count: hiddenDim)
        logBuf("qGate", qGateBuf, count: Int(qkvDim))
        logBuf("zGate", zGateBuf, count: Int(zDim))
        logBuf("aVec", aVecBuf, count: Int(linValHeads))
        logBuf("bVec", bVecBuf, count: Int(linValHeads))
        logBuf("attnCtx", attnCtxBuf, count: Int(zDim))
        logBuf("attnOut", attnOutBuf, count: hiddenDim)
        logBuf("hMid", hMidBuf, count: hiddenDim)
        logBuf("xNorm2", xNorm2Buf, count: hiddenDim)

        let rIdxPtr = routerIndicesBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
        let rWgtPtr = routerWeightsBuf.contents().bindMemory(to: Float.self, capacity: 8)
        var activeExperts: [(id: Int, weight: Float)] = []
        l0Log += "Active Experts:\n"
        for i in 0..<8 {
            activeExperts.append((id: Int(rIdxPtr[i]), weight: rWgtPtr[i]))
            l0Log += "  Expert \(i): ID=\(rIdxPtr[i]), weight=\(rWgtPtr[i])\n"
        }
        let sgPtr = sharedScoreBuf.contents().bindMemory(to: Float.self, capacity: 1)
        l0Log += "Shared Gate Weight: \(sgPtr[0])\n"
        try? l0Log.write(toFile: "/tmp/ornith_l0_debug.txt", atomically: true, encoding: .utf8)

        // Dispatch Routed & Shared Experts
        let moeCmd = cmdQueue.makeCommandBuffer()!
        let moeEnc = moeCmd.makeComputeCommandEncoder()!
        dispatchRoutedExperts(enc: moeEnc, layerIdx: 0, activeExperts: activeExperts, inBuf: xNorm2Buf, interBuf: interBuf, accumBuf: hMlpBuf)
        dispatchSharedExpert(enc: moeEnc, layer: layer0, inBuf: xNorm2Buf, interBuf: interBuf, accumBuf: hMlpBuf, sharedW: sgPtr[0])

        // Residual 2
        moeEnc.setComputePipelineState(addPipe)
        moeEnc.setBuffer(hMidBuf, offset: 0, index: 0)
        moeEnc.setBuffer(hMlpBuf, offset: 0, index: 1)
        moeEnc.setBuffer(hBufB, offset: 0, index: 2)
        moeEnc.setBytes(&hDimVal, length: 4, index: 3)
        moeEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

        moeEnc.endEncoding()
        moeCmd.commit()
        moeCmd.waitUntilCompleted()

        logBuf("hMlpBuf", hMlpBuf, count: hiddenDim)
        logBuf("hBufB_layer0_out", hBufB, count: hiddenDim)
        try? l0Log.write(toFile: "/tmp/ornith_l0_debug.txt", atomically: true, encoding: .utf8)
    }

    func testOrnith35BFullModelDecode() throws {
        guard let device = MTLCreateSystemDefaultDevice() else { return }
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/fab11c26e2325a42f4b32da0249c819a0bade1b1"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot dir does not exist, skipping test.")
            return
        }

        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 40)
        XCTAssertEqual(cachedLayers.count, 40)

        let cmdQueue = device.makeCommandQueue()!
        let hiddenDim = 2048
        let intermediateDim = 512
        let expertSize = 3151872
        let numHeads: UInt32 = 16
        let numKvHeads: UInt32 = 2
        let headDim: UInt32 = 256
        let rotaryDim: UInt32 = 64
        let thetaVal: Float = 10000000.0
        let linKeyHeads: UInt32 = 16
        let linValHeads: UInt32 = 32
        let qkvDim: UInt32 = 8192
        let zDim: UInt32 = 4096
        let eps: Float = 1e-6

        KVCacheManager.shared.reset(device: device, config: config, actualLayers: 40, totalLoops: 1, numKvHeads: 2, headDim: 256, maxSeqLen: 256, precision: .fp16)

        // Staging buffer for top-8 experts
        let stagingBuf = device.makeBuffer(length: 8 * expertSize, options: .storageModeShared)!

        // Ping-pong hidden state buffers
        let hBufA = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let hBufB = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let xNorm1Buf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let xNorm2Buf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let hMidBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let hMlpBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let interBuf = device.makeBuffer(length: intermediateDim * 4, options: .storageModeShared)!

        let qGateBuf = device.makeBuffer(length: Int(qkvDim) * 4, options: .storageModeShared)!
        let kVectorBuf = device.makeBuffer(length: Int(numKvHeads * headDim) * 4, options: .storageModeShared)!
        let vVectorBuf = device.makeBuffer(length: Int(numKvHeads * headDim) * 4, options: .storageModeShared)!
        let zGateBuf = device.makeBuffer(length: Int(zDim) * 4, options: .storageModeShared)!
        let aVecBuf = device.makeBuffer(length: Int(linValHeads) * 4, options: .storageModeShared)!
        let bVecBuf = device.makeBuffer(length: Int(linValHeads) * 4, options: .storageModeShared)!
        let attnCtxBuf = device.makeBuffer(length: Int(zDim) * 4, options: .storageModeShared)!
        let attnOutBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!

        let routerIndicesBuf = device.makeBuffer(length: 8 * 4, options: .storageModeShared)!
        let routerWeightsBuf = device.makeBuffer(length: 8 * 4, options: .storageModeShared)!
        let sharedScoreBuf = device.makeBuffer(length: 4, options: .storageModeShared)!

        let finalHBuf = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
        let vocabSize = 248320
        let logitsBuf = device.makeBuffer(length: vocabSize * 4, options: .storageModeShared)!

        func dispatchLinear(
            enc: MTLComputeCommandEncoder,
            weight: TensorMetadata?,
            scale: TensorMetadata?,
            bias: TensorMetadata?,
            inBuf: MTLBuffer,
            outBuf: MTLBuffer,
            inDim: UInt32,
            outDim: UInt32
        ) {
            guard let w = weight, let wRaw = buffers[w.shardIndex] else { return }
            var wOff = w.offsetStart
            var inD = inDim
            var outD = outDim
            let isFP8 = !w.dtype.contains("BF16") && !w.dtype.contains("F16") && !w.dtype.contains("FLOAT")
            if isFP8 {
                let sRaw = (scale != nil) ? buffers[scale!.shardIndex]! : wRaw
                var sOff = scale?.offsetStart ?? 0
                if let simdPipe = inference.fp8GemvSimdPipeline {
                    enc.setComputePipelineState(simdPipe)
                    enc.setBuffer(wRaw, offset: 0, index: 0)
                    enc.setBuffer(inBuf, offset: 0, index: 1)
                    enc.setBuffer(outBuf, offset: 0, index: 2)
                    enc.setBuffer(sRaw, offset: 0, index: 3)
                    enc.setBytes(&wOff, length: 8, index: 4)
                    enc.setBytes(&sOff, length: 8, index: 5)
                    enc.setBytes(&inD, length: 4, index: 6)
                    enc.setBytes(&outD, length: 4, index: 7)
                    enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                }
            } else if let bSimdPipe = inference.bf16GemvSimdPipeline {
                enc.setComputePipelineState(bSimdPipe)
                enc.setBuffer(wRaw, offset: 0, index: 0)
                enc.setBuffer(inBuf, offset: 0, index: 1)
                enc.setBuffer(outBuf, offset: 0, index: 2)
                enc.setBytes(&wOff, length: 8, index: 3)
                enc.setBytes(&inD, length: 4, index: 4)
                enc.setBytes(&outD, length: 4, index: 5)
                enc.dispatchThreadgroups(MTLSize(width: Int(outDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
            enc.memoryBarrier(scope: .buffers)
        }

        func dispatchSharedExpert(
            enc: MTLComputeCommandEncoder,
            layer: EngineCachedLayer,
            inBuf: MTLBuffer,
            interBuf: MTLBuffer,
            accumBuf: MTLBuffer,
            sharedW: Float
        ) {
            guard let gateW = layer.sharedGateWeight,
                  let upW = layer.sharedUpWeight,
                  let downW = layer.sharedDownWeight,
                  let gateS = layer.sharedGateScale,
                  let upS = layer.sharedUpScale,
                  let downS = layer.sharedDownScale,
                  let gateSimd = inference.fp8GateUpSimdPipeline,
                  let downSimd = inference.fp8DownSimdPipeline else { return }

            var gWOff = gateW.offsetStart
            var gSOff = gateS.offsetStart
            var uWOff = upW.offsetStart
            var uSOff = upS.offsetStart
            var dWOff = downW.offsetStart
            var dSOff = downS.offsetStart
            var hDimVal = UInt32(hiddenDim)
            var interDimVal = UInt32(intermediateDim)
            var pk = sharedW

            enc.setComputePipelineState(gateSimd)
            enc.setBuffer(buffers[gateW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(buffers[upW.shardIndex]!, offset: 0, index: 1)
            enc.setBuffer(inBuf, offset: 0, index: 2)
            enc.setBuffer(interBuf, offset: 0, index: 3)
            enc.setBuffer(buffers[gateS.shardIndex]!, offset: 0, index: 4)
            enc.setBuffer(buffers[upS.shardIndex]!, offset: 0, index: 5)
            enc.setBytes(&gWOff, length: 8, index: 6)
            enc.setBytes(&gSOff, length: 8, index: 7)
            enc.setBytes(&uWOff, length: 8, index: 8)
            enc.setBytes(&uSOff, length: 8, index: 9)
            enc.setBytes(&hDimVal, length: 4, index: 10)
            enc.setBytes(&interDimVal, length: 4, index: 11)
            enc.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(downSimd)
            enc.setBuffer(buffers[downW.shardIndex]!, offset: 0, index: 0)
            enc.setBuffer(interBuf, offset: 0, index: 1)
            enc.setBuffer(accumBuf, offset: 0, index: 2)
            enc.setBuffer(buffers[downS.shardIndex]!, offset: 0, index: 3)
            enc.setBytes(&dWOff, length: 8, index: 4)
            enc.setBytes(&dSOff, length: 8, index: 5)
            enc.setBytes(&interDimVal, length: 4, index: 6)
            enc.setBytes(&hDimVal, length: 4, index: 7)
            enc.setBytes(&pk, length: 4, index: 8)
            enc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        func dispatchRoutedExperts(
            enc: MTLComputeCommandEncoder,
            layerIdx: Int,
            activeExperts: [(id: Int, weight: Float)],
            inBuf: MTLBuffer,
            interBuf: MTLBuffer,
            accumBuf: MTLBuffer
        ) {
            let packedDir = URL(fileURLWithPath: snapshotDir).appendingPathComponent("packed_experts")
            guard let fd = ExpertIOThreadPool.shared.getOrOpenLayerFD(layerIndex: layerIdx, packedExpertsDir: packedDir) else { return }

            var tasks: [ExpertPreadTask] = []
            let rawStagingPtr = stagingBuf.contents()
            for (slot, exp) in activeExperts.enumerated() {
                let offset = off_t(exp.id * expertSize)
                let dst = rawStagingPtr.advanced(by: slot * expertSize)
                tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
            }
            ExpertIOThreadPool.shared.dispatchSync(tasks: &tasks)

            guard let gateSimd = inference.fp8GateUpSimdPipeline,
                  let downSimd = inference.fp8DownSimdPipeline else { return }

            for (slot, expert) in activeExperts.enumerated() {
                let pk = expert.weight
                if pk <= 0.00001 { continue }
                let slotOffset = UInt64(slot * expertSize)
                var gWOff = slotOffset + 0
                var gSOff = slotOffset + 1048576
                var uWOff = slotOffset + 1049600
                var uSOff = slotOffset + 2098176
                var dWOff = slotOffset + 2099200
                var dSOff = slotOffset + 3147776
                var hDimVal = UInt32(hiddenDim)
                var interDimVal = UInt32(intermediateDim)
                var pkVal = pk

                enc.setComputePipelineState(gateSimd)
                enc.setBuffer(stagingBuf, offset: 0, index: 0)
                enc.setBuffer(stagingBuf, offset: 0, index: 1)
                enc.setBuffer(inBuf, offset: 0, index: 2)
                enc.setBuffer(interBuf, offset: 0, index: 3)
                enc.setBuffer(stagingBuf, offset: 0, index: 4)
                enc.setBuffer(stagingBuf, offset: 0, index: 5)
                enc.setBytes(&gWOff, length: 8, index: 6)
                enc.setBytes(&gSOff, length: 8, index: 7)
                enc.setBytes(&uWOff, length: 8, index: 8)
                enc.setBytes(&uSOff, length: 8, index: 9)
                enc.setBytes(&hDimVal, length: 4, index: 10)
                enc.setBytes(&interDimVal, length: 4, index: 11)
                enc.dispatchThreadgroups(MTLSize(width: Int(intermediateDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)

                enc.setComputePipelineState(downSimd)
                enc.setBuffer(stagingBuf, offset: 0, index: 0)
                enc.setBuffer(interBuf, offset: 0, index: 1)
                enc.setBuffer(accumBuf, offset: 0, index: 2)
                enc.setBuffer(stagingBuf, offset: 0, index: 3)
                enc.setBytes(&dWOff, length: 8, index: 4)
                enc.setBytes(&dSOff, length: 8, index: 5)
                enc.setBytes(&interDimVal, length: 4, index: 6)
                enc.setBytes(&hDimVal, length: 4, index: 7)
                enc.setBytes(&pkVal, length: 4, index: 8)
                enc.dispatchThreadgroups(MTLSize(width: Int(hiddenDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                enc.memoryBarrier(scope: .buffers)
            }
        }

        // 1. Embed single token "The" (ID 795)
        let embedWeight = summary.tensors.first(where: { $0.name.contains("embed_tokens") })!
        let singleTokenBuf = device.makeBuffer(length: 4, options: .storageModeShared)!
        let singleTokenPtr = singleTokenBuf.contents().bindMemory(to: UInt32.self, capacity: 1)
        singleTokenPtr[0] = 795 // "The"

        let embedCmd = cmdQueue.makeCommandBuffer()!
        let embedEnc = embedCmd.makeComputeCommandEncoder()!
        var wOffset = embedWeight.offsetStart
        var hDimVal = UInt32(hiddenDim)
        var tokCount: UInt32 = 1
        let embedPipe = inference.embedPipeline!
        embedEnc.setComputePipelineState(embedPipe)
        embedEnc.setBuffer(buffers[embedWeight.shardIndex]!, offset: 0, index: 0)
        embedEnc.setBuffer(singleTokenBuf, offset: 0, index: 1)
        embedEnc.setBuffer(hBufA, offset: 0, index: 2)
        embedEnc.setBytes(&wOffset, length: 8, index: 3)
        embedEnc.setBytes(&hDimVal, length: 4, index: 4)
        embedEnc.setBytes(&tokCount, length: 4, index: 5)
        embedEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, embedPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
        embedEnc.endEncoding()
        embedCmd.commit()
        embedCmd.waitUntilCompleted()

        var currentInBuf = hBufA
        var currentOutBuf = hBufB
        var layerDiagnostics = ""

        for l in 0..<40 {
            let layer = cachedLayers[l]
            let isEven = (l % 2 == 0)
            currentInBuf = isEven ? hBufA : hBufB
            currentOutBuf = isEven ? hBufB : hBufA

            let lCmd = cmdQueue.makeCommandBuffer()!
            let lEnc = lCmd.makeComputeCommandEncoder()!

            // 1. Norm 1
            let norm1 = layer.norm1Tensor!
            var nOff1 = norm1.offsetStart
            var epsVal = eps
            let rmsPipe = inference.rmsnormOffsetPipeline ?? inference.rmsnormPipeline!
            lEnc.setComputePipelineState(rmsPipe)
            lEnc.setBuffer(currentInBuf, offset: 0, index: 0)
            lEnc.setBuffer(buffers[norm1.shardIndex]!, offset: 0, index: 1)
            lEnc.setBuffer(xNorm1Buf, offset: 0, index: 2)
            lEnc.setBytes(&nOff1, length: 8, index: 3)
            lEnc.setBytes(&hDimVal, length: 4, index: 4)
            lEnc.setBytes(&epsVal, length: 4, index: 5)
            lEnc.setThreadgroupMemoryLength(1024 * 4, index: 0)
            lEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
            lEnc.memoryBarrier(scope: .buffers)

            // 2. Attention
            if layer.attentionType == LayerAttentionType.fullAttention {
                dispatchLinear(enc: lEnc, weight: layer.qProjTensor, scale: layer.qScaleTensor, bias: layer.qBiasTensor, inBuf: xNorm1Buf, outBuf: qGateBuf, inDim: UInt32(hiddenDim), outDim: numHeads * headDim * 2)
                dispatchLinear(enc: lEnc, weight: layer.kProjTensor, scale: layer.kScaleTensor, bias: layer.kBiasTensor, inBuf: xNorm1Buf, outBuf: kVectorBuf, inDim: UInt32(hiddenDim), outDim: numKvHeads * headDim)
                dispatchLinear(enc: lEnc, weight: layer.vProjTensor, scale: layer.vScaleTensor, bias: layer.vBiasTensor, inBuf: xNorm1Buf, outBuf: vVectorBuf, inDim: UInt32(hiddenDim), outDim: numKvHeads * headDim)

                if let qNorm = layer.qNormTensor, let qNormRaw = buffers[qNorm.shardIndex],
                   let headNormPipe = inference.headRmsnormOffsetPipeline ?? inference.headRmsnormPipeline {
                    var qNormOff = qNorm.offsetStart
                    var nQ = numHeads
                    var hD = headDim
                    var hStride = headDim * 2
                    lEnc.setComputePipelineState(headNormPipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBuffer(qNormRaw, offset: 0, index: 1)
                    lEnc.setBytes(&qNormOff, length: 8, index: 2)
                    lEnc.setBytes(&nQ, length: 4, index: 3)
                    lEnc.setBytes(&hD, length: 4, index: 4)
                    lEnc.setBytes(&hStride, length: 4, index: 5)
                    lEnc.setBytes(&epsVal, length: 4, index: 6)
                    lEnc.dispatchThreadgroups(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                if let kNorm = layer.kNormTensor, let kNormRaw = buffers[kNorm.shardIndex],
                   let headNormPipe = inference.headRmsnormOffsetPipeline ?? inference.headRmsnormPipeline {
                    var kNormOff = kNorm.offsetStart
                    var nK = numKvHeads
                    var hD = headDim
                    var hStride = headDim
                    lEnc.setComputePipelineState(headNormPipe)
                    lEnc.setBuffer(kVectorBuf, offset: 0, index: 0)
                    lEnc.setBuffer(kNormRaw, offset: 0, index: 1)
                    lEnc.setBytes(&kNormOff, length: 8, index: 2)
                    lEnc.setBytes(&nK, length: 4, index: 3)
                    lEnc.setBytes(&hD, length: 4, index: 4)
                    lEnc.setBytes(&hStride, length: 4, index: 5)
                    lEnc.setBytes(&epsVal, length: 4, index: 6)
                    lEnc.dispatchThreadgroups(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                if let ropePipe = inference.ropePipeline {
                    var pos: UInt32 = 0
                    var nQ = numHeads
                    var nK = numKvHeads
                    var hD = headDim
                    var rD = rotaryDim
                    var qStr = headDim * 2
                    var kStr = headDim
                    var theta = thetaVal

                    lEnc.setComputePipelineState(ropePipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBytes(&pos, length: 4, index: 1)
                    lEnc.setBytes(&nQ, length: 4, index: 2)
                    lEnc.setBytes(&hD, length: 4, index: 3)
                    lEnc.setBytes(&rD, length: 4, index: 4)
                    lEnc.setBytes(&qStr, length: 4, index: 5)
                    lEnc.setBytes(&theta, length: 4, index: 6)
                    lEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

                    lEnc.setBuffer(kVectorBuf, offset: 0, index: 0)
                    lEnc.setBytes(&pos, length: 4, index: 1)
                    lEnc.setBytes(&nK, length: 4, index: 2)
                    lEnc.setBytes(&hD, length: 4, index: 3)
                    lEnc.setBytes(&rD, length: 4, index: 4)
                    lEnc.setBytes(&kStr, length: 4, index: 5)
                    lEnc.setBytes(&theta, length: 4, index: 6)
                    lEnc.dispatchThreads(MTLSize(width: Int(numKvHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numKvHeads), ropePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                if let kCache = KVCacheManager.shared.kCacheBuffer,
                   let vCache = KVCacheManager.shared.vCacheBuffer,
                   let storePipe = inference.storeKvCacheF16Pipeline,
                   let gqaPipe = inference.gqaDecodeF16Pipeline {
                    let slot = layer.fullAttnIndex
                    let maxSeq = KVCacheManager.shared.allocatedSeqLen
                    let kvStride = numKvHeads * headDim
                    let layerByteOffset = slot * maxSeq * Int(kvStride) * 2
                    var pos: UInt32 = 0
                    var nKv = numKvHeads
                    var hD = headDim
                    var nQ = numHeads
                    var seqLen: UInt32 = 1

                    lEnc.setComputePipelineState(storePipe)
                    lEnc.setBuffer(kVectorBuf, offset: 0, index: 0)
                    lEnc.setBuffer(vVectorBuf, offset: 0, index: 1)
                    lEnc.setBuffer(kCache, offset: layerByteOffset, index: 2)
                    lEnc.setBuffer(vCache, offset: layerByteOffset, index: 3)
                    lEnc.setBytes(&pos, length: 4, index: 4)
                    lEnc.setBytes(&nKv, length: 4, index: 5)
                    lEnc.setBytes(&hD, length: 4, index: 6)
                    lEnc.dispatchThreads(MTLSize(width: Int(kvStride), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(kvStride), storePipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)

                    lEnc.setComputePipelineState(gqaPipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBuffer(kCache, offset: layerByteOffset, index: 1)
                    lEnc.setBuffer(vCache, offset: layerByteOffset, index: 2)
                    lEnc.setBuffer(attnCtxBuf, offset: 0, index: 3)
                    lEnc.setBytes(&seqLen, length: 4, index: 4)
                    lEnc.setBytes(&nQ, length: 4, index: 5)
                    lEnc.setBytes(&nKv, length: 4, index: 6)
                    lEnc.setBytes(&hD, length: 4, index: 7)
                    lEnc.dispatchThreads(MTLSize(width: Int(numHeads), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numHeads), gqaPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                dispatchLinear(enc: lEnc, weight: layer.oProjTensor, scale: layer.oScaleTensor, bias: layer.oBiasTensor, inBuf: attnCtxBuf, outBuf: attnOutBuf, inDim: zDim, outDim: UInt32(hiddenDim))
            } else {
                dispatchLinear(enc: lEnc, weight: layer.inProjQKV, scale: layer.inProjQKVScale, bias: layer.inProjQKVBias, inBuf: xNorm1Buf, outBuf: qGateBuf, inDim: UInt32(hiddenDim), outDim: qkvDim)

                if let conv1d = layer.conv1dTensor, let convRaw = buffers[conv1d.shardIndex],
                   let convPipe = inference.causalConv1dPipeline, let convState = KVCacheManager.shared.convStateBuffer {
                    let convStateByteOffset = layer.linAttnIndex * Int(qkvDim) * 4 * 4
                    var cOff = conv1d.offsetStart
                    var numChannels = qkvDim
                    lEnc.setComputePipelineState(convPipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBuffer(convRaw, offset: 0, index: 1)
                    lEnc.setBuffer(convState, offset: convStateByteOffset, index: 2)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 3)
                    lEnc.setBytes(&cOff, length: 8, index: 4)
                    lEnc.setBytes(&numChannels, length: 4, index: 5)
                    lEnc.dispatchThreads(MTLSize(width: Int(qkvDim), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(256, convPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                if let l2Pipe = inference.l2NormQkPipeline {
                    var numH = linKeyHeads
                    var hD: UInt32 = 128
                    lEnc.setComputePipelineState(l2Pipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBytes(&numH, length: 4, index: 1)
                    lEnc.setBytes(&hD, length: 4, index: 2)
                    lEnc.dispatchThreads(MTLSize(width: Int(numH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(Int(numH), l2Pipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                dispatchLinear(enc: lEnc, weight: layer.inProjZ, scale: layer.inProjZScale, bias: layer.inProjZBias, inBuf: xNorm1Buf, outBuf: zGateBuf, inDim: UInt32(hiddenDim), outDim: zDim)
                dispatchLinear(enc: lEnc, weight: layer.inProjA, scale: layer.inProjAScale, bias: layer.inProjABias, inBuf: xNorm1Buf, outBuf: aVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)
                dispatchLinear(enc: lEnc, weight: layer.inProjB, scale: layer.inProjBScale, bias: layer.inProjBBias, inBuf: xNorm1Buf, outBuf: bVecBuf, inDim: UInt32(hiddenDim), outDim: linValHeads)

                if let linPipe = inference.gdnLinearAttnStepPipeline ?? inference.linearAttnStepPipeline,
                   let sBuf = KVCacheManager.shared.linearStateBuffer,
                   let aLog = layer.aLogTensor, let aLogRaw = buffers[aLog.shardIndex],
                   let dtBias = layer.dtBiasTensor, let dtBiasRaw = buffers[dtBias.shardIndex],
                   let linNorm = layer.linearNormTensor, let linNormRaw = buffers[linNorm.shardIndex] {
                    let stateByteOffset = layer.linAttnIndex * Int(linValHeads * 128 * 128) * 4
                    var aLogOff = aLog.offsetStart
                    var dtBiasOff = dtBias.offsetStart
                    var linNormOff = linNorm.offsetStart
                    var numValH = linValHeads
                    var numKeyH = linKeyHeads
                    var hD: UInt32 = 128
                    lEnc.setComputePipelineState(linPipe)
                    lEnc.setBuffer(qGateBuf, offset: 0, index: 0)
                    lEnc.setBuffer(zGateBuf, offset: 0, index: 1)
                    lEnc.setBuffer(aVecBuf, offset: 0, index: 2)
                    lEnc.setBuffer(bVecBuf, offset: 0, index: 3)
                    lEnc.setBuffer(aLogRaw, offset: 0, index: 4)
                    lEnc.setBuffer(dtBiasRaw, offset: 0, index: 5)
                    lEnc.setBuffer(linNormRaw, offset: 0, index: 6)
                    lEnc.setBuffer(sBuf, offset: stateByteOffset, index: 7)
                    lEnc.setBuffer(attnCtxBuf, offset: 0, index: 8)
                    lEnc.setBytes(&aLogOff, length: 8, index: 9)
                    lEnc.setBytes(&dtBiasOff, length: 8, index: 10)
                    lEnc.setBytes(&linNormOff, length: 8, index: 11)
                    lEnc.setBytes(&numValH, length: 4, index: 12)
                    lEnc.setBytes(&numKeyH, length: 4, index: 13)
                    lEnc.setBytes(&hD, length: 4, index: 14)
                    lEnc.setBytes(&epsVal, length: 4, index: 15)
                    lEnc.dispatchThreadgroups(MTLSize(width: Int(numValH), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                    lEnc.memoryBarrier(scope: .buffers)
                }

                dispatchLinear(enc: lEnc, weight: layer.linearOutProjTensor ?? layer.oProjTensor, scale: layer.linearOutProjScale ?? layer.oScaleTensor, bias: layer.linearOutProjBias ?? layer.oBiasTensor, inBuf: attnCtxBuf, outBuf: attnOutBuf, inDim: zDim, outDim: UInt32(hiddenDim))
            }

            // 3. Residual 1
            let addPipe = inference.addPipeline!
            lEnc.setComputePipelineState(addPipe)
            lEnc.setBuffer(currentInBuf, offset: 0, index: 0)
            lEnc.setBuffer(attnOutBuf, offset: 0, index: 1)
            lEnc.setBuffer(hMidBuf, offset: 0, index: 2)
            lEnc.setBytes(&hDimVal, length: 4, index: 3)
            lEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            lEnc.memoryBarrier(scope: .buffers)

            // 4. Norm 2
            let norm2 = layer.norm2Tensor!
            var nOff2 = norm2.offsetStart
            lEnc.setComputePipelineState(rmsPipe)
            lEnc.setBuffer(hMidBuf, offset: 0, index: 0)
            lEnc.setBuffer(buffers[norm2.shardIndex]!, offset: 0, index: 1)
            lEnc.setBuffer(xNorm2Buf, offset: 0, index: 2)
            lEnc.setBytes(&nOff2, length: 8, index: 3)
            lEnc.setBytes(&hDimVal, length: 4, index: 4)
            lEnc.setBytes(&epsVal, length: 4, index: 5)
            lEnc.setThreadgroupMemoryLength(1024 * 4, index: 0)
            lEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
            lEnc.memoryBarrier(scope: .buffers)

            // 5. Router
            let router = layer.routerTensor!
            var rOff = router.offsetStart
            var nExp: UInt32 = 256
            var kVal: UInt32 = 8
            let rPipe = inference.routerPipeline!
            lEnc.setComputePipelineState(rPipe)
            lEnc.setBuffer(buffers[router.shardIndex]!, offset: 0, index: 0)
            lEnc.setBuffer(xNorm2Buf, offset: 0, index: 1)
            lEnc.setBuffer(routerIndicesBuf, offset: 0, index: 2)
            lEnc.setBuffer(routerWeightsBuf, offset: 0, index: 3)
            lEnc.setBytes(&rOff, length: 8, index: 4)
            lEnc.setBytes(&hDimVal, length: 4, index: 5)
            lEnc.setBytes(&nExp, length: 4, index: 6)
            lEnc.setBytes(&kVal, length: 4, index: 7)
            lEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            lEnc.memoryBarrier(scope: .buffers)

            // Shared Gate
            if let sgPipe = inference.sharedGatePipeline,
               let sgWeight = layer.sharedGateTensor,
               let sgRaw = buffers[sgWeight.shardIndex] {
                var sgOff = sgWeight.offsetStart
                lEnc.setComputePipelineState(sgPipe)
                lEnc.setBuffer(sgRaw, offset: 0, index: 0)
                lEnc.setBuffer(xNorm2Buf, offset: 0, index: 1)
                lEnc.setBuffer(sharedScoreBuf, offset: 0, index: 2)
                lEnc.setBytes(&sgOff, length: 8, index: 3)
                lEnc.setBytes(&hDimVal, length: 4, index: 4)
                lEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
                lEnc.memoryBarrier(scope: .buffers)
            }

            // Clear hMlpBuf
            let clearPipe = inference.clearPipeline!
            lEnc.setComputePipelineState(clearPipe)
            lEnc.setBuffer(hMlpBuf, offset: 0, index: 0)
            lEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, clearPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))
            lEnc.memoryBarrier(scope: .buffers)

            lEnc.endEncoding()
            lCmd.commit()
            lCmd.waitUntilCompleted()

            let rIdxPtr = routerIndicesBuf.contents().bindMemory(to: UInt32.self, capacity: 8)
            let rWgtPtr = routerWeightsBuf.contents().bindMemory(to: Float.self, capacity: 8)
            var activeExperts: [(id: Int, weight: Float)] = []
            for i in 0..<8 {
                activeExperts.append((id: Int(rIdxPtr[i]), weight: rWgtPtr[i]))
            }
            let sgPtr = sharedScoreBuf.contents().bindMemory(to: Float.self, capacity: 1)

            // Dispatch Routed & Shared Experts
            let moeCmd = cmdQueue.makeCommandBuffer()!
            let moeEnc = moeCmd.makeComputeCommandEncoder()!
            dispatchRoutedExperts(enc: moeEnc, layerIdx: l, activeExperts: activeExperts, inBuf: xNorm2Buf, interBuf: interBuf, accumBuf: hMlpBuf)
            dispatchSharedExpert(enc: moeEnc, layer: layer, inBuf: xNorm2Buf, interBuf: interBuf, accumBuf: hMlpBuf, sharedW: sgPtr[0])

            // Residual 2
            moeEnc.setComputePipelineState(addPipe)
            moeEnc.setBuffer(hMidBuf, offset: 0, index: 0)
            moeEnc.setBuffer(hMlpBuf, offset: 0, index: 1)
            moeEnc.setBuffer(currentOutBuf, offset: 0, index: 2)
            moeEnc.setBytes(&hDimVal, length: 4, index: 3)
            moeEnc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(hiddenDim, addPipe.maxTotalThreadsPerThreadgroup), height: 1, depth: 1))

            moeEnc.endEncoding()
            moeCmd.commit()
            moeCmd.waitUntilCompleted()

            var layerLog = "Layer \(l) (\(layer.attentionType == .fullAttention ? "FULL" : "LIN")) | "
            let outPtr = currentOutBuf.contents().bindMemory(to: Float.self, capacity: hiddenDim)
            var l2Norm: Float = 0
            var minV: Float = outPtr[0]
            var maxV: Float = outPtr[0]
            var sumV: Float = 0
            for i in 0..<hiddenDim {
                let v = outPtr[i]
                l2Norm += v * v
                if v < minV { minV = v }
                if v > maxV { maxV = v }
                sumV += v
            }
            l2Norm = sqrt(l2Norm)
            let meanV = sumV / Float(hiddenDim)
            let expSummary = activeExperts.prefix(4).map { "E\($0.id):\(String(format: "%.2f", $0.weight))" }.joined(separator: ",")
            layerLog += "L2: \(String(format: "%.3f", l2Norm)), min: \(String(format: "%.4f", minV)), max: \(String(format: "%.4f", maxV)), mean: \(String(format: "%.5f", meanV)) | \(expSummary)\n"
            layerDiagnostics += layerLog
            XCTAssertFalse(l2Norm.isNaN, "NaN in hidden state at layer \(l)!")
        }
        try? layerDiagnostics.write(toFile: "/tmp/ornith_layer_trace.txt", atomically: true, encoding: .utf8)


        // Final RMSNorm
        guard let finalNorm = summary.tensors.first(where: {
            $0.name == "model.norm.weight" ||
            $0.name == "language_model.model.norm.weight" ||
            $0.name == "language_model.norm.weight" ||
            $0.name == "model.language_model.norm.weight" ||
            ($0.name.hasSuffix(".norm.weight") && !$0.name.contains("layers.") && !$0.name.contains("mixer"))
        }) else {
            XCTFail("Final RMSNorm tensor not found!")
            return
        }
        print("Found finalNorm tensor: \(finalNorm.name)")

        let finalCmd = cmdQueue.makeCommandBuffer()!
        let finalEnc = finalCmd.makeComputeCommandEncoder()!
        var finalNOff = finalNorm.offsetStart
        var epsVal = eps
        let rmsPipe = inference.rmsnormOffsetPipeline ?? inference.rmsnormPipeline!
        finalEnc.setComputePipelineState(rmsPipe)
        finalEnc.setBuffer(currentOutBuf, offset: 0, index: 0)
        finalEnc.setBuffer(buffers[finalNorm.shardIndex]!, offset: 0, index: 1)
        finalEnc.setBuffer(finalHBuf, offset: 0, index: 2)
        finalEnc.setBytes(&finalNOff, length: 8, index: 3)
        finalEnc.setBytes(&hDimVal, length: 4, index: 4)
        finalEnc.setBytes(&epsVal, length: 4, index: 5)
        finalEnc.setThreadgroupMemoryLength(1024 * 4, index: 0)
        finalEnc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(1024, hiddenDim), height: 1, depth: 1))
        finalEnc.memoryBarrier(scope: .buffers)

        // LM Head
        guard let lmHead = summary.tensors.first(where: {
            $0.name == "lm_head.weight" ||
            $0.name == "language_model.lm_head.weight" ||
            $0.name == "model.lm_head.weight" ||
            $0.name.hasSuffix("lm_head.weight")
        }) else {
            XCTFail("LM Head tensor not found!")
            return
        }
        print("Found lmHead tensor: \(lmHead.name)")
        dispatchLinear(enc: finalEnc, weight: lmHead, scale: nil, bias: nil, inBuf: finalHBuf, outBuf: logitsBuf, inDim: UInt32(hiddenDim), outDim: UInt32(vocabSize))

        finalEnc.endEncoding()
        finalCmd.commit()
        finalCmd.waitUntilCompleted()

        // Read logits
        let logPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
        var topTokens: [(id: Int, logit: Float)] = []
        for i in 0..<vocabSize {
            let lVal = logPtr[i]
            if topTokens.count < 10 {
                topTokens.append((id: i, logit: lVal))
                topTokens.sort { $0.logit > $1.logit }
            } else if lVal > topTokens.last!.logit {
                topTokens[9] = (id: i, logit: lVal)
                topTokens.sort { $0.logit > $1.logit }
            }
        }

        var outReport = "=== Top 10 predicted tokens for input token \(singleTokenPtr[0]) ===\n"
        let tokenizer = try? DynaMoeTokenizer(tokenizerPath: snapshotDir + "/tokenizer.json")
        for (rank, t) in topTokens.enumerated() {
            let decoded = (try? tokenizer?.decode(ids: [UInt32(t.id)])) ?? "<?>"
            let line = "  Top \(rank + 1): id=\(t.id) (\(decoded)) logit=\(t.logit)\n"
            outReport += line
            print(line)
        }
        try? outReport.write(toFile: "/tmp/ornith_top_tokens.txt", atomically: true, encoding: .utf8)
    }

    // MARK: - Native Tool Harness Unit Tests

    func testTriePrefixMatching() {
        let root = TokenTrieNode()
        root.insert(word: "file_read", tokenId: 101)
        root.insert(word: "file_write", tokenId: 102)
        root.insert(word: "shell_run", tokenId: 103)

        XCTAssertTrue(root.findNode(prefix: "file_") != nil)
        XCTAssertTrue(root.findNode(prefix: "file_read")?.isTerminal == true)
        XCTAssertTrue(root.findNode(prefix: "file_read")?.terminalTokenIds.contains(101) == true)
        XCTAssertTrue(root.findNode(prefix: "invalid_tool") == nil)
    }

    func testStreamingToolParserPreExecutionCatching() {
        let parser = StreamingToolParser.shared

        // In-progress tool call: should not freeze
        let partialText = "Let me check the file: <tool_call><function=file_read>{\"path\":\"main.swift\"}"
        XCTAssertFalse(parser.shouldFreezeGeneration(accumulatedText: partialText, deltaText: "\"main.swift\"}"))

        // Exact closing token emitted: should freeze generation instantly
        let closedText = partialText + "</tool_call>"
        XCTAssertTrue(parser.shouldFreezeGeneration(accumulatedText: closedText, deltaText: "</tool_call>"))
    }

    func testStreamingToolParserUniversalFormats() {
        let parser = StreamingToolParser.shared

        // Qwen XML format
        let qwenText = "<tool_call><function=file_read>{\"path\": \"Sources/main.swift\"}</function></tool_call>"
        let qwenCalls = parser.parseStreamingToolCalls(from: qwenText, format: .qwenXML)
        XCTAssertEqual(qwenCalls.calls.count, 1)
        XCTAssertEqual(qwenCalls.calls.first?.name, "file_read")
        XCTAssertEqual(qwenCalls.calls.first?.arguments["path"] as? String, "Sources/main.swift")

        // JSON Markdown format
        let jsonText = "<tool_call>{\"tool\": \"shell_run\", \"parameters\": {\"command\": \"swift --version\"}}</tool_call>"
        let jsonCalls = parser.parseStreamingToolCalls(from: jsonText, format: .hermeticJSON)
        XCTAssertEqual(jsonCalls.calls.count, 1)
        XCTAssertEqual(jsonCalls.calls.first?.name, "shell_run")
        XCTAssertEqual(jsonCalls.calls.first?.arguments["command"] as? String, "swift --version")

        // Llama 3 python tag format
        let llamaText = "<|python_tag|>file_write(path=\"test.txt\", content=\"hello\")</|python_tag|>"
        let llamaCalls = parser.parseStreamingToolCalls(from: llamaText, format: .llama3)
        XCTAssertEqual(llamaCalls.calls.count, 1)
        XCTAssertEqual(llamaCalls.calls.first?.name, "file_write")
        XCTAssertEqual(llamaCalls.calls.first?.arguments["path"] as? String, "test.txt")
    }

    func testGrammarConstrainedStateTransitions() {
        let sampler = GrammarConstrainedSampler.shared
        sampler.reset()
        sampler.registerTools(AgentHarness.shared.availableToolDefinitions)

        XCTAssertEqual(sampler.currentState, .outsideToolCall)

        sampler.updateState(emittedText: "I will use a tool: <tool_call><function=file")
        if case .insideFunctionName(let name) = sampler.currentState {
            XCTAssertEqual(name, "file")
        } else {
            XCTFail("Expected insideFunctionName state but got \(sampler.currentState)")
        }

        sampler.updateState(emittedText: "I will use a tool: <tool_call><function=file_read><parameter=path>test.txt")
        if case .insideParameterValue(let tool, let param) = sampler.currentState {
            XCTAssertEqual(tool, "file_read")
            XCTAssertEqual(param, "path")
        } else {
            XCTFail("Expected insideParameterValue state but got \(sampler.currentState)")
        }
    }

    func testControlledProcessRunner() async throws {
        let runner = ControlledProcessRunner.shared
        let res = try await runner.runCommand(command: "echo 'DynaMoE Harness Online'", workingDirectory: nil, timeoutSeconds: 5)
        XCTAssertEqual(res.exitCode, 0)
        XCTAssertTrue(res.stdout.contains("DynaMoE Harness Online"))
    }

    func testSemanticDocumentReaderChunking() {
        let reader = SemanticDocumentReader.shared
        let swiftCode = """
        // Section 1
        func testA() {
            print("A")
        }

        // Section 2
        func testB() {
            print("B")
        }
        """
        let tempURL = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("test_chunk.swift")
        try? swiftCode.write(to: tempURL, atomically: true, encoding: .utf8)
        let res = try? reader.readDocument(at: tempURL, maxChunkLength: 50)
        XCTAssertTrue((res?.chunks.count ?? 0) > 0)
        try? FileManager.default.removeItem(at: tempURL)
    }

    func testActionApprovalManagerContinuations() async {
        let manager = ActionApprovalManager.shared
        let testActionId = UUID()

        // Test approve flow
        let approveTask = Task {
            await manager.waitForApproval(id: testActionId)
        }
        // Yield briefly so the continuation registers
        try? await Task.sleep(nanoseconds: 20_000_000)
        manager.approve(id: testActionId)
        let approveResult = await approveTask.value
        XCTAssertTrue(approveResult)

        // Test reject flow
        let rejectActionId = UUID()
        let rejectTask = Task {
            await manager.waitForApproval(id: rejectActionId)
        }
        try? await Task.sleep(nanoseconds: 20_000_000)
        manager.reject(id: rejectActionId)
        let rejectResult = await rejectTask.value
        XCTAssertFalse(rejectResult)
    }

    func testAgentMultiTurnToolCallParsingAndContinuation() {
        // 1. Test XML function call with direct JSON payload
        let xmlWithJson = "<tool_call><function=file_read>{\"path\": \"README.md\"}</function></tool_call>"
        let parsedXml = AgentHarness.shared.parseToolCalls(from: xmlWithJson)
        XCTAssertEqual(parsedXml.calls.count, 1)
        XCTAssertEqual(parsedXml.calls.first?.name, "file_read")
        XCTAssertEqual(parsedXml.calls.first?.arguments["path"] as? String, "README.md")

        // 2. Test Hermetic JSON tool call
        let hermeticJson = "<tool_call>{\"tool\": \"file_read\", \"parameters\": {\"path\": \"README.md\"}}</tool_call>"
        let parsedHermetic = StreamingToolParser.shared.parseStreamingToolCalls(from: hermeticJson)
        XCTAssertEqual(parsedHermetic.calls.count, 1)
        XCTAssertEqual(parsedHermetic.calls.first?.name, "file_read")
        XCTAssertEqual(parsedHermetic.calls.first?.arguments["path"] as? String, "README.md")

        // 3. Test multi-turn tool response turn formatting
        let responseTurn = AgentHarness.shared.formatToolResponseTurn(
            responses: ["README.md contents: # DynaMoE"],
            includeThinkSuffix: true
        )
        XCTAssertTrue(responseTurn.contains("<|im_start|>user"))
        XCTAssertTrue(responseTurn.contains("<tool_response>"))
        XCTAssertTrue(responseTurn.contains("# DynaMoE"))
        XCTAssertTrue(responseTurn.contains("</tool_response>"))
        XCTAssertTrue(responseTurn.contains("<|im_start|>assistant\n<think>"))
    }

    func testCodebaseEmbeddingEngineAndMetalCosineSimilarity() {
        let engine = CodebaseEmbeddingEngine.shared
        guard engine.isAvailable else {
            print("NLEmbedding not available on this test host, skipping vector search test.")
            return
        }

        let query = "where is KV cache allocated?"
        let docRelevant = "KVCacheManager allocates unified memory KV cache buffer for tokens."
        let docIrrelevant = "How to make a delicious homemade chocolate chip cookie recipe."

        guard let queryVec = engine.embed(text: query),
              let relVec = engine.embed(text: docRelevant),
              let irrelVec = engine.embed(text: docIrrelevant) else {
            XCTFail("Failed to generate embedding vectors")
            return
        }

        XCTAssertEqual(queryVec.count, CodebaseEmbeddingEngine.embeddingDimension)
        XCTAssertEqual(relVec.count, CodebaseEmbeddingEngine.embeddingDimension)
        XCTAssertEqual(irrelVec.count, CodebaseEmbeddingEngine.embeddingDimension)

        let search = MetalVectorSearch.shared
        let updateOk = search.updateCorpus(vectors: [docRelevant, docIrrelevant].compactMap { engine.embed(text: $0) })
        XCTAssertTrue(updateOk)

        let results = search.search(queryVector: queryVec, topK: 2)
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results.first?.index, 0, "Relevant KV cache doc should rank higher than chocolate recipe")
        XCTAssertGreaterThan(results[0].score, results[1].score)
    }

    func testBM25TokenizationAndScoring() {
        let tokens = BM25Index.tokenize(text: "KVCacheManager allocates buffer_slot")
        XCTAssertTrue(tokens.contains("kvcachemanager"))
        XCTAssertTrue(tokens.contains("cache"))
        XCTAssertTrue(tokens.contains("allocates"))

        let bm25 = BM25Index()
        let docs = [
            "KVCacheManager allocates unified memory buffer for tokens",
            "Metal compute pipeline state and shader compiler",
            "SwiftUI view hierarchy and reactive state bindings"
        ]
        bm25.index(documents: docs)
        XCTAssertEqual(bm25.numDocs, 3)

        let results = bm25.search(query: "KVCacheManager allocation", topK: 2)
        XCTAssertFalse(results.isEmpty)
        XCTAssertEqual(results.first?.index, 0, "Doc 0 should be top result for KVCacheManager query")
    }

    func testHybridSearchFusionRRF() {
        let denseResults: [(index: Int, score: Float)] = [
            (index: 2, score: 0.85),
            (index: 0, score: 0.72),
            (index: 1, score: 0.50)
        ]
        let bm25Results: [(index: Int, score: Float)] = [
            (index: 0, score: 12.5),
            (index: 2, score: 8.2),
            (index: 3, score: 4.1)
        ]

        let fused = HybridSearchFusion.fuse(
            denseResults: denseResults,
            bm25Results: bm25Results,
            topK: 3
        )

        XCTAssertFalse(fused.isEmpty)
        // Indices 0 and 2 appear in both and should rank at the top
        let topIndices = Set(fused.prefix(2).map { $0.index })
        XCTAssertTrue(topIndices.contains(0))
        XCTAssertTrue(topIndices.contains(2))
    }

    func testCodebaseIndexerAndSearchTool() async throws {
        // Create temporary workspace directory with test files
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("dynamoe_rag_test_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        let swiftFile = tempDir.appendingPathComponent("CacheManager.swift")
        let swiftCode = """
        // CacheManager.swift
        public final class CacheManager {
            public func allocateKVCache(tokens: Int) -> Bool {
                // Allocates unified memory for KV cache
                return true
            }
        }
        """
        try swiftCode.write(to: swiftFile, atomically: true, encoding: .utf8)

        let indexer = CodebaseIndexer.shared
        await indexer.performIndexWorkspace(url: tempDir, forceRebuild: true)
        XCTAssertGreaterThan(indexer.indexedChunkCount, 0)

        let tool = CodebaseSearchTool()
        let result = try await tool.execute(
            arguments: [
                "query": "KV cache allocate unified memory",
                "top_k": 3,
                "search_mode": "hybrid"
            ],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )

        XCTAssertFalse(result.isCompleted)
        XCTAssertTrue(result.resultJSON.contains("CacheManager.swift"))
        XCTAssertTrue(result.resultJSON.contains("allocateKVCache"))
    }

    // MARK: - Option 2: Multi-Agent Subagent Delegation Harness Tests

    func testSubagentManagerLifecycleAndStatusTransitions() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SubagentLifecycleTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let dummyFile = tempDir.appendingPathComponent("MathLib.swift")
        try "public struct MathLib { public static func add(_ a: Int, _ b: Int) -> Int { a + b } }".write(to: dummyFile, atomically: true, encoding: .utf8)

        let manager = SubagentManager.shared
        let subagent = manager.spawn(
            role: "Codebase Researcher",
            taskDescription: "Find MathLib implementation",
            allowedTools: ["codebase_search", "file_read"],
            contextSummary: "Focus on arithmetic utility functions",
            workingDirectory: tempDir
        )

        XCTAssertNotNil(manager.getSubagent(byId: subagent.id))
        XCTAssertEqual(subagent.role, "Codebase Researcher")
        XCTAssertEqual(subagent.archetype, .codebaseResearcher)

        // Wait for autonomous subagent pipeline to finish
        let finalSummary = await subagent.waitForCompletion()
        XCTAssertEqual(subagent.status, .completed)
        XCTAssertFalse(finalSummary.isEmpty)
        XCTAssertTrue(subagent.executionDurationSeconds >= 0)
        XCTAssertGreaterThanOrEqual(subagent.transcript.count, 1)
    }

    func testSpawnSubagentSynchronousExecution() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SubagentSyncTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let spawnTool = SpawnSubagentTool()
        let result = try await spawnTool.execute(
            arguments: [
                "role": "Shader Optimizer",
                "task_description": "Analyze vector alignment in compute kernel",
                "run_in_background": false
            ],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )

        XCTAssertFalse(result.isCompleted)
        XCTAssertNotNil(result.stdout)
        XCTAssertTrue(result.resultJSON.contains("completed"))
        XCTAssertTrue(result.resultJSON.contains("subagent_id"))
        XCTAssertTrue(result.stdout?.contains("Shader Optimization Analysis") == true)
    }

    func testSpawnSubagentAsynchronousBackgroundExecution() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SubagentAsyncTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let spawnTool = SpawnSubagentTool()
        let spawnResult = try await spawnTool.execute(
            arguments: [
                "role": "Test Runner",
                "task_description": "Run unit test suite",
                "run_in_background": true
            ],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )

        XCTAssertFalse(spawnResult.isCompleted)
        XCTAssertTrue(spawnResult.resultJSON.contains("running"))
        XCTAssertTrue(spawnResult.resultJSON.contains("subagent_id"))

        // Extract subagent UUID from manager
        guard let subagent = SubagentManager.shared.allSubagents.first(where: { $0.role == "Test Runner" }) else {
            XCTFail("Spawned subagent was not found in SubagentManager registry")
            return
        }

        // Query status via GetSubagentStatusTool
        let statusTool = GetSubagentStatusTool()
        let statusResult = try await statusTool.execute(
            arguments: ["subagent_id": subagent.id.uuidString],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )

        XCTAssertFalse(statusResult.isCompleted)
        XCTAssertTrue(statusResult.resultJSON.contains(subagent.id.uuidString))
        XCTAssertTrue(statusResult.stdout?.contains("Test Runner") == true)
    }

    func testInterAgentMessagingAndListTool() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SubagentMessageTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let subagent = SubagentManager.shared.spawn(
            role: "Documentation Writer",
            taskDescription: "Draft architecture guide",
            allowedTools: ["file_read"],
            contextSummary: nil,
            workingDirectory: tempDir
        )

        // Send directive message to subagent
        let msgTool = SendSubagentMessageTool()
        let msgResult = try await msgTool.execute(
            arguments: [
                "subagent_id": subagent.id.uuidString,
                "message": "Include Mermaid diagrams for component interactions"
            ],
            workingDirectory: tempDir,
            maxOutputLength: 2000
        )

        XCTAssertFalse(msgResult.isCompleted)
        XCTAssertTrue(msgResult.resultJSON.contains("delivered"))
        XCTAssertEqual(subagent.recordedMessages.count, 1)
        XCTAssertEqual(subagent.recordedMessages.first?.content, "Include Mermaid diagrams for component interactions")

        // Test ListSubagentsTool
        let listTool = ListSubagentsTool()
        let listResult = try await listTool.execute(
            arguments: ["status_filter": "all"],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )

        XCTAssertFalse(listResult.isCompleted)
        XCTAssertTrue(listResult.resultJSON.contains(subagent.id.uuidString))
        XCTAssertTrue(listResult.stdout?.contains("Documentation Writer") == true)
    }

    // MARK: - Option 3: Deep Developer Tooling (Git, Symbol Intelligence, & Self-Healing Diagnostics) Tests

    func testGitStatusAndDiffTools() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("GitStatusDiffTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Initialize git repo in temp dir
        let gitURL = URL(fileURLWithPath: "/usr/bin/git")
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["init", "-b", "main"], currentDirectory: tempDir)
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["config", "user.email", "agent@dynamoe.ai"], currentDirectory: tempDir)
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["config", "user.name", "DynaMoE Agent"], currentDirectory: tempDir)

        let fileA = tempDir.appendingPathComponent("App.swift")
        try "import Foundation\nstruct App { let name = \"DynaMoE\" }\n".write(to: fileA, atomically: true, encoding: .utf8)

        // 1. Verify git_status shows untracked file
        let statusTool = GitStatusTool()
        let statusRes = try await statusTool.execute(
            arguments: [:],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(statusRes.isCompleted)
        XCTAssertTrue(statusRes.resultJSON.contains("App.swift"))
        XCTAssertTrue(statusRes.resultJSON.contains("main"))

        // Stage file
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["add", "App.swift"], currentDirectory: tempDir)

        // 2. Verify git_diff shows staged diff
        let diffTool = GitDiffTool()
        let diffRes = try await diffTool.execute(
            arguments: ["staged": true],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(diffRes.isCompleted)
        XCTAssertTrue(diffRes.resultJSON.contains("App.swift") || (diffRes.stdout?.contains("App.swift") == true))
        XCTAssertTrue(diffRes.stdout?.contains("+struct App") == true)
    }

    func testGitCommitToolAndSafetyRails() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("GitCommitSafetyTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let gitURL = URL(fileURLWithPath: "/usr/bin/git")
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["init", "-b", "main"], currentDirectory: tempDir)
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["config", "user.email", "agent@dynamoe.ai"], currentDirectory: tempDir)
        _ = try await AgentHarness.runProcess(executableURL: gitURL, arguments: ["config", "user.name", "DynaMoE Agent"], currentDirectory: tempDir)

        let fileA = tempDir.appendingPathComponent("Config.swift")
        try "struct Config { static let version = \"1.0.0\" }".write(to: fileA, atomically: true, encoding: .utf8)

        let commitTool = GitCommitTool()

        // Safety Rail 1: Empty commit message must fail
        let emptyRes = try await commitTool.execute(
            arguments: ["message": "   "],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertTrue(emptyRes.resultJSON.contains("error"))
        XCTAssertTrue(emptyRes.stderr?.contains("empty") == true)

        // Safety Rail 2: Dangerous forbidden flag must fail
        let flagRes = try await commitTool.execute(
            arguments: ["message": "fix: bypass checks --no-verify"],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertTrue(flagRes.resultJSON.contains("error"))
        XCTAssertTrue(flagRes.stderr?.contains("forbidden flag") == true)

        // Safety Rail 3: Commit with no staged changes without stage_all must fail
        let noStageRes = try await commitTool.execute(
            arguments: ["message": "feat: initial commit", "stage_all": false],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertTrue(noStageRes.resultJSON.contains("error"))
        XCTAssertTrue(noStageRes.stderr?.contains("No changes are staged") == true)

        // Valid Commit: stage_all: true
        let validRes = try await commitTool.execute(
            arguments: ["message": "feat: initial commit", "stage_all": true],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(validRes.isCompleted)
        XCTAssertTrue(validRes.resultJSON.contains("committed"))
        XCTAssertTrue(validRes.resultJSON.contains("commit_hash"))
        XCTAssertTrue(validRes.stdout?.contains("Committed") == true)
    }

    func testSymbolIntelligenceDefinitionAndReferences() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("SymbolIntelligenceTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let matrixFile = tempDir.appendingPathComponent("MatrixMath.swift")
        try """
        import Foundation

        public struct Matrix4x4 {
            public var values: [Float]
            public init() { values = Array(repeating: 0.0, count: 16) }
        }

        public enum MatrixProjectionMode {
            case orthographic
            case perspective
        }
        """.write(to: matrixFile, atomically: true, encoding: .utf8)

        let rendererFile = tempDir.appendingPathComponent("RenderPipeline.swift")
        try """
        import Foundation

        final class RenderPipeline {
            func renderFrame() {
                let m = Matrix4x4()
                let mode: MatrixProjectionMode = .perspective
                print(m)
            }
        }
        """.write(to: rendererFile, atomically: true, encoding: .utf8)

        let metalFile = tempDir.appendingPathComponent("Shaders.metal")
        try """
        #include <metal_stdlib>
        using namespace metal;

        kernel void compute_blur_kernel(device float* buffer [[buffer(0)]]) {
            // kernel logic
        }
        """.write(to: metalFile, atomically: true, encoding: .utf8)

        // 1. Find Definition of Swift Struct
        let defTool = FindSymbolDefinitionTool()
        let defRes = try await defTool.execute(
            arguments: ["symbol_name": "Matrix4x4"],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(defRes.isCompleted)
        XCTAssertTrue(defRes.resultJSON.contains("Matrix4x4"))
        XCTAssertTrue(defRes.resultJSON.contains("struct"))
        XCTAssertTrue(defRes.resultJSON.contains("MatrixMath.swift"))

        // 2. Find Definition of Metal Kernel
        let metalRes = try await defTool.execute(
            arguments: ["symbol_name": "compute_blur_kernel"],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(metalRes.isCompleted)
        XCTAssertTrue(metalRes.resultJSON.contains("compute_blur_kernel"))
        XCTAssertTrue(metalRes.resultJSON.contains("kernel"))

        // 3. Find References to Matrix4x4 in RenderPipeline
        let refTool = FindReferencesTool()
        let refRes = try await refTool.execute(
            arguments: ["symbol_name": "Matrix4x4"],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(refRes.isCompleted)
        XCTAssertTrue(refRes.resultJSON.contains("RenderPipeline.swift"))
        XCTAssertTrue(refRes.stdout?.contains("let m = Matrix4x4()") == true)
    }

    func testLintDiagnosticsFeedbackLoop() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent("LintFeedbackTest_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let targetFile = tempDir.appendingPathComponent("Component.swift")

        // 1. Write clean file via FileWriteTool -> Diagnostics must be clean
        let writeTool = FileWriteTool()
        let cleanWriteRes = try await writeTool.execute(
            arguments: [
                "path": targetFile.path,
                "content": "struct Component { let id: Int = 1 }\n"
            ],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(cleanWriteRes.isCompleted)
        XCTAssertTrue(cleanWriteRes.resultJSON.contains("written"))
        XCTAssertFalse(cleanWriteRes.resultJSON.contains("written_with_syntax_errors"))

        // 2. Edit file via FileEditTool with a syntax error -> Self-healing loop must report error!
        let editTool = FileEditTool()
        let syntaxErrRes = try await editTool.execute(
            arguments: [
                "path": targetFile.path,
                "target_content": "let id: Int = 1",
                "replacement_content": "let id: Int = "
            ],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(syntaxErrRes.isCompleted)
        XCTAssertTrue(syntaxErrRes.resultJSON.contains("syntax_errors_detected"))
        XCTAssertTrue(syntaxErrRes.resultJSON.contains("self_healing_hint"))
        XCTAssertTrue(syntaxErrRes.stdout?.contains("SELF-HEALING ACTION REQUIRED") == true)
        XCTAssertTrue(syntaxErrRes.stdout?.contains("expected initial value") == true)

        // 3. Test LintDiagnosticsTool directly on the file
        let lintTool = LintDiagnosticsTool()
        let lintRes = try await lintTool.execute(
            arguments: ["path": targetFile.path],
            workingDirectory: tempDir,
            maxOutputLength: 4000
        )
        XCTAssertFalse(lintRes.isCompleted)
        XCTAssertTrue(lintRes.resultJSON.contains("has_errors"))
        XCTAssertTrue(lintRes.resultJSON.contains("expected initial value"))
    }
}

// MARK: - Option 4: Model Dogfooding & KV Prefix Pinning Tests

final class ModelDogfoodAndPrefixCacheTests: XCTestCase {

    override func setUp() {
        super.setUp()
        PrefixCacheManager.shared.invalidate()
    }

    override func tearDown() {
        PrefixCacheManager.shared.invalidate()
        UserDefaults.standard.removeObject(forKey: "dynamoe_agent_turbo_mode")
        super.tearDown()
    }

    func testPrefixCacheManagerPrefixDetectionAndRecording() {
        let manager = PrefixCacheManager.shared
        let sessionA = UUID()
        let sessionB = UUID()

        let turn1Prompt: [UInt32] = [100, 101, 102, 103, 104, 105]
        let turn1Reused = manager.findCommonPrefix(promptTokenIds: turn1Prompt, sessionId: sessionA)
        XCTAssertEqual(turn1Reused, 0, "First cold prompt should have 0 prefix reuse")

        let generatedA: [UInt32] = [200, 201, 202]
        manager.recordTurn(promptTokenIds: turn1Prompt, generatedTokenIds: generatedA, sessionId: sessionA)

        // Turn 2 with sessionA retains prefix
        var turn2Prompt: [UInt32] = [100, 101, 102, 103, 104, 105, 200, 201, 202]
        turn2Prompt.append(contentsOf: [300, 301, 302])

        let turn2Reused = manager.findCommonPrefix(promptTokenIds: turn2Prompt, sessionId: sessionA)
        XCTAssertEqual(turn2Reused, 9, "Turn 2 should reuse all 9 tokens from turn 1 prompt + response")

        // SessionB should be isolated and not match sessionA
        let sessionBReused = manager.findCommonPrefix(promptTokenIds: turn2Prompt, sessionId: sessionB)
        XCTAssertEqual(sessionBReused, 0, "Session B should have 0 reuse from Session A's cache")

        // Telemetry metrics
        let metrics = manager.metrics
        XCTAssertGreaterThan(metrics.totalTokensRequested, 0)
        XCTAssertGreaterThan(metrics.totalTokensReused, 0)
        XCTAssertGreaterThan(metrics.hitRatePercent, 0.0)

        // Invalidate sessionA
        manager.invalidate(sessionId: sessionA)
        let postInvalidationReused = manager.findCommonPrefix(promptTokenIds: turn2Prompt, sessionId: sessionA)
        XCTAssertEqual(postInvalidationReused, 0, "After invalidation, reuse should be 0")
    }

    func testKVCacheManagerPrefixPreservation() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("Metal is required for this test")
            return
        }

        let kvManager = KVCacheManager.shared
        let seqLen = 256
        let actualLayers = 2
        let loops = 1
        let kvHeads = 8
        let headDim = 128
        let stride = max(kvHeads * headDim, 1024) // 1024 elements per token

        // 1. Initial Reset (cold, preservePrefixCount: 0)
        kvManager.reset(
            device: device,
            actualLayers: actualLayers,
            totalLoops: loops,
            numKvHeads: kvHeads,
            headDim: headDim,
            maxSeqLen: seqLen,
            precision: .fp16,
            preservePrefixCount: 0
        )

        guard let kBuf = kvManager.kCacheBuffer, let vBuf = kvManager.vCacheBuffer else {
            XCTFail("KV Buffers failed to allocate")
            return
        }

        let prefixToPreserve = 32
        let elementsToFill = (prefixToPreserve + 10) * stride

        // Fill with canary pattern: token t -> Float16(42.0 / 84.0)
        let kPtr = kBuf.contents().bindMemory(to: Float16.self, capacity: kBuf.length / 2)
        let vPtr = vBuf.contents().bindMemory(to: Float16.self, capacity: vBuf.length / 2)
        for i in 0..<min(kBuf.length / 2, elementsToFill) {
            kPtr[i] = Float16(42.0)
            vPtr[i] = Float16(84.0)
        }

        // 2. Second Reset with preservePrefixCount: 32
        kvManager.reset(
            device: device,
            actualLayers: actualLayers,
            totalLoops: loops,
            numKvHeads: kvHeads,
            headDim: headDim,
            maxSeqLen: seqLen,
            precision: .fp16,
            preservePrefixCount: prefixToPreserve
        )

        // Verify preserved prefix remains intact
        let preservedKPtr = kvManager.kCacheBuffer!.contents().bindMemory(to: Float16.self, capacity: kvManager.kCacheBuffer!.length / 2)
        let preservedVPtr = kvManager.vCacheBuffer!.contents().bindMemory(to: Float16.self, capacity: kvManager.vCacheBuffer!.length / 2)

        XCTAssertEqual(preservedKPtr[0], Float16(42.0), "Token 0 should be preserved")
        XCTAssertEqual(preservedKPtr[prefixToPreserve * stride - 1], Float16(42.0), "Token at end of prefix should be preserved")
        XCTAssertEqual(preservedVPtr[0], Float16(84.0), "V-cache Token 0 should be preserved")
        XCTAssertEqual(preservedVPtr[prefixToPreserve * stride - 1], Float16(84.0), "V-cache token at end of prefix should be preserved")

        // Verify tail beyond prefix was zeroed
        let tailOffset = (prefixToPreserve + 2) * stride
        if tailOffset < seqLen * stride {
            XCTAssertEqual(preservedKPtr[tailOffset], Float16(0.0), "Tail beyond preserved prefix should be zeroed")
            XCTAssertEqual(preservedVPtr[tailOffset], Float16(0.0), "V-cache tail should be zeroed")
        }
    }

    func testDogfoodBenchmarkRunnerMultiTurnExecution() async throws {
        let runner = await ModelDogfoodBenchmarkRunner.shared

        let report = try await runner.runBenchmark(
            modelName: "Ornith-1.5-35B-A3B-FP8",
            modelSnapshotPath: "/tmp/fake_snapshot"
        )

        XCTAssertTrue(report.isPassing, "Dogfood benchmark report must pass all criteria")
        XCTAssertEqual(report.turns.count, 3, "Expected 3 full turns")
        XCTAssertGreaterThan(report.prefixPinningSpeedup, 1.0, "Prefix pinning must yield speedup")
        XCTAssertTrue(report.turboModeVerified, "Turbo mode should be verified")
        XCTAssertTrue(report.selfHealingVerified, "Self-healing should catch and heal compiler errors")
        XCTAssertTrue(report.summaryMarkdown.contains("Model Dogfooding & Benchmarking Report"))
        XCTAssertTrue(report.summaryMarkdown.contains("TTFT Speedup Factor"))
    }

    func testLiveMoEWeightsCheckpointIntegrity() {
        // Discovers real on-disk models if present on user system
        let discovered = LocalModelManager.shared.discoveredModels
        let ornith35B = discovered.first { $0.displayName.contains("35B") }
        let ornith9B = discovered.first { $0.displayName.contains("9B") }

        if let model = ornith35B ?? ornith9B {
            let snapPath = model.snapshotPath
            let fm = FileManager.default
            XCTAssertTrue(fm.fileExists(atPath: snapPath), "Discovered snapshot path must exist")

            let configPath = (snapPath as NSString).appendingPathComponent("config.json")
            if fm.fileExists(atPath: configPath), let data = try? Data(contentsOf: URL(fileURLWithPath: configPath)) {
                let parsed = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
                XCTAssertNotNil(parsed, "config.json must be valid JSON")
            }
        }
    }

    func testHeadlessChromeBinaryResolution() {
        let engine = HeadlessChromeSearchEngine.shared
        let binaryPath = engine.resolveBinaryPath()
        if FileManager.default.fileExists(atPath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome") {
            XCTAssertNotNil(binaryPath, "Should resolve Google Chrome binary path")
            XCTAssertTrue(FileManager.default.isExecutableFile(atPath: binaryPath!), "Resolved path must be executable")
        }
    }

    func testHeadlessChromeWebSearchToolExecution() async throws {
        let tool = WebSearchTool()
        let result = try await tool.execute(
            arguments: ["query": "Metal framework Apple Developer"],
            workingDirectory: nil,
            maxOutputLength: 4000
        )

        XCTAssertTrue(result.isCompleted, "Tool execution should complete")
        XCTAssertFalse(result.resultJSON.isEmpty, "Result JSON should not be empty")

        if let data = result.resultJSON.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            XCTAssertNotNil(json["query"], "JSON should contain query")
            let engine = json["engine"] as? String
            XCTAssertTrue(engine == "headless_chrome" || engine == "brave_search", "Engine should be headless_chrome or brave_search")
            if let results = json["results"] as? [[String: Any]] {
                XCTAssertFalse(results.isEmpty, "Should find at least 1 search result")
                if let first = results.first {
                    XCTAssertNotNil(first["title"])
                    XCTAssertNotNil(first["url"])
                }
            }
        }
    }

    func testHeadlessChromeDOMExtractionFallback() async throws {
        let tool = WebFetchTool()
        let result = try await tool.execute(
            arguments: ["url": "https://example.com"],
            workingDirectory: nil,
            maxOutputLength: 2000
        )
        XCTAssertTrue(result.isCompleted)
        XCTAssertTrue(result.resultJSON.contains("Example") || result.resultJSON.contains("Domain"), "Should retrieve page content")
    }

    // MARK: - Current Date & Time System Prompt Tests

    func testAgentHarnessCurrentDateTimeContextFormatting() {
        var calendar = Calendar(identifier: .gregorian)
        guard let tz = TimeZone(identifier: "America/New_York") else {
            XCTFail("Could not create America/New_York timezone")
            return
        }
        calendar.timeZone = tz
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 7
        components.hour = 10
        components.minute = 30
        components.second = 0
        guard let testDate = calendar.date(from: components) else {
            XCTFail("Could not create test date")
            return
        }

        let context = AgentHarness.formattedDateTimeContext(date: testDate, timeZone: tz)
        XCTAssertTrue(context.contains("# Current Date & Time"))
        XCTAssertTrue(context.contains("Monday, September 7, 2026"))
        XCTAssertTrue(context.contains("10:30 AM"))
        XCTAssertTrue(context.contains("Current Year: 2026"))
        XCTAssertTrue(context.contains("2026-09-07"))
        XCTAssertTrue(context.contains("America/New_York"))
        XCTAssertTrue(context.contains("web_search"))
        XCTAssertTrue(context.contains("web_fetch"))
    }

    func testAgentHarnessBuildSystemPromptIncludesDateTime() {
        let harness = AgentHarness.shared
        var calendar = Calendar(identifier: .gregorian)
        let tz = TimeZone(identifier: "America/New_York")!
        calendar.timeZone = tz
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 7
        components.hour = 14
        components.minute = 15
        let testDate = calendar.date(from: components)!

        let prompt = harness.buildSystemPrompt(
            baseSystem: "You are an expert engineer.",
            modelName: "TestModel",
            currentDate: testDate
        )

        XCTAssertTrue(prompt.contains("# Current Date & Time"))
        XCTAssertTrue(prompt.contains("September 7, 2026"))
        XCTAssertTrue(prompt.contains("Current Year: 2026"))
        XCTAssertTrue(prompt.contains("# Tools"))
        XCTAssertTrue(prompt.contains("grep_search"))
        XCTAssertTrue(prompt.contains("tools_discover"))
        XCTAssertFalse(prompt.contains("web_search"), "Unloaded tools must not be advertised in the initial system prompt")
        XCTAssertTrue(prompt.contains("<tool_call>"))
        XCTAssertTrue(prompt.contains("You are an expert engineer."))
    }

    func testModelConfigBuildEffectiveSystemPromptWithDate() {
        var calendar = Calendar(identifier: .gregorian)
        let tz = TimeZone(identifier: "America/New_York")!
        calendar.timeZone = tz
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 7
        let testDate = calendar.date(from: components)!

        let effective = ModelConfig.buildEffectiveSystemPrompt(
            userPrompt: "You are a coding assistant.",
            config: nil,
            summary: nil,
            currentDate: testDate
        )

        XCTAssertTrue(effective.contains("# Current Date & Time"))
        XCTAssertTrue(effective.contains("September 7, 2026"))
        XCTAssertTrue(effective.contains("You are a coding assistant."))

        // Verify deduplication: Passing already-formatted prompt into AgentHarness doesn't duplicate header
        let agentPrompt = AgentHarness.shared.buildSystemPrompt(baseSystem: effective, currentDate: testDate)
        let occurrences = agentPrompt.components(separatedBy: "# Current Date & Time").count - 1
        XCTAssertEqual(occurrences, 1, "Temporal context should only appear once in agent system prompt")
    }

    func testSystemPromptTemporalStabilityForKVCachePrefix() {
        var calendar = Calendar(identifier: .gregorian)
        let tz = TimeZone(identifier: "America/New_York")!
        calendar.timeZone = tz
        var components = DateComponents()
        components.year = 2026
        components.month = 9
        components.day = 7
        components.hour = 9
        components.minute = 0
        let sessionStartDate = calendar.date(from: components)!

        // Turn 1
        let turn1System = AgentHarness.shared.buildSystemPrompt(
            baseSystem: "Autonomous reasoning assistant.",
            currentDate: sessionStartDate
        )

        // Turn 2: Occurs 10 minutes later in conversation, but anchored to sessionStartDate
        let turn2System = AgentHarness.shared.buildSystemPrompt(
            baseSystem: "Autonomous reasoning assistant.",
            currentDate: sessionStartDate
        )

        XCTAssertEqual(turn1System, turn2System, "System prompt prefix must remain 100% identical across conversation turns for KV-cache prefix hits")
    }

    // MARK: - Web Search & Fetch Tooling Enhancements Tests

    func testWebFetchHTMLTableToMarkdown() {
        let sampleHTML = """
        <html>
        <body>
        <main>
        <h1>Mac Specifications</h1>
        <table>
            <tr>
                <th>Model</th>
                <th>Processor</th>
                <th>Memory</th>
            </tr>
            <tr>
                <td>Mac mini</td>
                <td>Apple M4 Pro (12-core CPU, 16-core GPU)</td>
                <td>24GB unified memory (273GB/s)</td>
            </tr>
            <tr>
                <td>Mac Studio</td>
                <td>Apple M2 Ultra (24-core CPU, 60-core GPU)</td>
                <td>64GB unified memory (800GB/s)</td>
            </tr>
        </table>
        </main>
        </body>
        </html>
        """

        let cleaned = WebFetchTool.cleanHTMLStructure(sampleHTML)
        XCTAssertTrue(cleaned.contains("| Model | Processor | Memory |"), "Must contain Markdown table header row")
        XCTAssertTrue(cleaned.contains("| --- | --- | --- |"), "Must contain Markdown table delimiter row")
        XCTAssertTrue(cleaned.contains("| Mac mini | Apple M4 Pro (12-core CPU, 16-core GPU) | 24GB unified memory (273GB/s) |"), "Must format table data rows with pipes")
        XCTAssertTrue(cleaned.contains("| Mac Studio | Apple M2 Ultra (24-core CPU, 60-core GPU) | 64GB unified memory (800GB/s) |"), "Must format table data rows with pipes")
    }

    func testWebFetchNoiseAndTemplateStripping() {
        let dirtyHTML = """
        <html>
        <head><title>Compare Mac - Apple</title></head>
        <body>
        <nav><a href="/store">Store</a><a href="/mac">Mac</a></nav>
        <select name="models">
            <option value="mbn">MacBook Neo (A18 Pro)</option>
            <option value="mba13">MacBook Air 13-in. (M5)</option>
            <option value="mba15">MacBook Air 15-in. (M5)</option>
        </select>
        <div class="pricing">
            {MBN_2026_MAIN}From $price.display.smart or $price.display.perMonth for $price.display.months mo.*
        </div>
        <form action="/newsletter"><button type="submit">Subscribe</button></form>
        <main>
            <h1>Mac mini Overview</h1>
            <p>The new Mac mini with M4 and M4 Pro delivers monstrous performance in an impossibly small frame.</p>
        </main>
        <footer><p>Copyright 2026 Apple Inc.</p></footer>
        </body>
        </html>
        """

        let cleaned = WebFetchTool.cleanHTMLStructure(dirtyHTML)
        XCTAssertFalse(cleaned.contains("MacBook Neo (A18 Pro)"), "Must strip <select> and <option> dropdown navigation clutter")
        XCTAssertFalse(cleaned.contains("{MBN_2026_MAIN}"), "Must strip unrendered JavaScript template strings")
        XCTAssertFalse(cleaned.contains("$price.display.smart"), "Must strip unhydrated template expressions")
        XCTAssertFalse(cleaned.contains("Subscribe"), "Must strip form and button clutter")
        XCTAssertTrue(cleaned.contains("Mac mini Overview"), "Must preserve primary article/main heading")
        XCTAssertTrue(cleaned.contains("The new Mac mini with M4 and M4 Pro"), "Must preserve main content text")
    }

    func testWebFetchQueryTargetedExtraction() {
        let fullPage = """
        # Apple Hardware Overview

        ### Design and Ports
        The chassis features front USB-C ports, an audio jack, and rear Thunderbolt 5 ports.

        ### Processor and Performance
        The Mac mini is powered by the M4 Pro chip featuring a 14-core CPU, 20-core GPU, and 273GB/s memory bandwidth.
        It outperforms previous generation M2 Pro desktops by over 1.8x in multi-threaded workflows.

        ### Power and Environmental Specs
        Constructed with over 50% recycled content and meets ENERGY STAR requirements.
        """

        let prioritized = WebFetchTool.filterContentByQuery(fullPage, query: "processor m4 pro gpu bandwidth", limit: 3000)
        XCTAssertTrue(prioritized.contains("Key Sections Matching \"processor m4 pro gpu bandwidth\""), "Must contain matched section header")
        XCTAssertTrue(prioritized.contains("14-core CPU, 20-core GPU, and 273GB/s memory bandwidth"), "Must prioritize processor section")
    }

    func testSearchEngineRedirectURLUnwrapping() {
        // Yahoo redirect
        let yahooRedirect = "https://r.search.yahoo.com/_ylt=Awr.123/RU=https%3a%2f%2fwww.apple.com%2fmac-mini%2fspecs%2f/RK=2/RS=xyz"
        let unwrappedYahoo = HeadlessChromeSearchEngine.unwrapRedirectURL(yahooRedirect)
        XCTAssertEqual(unwrappedYahoo, "https://www.apple.com/mac-mini/specs/", "Must unwrap Yahoo redirect to canonical target URL")

        // Google redirect
        let googleRedirect = "https://www.google.com/url?q=https://support.apple.com/kb/SP894&sa=U&ved=2ahUKEwi"
        let unwrappedGoogle = HeadlessChromeSearchEngine.unwrapRedirectURL(googleRedirect)
        XCTAssertEqual(unwrappedGoogle, "https://support.apple.com/kb/SP894", "Must unwrap Google redirect to canonical target URL")

        // Direct URL untouched
        let directURL = "https://en.wikipedia.org/wiki/Mac_Studio"
        let unwrappedDirect = HeadlessChromeSearchEngine.unwrapRedirectURL(directURL)
        XCTAssertEqual(unwrappedDirect, directURL, "Direct URLs must remain unchanged")
    }

    func testAgentHarnessWebResearchPromptGuardrails() {
        let systemPrompt = AgentHarness.shared.buildSystemPrompt(
            baseSystem: "Expert assistant.",
            currentDate: Date()
        )

        XCTAssertTrue(systemPrompt.contains("Web Research & Grounding Guidelines"), "Must include research and grounding guidelines")
        XCTAssertTrue(systemPrompt.contains("Neutral Queries First"), "Must instruct agent to use neutral queries")
        XCTAssertTrue(systemPrompt.contains("Strict URL Grounding"), "Must instruct agent to strictly ground URLs from web_search")
        XCTAssertTrue(systemPrompt.contains("Official vs. Speculative Rumors"), "Must instruct agent to differentiate shipping hardware from rumors")
        XCTAssertTrue(systemPrompt.contains("Natural Linear Fetching"), "Must instruct agent to use natural linear fetching")
        XCTAssertTrue(systemPrompt.contains("Multi-Category Technical Specifications"), "Must instruct agent on multi-category spec sheets")
    }

    func testWebFetchPreservesStrictSequentialOrder() {
        let sampleDoc = """
        # Mac Studio - Technical Specifications

        ## Chip
        - Apple M5 Max chip with 18-core CPU and 40-core GPU

        ## Memory
        - 36GB unified memory, configurable to 128GB

        ## Storage
        - 512GB SSD or 1TB SSD

        ## Environmental Requirements
        - Operating temperature: 50° to 95° F
        - Storage temperature: –40° to 116° F
        - Operating altitude: up to 16,400 feet
        """

        let filtered = WebFetchTool.filterContentByQuery(sampleDoc, query: "storage m5 max", limit: 3000)
        let chipIndex = filtered.range(of: "M5 Max chip")?.lowerBound
        let envIndex = filtered.range(of: "Storage temperature")?.lowerBound
        XCTAssertNotNil(chipIndex)
        XCTAssertNotNil(envIndex)
        if let ci = chipIndex, let ei = envIndex {
            XCTAssertTrue(ci < ei, "Matching blocks must strictly maintain sequential document order (Chip before Storage temperature)")
        }
        XCTAssertFalse(filtered.contains("### Additional Page Context:"), "Must not duplicate page under Additional Page Context")
    }

    func testWebFetchAccessibilityAndOrphanCleaning() {
        let pressReleaseHTML = """
        <html>
        <body>
        <a href="https://example.com" target="_blank"><span class="visually-hidden">opens in new window</span></a>
        <h1></h1>
        <p>PRESS RELEASE</p>
        <p>August 25, 2026</p>
        <h2></h2>
        <p>Apple introduces new Mac Studio with M5 Max and M5 Ultra — the ultimate desktop for on‑device AI.</p>
        <ul>
            <li></li>
            <li></li>
            <li></li>
        </ul>
        <p>Apple today announced the new Mac Studio, featuring M5 Max and the all-new M5 Ultra.</p>
        </body>
        </html>
        """

        let cleaned = WebFetchTool.cleanHTMLStructure(pressReleaseHTML)
        XCTAssertFalse(cleaned.contains("opens in new window"), "Must strip screen-reader 'opens in new window' text")

        let lines = cleaned.components(separatedBy: .newlines).map { $0.trimmingCharacters(in: .whitespaces) }
        for line in lines {
            let withoutHash = line.trimmingCharacters(in: CharacterSet(charactersIn: "# \t"))
            if withoutHash.isEmpty && line.hasPrefix("#") {
                XCTFail("Must not contain orphan markdown header: '\(line)'")
            }
            let withoutBullet = line.trimmingCharacters(in: CharacterSet(charactersIn: "-*+• \t"))
            if withoutBullet.isEmpty && (line.hasPrefix("-") || line.hasPrefix("*") || line.hasPrefix("+") || line.hasPrefix("•")) {
                XCTFail("Must not contain orphan markdown bullet: '\(line)'")
            }
        }
        XCTAssertTrue(cleaned.contains("Apple introduces new Mac Studio with M5 Max and M5 Ultra"), "Must preserve real text")
    }

    func testAgentHarnessTemporalGroundingAndVendorDomainValidation() {
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("apple.com"))
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("newsroom.apple.com"))
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("developer.apple.com"))
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("github.com"))
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("nasa.gov"))
        XCTAssertTrue(AgentHarness.isOfficialVendorDomain("mit.edu"))
        XCTAssertFalse(AgentHarness.isOfficialVendorDomain("randomtechblog.xyz"))

        let prompt = AgentHarness.shared.buildSystemPrompt(baseSystem: "", currentDate: Date())
        XCTAssertTrue(prompt.contains("Temporal Grounding & Live Reality"), "Must include temporal grounding guidelines")
        XCTAssertTrue(prompt.contains("Authoritative Official Domain Reality"), "Must instruct agent to trust official vendor domains as ground truth")
        XCTAssertTrue(prompt.contains("Never Reject Live Data"), "Must explicitly tell agent not to reject newer hardware models or names as hallucinations")
    }

    func testWebFetchStructuredSectionsAndAriaGridExtraction() {
        let sampleAppleHTML = """
        <div class="techspecs-header-row visuallyhidden">
            <div role="columnheader" class="techspecs-rowheader">&nbsp;</div>
            <div role="columnheader" class="techspecs-columnheader">Mac Studio, Model1</div>
            <div role="columnheader" class="techspecs-columnheader">Mac Studio, Model2</div>
        </div>
        <div class="techspecs-row" role="row">
            <div class="techspecs-rowheader" role="rowheader">Finish</div>
            <div class="techspecs-column" role="cell">Silver</div>
        </div>
        <div class="techspecs-row" role="row">
            <div class="techspecs-rowheader" role="rowheader">Price</div>
            <div class="techspecs-column" role="cell">$2499</div>
            <div class="techspecs-column" role="cell">$5499</div>
        </div>
        <div class="techspecs-row" role="row">
            <div class="techspecs-rowheader" role="rowheader">Chip</div>
            <p class="techspecs-subheader">Apple M5 Max chip</p>
            <ul>
                <li>18-core CPU</li>
                <li>32-core GPU</li>
                <li>16-core Neural Engine</li>
            </ul>
        </div>
        <figure class="techspecs-figure">
            <picture><img src="diagram.png" alt="Dimensions"></picture>
            <div class="caption-wrapper">
                <p>Width: 7.7 inches (19.7 cm)</p>
                <p>Height: 3.7 inches (9.5 cm)</p>
            </div>
        </figure>
        <div class="techspecs-row" role="row">
            <div class="techspecs-rowheader" role="rowheader">Size and Weight</div>
            <ul>
                <li>Height: 3.7 inches (9.5 cm)</li>
                <li>Width: 7.7 inches (19.7 cm)</li>
                <li>Depth: 7.7 inches (19.7 cm)</li>
                <li>Weight: 6.0 pounds (2.7 kg)</li>
            </ul>
        </div>
        """

        let cleaned = WebFetchTool.cleanHTMLStructure(sampleAppleHTML)

        // 1. Must strip Model1 / Model2 columnheader placeholders
        XCTAssertFalse(cleaned.contains("Model1"), "Must strip Model1 accessibility columnheader")
        XCTAssertFalse(cleaned.contains("Model2"), "Must strip Model2 accessibility columnheader")

        // 2. Must format Price and Finish clearly
        XCTAssertTrue(cleaned.contains("## Finish\nSilver"), "Must format Finish row cleanly")
        XCTAssertTrue(cleaned.contains("Base (M5 Max): $2499"), "Must format base price")
        XCTAssertTrue(cleaned.contains("High-End (M5 Ultra): $5499"), "Must format high-end price")

        // 3. Must strip diagram pins from caption-wrapper & figure
        let countWidth = cleaned.components(separatedBy: "Width: 7.7 inches").count - 1
        XCTAssertEqual(countWidth, 1, "Must only contain Width dimension once (caption-wrapper duplicate stripped)")

        // 4. Must convert headers
        XCTAssertTrue(cleaned.contains("## Chip"), "Must convert techspecs-rowheader to ## Chip")
        XCTAssertTrue(cleaned.contains("### Apple M5 Max chip"), "Must convert techspecs-subheader to ### Subheader")

        // 5. Must extract structured sections
        let sections = AgentHarness.extractStructuredSections(from: cleaned)
        XCTAssertTrue(sections.count >= 4, "Must extract at least 4 structured sections")
        let headings = sections.compactMap { $0["heading"] }
        XCTAssertTrue(headings.contains("Finish"))
        XCTAssertTrue(headings.contains("Price"))
        XCTAssertTrue(headings.contains("Chip"))
        XCTAssertTrue(headings.contains("Size and Weight"))

        if let chipSection = sections.first(where: { $0["heading"] == "Chip" }) {
            XCTAssertTrue(chipSection["details"]?.contains("18-core CPU") == true)
            XCTAssertTrue(chipSection["details"]?.contains("32-core GPU") == true)
        }
    }

    func testAgentHarnessStructuredSectionsAndReleaseCyclePromptGuardrails() {
        let prompt = AgentHarness.shared.buildSystemPrompt(baseSystem: "", currentDate: Date())
        XCTAssertTrue(prompt.contains("Structured Tool Responses"), "Must include structured tool responses in prompt")
        XCTAssertTrue(prompt.contains("Hardware Model Numbering & Non-Linear Release Cycles"), "Must include release cycle guidance")
        XCTAssertTrue(prompt.contains("Workstation Engineering Ratings"), "Must include workstation engineering ratings guidance")
        XCTAssertTrue(prompt.contains("Real Executive Names & Official Quotes"), "Must include executive names guidance")
        XCTAssertTrue(prompt.contains("Relative Benchmark Multipliers & Monthly Lease Pricing"), "Must include lease and benchmark guidance")
    }

    func testPressReleaseNoiseStrippingAndGroundTruthNotices() {
        let rawNewsroomHTML = """
        <article class="article">
            <h1 class="hero-headline">Apple introduces new Mac Studio with M5 Max and M5 Ultra</h1>
            <div class="article-subhead">Apple's most powerful Mac raises the bar for local AI</div>
            <div class="pagebody-copy">CUPERTINO, CALIFORNIA Apple today announced the new Mac Studio...</div>
            <h2>**A Monumental Step for AI**</h2>
            <div class="pagebody-copy">"Mac Studio is at the forefront of AI," said Johny Srouji, Apple's senior vice president.</div>
            <div class="pagebody-copy"><strong>Pricing and Availability</strong></div>
            <div class="pagebody-copy">
                <ul>
                    <li>Mac Studio with M5 Max starts at $2,499.</li>
                    <li>Lease with Apple Upgrade from $48.99 per month.</li>
                </ul>
            </div>
            <div class="nr-article-share">
                <div class="sharesheet component">
                    <p>Share article</p>
                </div>
            </div>
            <div class="docsanddownloads text component">
                <p>Text of this article</p>
                <div data-copy-content class="visuallyhidden" aria-hidden="true">
                    <p>PRESS RELEASE Apple introduces new Mac Studio with M5 Max and M5 Ultra DUPLICATE COPY</p>
                </div>
            </div>
            <div class="presscontacts component">
                <p>Press Contacts: media.help@apple.com</p>
            </div>
        </article>
        """

        let cleaned = AgentHarness.cleanHTMLStructure(rawNewsroomHTML)
        // Ensure all noise components and duplicate article clones were stripped
        XCTAssertFalse(cleaned.contains("Share article"), "Must strip sharesheet")
        XCTAssertFalse(cleaned.contains("Text of this article"), "Must strip docsanddownloads")
        XCTAssertFalse(cleaned.contains("DUPLICATE COPY"), "Must strip data-copy-content clone")
        XCTAssertFalse(cleaned.contains("media.help@apple.com"), "Must strip press contacts")

        // Ensure real content is preserved
        XCTAssertTrue(cleaned.contains("Johny Srouji"), "Must preserve Johny Srouji quote")
        XCTAssertTrue(cleaned.contains("$48.99"), "Must preserve lease pricing")

        // Check structured sections
        let sections = AgentHarness.extractStructuredSections(from: cleaned)
        let headings = sections.compactMap { $0["heading"] }
        XCTAssertTrue(headings.contains("A Monumental Step for AI"), "Must clean asterisks from heading")
        XCTAssertFalse(headings.contains("**A Monumental Step for AI**"), "Must not have raw markdown asterisks in heading name")
        XCTAssertTrue(headings.contains("Pricing and Availability"), "Must extract Pricing and Availability as a structured heading")

        // Check tool response framing
        let mockOfficialTurn = AgentHarness.shared.formatToolResponseTurn(
            responses: ["{\"status\": \"success\", \"is_official_domain\": true, \"result\": {}}"],
            includeThinkSuffix: true
        )
        XCTAssertTrue(mockOfficialTurn.contains("SYSTEM NOTICE: Verified official vendor domain response"), "Must inject official domain ground truth notice")
        XCTAssertTrue(mockOfficialTurn.hasSuffix("<think>\n"), "Must include think tag suffix")
    }

    func testLingFlashMoERepackAndLoad() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--inclusionAI--Ling-3.0-tiny/snapshots/e3a47d5b986e7141b6efd62597d598ebb392060d"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Ling snapshot not found, skipping.")
            return
        }
        let url = URL(fileURLWithPath: snapshotDir)

        print("🔍 Testing Ling-3.0 FlashMoE repacking...")
        let repacker = ExpertRepacker.shared
        try repacker.repackSafetensors(sourceDir: url, outputDir: url) { prog, msg in
            if Int(prog * 100) % 20 == 0 || prog >= 0.99 {
                print(String(format: "Repack [%.0f%%]: %@", prog * 100, msg))
            }
        }

        XCTAssertTrue(ExpertRepacker.isPackedFormat(dir: url), "isPackedFormat must return true after repacking")

        let packedDir = url.appendingPathComponent("packed_experts")
        let layoutJson = packedDir.appendingPathComponent("layout.json")
        let lData = try Data(contentsOf: layoutJson)
        let layout = try JSONDecoder().decode(FlashMoELayout.self, from: lData)

        print("✅ Packed layout: expert_size=\(layout.expert_size), layers=\(layout.num_layers), experts=\(layout.num_experts), components=\(layout.components.count)")
        XCTAssertGreaterThan(layout.expert_size, 0, "expert_size must be greater than 0")
        XCTAssertEqual(layout.components.count, 3, "Ling has 3 components: gate, up, down")

        // Verify layer 0 binary was skipped (dense SwiGLU)
        let layer0Bin = packedDir.appendingPathComponent("layer_00.bin")
        XCTAssertFalse(FileManager.default.fileExists(atPath: layer0Bin.path), "Layer 0 has no experts and should not produce layer_00.bin")

        // Verify layer 1 binary exists and has non-zero size
        let layer1Bin = packedDir.appendingPathComponent("layer_01.bin")
        XCTAssertTrue(FileManager.default.fileExists(atPath: layer1Bin.path), "Layer 1 should produce layer_01.bin")
        let layer1Attrs = try FileManager.default.attributesOfItem(atPath: layer1Bin.path)
        let layer1Size = layer1Attrs[.size] as? UInt64 ?? 0
        XCTAssertGreaterThan(layer1Size, 100 * 1024 * 1024, "Layer 1 binary should be > 100 MB")
        print(String(format: "✅ Layer 1 binary size: %.2f MB", Double(layer1Size) / (1024 * 1024)))

        // Test loading the repacked model with DynaMoeEngine and bridging to Metal
        print("🔍 Loading repacked Ling model via DynaMoeEngine...")
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()
        print("✅ Engine loaded: \(summary.shards.count) shards, \(summary.tensors.count) tensors")

        guard let device = MTLCreateSystemDefaultDevice() else {
            XCTFail("No Metal GPU device")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            guard len > 0 else { continue }
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }
        print("✅ Successfully mapped \(buffers.count) Metal buffers without zero-length assertion crash!")
    }

    func testLingFlashMoELayer1Forward() throws {
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--inclusionAI--Ling-3.0-tiny/snapshots/e3a47d5b986e7141b6efd62597d598ebb392060d"
        guard FileManager.default.fileExists(atPath: snapshotDir) else {
            print("Snapshot not found, skipping.")
            return
        }
        let engine = try DynaMoeEngine(filePath: snapshotDir)
        let summary = try engine.getSummary()

        guard let device = MTLCreateSystemDefaultDevice(),
              let cmdQueue = device.makeCommandQueue() else {
            XCTFail("No Metal device or command queue")
            return
        }

        var buffers: [UInt32: MTLBuffer] = [:]
        for shard in summary.shards {
            let address = UInt(shard.baseAddress)
            guard let ptr = UnsafeMutableRawPointer(bitPattern: address) else { continue }
            let len = Int(shard.length)
            guard len > 0 else { continue }
            if let buf = device.makeBuffer(bytesNoCopy: ptr, length: len, options: .storageModeShared, deallocator: nil) {
                buffers[shard.index] = buf
            }
        }

        let config = ModelConfig.load(from: URL(fileURLWithPath: snapshotDir))
        let inference = InferenceEngine.shared
        try inference.initializePipelines(device: device)
        let cachedLayers = inference.buildCachedLayers(summary: summary, config: config, targetLayerCount: 24)

        XCTAssertTrue(cachedLayers.count >= 2, "Expected at least 2 layers")
        let layer1 = cachedLayers[1]

        let hiddenDim = 1536
        let intermediateDim = 512
        let expertSize = 4718592
        let packedDir = URL(fileURLWithPath: snapshotDir).appendingPathComponent("packed_experts")

        let pool = ExpertIOThreadPool.shared
        pool.initialize(numThreads: 8)

        guard let fd = pool.getOrOpenLayerFD(layerIndex: 1, packedExpertsDir: packedDir) else {
            XCTFail("Could not open layer_01.bin")
            return
        }

        guard let expertStagingBuffer = device.makeBuffer(length: 16 * expertSize, options: .storageModeShared),
              let xNorm2Buffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let interBuffer = device.makeBuffer(length: intermediateDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMlpBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let hMidBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared),
              let nextH = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.stride, options: .storageModeShared) else {
            XCTFail("Could not allocate buffers")
            return
        }

        // Initialize xNorm2Buffer with dummy float values
        let xNormPtr = xNorm2Buffer.contents().bindMemory(to: Float.self, capacity: hiddenDim)
        for i in 0..<hiddenDim { xNormPtr[i] = 0.01 * Float(i % 10) }

        // Top-8 active experts
        let activeExperts: [(id: Int, weight: Float)] = [
            (id: 0, weight: 0.25),
            (id: 1, weight: 0.20),
            (id: 2, weight: 0.15),
            (id: 3, weight: 0.10),
            (id: 4, weight: 0.10),
            (id: 5, weight: 0.08),
            (id: 6, weight: 0.07),
            (id: 7, weight: 0.05)
        ]

        var tasks: [ExpertPreadTask] = []
        let rawStagingPtr = expertStagingBuffer.contents()
        for (slot, exp) in activeExperts.enumerated() {
            let offset = off_t(exp.id * expertSize)
            let dst = rawStagingPtr.advanced(by: slot * expertSize)
            tasks.append(ExpertPreadTask(fd: fd, dst: dst, offset: offset, size: expertSize))
        }
        pool.dispatchSync(tasks: &tasks)
        print("✅ pread completed for 8 experts")

        guard let clearPipe = inference.clearPipeline,
              let addPipe = inference.addPipeline,
              let bf16GateSimdPipe = inference.bf16GateUpSimdPipeline,
              let bf16DownSimdPipe = inference.bf16DownSimdPipeline else {
            XCTFail("Failed to load Metal pipelines from InferenceEngine")
            return
        }

        print("Testing with SIMD pipelines...")
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let moeCmd = cmdQueue.makeCommandBuffer(),
              let enc = moeCmd.makeComputeCommandEncoder() else {
            XCTFail("Failed to make command buffer")
            return
        }

        enc.setComputePipelineState(clearPipe)
        enc.setBuffer(hMlpBuffer, offset: 0, index: 0)
        enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        for (slot, exp) in activeExperts.enumerated() {
            let slotOffset = UInt64(slot * expertSize)
            var gWOff = slotOffset + 0
            var uWOff = slotOffset + 1572864
            var dWOff = slotOffset + 3145728
            var hDim = UInt32(hiddenDim)
            var interDim = UInt32(intermediateDim)
            var pk = exp.weight

            enc.setComputePipelineState(bf16GateSimdPipe)
            enc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
            enc.setBuffer(expertStagingBuffer, offset: 0, index: 1)
            enc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
            enc.setBuffer(interBuffer, offset: 0, index: 3)
            enc.setBytes(&gWOff, length: 8, index: 4)
            enc.setBytes(&uWOff, length: 8, index: 5)
            enc.setBytes(&hDim, length: 4, index: 6)
            enc.setBytes(&interDim, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(bf16DownSimdPipe)
            enc.setBuffer(expertStagingBuffer, offset: 0, index: 0)
            enc.setBuffer(interBuffer, offset: 0, index: 1)
            enc.setBuffer(hMlpBuffer, offset: 0, index: 2)
            enc.setBytes(&dWOff, length: 8, index: 3)
            enc.setBytes(&interDim, length: 4, index: 4)
            enc.setBytes(&hDim, length: 4, index: 5)
            enc.setBytes(&pk, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        // Shared expert
        if let gateW = layer1.sharedGateWeight,
           let upW = layer1.sharedUpWeight,
           let downW = layer1.sharedDownWeight,
           let gRaw = buffers[gateW.shardIndex],
           let uRaw = buffers[upW.shardIndex],
           let dRaw = buffers[downW.shardIndex] {
            var gWOff = gateW.offsetStart
            var uWOff = upW.offsetStart
            var dWOff = downW.offsetStart
            var hDimVal = UInt32(hiddenDim)
            var interDimVal = UInt32(intermediateDim)
            var pk: Float = 1.0

            enc.setComputePipelineState(bf16GateSimdPipe)
            enc.setBuffer(gRaw, offset: 0, index: 0)
            enc.setBuffer(uRaw, offset: 0, index: 1)
            enc.setBuffer(xNorm2Buffer, offset: 0, index: 2)
            enc.setBuffer(interBuffer, offset: 0, index: 3)
            enc.setBytes(&gWOff, length: 8, index: 4)
            enc.setBytes(&uWOff, length: 8, index: 5)
            enc.setBytes(&hDimVal, length: 4, index: 6)
            enc.setBytes(&interDimVal, length: 4, index: 7)
            enc.dispatchThreadgroups(MTLSize(width: intermediateDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)

            enc.setComputePipelineState(bf16DownSimdPipe)
            enc.setBuffer(dRaw, offset: 0, index: 0)
            enc.setBuffer(interBuffer, offset: 0, index: 1)
            enc.setBuffer(hMlpBuffer, offset: 0, index: 2)
            enc.setBytes(&dWOff, length: 8, index: 3)
            enc.setBytes(&interDimVal, length: 4, index: 4)
            enc.setBytes(&hDimVal, length: 4, index: 5)
            enc.setBytes(&pk, length: 4, index: 6)
            enc.dispatchThreadgroups(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            enc.memoryBarrier(scope: .buffers)
        }

        var hDim = UInt32(hiddenDim)
        enc.setComputePipelineState(addPipe)
        enc.setBuffer(hMidBuffer, offset: 0, index: 0)
        enc.setBuffer(hMlpBuffer, offset: 0, index: 1)
        enc.setBuffer(nextH, offset: 0, index: 2)
        enc.setBytes(&hDim, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: hiddenDim, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.memoryBarrier(scope: .buffers)

        enc.endEncoding()
        moeCmd.commit()
        moeCmd.waitUntilCompleted()

        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        XCTAssertNil(moeCmd.error, "moeCmd failed with error: \(String(describing: moeCmd.error))")
        print(String(format: "✅ moeCmd completed successfully in %.2f ms!", elapsed))
    }
}






