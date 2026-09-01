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
        let snapshotDir = "/Users/derekparris/.cache/huggingface/hub/models--ornith-ai--Ornith-1.5-35B-A3B-FP8/snapshots/0e048080ccd0ccf4296bfea5638036c196dccc0c"
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
            !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            !$0.name.contains("scale") && !$0.name.contains("scales") &&
            !$0.name.contains("bias") && !$0.name.contains("biases")
        })!
        let embedScale = summary.tensors.first(where: {
            !$0.name.hasPrefix("visual.") && !$0.name.hasPrefix("mtp.") &&
            ($0.name.contains("embed_tokens") || $0.name.hasSuffix("embed.weight") || $0.name.contains("wte")) &&
            ($0.name.contains("scale") || $0.name.contains("scales"))
        })

        let finalHcNormWeight = summary.tensors.first(where: {
            $0.name.contains("hyper_connection_mixer") && $0.name.contains("hc_norm")
        })
        let finalHcDownWeight = summary.tensors.first(where: {
            $0.name.contains("hyper_connection_mixer") && $0.name.contains("input_mix_weight_down")
        })
        let finalHcUpWeight = summary.tensors.first(where: {
            $0.name.contains("hyper_connection_mixer") && $0.name.contains("input_mix_weight_up")
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

        print("🔍 [DIAGNOSTIC] Final HC Mixer: norm=\(finalHcNormWeight?.name ?? "nil"), down=\(finalHcDownWeight?.name ?? "nil"), up=\(finalHcUpWeight?.name ?? "nil")")
        print("🔍 [DIAGNOSTIC] LM Head: tensor=\(lmHeadTensorCandidate.name), dtype=\(lmHeadTensorCandidate.dtype), shape=\(lmHeadTensorCandidate.shapeDisplay), scale=\(lmHeadScale?.name ?? "nil"), bias=\(lmHeadBias?.name ?? "nil")")

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

        for (rank, item) in topTokens.enumerated() {
            print("  Top \(rank + 1): Token \(item.id) with logit \(item.logit)")
        }
        print("🎉 [SUCCESS] Qwen 3.8 Flash Next LM Head & Logits computed cleanly!")
    }
}




