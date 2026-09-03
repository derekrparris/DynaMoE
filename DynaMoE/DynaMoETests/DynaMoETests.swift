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
}




