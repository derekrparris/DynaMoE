//
//  InferenceEngine.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/25/26.
//

import Foundation
import Metal

public enum MemoryExecutionMode: String, CaseIterable, Identifiable, Codable {
    case autoDetect = "Auto (Smart)"
    case residentRAM = "Full RAM"
    case dynamicStreaming = "SSD Streaming"

    public var id: String { rawValue }

    public var description: String {
        switch self {
        case .autoDetect:
            return "Automatically pins all weights in RAM if memory allows, or enables active-expert SSD streaming on constrained devices."
        case .residentRAM:
            return "Pre-faults and locks all SafeTensors weights into Unified RAM. Bypasses disk I/O for maximum generation tokens/second."
        case .dynamicStreaming:
            return "Retains dense backbone in RAM while streaming active MoE experts on-demand from NVMe storage using LRU working set management."
        }
    }

    public var icon: String {
        switch self {
        case .autoDetect: return "wand.and.stars"
        case .residentRAM: return "bolt.fill"
        case .dynamicStreaming: return "externaldrive.fill"
        }
    }

    /// Resolves actual operational mode against physical unified RAM
    public func resolveEffectiveMode(modelFootprintGB: Double) -> MemoryExecutionMode {
        if self != .autoDetect { return self }
        let physicalRamBytes = ProcessInfo.processInfo.physicalMemory
        let physicalRamGB = Double(physicalRamBytes) / (1024.0 * 1024.0 * 1024.0)

        // If physical RAM has >= 25% headroom over model footprint, use Full RAM
        if physicalRamGB >= (modelFootprintGB * 1.25) {
            return .residentRAM
        } else {
            return .dynamicStreaming
        }
    }
}

public struct EngineCachedLayer {
    public let layerIndex: UInt32
    public let attentionType: LayerAttentionType
    public let mlpType: LayerMlpType
    public let fullAttnIndex: Int
    public let linAttnIndex: Int

    // Normalizations
    public let norm1Tensor: TensorMetadata?
    public let norm2Tensor: TensorMetadata?

    // Dense MLP (for standard Dense Transformer models)
    public let denseGateWeight: TensorMetadata?
    public let denseGateScale: TensorMetadata?
    public let denseGateBias: TensorMetadata?
    public let denseUpWeight: TensorMetadata?
    public let denseUpScale: TensorMetadata?
    public let denseUpBias: TensorMetadata?
    public let denseDownWeight: TensorMetadata?
    public let denseDownScale: TensorMetadata?
    public let denseDownBias: TensorMetadata?

    // MoE Routers & Shared Gate (for MoE architectures)
    public let routerTensor: TensorMetadata?
    public let routerScale: TensorMetadata?
    public let routerBias: TensorMetadata?
    public let sharedGateTensor: TensorMetadata?
    public let sharedGateTensorScale: TensorMetadata?
    public let sharedGateTensorBias: TensorMetadata?

    // Full Attention Tensors (GQA)
    public let qProjTensor: TensorMetadata?
    public let qScaleTensor: TensorMetadata?
    public let qBiasTensor: TensorMetadata?
    public let kProjTensor: TensorMetadata?
    public let kScaleTensor: TensorMetadata?
    public let kBiasTensor: TensorMetadata?
    public let vProjTensor: TensorMetadata?
    public let vScaleTensor: TensorMetadata?
    public let vBiasTensor: TensorMetadata?
    public let qNormTensor: TensorMetadata?
    public let kNormTensor: TensorMetadata?
    public let oProjTensor: TensorMetadata?
    public let oScaleTensor: TensorMetadata?
    public let oBiasTensor: TensorMetadata?

    // Linear Attention Tensors (GatedDeltaNet SSM)
    public let inProjQKV: TensorMetadata?
    public let inProjQKVScale: TensorMetadata?
    public let inProjQKVBias: TensorMetadata?
    public let conv1dTensor: TensorMetadata?
    public let inProjZ: TensorMetadata?
    public let inProjZScale: TensorMetadata?
    public let inProjZBias: TensorMetadata?
    public let inProjA: TensorMetadata?
    public let inProjAScale: TensorMetadata?
    public let inProjABias: TensorMetadata?
    public let inProjB: TensorMetadata?
    public let inProjBScale: TensorMetadata?
    public let inProjBBias: TensorMetadata?
    public let aLogTensor: TensorMetadata?
    public let dtBiasTensor: TensorMetadata?
    public let linearNormTensor: TensorMetadata?
    public let linearOutProjTensor: TensorMetadata?
    public let linearOutProjScale: TensorMetadata?
    public let linearOutProjBias: TensorMetadata?

    // Shared Expert MLP
    public let sharedGateWeight: TensorMetadata?
    public let sharedUpWeight: TensorMetadata?
    public let sharedDownWeight: TensorMetadata?
    public let sharedGateScale: TensorMetadata?
    public let sharedUpScale: TensorMetadata?
    public let sharedDownScale: TensorMetadata?
    public let sharedGateBias: TensorMetadata?
    public let sharedUpBias: TensorMetadata?
    public let sharedDownBias: TensorMetadata?

    // Routed MoE Experts
    public let expertGateWeights: [Int: TensorMetadata]
    public let expertUpWeights: [Int: TensorMetadata]
    public let expertDownWeights: [Int: TensorMetadata]
    public let expertGateScales: [Int: TensorMetadata]
    public let expertUpScales: [Int: TensorMetadata]
    public let expertDownScales: [Int: TensorMetadata]
    public let expertGateBiases: [Int: TensorMetadata]
    public let expertUpBiases: [Int: TensorMetadata]
    public let expertDownBiases: [Int: TensorMetadata]
    public let intermediateDim: UInt32

    // Gated Residuals & QSA Indexer (Qwen 3.8 Flash Next)
    public let grReadWeight: TensorMetadata?
    public let grWriteScale: TensorMetadata?
    public let qsaMqaIndexerWeight: TensorMetadata?
}

public final class InferenceEngine {
    public static let shared = InferenceEngine()

    public private(set) var defaultLibrary: MTLLibrary?
    public private(set) var device: MTLDevice?

    // Compute Pipelines
    public var embedPipeline: MTLComputePipelineState?
    public var embedQ4Pipeline: MTLComputePipelineState?
    public var embedQ8Pipeline: MTLComputePipelineState?
    public var embedMXFP8Pipeline: MTLComputePipelineState?
    public var rmsnormPipeline: MTLComputePipelineState?
    public var rmsnormF16Pipeline: MTLComputePipelineState?
    public var headRmsnormPipeline: MTLComputePipelineState?
    public var headRmsnormF16Pipeline: MTLComputePipelineState?
    public var addPipeline: MTLComputePipelineState?
    public var clearPipeline: MTLComputePipelineState?
    public var ropePipeline: MTLComputePipelineState?
    public var storeKvCachePipeline: MTLComputePipelineState?
    public var storeKvCacheF16Pipeline: MTLComputePipelineState?
    public var storeKvCacheFP8Pipeline: MTLComputePipelineState?
    public var gqaDecodePipeline: MTLComputePipelineState?
    public var gqaDecodeF16Pipeline: MTLComputePipelineState?
    public var gqaDecodeFP8Pipeline: MTLComputePipelineState?
    public var gqaStandardPipeline: MTLComputePipelineState?
    public var gqaStandardF16Pipeline: MTLComputePipelineState?
    public var gqaStandardFP8Pipeline: MTLComputePipelineState?
    public var causalConv1dPipeline: MTLComputePipelineState?
    public var l2NormQkPipeline: MTLComputePipelineState?
    public var linearAttnStepPipeline: MTLComputePipelineState?
    public var routerPipeline: MTLComputePipelineState?
    public var routerQ4Pipeline: MTLComputePipelineState?
    public var routerQ8Pipeline: MTLComputePipelineState?
    public var q4GemvPipeline: MTLComputePipelineState?
    public var q8GemvPipeline: MTLComputePipelineState?
    public var fp8GemvPipeline: MTLComputePipelineState?
    public var bf16GemvSimdPipeline: MTLComputePipelineState?
    public var gemvBF16Pipeline: MTLComputePipelineState?
    public var mxfp8GemvPipeline: MTLComputePipelineState?
    public var mxfp8GemvSimdPipeline: MTLComputePipelineState?
    public var q4GateUpPipeline: MTLComputePipelineState?
    public var q4DownPipeline: MTLComputePipelineState?
    public var q8GateUpPipeline: MTLComputePipelineState?
    public var q8DownPipeline: MTLComputePipelineState?
    public var fp8GateUpPipeline: MTLComputePipelineState?
    public var fp8DownPipeline: MTLComputePipelineState?
    public var fp8GateUpSimdPipeline: MTLComputePipelineState?
    public var fp8DownSimdPipeline: MTLComputePipelineState?
    public var mxfp8GateUpPipeline: MTLComputePipelineState?
    public var mxfp8DownPipeline: MTLComputePipelineState?
    public var mxfp8GateUpSimdPipeline: MTLComputePipelineState?
    public var mxfp8DownSimdPipeline: MTLComputePipelineState?
    public var bf16GateUpPipeline: MTLComputePipelineState?
    public var bf16DownPipeline: MTLComputePipelineState?

    // Qwen 3.8 Flash Next Specialized Pipelines
    public var router512Pipeline: MTLComputePipelineState?
    public var router512Q4Pipeline: MTLComputePipelineState?
    public var gdnLinearAttnStepPipeline: MTLComputePipelineState?
    public var qsaMqaIndexerPipeline: MTLComputePipelineState?
    public var gatedResidualBlendPipeline: MTLComputePipelineState?
    public var fuseNgramPlePipeline: MTLComputePipelineState?

    private init() {}

    public func initializePipelines(device: MTLDevice) throws {
        self.device = device
        guard let defaultLib = device.makeDefaultLibrary() else {
            throw NSError(domain: "InferenceEngine", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load default Metal library."])
        }
        self.defaultLibrary = defaultLib

        // Base Kernels
        if let embedFunc = defaultLib.makeFunction(name: "lookup_embeddings_bf16") {
            embedPipeline = try device.makeComputePipelineState(function: embedFunc)
        }
        if let embedQ4Func = defaultLib.makeFunction(name: "lookup_embeddings_q4") {
            embedQ4Pipeline = try device.makeComputePipelineState(function: embedQ4Func)
        }
        if let embedQ8Func = defaultLib.makeFunction(name: "lookup_embeddings_q8") {
            embedQ8Pipeline = try device.makeComputePipelineState(function: embedQ8Func)
        }
        if let embedMXFP8Func = defaultLib.makeFunction(name: "lookup_embeddings_mxfp8") {
            embedMXFP8Pipeline = try device.makeComputePipelineState(function: embedMXFP8Func)
        }
        if let rmsFunc = defaultLib.makeFunction(name: "rmsnorm_bf16") {
            rmsnormPipeline = try device.makeComputePipelineState(function: rmsFunc)
        }
        if let rmsF16Func = defaultLib.makeFunction(name: "rmsnorm_f16") {
            rmsnormF16Pipeline = try device.makeComputePipelineState(function: rmsF16Func)
        }
        if let hrmsFunc = defaultLib.makeFunction(name: "per_head_rmsnorm_bf16") {
            headRmsnormPipeline = try device.makeComputePipelineState(function: hrmsFunc)
        }
        if let hrmsF16Func = defaultLib.makeFunction(name: "per_head_rmsnorm_f16") {
            headRmsnormF16Pipeline = try device.makeComputePipelineState(function: hrmsF16Func)
        }
        if let addFunc = defaultLib.makeFunction(name: "vector_add_f32") {
            addPipeline = try device.makeComputePipelineState(function: addFunc)
        }
        if let clearFunc = defaultLib.makeFunction(name: "clear_buffer_f32") {
            clearPipeline = try device.makeComputePipelineState(function: clearFunc)
        }
        if let ropeFunc = defaultLib.makeFunction(name: "apply_rope_qwen") {
            ropePipeline = try device.makeComputePipelineState(function: ropeFunc)
        }
        if let storeKvFunc = defaultLib.makeFunction(name: "store_kv_cache") {
            storeKvCachePipeline = try device.makeComputePipelineState(function: storeKvFunc)
        }
        if let storeKvF16Func = defaultLib.makeFunction(name: "store_kv_cache_f16") {
            storeKvCacheF16Pipeline = try device.makeComputePipelineState(function: storeKvF16Func)
        }
        if let storeKvFP8Func = defaultLib.makeFunction(name: "store_kv_cache_fp8") {
            storeKvCacheFP8Pipeline = try device.makeComputePipelineState(function: storeKvFP8Func)
        }
        if let gqaFunc = defaultLib.makeFunction(name: "gqa_attention_decode_fused") {
            gqaDecodePipeline = try device.makeComputePipelineState(function: gqaFunc)
        }
        if let gqaF16Func = defaultLib.makeFunction(name: "gqa_attention_decode_fused_f16") {
            gqaDecodeF16Pipeline = try device.makeComputePipelineState(function: gqaF16Func)
        }
        if let gqaFP8Func = defaultLib.makeFunction(name: "gqa_attention_decode_fused_fp8") {
            gqaDecodeFP8Pipeline = try device.makeComputePipelineState(function: gqaFP8Func)
        }
        if let gqaStdFunc = defaultLib.makeFunction(name: "gqa_attention_decode_standard") {
            gqaStandardPipeline = try device.makeComputePipelineState(function: gqaStdFunc)
        }
        if let gqaStdF16Func = defaultLib.makeFunction(name: "gqa_attention_decode_standard_f16") {
            gqaStandardF16Pipeline = try device.makeComputePipelineState(function: gqaStdF16Func)
        }
        if let gqaStdFP8Func = defaultLib.makeFunction(name: "gqa_attention_decode_standard_fp8") {
            gqaStandardFP8Pipeline = try device.makeComputePipelineState(function: gqaStdFP8Func)
        }
        if let convFunc = defaultLib.makeFunction(name: "causal_conv1d_silu") {
            causalConv1dPipeline = try device.makeComputePipelineState(function: convFunc)
        }
        if let l2Func = defaultLib.makeFunction(name: "l2_norm_qk") {
            l2NormQkPipeline = try device.makeComputePipelineState(function: l2Func)
        }
        if let linStepFunc = defaultLib.makeFunction(name: "linear_attention_recurrent_step") {
            linearAttnStepPipeline = try device.makeComputePipelineState(function: linStepFunc)
        }
        if let rFunc = defaultLib.makeFunction(name: "moe_router_topk") {
            routerPipeline = try device.makeComputePipelineState(function: rFunc)
        }
        if let rQ4Func = defaultLib.makeFunction(name: "moe_router_topk_q4") {
            routerQ4Pipeline = try device.makeComputePipelineState(function: rQ4Func)
        }
        if let rQ8Func = defaultLib.makeFunction(name: "moe_router_topk_q8") {
            routerQ8Pipeline = try device.makeComputePipelineState(function: rQ8Func)
        }
        if let q4GemvFunc = defaultLib.makeFunction(name: "q4_gemv") {
            q4GemvPipeline = try device.makeComputePipelineState(function: q4GemvFunc)
        }
        if let q8GemvFunc = defaultLib.makeFunction(name: "q8_gemv") {
            q8GemvPipeline = try device.makeComputePipelineState(function: q8GemvFunc)
        }
        if let mxfp8GemvFunc = defaultLib.makeFunction(name: "mxfp8_gemv") {
            mxfp8GemvPipeline = try device.makeComputePipelineState(function: mxfp8GemvFunc)
        }
        if let mxfp8GemvSimdFunc = defaultLib.makeFunction(name: "mxfp8_gemv_simd") {
            mxfp8GemvSimdPipeline = try device.makeComputePipelineState(function: mxfp8GemvSimdFunc)
        }
        if let fp8GemvFunc = defaultLib.makeFunction(name: "mxfp8_gemv") ?? defaultLib.makeFunction(name: "fp8_gemv") {
            fp8GemvPipeline = try device.makeComputePipelineState(function: fp8GemvFunc)
        }
        if let bf16SimdFunc = defaultLib.makeFunction(name: "bf16_gemv_simd") {
            bf16GemvSimdPipeline = try device.makeComputePipelineState(function: bf16SimdFunc)
        }
        if let bGemvFunc = defaultLib.makeFunction(name: "gemv_bf16") {
            gemvBF16Pipeline = try device.makeComputePipelineState(function: bGemvFunc)
        }
        if let q4GateUpFunc = defaultLib.makeFunction(name: "q4_swiglu_gate_up") {
            q4GateUpPipeline = try device.makeComputePipelineState(function: q4GateUpFunc)
        }
        if let q4DownFunc = defaultLib.makeFunction(name: "q4_down_proj_accumulate") {
            q4DownPipeline = try device.makeComputePipelineState(function: q4DownFunc)
        }
        if let q8GateUpFunc = defaultLib.makeFunction(name: "q8_swiglu_gate_up") {
            q8GateUpPipeline = try device.makeComputePipelineState(function: q8GateUpFunc)
        }
        if let q8DownFunc = defaultLib.makeFunction(name: "q8_down_proj_accumulate") {
            q8DownPipeline = try device.makeComputePipelineState(function: q8DownFunc)
        }
        if let fp8GateUpFunc = defaultLib.makeFunction(name: "fp8_swiglu_gate_up") {
            fp8GateUpPipeline = try device.makeComputePipelineState(function: fp8GateUpFunc)
        }
        if let fp8DownFunc = defaultLib.makeFunction(name: "fp8_down_proj_accumulate") {
            fp8DownPipeline = try device.makeComputePipelineState(function: fp8DownFunc)
        }
        if let fp8GateSimdFunc = defaultLib.makeFunction(name: "fp8_swiglu_gate_up_simd") {
            fp8GateUpSimdPipeline = try device.makeComputePipelineState(function: fp8GateSimdFunc)
        }
        if let fp8DownSimdFunc = defaultLib.makeFunction(name: "fp8_down_proj_accumulate_simd") {
            fp8DownSimdPipeline = try device.makeComputePipelineState(function: fp8DownSimdFunc)
        }
        if let mxfp8GateUpFunc = defaultLib.makeFunction(name: "mxfp8_swiglu_gate_up") {
            mxfp8GateUpPipeline = try device.makeComputePipelineState(function: mxfp8GateUpFunc)
        }
        if let mxfp8DownFunc = defaultLib.makeFunction(name: "mxfp8_down_proj_accumulate") {
            mxfp8DownPipeline = try device.makeComputePipelineState(function: mxfp8DownFunc)
        }
        if let mxfp8GateSimdFunc = defaultLib.makeFunction(name: "mxfp8_swiglu_gate_up_simd") {
            mxfp8GateUpSimdPipeline = try device.makeComputePipelineState(function: mxfp8GateSimdFunc)
        }
        if let mxfp8DownSimdFunc = defaultLib.makeFunction(name: "mxfp8_down_proj_accumulate_simd") {
            mxfp8DownSimdPipeline = try device.makeComputePipelineState(function: mxfp8DownSimdFunc)
        }
        if let bf16GateUpFunc = defaultLib.makeFunction(name: "bf16_swiglu_gate_up") {
            bf16GateUpPipeline = try device.makeComputePipelineState(function: bf16GateUpFunc)
        }
        if let bf16DownFunc = defaultLib.makeFunction(name: "bf16_down_proj_accumulate") {
            bf16DownPipeline = try device.makeComputePipelineState(function: bf16DownFunc)
        }

        // Qwen 3.8 Flash Next Pipelines
        if let r512Func = defaultLib.makeFunction(name: "moe_router_topk_512_bf16") {
            router512Pipeline = try device.makeComputePipelineState(function: r512Func)
        }
        if let r512Q4Func = defaultLib.makeFunction(name: "moe_router_topk_512_q4") {
            router512Q4Pipeline = try device.makeComputePipelineState(function: r512Q4Func)
        }
        if let gdnStepFunc = defaultLib.makeFunction(name: "gdn_linear_attention_recurrent_step") {
            gdnLinearAttnStepPipeline = try device.makeComputePipelineState(function: gdnStepFunc)
        }
        if let qsaIdxFunc = defaultLib.makeFunction(name: "qsa_mqa_indexer_score_blocks") {
            qsaMqaIndexerPipeline = try device.makeComputePipelineState(function: qsaIdxFunc)
        }
        if let grBlendFunc = defaultLib.makeFunction(name: "gated_residual_blend_4stream") {
            gatedResidualBlendPipeline = try device.makeComputePipelineState(function: grBlendFunc)
        }
        if let pleFunc = defaultLib.makeFunction(name: "fuse_ngram_ple_embedding") {
            fuseNgramPlePipeline = try device.makeComputePipelineState(function: pleFunc)
        }
    }

    /// Builds structured layer representations inspecting tensors for Dense, Standard MoE, or Hybrid SSM-MoE
    public func buildCachedLayers(summary: ModelSummary, config: ModelConfig?, targetLayerCount: Int = 40) -> [EngineCachedLayer] {
        var tensorsByLayer: [UInt32: [TensorMetadata]] = [:]
        for t in summary.tensors {
            if t.name.hasPrefix("mtp.") || t.name.hasPrefix("visual.") { continue }
            if let l = t.layerIndex {
                tensorsByLayer[l, default: []].append(t)
            }
        }

        let numLayers = min(targetLayerCount, Int(summary.layerCount > 0 ? summary.layerCount : 40))
        let arch = config?.resolveArchitectureType(summary: summary) ?? (summary.maxExpertId > 0 ? .hybridSsmMoe : .denseTransformer)
        let layerAttnTypes = config?.resolveLayerAttentionTypes(totalLayers: numLayers) ?? (arch.isHybridSsm ? (0..<numLayers).map { ($0 % 4 == 3) ? .fullAttention : .linearAttention } : Array(repeating: .fullAttention, count: numLayers))

        var cached: [EngineCachedLayer] = []
        var fullCount = 0
        var linCount = 0

        for l in 0..<numLayers {
            let layerTensors = tensorsByLayer[UInt32(l)] ?? []
            let attnType = (l < layerAttnTypes.count) ? layerAttnTypes[l] : .fullAttention

            let fullIdx = (attnType == .fullAttention) ? fullCount : 0
            let linIdx = (attnType == .linearAttention) ? linCount : 0
            if attnType == .fullAttention { fullCount += 1 } else { linCount += 1 }

            let norm1 = layerTensors.first(where: { $0.name.contains("input_layernorm") || $0.name.contains("norm1") || $0.name.contains("attn_norm") })
            let norm2 = layerTensors.first(where: { $0.name.contains("post_attention_layernorm") || $0.name.contains("norm2") || $0.name.contains("ffn_norm") })

            // Router
            let router = layerTensors.first(where: { ($0.category == "MoE Router" || ($0.name.contains("mlp.gate") && !$0.name.contains("switch_mlp") && !$0.name.contains("proj") && !$0.name.contains("shared"))) && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let routerScale = layerTensors.first(where: { ($0.category == "MoE Router" || ($0.name.contains("mlp.gate") && !$0.name.contains("switch_mlp") && !$0.name.contains("proj") && !$0.name.contains("shared"))) && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let routerBias = layerTensors.first(where: { ($0.category == "MoE Router" || ($0.name.contains("mlp.gate") && !$0.name.contains("switch_mlp") && !$0.name.contains("proj") && !$0.name.contains("shared"))) && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Shared Gate
            let sharedGate = layerTensors.first(where: { ($0.category == "Shared Expert Gate" || $0.name.contains("shared_expert_gate")) && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let sharedGateScale = layerTensors.first(where: { ($0.category == "Shared Expert Gate" || $0.name.contains("shared_expert_gate")) && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let sharedGateBias = layerTensors.first(where: { ($0.category == "Shared Expert Gate" || $0.name.contains("shared_expert_gate")) && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Dense MLP
            let dGate = layerTensors.first(where: { ($0.name.contains("mlp.gate_proj") || $0.name.contains("feed_forward.w1")) && !$0.name.contains("experts") && !$0.name.contains("shared") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let dGateS = layerTensors.first(where: { ($0.name.contains("mlp.gate_proj") || $0.name.contains("feed_forward.w1")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let dGateB = layerTensors.first(where: { ($0.name.contains("mlp.gate_proj") || $0.name.contains("feed_forward.w1")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let dUp = layerTensors.first(where: { ($0.name.contains("mlp.up_proj") || $0.name.contains("feed_forward.w3")) && !$0.name.contains("experts") && !$0.name.contains("shared") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let dUpS = layerTensors.first(where: { ($0.name.contains("mlp.up_proj") || $0.name.contains("feed_forward.w3")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let dUpB = layerTensors.first(where: { ($0.name.contains("mlp.up_proj") || $0.name.contains("feed_forward.w3")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let dDown = layerTensors.first(where: { ($0.name.contains("mlp.down_proj") || $0.name.contains("feed_forward.w2")) && !$0.name.contains("experts") && !$0.name.contains("shared") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let dDownS = layerTensors.first(where: { ($0.name.contains("mlp.down_proj") || $0.name.contains("feed_forward.w2")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let dDownB = layerTensors.first(where: { ($0.name.contains("mlp.down_proj") || $0.name.contains("feed_forward.w2")) && !$0.name.contains("experts") && !$0.name.contains("shared") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Full Attention Tensors
            let qProj = layerTensors.first(where: { $0.name.contains("self_attn.q_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let qScale = layerTensors.first(where: { $0.name.contains("self_attn.q_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let qBias = layerTensors.first(where: { $0.name.contains("self_attn.q_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let kProj = layerTensors.first(where: { $0.name.contains("self_attn.k_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let kScale = layerTensors.first(where: { $0.name.contains("self_attn.k_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let kBias = layerTensors.first(where: { $0.name.contains("self_attn.k_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let vProj = layerTensors.first(where: { $0.name.contains("self_attn.v_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let vScale = layerTensors.first(where: { $0.name.contains("self_attn.v_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let vBias = layerTensors.first(where: { $0.name.contains("self_attn.v_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let qNorm = layerTensors.first(where: { $0.name.contains("self_attn.q_norm") })
            let kNorm = layerTensors.first(where: { $0.name.contains("self_attn.k_norm") })

            let oProj = layerTensors.first(where: { ($0.name.contains("self_attn.o_proj") || $0.name.contains("linear_attn.out_proj") || $0.name.contains("o_proj") || $0.name.contains("out_proj")) && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let oScale = layerTensors.first(where: { ($0.name.contains("self_attn.o_proj") || $0.name.contains("o_proj") || $0.name.contains("out_proj")) && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let oBias = layerTensors.first(where: { ($0.name.contains("self_attn.o_proj") || $0.name.contains("o_proj") || $0.name.contains("out_proj")) && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Linear Attention Tensors
            let inQKV = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_qkv") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let inQKVScale = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_qkv") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let inQKVBias = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_qkv") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let conv1d = layerTensors.first(where: { $0.name.contains("linear_attn.conv1d.weight") || $0.name.contains("conv1d.weight") })

            let inZ = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_z") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let inZScale = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_z") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let inZBias = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_z") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let inA = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_a") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let inAScale = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_a") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let inABias = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_a") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let inB = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_b") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let inBScale = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_b") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let inBBias = layerTensors.first(where: { $0.name.contains("linear_attn.in_proj_b") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            let aLog = layerTensors.first(where: { $0.name.contains("linear_attn.A_log") })
            let dtBias = layerTensors.first(where: { $0.name.contains("linear_attn.dt_bias") })
            let linNorm = layerTensors.first(where: { $0.name.contains("linear_attn.norm") })

            let linOut = layerTensors.first(where: { $0.name.contains("linear_attn.out_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let linOutScale = layerTensors.first(where: { $0.name.contains("linear_attn.out_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let linOutBias = layerTensors.first(where: { $0.name.contains("linear_attn.out_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Shared Expert
            let sharedGateW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("gate_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let sharedUpW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("up_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let sharedDownW = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("down_proj") && !$0.name.contains("scale") && !$0.name.contains("bias") })

            let sharedGateS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("gate_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let sharedUpS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("up_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let sharedDownS = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("down_proj") && ($0.name.contains("scale") || $0.name.contains("scales")) })

            let sharedGateB = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("gate_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })
            let sharedUpB = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("up_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })
            let sharedDownB = layerTensors.first(where: { $0.name.contains("shared_expert") && $0.name.contains("down_proj") && ($0.name.contains("bias") || $0.name.contains("biases")) })

            // Routed MoE Experts
            var expGW: [Int: TensorMetadata] = [:]
            var expUW: [Int: TensorMetadata] = [:]
            var expDW: [Int: TensorMetadata] = [:]
            var expGS: [Int: TensorMetadata] = [:]
            var expUS: [Int: TensorMetadata] = [:]
            var expDS: [Int: TensorMetadata] = [:]
            var expGB: [Int: TensorMetadata] = [:]
            var expUB: [Int: TensorMetadata] = [:]
            var expDB: [Int: TensorMetadata] = [:]

            for t in layerTensors {
                guard let expId = t.expertId else { continue }
                let exp = Int(expId)
                let isScale = t.name.contains("scale") || t.name.contains("scales")
                let isBias = t.name.contains("bias") || t.name.contains("biases")

                if t.name.contains("gate_proj") {
                    if isScale { expGS[exp] = t }
                    else if isBias { expGB[exp] = t }
                    else { expGW[exp] = t }
                } else if t.name.contains("up_proj") {
                    if isScale { expUS[exp] = t }
                    else if isBias { expUB[exp] = t }
                    else { expUW[exp] = t }
                } else if t.name.contains("down_proj") {
                    if isScale { expDS[exp] = t }
                    else if isBias { expDB[exp] = t }
                    else { expDW[exp] = t }
                }
            }

            var interDim: UInt32 = 512
            if let targetGate = dGate ?? sharedGateW ?? expGW.values.first {
                let cleanShape = targetGate.shapeDisplay.replacingOccurrences(of: "[", with: "").replacingOccurrences(of: "]", with: "").replacingOccurrences(of: " ", with: "")
                let parts = cleanShape.split(separator: ",")
                if let first = parts.first, let parsed = UInt32(first), parsed > 0 {
                    interDim = parsed
                }
            } else if let cfgInter = config?.intermediateSize {
                interDim = UInt32(cfgInter)
            }

            let mlpType: LayerMlpType = (arch == .denseTransformer || expGW.isEmpty) ? .denseMlp : .moeExperts

            let grRead = layerTensors.first(where: { $0.name.contains("gated_residual.read") || $0.name.contains("residual_gate.read") })
            let grWrite = layerTensors.first(where: { $0.name.contains("gated_residual.write") || $0.name.contains("residual_gate.write") })
            let qsaMqaIdx = layerTensors.first(where: { $0.name.contains("qsa.indexer") || $0.name.contains("self_attn.indexer") })

            cached.append(EngineCachedLayer(
                layerIndex: UInt32(l),
                attentionType: attnType,
                mlpType: mlpType,
                fullAttnIndex: fullIdx,
                linAttnIndex: linIdx,
                norm1Tensor: norm1,
                norm2Tensor: norm2,
                denseGateWeight: dGate,
                denseGateScale: dGateS,
                denseGateBias: dGateB,
                denseUpWeight: dUp,
                denseUpScale: dUpS,
                denseUpBias: dUpB,
                denseDownWeight: dDown,
                denseDownScale: dDownS,
                denseDownBias: dDownB,
                routerTensor: router,
                routerScale: routerScale,
                routerBias: routerBias,
                sharedGateTensor: sharedGate,
                sharedGateTensorScale: sharedGateScale,
                sharedGateTensorBias: sharedGateBias,
                qProjTensor: qProj,
                qScaleTensor: qScale,
                qBiasTensor: qBias,
                kProjTensor: kProj,
                kScaleTensor: kScale,
                kBiasTensor: kBias,
                vProjTensor: vProj,
                vScaleTensor: vScale,
                vBiasTensor: vBias,
                qNormTensor: qNorm,
                kNormTensor: kNorm,
                oProjTensor: oProj,
                oScaleTensor: oScale,
                oBiasTensor: oBias,
                inProjQKV: inQKV,
                inProjQKVScale: inQKVScale,
                inProjQKVBias: inQKVBias,
                conv1dTensor: conv1d,
                inProjZ: inZ,
                inProjZScale: inZScale,
                inProjZBias: inZBias,
                inProjA: inA,
                inProjAScale: inAScale,
                inProjABias: inABias,
                inProjB: inB,
                inProjBScale: inBScale,
                inProjBBias: inBBias,
                aLogTensor: aLog,
                dtBiasTensor: dtBias,
                linearNormTensor: linNorm,
                linearOutProjTensor: linOut,
                linearOutProjScale: linOutScale,
                linearOutProjBias: linOutBias,
                sharedGateWeight: sharedGateW,
                sharedUpWeight: sharedUpW,
                sharedDownWeight: sharedDownW,
                sharedGateScale: sharedGateS,
                sharedUpScale: sharedUpS,
                sharedDownScale: sharedDownS,
                sharedGateBias: sharedGateB,
                sharedUpBias: sharedUpB,
                sharedDownBias: sharedDownB,
                expertGateWeights: expGW,
                expertUpWeights: expUW,
                expertDownWeights: expDW,
                expertGateScales: expGS,
                expertUpScales: expUS,
                expertDownScales: expDS,
                expertGateBiases: expGB,
                expertUpBiases: expUB,
                expertDownBiases: expDB,
                intermediateDim: interDim,
                grReadWeight: grRead,
                grWriteScale: grWrite,
                qsaMqaIndexerWeight: qsaMqaIdx
            ))
        }

        return cached
    }
}

extension EngineCachedLayer {
    public var isFullAttention: Bool {
        return attentionType == .fullAttention
    }

    public var backboneTensors: [TensorMetadata] {
        var list: [TensorMetadata] = []
        let candidates: [TensorMetadata?] = [
            norm1Tensor, norm2Tensor,
            denseGateWeight, denseGateScale, denseGateBias,
            denseUpWeight, denseUpScale, denseUpBias,
            denseDownWeight, denseDownScale, denseDownBias,
            routerTensor, routerScale, routerBias,
            sharedGateTensor, sharedGateTensorScale, sharedGateTensorBias,
            qProjTensor, qScaleTensor, qBiasTensor,
            kProjTensor, kScaleTensor, kBiasTensor,
            vProjTensor, vScaleTensor, vBiasTensor,
            qNormTensor, kNormTensor,
            oProjTensor, oScaleTensor, oBiasTensor,
            inProjQKV, inProjQKVScale, inProjQKVBias,
            conv1dTensor,
            inProjZ, inProjZScale, inProjZBias,
            inProjA, inProjAScale, inProjABias,
            inProjB, inProjBScale, inProjBBias,
            aLogTensor, dtBiasTensor, linearNormTensor,
            linearOutProjTensor, linearOutProjScale, linearOutProjBias,
            sharedGateWeight, sharedGateScale, sharedGateBias,
            sharedUpWeight, sharedUpScale, sharedUpBias,
            sharedDownWeight, sharedDownScale, sharedDownBias
        ]
        for c in candidates {
            if let t = c {
                list.append(t)
            }
        }
        return list
    }
}

// MARK: - Expert Staging Buffers & Memory Pinning
extension InferenceEngine {
    public struct ExpertStagingBuffers {
        public let bufferA: MTLBuffer
        public let bufferB: MTLBuffer
        public let maxK: Int
        public let expertSizeBytes: Int

        public func bufferForSlot(_ slot: Int) -> MTLBuffer {
            return (slot % 2 == 0) ? bufferA : bufferB
        }
    }

    /// Allocates pre-allocated Metal shared memory staging pools for double-buffered expert streaming
    public func allocateStagingBuffers(device: MTLDevice, maxK: Int = 8, expertSizeBytes: Int = 2 * 1024 * 1024) -> ExpertStagingBuffers? {
        let totalBytes = maxK * expertSizeBytes
        guard let bufA = device.makeBuffer(length: totalBytes, options: .storageModeShared),
              let bufB = device.makeBuffer(length: totalBytes, options: .storageModeShared) else {
            return nil
        }
        return ExpertStagingBuffers(bufferA: bufA, bufferB: bufB, maxK: maxK, expertSizeBytes: expertSizeBytes)
    }

    /// Advises kernel to lock/keep non-expert backbone buffer resident in RAM
    public func pinBackboneBuffer(_ buffer: MTLBuffer) {
        let ptr = buffer.contents()
        let len = buffer.length
        #if os(macOS)
        _ = posix_madvise(ptr, len, POSIX_MADV_WILLNEED)
        #endif
    }
}

public typealias CachedLayer = EngineCachedLayer


