//
//  InferenceEngine.swift
//  DynaMoE
//
//  Created by Derek Parris on 8/25/26.
//

import Foundation
import Metal
import Accelerate

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

    // Hyper-Connections (Qwen 3.8 Flash Next)
    public let attnHcNorm: TensorMetadata?
    public let attnHcDownWeight: TensorMetadata?
    public let attnHcUpWeight: TensorMetadata?
    public let attnHcInjectWeight: TensorMetadata?
    public let mlpHcNorm: TensorMetadata?
    public let mlpHcDownWeight: TensorMetadata?
    public let mlpHcUpWeight: TensorMetadata?
    public let mlpHcInjectWeight: TensorMetadata?
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
    public var rmsnormOffsetPipeline: MTLComputePipelineState?
    public var rmsnormOffsetF16Pipeline: MTLComputePipelineState?
    public var headRmsnormPipeline: MTLComputePipelineState?
    public var headRmsnormF16Pipeline: MTLComputePipelineState?
    public var headRmsnormOffsetPipeline: MTLComputePipelineState?
    public var headRmsnormOffsetF16Pipeline: MTLComputePipelineState?
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
    public var linearAttnStepSigmoidPipeline: MTLComputePipelineState?
    public var routerPipeline: MTLComputePipelineState?
    public var routerQ4Pipeline: MTLComputePipelineState?
    public var routerQ8Pipeline: MTLComputePipelineState?
    public var sharedGatePipeline: MTLComputePipelineState?
    public var q4GemvPipeline: MTLComputePipelineState?
    public var q8GemvPipeline: MTLComputePipelineState?
    public var fp8GemvPipeline: MTLComputePipelineState?
    public var fp8GemvSimdPipeline: MTLComputePipelineState?
    public var bf16GemvSimdPipeline: MTLComputePipelineState?
    public var gemvBF16Pipeline: MTLComputePipelineState?
    public var mxfp8GemvPipeline: MTLComputePipelineState?
    public var mxfp8GemvSimdPipeline: MTLComputePipelineState?
    public var q4GateUpPipeline: MTLComputePipelineState?
    public var q4DownPipeline: MTLComputePipelineState?
    public var q8GateUpPipeline: MTLComputePipelineState?
    public var q8DownPipeline: MTLComputePipelineState?
    public var q4GateQ8UpPipeline: MTLComputePipelineState?
    public var q8GateQ4UpPipeline: MTLComputePipelineState?
    public var fp8GateUpPipeline: MTLComputePipelineState?
    public var fp8DownPipeline: MTLComputePipelineState?
    public var fp8GateUpSimdPipeline: MTLComputePipelineState?
    public var fp8DownSimdPipeline: MTLComputePipelineState?
    public var fp8BlockGateUpPipeline: MTLComputePipelineState?
    public var fp8BlockDownPipeline: MTLComputePipelineState?
    public var fp8BlockGateUpSimdPipeline: MTLComputePipelineState?
    public var fp8BlockDownSimdPipeline: MTLComputePipelineState?
    public var fp8BlockGemvPipeline: MTLComputePipelineState?
    public var fp8BlockGemvSimdPipeline: MTLComputePipelineState?
    public var mxfp8GateUpPipeline: MTLComputePipelineState?
    public var mxfp8DownPipeline: MTLComputePipelineState?
    public var mxfp8GateUpSimdPipeline: MTLComputePipelineState?
    public var mxfp8DownSimdPipeline: MTLComputePipelineState?
    public var bf16GateUpPipeline: MTLComputePipelineState?
    public var bf16DownPipeline: MTLComputePipelineState?
    public var fp8GateUpBatchedPipeline: MTLComputePipelineState?
    public var fp8DownBatchedPipeline: MTLComputePipelineState?
    public var fp8BlockGateUpBatchedPipeline: MTLComputePipelineState?
    public var fp8BlockDownBatchedPipeline: MTLComputePipelineState?
    public var bf16GateUpBatchedPipeline: MTLComputePipelineState?
    public var bf16DownBatchedPipeline: MTLComputePipelineState?

    // Qwen 3.8 Flash Next Specialized Pipelines
    public var router512Pipeline: MTLComputePipelineState?
    public var router512Q4Pipeline: MTLComputePipelineState?
    public var gdnLinearAttnStepPipeline: MTLComputePipelineState?
    public var gdnLinearAttnStepSigmoidPipeline: MTLComputePipelineState?
    public var gdnLinearAttnSeqPipeline: MTLComputePipelineState?
    public var gdnLinearAttnSeqSigmoidPipeline: MTLComputePipelineState?
    public var qsaMqaIndexerPipeline: MTLComputePipelineState?
    public var gatedResidualBlendPipeline: MTLComputePipelineState?
    public var fuseNgramPlePipeline: MTLComputePipelineState?
    public var fusedInit4StreamsPipeline: MTLComputePipelineState?
    public var extractStream0Pipeline: MTLComputePipelineState?
    public var hcNormPipeline: MTLComputePipelineState?
    public var hcDownProjPipeline: MTLComputePipelineState?
    public var hcUpBlendPipeline: MTLComputePipelineState?
    public var hcInjectScalePipeline: MTLComputePipelineState?
    public var hcInjectPipeline: MTLComputePipelineState?

    // JetSpec Speculative Decoding Pipelines
    public var jetDraftHeadPredictPipeline: MTLComputePipelineState?
    public var gqaAttentionTreeVerifyStandardPipeline: MTLComputePipelineState?
    public var gqaAttentionTreeVerifyStandardF16Pipeline: MTLComputePipelineState?
    public var gqaAttentionTreeVerifyFusedPipeline: MTLComputePipelineState?
    public var gqaAttentionTreeVerifyFusedF16Pipeline: MTLComputePipelineState?
    public var gdnLinearAttnTreeStepPipeline: MTLComputePipelineState?
    public var gdnLinearAttnTreeStepSigmoidPipeline: MTLComputePipelineState?
    public var gatherGdnTreeParentStatesPipeline: MTLComputePipelineState?
    public var commitGdnTreeWinningStatePipeline: MTLComputePipelineState?
    public var applyRopeTreePipeline: MTLComputePipelineState?
    public var compactKvCacheSlotsF32Pipeline: MTLComputePipelineState?
    public var compactKvCacheSlotsF16Pipeline: MTLComputePipelineState?

    private init() {}

    public func initializePipelines(device: MTLDevice) throws {
        self.device = device
        guard let defaultLib = device.makeDefaultLibrary() ?? (try? device.makeDefaultLibrary(bundle: Bundle(for: InferenceEngine.self))) else {
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
        if let rmsOffsetFunc = defaultLib.makeFunction(name: "rmsnorm_offset_bf16") {
            rmsnormOffsetPipeline = try device.makeComputePipelineState(function: rmsOffsetFunc)
        }
        if let rmsOffsetF16Func = defaultLib.makeFunction(name: "rmsnorm_offset_f16") {
            rmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: rmsOffsetF16Func)
        }
        if let hrmsFunc = defaultLib.makeFunction(name: "per_head_rmsnorm_bf16") {
            headRmsnormPipeline = try device.makeComputePipelineState(function: hrmsFunc)
        }
        if let hrmsF16Func = defaultLib.makeFunction(name: "per_head_rmsnorm_f16") {
            headRmsnormF16Pipeline = try device.makeComputePipelineState(function: hrmsF16Func)
        }
        if let hrmsOffsetFunc = defaultLib.makeFunction(name: "per_head_rmsnorm_offset_bf16") {
            headRmsnormOffsetPipeline = try device.makeComputePipelineState(function: hrmsOffsetFunc)
        }
        if let hrmsOffsetF16Func = defaultLib.makeFunction(name: "per_head_rmsnorm_offset_f16") {
            headRmsnormOffsetF16Pipeline = try device.makeComputePipelineState(function: hrmsOffsetF16Func)
        }
        if let addFunc = defaultLib.makeFunction(name: "vector_add_f32") {
            addPipeline = try device.makeComputePipelineState(function: addFunc)
        }
        if let clearFunc = defaultLib.makeFunction(name: "clear_vector_f32") ?? defaultLib.makeFunction(name: "clear_buffer_f32") {
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
        if let linStepSigFunc = defaultLib.makeFunction(name: "linear_attention_recurrent_step_sigmoid") {
            linearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: linStepSigFunc)
        }
        if let rFunc = defaultLib.makeFunction(name: "moe_router_topk_bf16") ?? defaultLib.makeFunction(name: "moe_router_topk") {
            routerPipeline = try device.makeComputePipelineState(function: rFunc)
        }
        if let rQ4Func = defaultLib.makeFunction(name: "moe_router_topk_q4") {
            routerQ4Pipeline = try device.makeComputePipelineState(function: rQ4Func)
        }
        if let rQ8Func = defaultLib.makeFunction(name: "moe_router_topk_q8") {
            routerQ8Pipeline = try device.makeComputePipelineState(function: rQ8Func)
        }
        if let sgFunc = defaultLib.makeFunction(name: "moe_shared_gate_bf16") {
            sharedGatePipeline = try device.makeComputePipelineState(function: sgFunc)
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
        if let fp8SimdFunc = defaultLib.makeFunction(name: "fp8_gemv_simd") {
            fp8GemvSimdPipeline = try device.makeComputePipelineState(function: fp8SimdFunc)
        }
        if let fp8GemvFunc = defaultLib.makeFunction(name: "fp8_gemv") ?? defaultLib.makeFunction(name: "mxfp8_gemv") {
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
        if let q4q8Func = defaultLib.makeFunction(name: "q4_gate_q8_up_swiglu") {
            q4GateQ8UpPipeline = try device.makeComputePipelineState(function: q4q8Func)
        }
        if let q8q4Func = defaultLib.makeFunction(name: "q8_gate_q4_up_swiglu") {
            q8GateQ4UpPipeline = try device.makeComputePipelineState(function: q8q4Func)
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
        if let fp8BlockGateSimdFunc = defaultLib.makeFunction(name: "fp8_block_swiglu_gate_up_simd") {
            fp8BlockGateUpSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGateSimdFunc)
        }
        if let fp8BlockDownSimdFunc = defaultLib.makeFunction(name: "fp8_block_down_proj_accumulate_simd") {
            fp8BlockDownSimdPipeline = try device.makeComputePipelineState(function: fp8BlockDownSimdFunc)
        }
        if let fp8BlockGemvFunc = defaultLib.makeFunction(name: "fp8_block_gemv") {
            fp8BlockGemvPipeline = try device.makeComputePipelineState(function: fp8BlockGemvFunc)
        }
        if let fp8BlockGemvSimdFunc = defaultLib.makeFunction(name: "fp8_block_gemv_simd") {
            fp8BlockGemvSimdPipeline = try device.makeComputePipelineState(function: fp8BlockGemvSimdFunc)
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
        if let fp8GateBatchedFunc = defaultLib.makeFunction(name: "fp8_swiglu_gate_up_batched") {
            fp8GateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8GateBatchedFunc)
        }
        if let fp8DownBatchedFunc = defaultLib.makeFunction(name: "fp8_down_proj_accumulate_batched") {
            fp8DownBatchedPipeline = try device.makeComputePipelineState(function: fp8DownBatchedFunc)
        }
        if let fp8BlockGateBatchedFunc = defaultLib.makeFunction(name: "fp8_block_swiglu_gate_up_batched") {
            fp8BlockGateUpBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockGateBatchedFunc)
        }
        if let fp8BlockDownBatchedFunc = defaultLib.makeFunction(name: "fp8_block_down_proj_accumulate_batched") {
            fp8BlockDownBatchedPipeline = try device.makeComputePipelineState(function: fp8BlockDownBatchedFunc)
        }
        if let bf16GateBatchedFunc = defaultLib.makeFunction(name: "bf16_swiglu_gate_up_batched") {
            bf16GateUpBatchedPipeline = try device.makeComputePipelineState(function: bf16GateBatchedFunc)
        }
        if let bf16DownBatchedFunc = defaultLib.makeFunction(name: "bf16_down_proj_accumulate_batched") {
            bf16DownBatchedPipeline = try device.makeComputePipelineState(function: bf16DownBatchedFunc)
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
        if let gdnStepSigFunc = defaultLib.makeFunction(name: "gdn_linear_attention_recurrent_step_sigmoid") {
            gdnLinearAttnStepSigmoidPipeline = try device.makeComputePipelineState(function: gdnStepSigFunc)
        }
        if let gdnSeqFunc = defaultLib.makeFunction(name: "linear_attention_recurrent_sequence") {
            gdnLinearAttnSeqPipeline = try device.makeComputePipelineState(function: gdnSeqFunc)
        }
        if let gdnSeqSigFunc = defaultLib.makeFunction(name: "linear_attention_recurrent_sequence_sigmoid") {
            gdnLinearAttnSeqSigmoidPipeline = try device.makeComputePipelineState(function: gdnSeqSigFunc)
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
        if let init4StreamsFunc = defaultLib.makeFunction(name: "fused_init_4streams") {
            fusedInit4StreamsPipeline = try device.makeComputePipelineState(function: init4StreamsFunc)
        }
        if let extract0Func = defaultLib.makeFunction(name: "extract_stream0") {
            extractStream0Pipeline = try device.makeComputePipelineState(function: extract0Func)
        }
        if let hcNormFunc = defaultLib.makeFunction(name: "hyper_connection_norm_bf16") {
            hcNormPipeline = try device.makeComputePipelineState(function: hcNormFunc)
        }
        if let hcDownFunc = defaultLib.makeFunction(name: "hyper_connection_down_proj_bf16") {
            hcDownProjPipeline = try device.makeComputePipelineState(function: hcDownFunc)
        }
        if let hcUpFunc = defaultLib.makeFunction(name: "hyper_connection_up_proj_blend_bf16") {
            hcUpBlendPipeline = try device.makeComputePipelineState(function: hcUpFunc)
        }
        if let hcInjScaleFunc = defaultLib.makeFunction(name: "hyper_connection_inject_scale_bf16") {
            hcInjectScalePipeline = try device.makeComputePipelineState(function: hcInjScaleFunc)
        }
        if let hcInjFunc = defaultLib.makeFunction(name: "hyper_connection_inject_bf16") {
            hcInjectPipeline = try device.makeComputePipelineState(function: hcInjFunc)
        }

        // JetSpec Speculative Decoding Kernels
        if let jetDraftFunc = defaultLib.makeFunction(name: "jet_draft_head_predict_bf16") {
            jetDraftHeadPredictPipeline = try device.makeComputePipelineState(function: jetDraftFunc)
        }
        if let gqaTreeStdFunc = defaultLib.makeFunction(name: "gqa_attention_tree_verify_standard") {
            gqaAttentionTreeVerifyStandardPipeline = try device.makeComputePipelineState(function: gqaTreeStdFunc)
        }
        if let gqaTreeStdF16Func = defaultLib.makeFunction(name: "gqa_attention_tree_verify_standard_f16") {
            gqaAttentionTreeVerifyStandardF16Pipeline = try device.makeComputePipelineState(function: gqaTreeStdF16Func)
        }
        if let gqaTreeFusedFunc = defaultLib.makeFunction(name: "gqa_attention_tree_verify_fused") {
            gqaAttentionTreeVerifyFusedPipeline = try device.makeComputePipelineState(function: gqaTreeFusedFunc)
        }
        if let gqaTreeFusedF16Func = defaultLib.makeFunction(name: "gqa_attention_tree_verify_fused_f16") {
            gqaAttentionTreeVerifyFusedF16Pipeline = try device.makeComputePipelineState(function: gqaTreeFusedF16Func)
        }
        if let gdnTreeStepFunc = defaultLib.makeFunction(name: "gdn_linear_attention_tree_step") {
            gdnLinearAttnTreeStepPipeline = try device.makeComputePipelineState(function: gdnTreeStepFunc)
        }
        if let gdnTreeStepSigFunc = defaultLib.makeFunction(name: "gdn_linear_attention_tree_step_sigmoid") {
            gdnLinearAttnTreeStepSigmoidPipeline = try device.makeComputePipelineState(function: gdnTreeStepSigFunc)
        }
        if let gatherGdnFunc = defaultLib.makeFunction(name: "gather_gdn_tree_parent_states") {
            gatherGdnTreeParentStatesPipeline = try device.makeComputePipelineState(function: gatherGdnFunc)
        }
        if let commitGdnFunc = defaultLib.makeFunction(name: "commit_gdn_tree_winning_state") {
            commitGdnTreeWinningStatePipeline = try device.makeComputePipelineState(function: commitGdnFunc)
        }
        if let ropeTreeFunc = defaultLib.makeFunction(name: "apply_rope_tree") {
            applyRopeTreePipeline = try device.makeComputePipelineState(function: ropeTreeFunc)
        }
        if let compactF32Func = defaultLib.makeFunction(name: "compact_kv_cache_slots_f32") {
            compactKvCacheSlotsF32Pipeline = try device.makeComputePipelineState(function: compactF32Func)
        }
        if let compactF16Func = defaultLib.makeFunction(name: "compact_kv_cache_slots_f16") {
            compactKvCacheSlotsF16Pipeline = try device.makeComputePipelineState(function: compactF16Func)
        }
    }

    /// Builds structured layer representations inspecting tensors for Dense, Standard MoE, or Hybrid SSM-MoE
    public func buildCachedLayers(summary: ModelSummary, config: ModelConfig?, targetLayerCount: Int? = nil) -> [EngineCachedLayer] {
        var tensorsByLayer: [UInt32: [TensorMetadata]] = [:]
        for t in summary.tensors {
            if t.name.contains("mtp") || t.name.contains("visual") { continue }
            if let l = t.layerIndex {
                tensorsByLayer[l, default: []].append(t)
            }
        }

        let totalModelLayers = summary.layerCount > 0 ? Int(summary.layerCount) : (config?.effectiveNumHiddenLayers ?? 48)
        let numLayers = targetLayerCount != nil ? min(targetLayerCount!, totalModelLayers) : totalModelLayers
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

            let aLog = layerTensors.first(where: { $0.name.lowercased().contains("linear_attn.a_log") || $0.name.contains("A_log") || $0.name.contains("a_log") })
            let dtBias = layerTensors.first(where: { $0.name.lowercased().contains("linear_attn.dt_bias") || $0.name.contains("dt_bias") })
            let linNorm = layerTensors.first(where: { $0.name.lowercased().contains("linear_attn.norm") || $0.name.contains("linear_attn_norm") })

            let linOut = layerTensors.first(where: { ($0.name.contains("linear_attn.out_proj") || $0.name.contains("linear_attn.o_proj")) && !$0.name.contains("scale") && !$0.name.contains("bias") })
            let linOutScale = layerTensors.first(where: { ($0.name.contains("linear_attn.out_proj") || $0.name.contains("linear_attn.o_proj")) && ($0.name.contains("scale") || $0.name.contains("scales")) })
            let linOutBias = layerTensors.first(where: { ($0.name.contains("linear_attn.out_proj") || $0.name.contains("linear_attn.o_proj")) && ($0.name.contains("bias") || $0.name.contains("biases")) })

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

            let attnHcNorm = layerTensors.first(where: { $0.name.contains("attn_hyper_connection.hc_norm") || $0.name.contains("attn_hc.norm") || $0.name.contains("attn_norm_hc") })
            let attnHcDown = layerTensors.first(where: { $0.name.contains("attn_hyper_connection.input_mix_weight_down") || $0.name.contains("attn_hc.down") || $0.name.contains("attn_hc_down") })
            let attnHcUp = layerTensors.first(where: { $0.name.contains("attn_hyper_connection.input_mix_weight_up") || $0.name.contains("attn_hc.up") || $0.name.contains("attn_hc_up") })
            let attnHcInject = layerTensors.first(where: { $0.name.contains("attn_hyper_connection.block_inject_weight") || $0.name.contains("attn_hc.inject") || $0.name.contains("attn_hc_inject") })

            let mlpHcNorm = layerTensors.first(where: { $0.name.contains("mlp_hyper_connection.hc_norm") || $0.name.contains("mlp_hc.norm") || $0.name.contains("mlp_norm_hc") })
            let mlpHcDown = layerTensors.first(where: { $0.name.contains("mlp_hyper_connection.input_mix_weight_down") || $0.name.contains("mlp_hc.down") || $0.name.contains("mlp_hc_down") })
            let mlpHcUp = layerTensors.first(where: { $0.name.contains("mlp_hyper_connection.input_mix_weight_up") || $0.name.contains("mlp_hc.up") || $0.name.contains("mlp_hc_up") })
            let mlpHcInject = layerTensors.first(where: { $0.name.contains("mlp_hyper_connection.block_inject_weight") || $0.name.contains("mlp_hc.inject") || $0.name.contains("mlp_hc_inject") })

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
                qsaMqaIndexerWeight: qsaMqaIdx,
                attnHcNorm: attnHcNorm,
                attnHcDownWeight: attnHcDown,
                attnHcUpWeight: attnHcUp,
                attnHcInjectWeight: attnHcInject,
                mlpHcNorm: mlpHcNorm,
                mlpHcDownWeight: mlpHcDown,
                mlpHcUpWeight: mlpHcUp,
                mlpHcInjectWeight: mlpHcInject
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
            sharedDownWeight, sharedDownScale, sharedDownBias,
            attnHcNorm, attnHcDownWeight, attnHcUpWeight, attnHcInjectWeight,
            mlpHcNorm, mlpHcDownWeight, mlpHcUpWeight, mlpHcInjectWeight
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

// MARK: - JetSpec Speculative Staging Buffers & Topology
extension InferenceEngine {
    public struct JetSpecStagingBuffers {
        public let treeMaskBuffer: MTLBuffer        // [maxNodes * maxNodes] f32
        public let candidateTokensBuffer: MTLBuffer // [maxNodes] u32
        public let parentIndicesBuffer: MTLBuffer   // [maxNodes] u32
        public let depthsBuffer: MTLBuffer          // [maxNodes] u32
        public let draftLogitsBuffer: MTLBuffer     // [maxNodes * vocabSize] f32
        public let targetLogitsBuffer: MTLBuffer    // [maxNodes * vocabSize] f32
        public let treeHiddenBuffer: MTLBuffer      // [maxNodes * hiddenDim] f32
        public let treeAttnOutBuffer: MTLBuffer     // [maxNodes * hiddenDim] f32

        // Scratch buffers for multi-node parallel tree forward pass
        public let treeXNorm1Buffer: MTLBuffer      // [maxNodes * hiddenDim] f32
        public let treeQGateBuffer: MTLBuffer       // [maxNodes * maxQkvDim] f32
        public let treeZGateBuffer: MTLBuffer       // [maxNodes * maxZDim] f32
        public let treeAttnCtxBuffer: MTLBuffer     // [maxNodes * maxZDim] f32
        public let treeKVectorBuffer: MTLBuffer     // [maxNodes * kvStride] f32
        public let treeVVectorBuffer: MTLBuffer     // [maxNodes * kvStride] f32
        public let treeAVectorBuffer: MTLBuffer     // [maxNodes * 64] f32
        public let treeBVectorBuffer: MTLBuffer     // [maxNodes * 64] f32
        public let treeHMidBuffer: MTLBuffer        // [maxNodes * hiddenDim] f32
        public let treeXNorm2Buffer: MTLBuffer      // [maxNodes * hiddenDim] f32
        public let treeInterBuffer: MTLBuffer       // [maxNodes * intermediateDim] f32
        public let treeHMlpBuffer: MTLBuffer        // [maxNodes * hiddenDim] f32
        public let treeRouterIndicesBuffer: MTLBuffer // [maxNodes * topK] u32
        public let treeRouterWeightsBuffer: MTLBuffer // [maxNodes * topK] f32
        public let treeGdnParentStateBuffer: MTLBuffer // [maxNodes * 48 * 128 * 128] f32
        public let treeGdnOutStateBuffer: MTLBuffer    // [maxNodes * 48 * 128 * 128] f32

        public let maxNodes: Int
        public let vocabSize: Int
        public let hiddenDim: Int
        public let intermediateDim: Int
        public let kvStride: Int
        public let maxLinearLayers: Int
        public let linValHeads: Int
    }

    /// Allocates shared memory buffers for JetSpec speculative tree expansion and parallel verification
    public func allocateJetSpecBuffers(
        device: MTLDevice,
        maxNodes: Int = 16,
        vocabSize: Int = 152064,
        hiddenDim: Int = 4096,
        intermediateDim: Int = 14336,
        maxQkvDim: Int = 8192,
        maxZDim: Int = 8192,
        kvStride: Int = 2048,
        topK: Int = 8,
        maxLinearLayers: Int = 32,
        linValHeads: Int = 48
    ) -> JetSpecStagingBuffers? {
        let safeTopK = max(1, topK)
        let safeInterDim = max(1, intermediateDim)
        let maskBytes = max(16, maxNodes * maxNodes * MemoryLayout<Float>.stride)
        let u32Bytes = max(16, maxNodes * MemoryLayout<UInt32>.stride)
        let logitsBytes = max(16, maxNodes * vocabSize * MemoryLayout<Float>.stride)
        let hiddenBytes = max(16, maxNodes * hiddenDim * MemoryLayout<Float>.stride)
        let qkvBytes = max(16, maxNodes * maxQkvDim * MemoryLayout<Float>.stride)
        let zBytes = max(16, maxNodes * maxZDim * MemoryLayout<Float>.stride)
        let kvBytes = max(16, maxNodes * kvStride * MemoryLayout<Float>.stride)
        let abBytes = max(16, maxNodes * 64 * MemoryLayout<Float>.stride)
        let interBytes = max(16, maxNodes * safeInterDim * MemoryLayout<Float>.stride)
        let routerBytes = max(16, maxNodes * safeTopK * MemoryLayout<UInt32>.stride)
        let routerWBytes = max(16, maxNodes * safeTopK * MemoryLayout<Float>.stride)
        let gdnParentBytes = max(16, maxNodes * linValHeads * 128 * 128 * MemoryLayout<Float>.stride)
        let gdnOutStateBytes = max(16, maxLinearLayers * maxNodes * linValHeads * 128 * 128 * MemoryLayout<Float>.stride)

        guard let maskBuf = device.makeBuffer(length: maskBytes, options: .storageModeShared),
              let tokensBuf = device.makeBuffer(length: u32Bytes, options: .storageModeShared),
              let parentBuf = device.makeBuffer(length: u32Bytes, options: .storageModeShared),
              let depthsBuf = device.makeBuffer(length: u32Bytes, options: .storageModeShared),
              let draftLogitsBuf = device.makeBuffer(length: logitsBytes, options: .storageModeShared),
              let targetLogitsBuf = device.makeBuffer(length: logitsBytes, options: .storageModeShared),
              let hiddenBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let attnOutBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let xNorm1Buf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let qGateBuf = device.makeBuffer(length: qkvBytes, options: .storageModeShared),
              let zGateBuf = device.makeBuffer(length: zBytes, options: .storageModeShared),
              let attnCtxBuf = device.makeBuffer(length: zBytes, options: .storageModeShared),
              let kVecBuf = device.makeBuffer(length: kvBytes, options: .storageModeShared),
              let vVecBuf = device.makeBuffer(length: kvBytes, options: .storageModeShared),
              let aVecBuf = device.makeBuffer(length: abBytes, options: .storageModeShared),
              let bVecBuf = device.makeBuffer(length: abBytes, options: .storageModeShared),
              let hMidBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let xNorm2Buf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let interBuf = device.makeBuffer(length: interBytes, options: .storageModeShared),
              let hMlpBuf = device.makeBuffer(length: hiddenBytes, options: .storageModeShared),
              let routerIdxBuf = device.makeBuffer(length: routerBytes, options: .storageModeShared),
              let routerWBuf = device.makeBuffer(length: routerWBytes, options: .storageModeShared),
              let gdnParentBuf = device.makeBuffer(length: gdnParentBytes, options: .storageModeShared),
              let gdnOutBuf = device.makeBuffer(length: gdnOutStateBytes, options: .storageModeShared) else {
            return nil
        }

        return JetSpecStagingBuffers(
            treeMaskBuffer: maskBuf,
            candidateTokensBuffer: tokensBuf,
            parentIndicesBuffer: parentBuf,
            depthsBuffer: depthsBuf,
            draftLogitsBuffer: draftLogitsBuf,
            targetLogitsBuffer: targetLogitsBuf,
            treeHiddenBuffer: hiddenBuf,
            treeAttnOutBuffer: attnOutBuf,
            treeXNorm1Buffer: xNorm1Buf,
            treeQGateBuffer: qGateBuf,
            treeZGateBuffer: zGateBuf,
            treeAttnCtxBuffer: attnCtxBuf,
            treeKVectorBuffer: kVecBuf,
            treeVVectorBuffer: vVecBuf,
            treeAVectorBuffer: aVecBuf,
            treeBVectorBuffer: bVecBuf,
            treeHMidBuffer: hMidBuf,
            treeXNorm2Buffer: xNorm2Buf,
            treeInterBuffer: interBuf,
            treeHMlpBuffer: hMlpBuf,
            treeRouterIndicesBuffer: routerIdxBuf,
            treeRouterWeightsBuffer: routerWBuf,
            treeGdnParentStateBuffer: gdnParentBuf,
            treeGdnOutStateBuffer: gdnOutBuf,
            maxNodes: maxNodes,
            vocabSize: vocabSize,
            hiddenDim: hiddenDim,
            intermediateDim: intermediateDim,
            kvStride: kvStride,
            maxLinearLayers: maxLinearLayers,
            linValHeads: linValHeads
        )
    }

    public static func sampleNextToken(
        logits: UnsafeMutablePointer<Float>,
        vocabSize: Int,
        contextTokens: [UInt32],
        temperature: Float,
        topP: Float,
        minP: Float,
        topK: Int,
        repetitionPenalty: Float,
        presencePenalty: Float = 0.0
    ) -> UInt32 {
        // 1. Direct repetition and presence penalties to recent context tokens (no Set lookup across 166k items)
        let recent = contextTokens.suffix(256)
        var origVals: [(Int, Float)] = []
        if repetitionPenalty > 1.001 || abs(presencePenalty) > 0.001 {
            var seen = Set<Int>()
            origVals.reserveCapacity(recent.count)
            for tok in recent {
                let v = Int(tok)
                guard v < vocabSize else { continue }
                if seen.insert(v).inserted {
                    let l = logits[v]
                    origVals.append((v, l))
                    var modified = l
                    if repetitionPenalty > 1.001 {
                        modified = modified > 0 ? (modified / repetitionPenalty) : (modified * repetitionPenalty)
                    }
                    if abs(presencePenalty) > 0.001 {
                        modified -= presencePenalty
                    }
                    logits[v] = modified
                }
            }
        }
        defer {
            for (v, orig) in origVals {
                logits[v] = orig
            }
        }

        // Hardware-Vectorized Greedy Fast Path (temperature <= 0.01) using vDSP_maxvi
        if temperature <= 0.01 {
            var bestIdx: vDSP_Length = 0
            var bestVal: Float = 0
            vDSP_maxvi(logits, 1, &bestVal, &bestIdx, vDSP_Length(vocabSize))
            return UInt32(bestIdx)
        }

        // 2. High-performance top-K selection using Min-Heap (O(log K) per candidate without array shifting)
        let effectiveTopK = max(1, min(topK, vocabSize))
        var heap: [(id: Int, logit: Float)] = []
        heap.reserveCapacity(effectiveTopK)

        for v in 0..<effectiveTopK {
            heap.append((id: v, logit: logits[v]))
        }
        // Build min-heap (root at index 0 is minimum element)
        if effectiveTopK > 1 {
            for i in stride(from: (effectiveTopK / 2) - 1, through: 0, by: -1) {
                var parent = i
                while true {
                    let left = 2 * parent + 1
                    let right = left + 1
                    var smallest = parent
                    if left < effectiveTopK && heap[left].logit < heap[smallest].logit {
                        smallest = left
                    }
                    if right < effectiveTopK && heap[right].logit < heap[smallest].logit {
                        smallest = right
                    }
                    if smallest != parent {
                        heap.swapAt(parent, smallest)
                        parent = smallest
                    } else {
                        break
                    }
                }
            }
        }

        // Stream through remaining logits, updating min-heap in O(log K) without array shifting
        var minLogit = heap[0].logit
        for v in effectiveTopK..<vocabSize {
            let logit = logits[v]
            if logit > minLogit {
                heap[0] = (id: v, logit: logit)
                var parent = 0
                while true {
                    let left = 2 * parent + 1
                    let right = left + 1
                    var smallest = parent
                    if left < effectiveTopK && heap[left].logit < heap[smallest].logit {
                        smallest = left
                    }
                    if right < effectiveTopK && heap[right].logit < heap[smallest].logit {
                        smallest = right
                    }
                    if smallest != parent {
                        heap.swapAt(parent, smallest)
                        parent = smallest
                    } else {
                        break
                    }
                }
                minLogit = heap[0].logit
            }
        }

        // Restore context logits
        for (v, orig) in origVals {
            logits[v] = orig
        }

        let candidates = heap.sorted(by: { $0.logit > $1.logit })
        guard let first = candidates.first else { return 0 }

        let count = candidates.count
        var logitVec = candidates.map { $0.logit }
        var scaledLogits = [Float](repeating: 0, count: count)
        var probs = [Float](repeating: 0, count: count)

        let invTemp = 1.0 / max(temperature, 0.01)
        let maxLogit = first.logit

        // Vectorized: (logits - maxLogit) * invTemp using Accelerate vDSP
        var negMaxLogit = -maxLogit
        var tempScale = invTemp
        vDSP_vsadd(logitVec, 1, &negMaxLogit, &scaledLogits, 1, vDSP_Length(count))
        vDSP_vsmul(scaledLogits, 1, &tempScale, &scaledLogits, 1, vDSP_Length(count))

        // Vectorized exponential: probs = exp(scaledLogits)
        var n = Int32(count)
        vvexpf(&probs, scaledLogits, &n)

        // Vectorized sum of exponents
        var expSum: Float = 0
        vDSP_sve(probs, 1, &expSum, vDSP_Length(count))

        if expSum <= 0.0 {
            return UInt32(first.id)
        }

        // Vectorized normalization: probs = probs / expSum
        var invExpSum = 1.0 / expSum
        vDSP_vsmul(probs, 1, &invExpSum, &probs, 1, vDSP_Length(count))

        // 3. Adaptive Min-P Dynamic Truncation
        let maxProb = probs[0]
        let minPThreshold = maxProb * max(0.0, min(minP, 1.0))
        var minPFilteredCount = count
        for i in 0..<count {
            if probs[i] < minPThreshold {
                minPFilteredCount = max(1, i)
                break
            }
        }

        // 4. Top-P (Nucleus) Truncation on top of Min-P
        var cumulativeProb: Float = 0.0
        var cutoffIndex = minPFilteredCount - 1
        for i in 0..<minPFilteredCount {
            cumulativeProb += probs[i]
            if cumulativeProb >= topP {
                cutoffIndex = i
                break
            }
        }

        var nucleusSum: Float = 0
        vDSP_sve(probs, 1, &nucleusSum, vDSP_Length(cutoffIndex + 1))
        let randomVal = Float.random(in: 0..<1.0) * nucleusSum
        var runningSum: Float = 0.0
        for i in 0...cutoffIndex {
            runningSum += probs[i]
            if runningSum >= randomVal {
                return UInt32(candidates[i].id)
            }
        }

        return UInt32(candidates[0].id)
    }
}

public typealias CachedLayer = EngineCachedLayer
