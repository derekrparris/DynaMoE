//
//  ComputeShaders.metal
//  DynaMoE
//
//  Created by Derek Parris on 8/18/26.
//

#include <metal_stdlib>
using namespace metal;

/// Unpacks an FP8 (E4M3) byte into an unscaled FP32 float
inline float unpack_e4m3(uchar u) {
    uint sign = (u >> 7) & 0x01;
    uint exp  = (u >> 3) & 0x0F;
    uint mant = u & 0x07;
    
    float val = 0.0f;
    if (exp == 0) {
        val = ((float)mant / 8.0f) * 0.015625f; // Subnormal (2^-6)
    } else {
        val = (1.0f + ((float)mant / 8.0f)) * pow(2.0f, (float)exp - 7.0f); // Normalized
    }
    return sign ? -val : val;
}

/// Decodes an OCP MXFP8 E8M0 scale byte into a floating-point multiplier
inline float decode_e8m0_scale(uchar s) {
    if (s == 0) return 0.0f;
    return exp2((float)s - 127.0f); // 2^(s - 127)
}

/// Converts a BF16 (bfloat16) word into an IEEE FP32 float
inline float bf16_to_fp32(ushort u) {
    uint32_t bits = ((uint32_t)u) << 16;
    return as_type<float>(bits);
}

/// MSL Kernel: Dequantizes per-row scaled FP8 weights for preview
kernel void dequantize_fp8_row_scaled(
    device const uchar* rawBaseBuffer [[buffer(0)]],
    device float* outputPreview [[buffer(1)]],
    device const ushort* rawScaleBuffer [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint64_t& scaleOffset [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float scale = bf16_to_fp32(rawScaleBuffer[scaleOffset / 2]);
    uchar rawFp8Byte = rawBaseBuffer[weightOffset + id];
    outputPreview[id] = unpack_e4m3(rawFp8Byte) * scale;
}

/// MSL Kernel: Dequantizes raw BF16 weights for preview
kernel void dequantize_bf16_preview(
    device const ushort* rawBaseBuffer [[buffer(0)]],
    device float* outputPreview [[buffer(1)]],
    constant uint64_t& weightOffset [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    ushort rawBf16 = rawBaseBuffer[(weightOffset / 2) + id];
    outputPreview[id] = bf16_to_fp32(rawBf16);
}

/// MSL Kernel: Unpacks 4 FP8 weights per U32 word and scales them via paired U8 block scales
kernel void dequantize_mxfp8_paired(
    device const uchar* rawBaseBuffer [[buffer(0)]],
    device float* outputPreview [[buffer(1)]],
    constant uint64_t& weightOffset [[buffer(2)]],
    constant uint64_t& scaleOffset [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    // 1. Extract raw FP8 byte (4 packed weights per U32 word)
    uint32_t u32_index = id / 4;
    uint32_t byte_in_u32 = id % 4;
    
    device const uint32_t* u32Weights = (device const uint32_t*)(rawBaseBuffer + weightOffset);
    uint32_t packedWord = u32Weights[u32_index];
    uchar rawFp8Byte = (packedWord >> (byte_in_u32 * 8)) & 0xFF;
    
    // 2. Fetch block scale (1 U8 scale per 8 weights)
    uint32_t scale_index = id / 8;
    device const uchar* scales = rawBaseBuffer + scaleOffset;
    uchar scaleByte = scales[scale_index];
    
    // 3. Dequantize FP8 weight and apply block scaling
    float unscaledWeight = unpack_e4m3(rawFp8Byte);
    float blockScale = decode_e8m0_scale(scaleByte);
    
    outputPreview[id] = unscaledWeight * blockScale;
}

/// MSL Kernel: Looks up token IDs in mapped FP8 (compressed-tensors) embedding tables with BF16 row scales
kernel void lookup_embeddings_fp8(
    device const uchar* rawBaseBuffer [[buffer(0)]],
    device const uint32_t* tokenIds [[buffer(1)]],
    device float* outputHiddenStates [[buffer(2)]],
    device const ushort* rawScaleBuffer [[buffer(3)]],
    constant uint64_t& weightOffset [[buffer(4)]],
    constant uint64_t& scaleOffset [[buffer(5)]],
    constant uint32_t& hiddenDim [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint tokenIdx = id / hiddenDim;
    uint dimIdx = id % hiddenDim;
    uint32_t targetTokenId = tokenIds[tokenIdx];
    
    float rowScale = bf16_to_fp32(rawScaleBuffer[(scaleOffset / 2) + targetTokenId]);
    uint64_t globalIdx = weightOffset + ((uint64_t)targetTokenId * hiddenDim) + dimIdx;
    uchar rawFp8Byte = rawBaseBuffer[globalIdx];
    
    outputHiddenStates[id] = unpack_e4m3(rawFp8Byte) * rowScale;
}

/// MSL Kernel: Looks up token IDs in mapped MXFP8 embedding tables and outputs hidden state h_0
kernel void lookup_embeddings_mxfp8(
    device const uchar* rawBaseBuffer [[buffer(0)]],
    device const uint32_t* tokenIds [[buffer(1)]],
    device float* outputHiddenStates [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint64_t& scaleOffset [[buffer(4)]],
    constant uint32_t& hiddenDim [[buffer(5)]], // e.g., 2048
    uint id [[thread_position_in_grid]]
) {
    uint tokenIdx = id / hiddenDim;
    uint dimIdx = id % hiddenDim;
    uint32_t targetTokenId = tokenIds[tokenIdx];
    
    // 1. Calculate row offsets for target token ID
    // 2048 hidden floats stored as 512 packed U32 words (4 bytes each)
    uint32_t u32PerRow = hiddenDim / 4;
    uint32_t u32GlobalIdx = (targetTokenId * u32PerRow) + (dimIdx / 4);
    uint32_t byteInU32 = dimIdx % 4;
    
    device const uint32_t* u32Weights = (device const uint32_t*)(rawBaseBuffer + weightOffset);
    uint32_t packedWord = u32Weights[u32GlobalIdx];
    uchar rawFp8Byte = (packedWord >> (byteInU32 * 8)) & 0xFF;
    
    // 2. Fetch block scale (32 weights share 1 scale byte for 2048 dim / 64 scales)
    uint32_t scalesPerRow = hiddenDim / 32;
    uint32_t scaleGlobalIdx = (targetTokenId * scalesPerRow) + (dimIdx / 32);
    
    device const uchar* scales = rawBaseBuffer + scaleOffset;
    uchar scaleByte = scales[scaleGlobalIdx];
    
    // 3. Dequantize and write to output hidden state h_0
    float unscaledWeight = unpack_e4m3(rawFp8Byte);
    float blockScale = decode_e8m0_scale(scaleByte);
    
    outputHiddenStates[id] = unscaledWeight * blockScale;
}

/// MSL Kernel: Unpacks BF16 (bfloat16) embedding rows into FP32 h_0 hidden state
kernel void lookup_embeddings_bf16(
    device const ushort* rawBaseBuffer [[buffer(0)]],
    device const uint32_t* tokenIds [[buffer(1)]],
    device float* outputHiddenStates [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint32_t& hiddenDim [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    uint tokenIdx = id / hiddenDim;
    uint dimIdx = id % hiddenDim;
    uint32_t targetTokenId = tokenIds[tokenIdx];
    
    // Calculate 16-bit word offset for target token row
    uint64_t baseUshortOffset = weightOffset / 2;
    uint64_t globalIdx = baseUshortOffset + ((uint64_t)targetTokenId * hiddenDim) + dimIdx;
    
    // Convert BF16 to FP32 by bit-shifting top 16 bits
    ushort rawBf16 = rawBaseBuffer[globalIdx];
    outputHiddenStates[id] = bf16_to_fp32(rawBf16);
}

/// MSL Kernel: MoE Top-K Router
/// Computes expert gating logits (z = W_gate * h), extracts top-K expert IDs, and applies normalized Softmax.
kernel void moe_router_topk_bf16(
    device const ushort* rawBaseBuffer [[buffer(0)]],
    device const float* inputHiddenState [[buffer(1)]],
    device uint32_t* outExpertIndices [[buffer(2)]],
    device float* outRoutingWeights [[buffer(3)]],
    constant uint64_t& gateWeightOffset [[buffer(4)]],
    constant uint32_t& hiddenDim [[buffer(5)]],
    constant uint32_t& numExperts [[buffer(6)]],
    constant uint32_t& topK [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tokenIdx [[threadgroup_position_in_grid]]
) {
    // Shared threadgroup memory for expert logits (supports up to 256 experts)
    threadgroup float sharedLogits[256];
    
    // 1. Each thread computes the dot-product logit for expert `tid`
    if (tid < numExperts && tid < 256) {
        uint64_t baseUshortOffset = gateWeightOffset / 2;
        uint64_t rowOffset = baseUshortOffset + ((uint64_t)tid * (uint64_t)hiddenDim);
        device const float* tokenH = inputHiddenState + (tokenIdx * hiddenDim);
        device const ushort* wRow = rawBaseBuffer + rowOffset;
        
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        uint32_t num4 = hiddenDim / 4;
        for (uint32_t i = 0; i < num4; i++) {
            uint32_t d = i * 4;
            sum0 += bf16_to_fp32(wRow[d + 0]) * tokenH[d + 0];
            sum0 += bf16_to_fp32(wRow[d + 1]) * tokenH[d + 1];
            sum1 += bf16_to_fp32(wRow[d + 2]) * tokenH[d + 2];
            sum1 += bf16_to_fp32(wRow[d + 3]) * tokenH[d + 3];
        }
        sharedLogits[tid] = sum0 + sum1;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Thread 0 extracts Top-K experts and computes numerically stable Softmax
    if (tid == 0) {
        uint32_t topIndices[32];
        float topValues[32];
        uint32_t k = min(topK, (uint32_t)32);
        
        for (uint32_t i = 0; i < k; i++) {
            topValues[i] = -INFINITY;
            topIndices[i] = 0;
        }
        
        uint32_t validExperts = min(numExperts, (uint32_t)256);
        for (uint32_t e = 0; e < validExperts; e++) {
            float val = sharedLogits[e];
            if (val > topValues[k - 1]) {
                int insertPos = (int)k - 1;
                while (insertPos > 0 && val > topValues[insertPos - 1]) {
                    topValues[insertPos] = topValues[insertPos - 1];
                    topIndices[insertPos] = topIndices[insertPos - 1];
                    insertPos--;
                }
                topValues[insertPos] = val;
                topIndices[insertPos] = e;
            }
        }
        
        // Softmax over selected Top-K values
        float maxVal = topValues[0];
        float sumExp = 0.0f;
        float exps[32];
        for (uint32_t i = 0; i < k; i++) {
            exps[i] = exp(topValues[i] - maxVal);
            sumExp += exps[i];
        }
        
        float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        
        device uint32_t* tokenOutIndices = outExpertIndices + (tokenIdx * topK);
        device float* tokenOutWeights = outRoutingWeights + (tokenIdx * topK);
        
        for (uint32_t i = 0; i < k; i++) {
            tokenOutIndices[i] = topIndices[i];
            tokenOutWeights[i] = exps[i] * invSum;
        }
    }
}

/// MSL Kernel: Computes Sigmoid activation for the Shared Expert Gate
kernel void moe_shared_gate_bf16(
    device const ushort* rawBaseBuffer [[buffer(0)]],
    device const float* inputHiddenState [[buffer(1)]],
    device float* outSharedWeight [[buffer(2)]],
    constant uint64_t& gateWeightOffset [[buffer(3)]],
    constant uint32_t& hiddenDim [[buffer(4)]],
    uint tokenIdx [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]]
) {
    if (tid != 0) return;
    
    uint64_t baseUshortOffset = gateWeightOffset / 2;
    device const float* tokenH = inputHiddenState + (tokenIdx * hiddenDim);
    
    float sum = 0.0f;
    for (uint32_t d = 0; d < hiddenDim; d++) {
        float w = bf16_to_fp32(rawBaseBuffer[baseUshortOffset + d]);
        sum += w * tokenH[d];
    }
    
    // Sigmoid: 1 / (1 + exp(-sum))
    float sig = 1.0f / (1.0f + exp(-sum));
    outSharedWeight[tokenIdx] = sig;
}

/// MSL Kernel: Zeros a float buffer in GPU memory
kernel void clear_vector_f32(
    device float* buffer [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    buffer[id] = 0.0f;
}

/// MSL Kernel: Fused Per-Channel FP8 (E4M3FN with BF16 row scale) SwiGLU Gate & Up Projections
kernel void fp8_swiglu_gate_up(
    device const uchar* rawGateBuffer [[buffer(0)]],
    device const uchar* rawUpBuffer [[buffer(1)]],
    device const float* inputVector [[buffer(2)]],
    device float* intermediateOutput [[buffer(3)]],
    device const ushort* rawGateScaleBuffer [[buffer(4)]],
    device const ushort* rawUpScaleBuffer [[buffer(5)]],
    constant uint64_t& gateWeightOffset [[buffer(6)]],
    constant uint64_t& gateScaleOffset [[buffer(7)]],
    constant uint64_t& upWeightOffset [[buffer(8)]],
    constant uint64_t& upScaleOffset [[buffer(9)]],
    constant uint32_t& hiddenDim [[buffer(10)]],
    constant uint32_t& intermediateDim [[buffer(11)]],
    uint r [[thread_position_in_grid]]
) {
    if (r >= intermediateDim) return;

    // 1. Fetch BF16 per-row scale
    float gateScale = bf16_to_fp32(rawGateScaleBuffer[(gateScaleOffset / 2) + r]);
    float upScale   = bf16_to_fp32(rawUpScaleBuffer[(upScaleOffset / 2) + r]);

    // 2. Vectorized dot products over hiddenDim in chunks of 4
    device const uchar* gRow = rawGateBuffer + gateWeightOffset + ((uint64_t)r * hiddenDim);
    device const uchar* uRow = rawUpBuffer + upWeightOffset + ((uint64_t)r * hiddenDim);

    float gate_dot0 = 0.0f;
    float gate_dot1 = 0.0f;
    float up_dot0   = 0.0f;
    float up_dot1   = 0.0f;

    uint32_t num4 = hiddenDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t baseD = i * 4;
        uchar4 g4 = *(device const uchar4*)(gRow + baseD);
        uchar4 u4 = *(device const uchar4*)(uRow + baseD);
        float4 in4 = *(device const float4*)(inputVector + baseD);

        gate_dot0 += (unpack_e4m3(g4.x) * in4.x) + (unpack_e4m3(g4.y) * in4.y);
        gate_dot1 += (unpack_e4m3(g4.z) * in4.z) + (unpack_e4m3(g4.w) * in4.w);

        up_dot0 += (unpack_e4m3(u4.x) * in4.x) + (unpack_e4m3(u4.y) * in4.y);
        up_dot1 += (unpack_e4m3(u4.z) * in4.z) + (unpack_e4m3(u4.w) * in4.w);
    }

    float finalGate = (gate_dot0 + gate_dot1) * gateScale;
    float finalUp   = (up_dot0 + up_dot1) * upScale;

    // 3. SwiGLU activation: silu(gate) * up
    float silu_gate = finalGate / (1.0f + exp(-finalGate));
    intermediateOutput[r] = silu_gate * finalUp;
}

/// MSL Kernel: Per-Channel FP8 Down-Projection with Weighted Accumulation
kernel void fp8_down_proj_accumulate(
    device const uchar* rawDownBuffer [[buffer(0)]],
    device const float* intermediateVector [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    device const ushort* rawDownScaleBuffer [[buffer(3)]],
    constant uint64_t& downWeightOffset [[buffer(4)]],
    constant uint64_t& downScaleOffset [[buffer(5)]],
    constant uint32_t& intermediateDim [[buffer(6)]],
    constant uint32_t& hiddenDim [[buffer(7)]],
    constant float& routingWeight [[buffer(8)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= hiddenDim) return;

    // 1. Fetch BF16 per-row scale
    float downScale = bf16_to_fp32(rawDownScaleBuffer[(downScaleOffset / 2) + d]);

    // 2. Vectorized dot product over intermediateDim in chunks of 4
    device const uchar* dRow = rawDownBuffer + downWeightOffset + ((uint64_t)d * intermediateDim);

    float down_dot0 = 0.0f;
    float down_dot1 = 0.0f;
    uint32_t num4 = intermediateDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t baseI = i * 4;
        uchar4 d4 = *(device const uchar4*)(dRow + baseI);
        float4 in4 = *(device const float4*)(intermediateVector + baseI);

        down_dot0 += (unpack_e4m3(d4.x) * in4.x) + (unpack_e4m3(d4.y) * in4.y);
        down_dot1 += (unpack_e4m3(d4.z) * in4.z) + (unpack_e4m3(d4.w) * in4.w);
    }

    float finalDown = (down_dot0 + down_dot1) * downScale;
    outputAccumulator[d] += routingWeight * finalDown;
}

/// MSL Kernel: General Per-Channel FP8 GEMV (out = (W_fp8 * in) * scale_bf16)
kernel void fp8_gemv(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const float* inputVector [[buffer(1)]],
    device float* outputVector [[buffer(2)]],
    device const ushort* rawScaleBuffer [[buffer(3)]],
    constant uint64_t& weightOffset [[buffer(4)]],
    constant uint64_t& scaleOffset [[buffer(5)]],
    constant uint32_t& inDim [[buffer(6)]],
    constant uint32_t& outDim [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= outDim) return;

    float rowScale = bf16_to_fp32(rawScaleBuffer[(scaleOffset / 2) + row]);
    device const uchar* rPtr = rawWeightBuffer + weightOffset + ((uint64_t)row * inDim);

    float dot0 = 0.0f;
    float dot1 = 0.0f;
    uint32_t num4 = inDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t base = i * 4;
        uchar4 w4 = *(device const uchar4*)(rPtr + base);
        float4 in4 = *(device const float4*)(inputVector + base);
        dot0 += (unpack_e4m3(w4.x) * in4.x) + (unpack_e4m3(w4.y) * in4.y);
        dot1 += (unpack_e4m3(w4.z) * in4.z) + (unpack_e4m3(w4.w) * in4.w);
    }

    outputVector[row] = (dot0 + dot1) * rowScale;
}

/// MSL Kernel: Fused BF16 SwiGLU Gate & Up Projections
kernel void bf16_swiglu_gate_up(
    device const ushort* rawGateBuffer [[buffer(0)]],
    device const ushort* rawUpBuffer [[buffer(1)]],
    device const float* inputVector [[buffer(2)]],
    device float* intermediateOutput [[buffer(3)]],
    constant uint64_t& gateWeightOffset [[buffer(4)]],
    constant uint64_t& upWeightOffset [[buffer(5)]],
    constant uint32_t& hiddenDim [[buffer(6)]],
    constant uint32_t& intermediateDim [[buffer(7)]],
    uint r [[thread_position_in_grid]]
) {
    if (r >= intermediateDim) return;

    uint64_t gateRowStart = (gateWeightOffset / 2) + ((uint64_t)r * hiddenDim);
    uint64_t upRowStart   = (upWeightOffset / 2) + ((uint64_t)r * hiddenDim);

    float gate_dot = 0.0f;
    float up_dot   = 0.0f;

    uint32_t num4 = hiddenDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t d = i * 4;
        gate_dot += bf16_to_fp32(rawGateBuffer[gateRowStart + d + 0]) * inputVector[d + 0];
        gate_dot += bf16_to_fp32(rawGateBuffer[gateRowStart + d + 1]) * inputVector[d + 1];
        gate_dot += bf16_to_fp32(rawGateBuffer[gateRowStart + d + 2]) * inputVector[d + 2];
        gate_dot += bf16_to_fp32(rawGateBuffer[gateRowStart + d + 3]) * inputVector[d + 3];

        up_dot += bf16_to_fp32(rawUpBuffer[upRowStart + d + 0]) * inputVector[d + 0];
        up_dot += bf16_to_fp32(rawUpBuffer[upRowStart + d + 1]) * inputVector[d + 1];
        up_dot += bf16_to_fp32(rawUpBuffer[upRowStart + d + 2]) * inputVector[d + 2];
        up_dot += bf16_to_fp32(rawUpBuffer[upRowStart + d + 3]) * inputVector[d + 3];
    }

    float silu_gate = gate_dot / (1.0f + exp(-gate_dot));
    intermediateOutput[r] = silu_gate * up_dot;
}

/// MSL Kernel: BF16 Down-Projection with Weighted Accumulation
kernel void bf16_down_proj_accumulate(
    device const ushort* rawDownBuffer [[buffer(0)]],
    device const float* intermediateVector [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    constant uint64_t& downWeightOffset [[buffer(3)]],
    constant uint32_t& intermediateDim [[buffer(4)]],
    constant uint32_t& hiddenDim [[buffer(5)]],
    constant float& routingWeight [[buffer(6)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= hiddenDim) return;

    uint64_t downRowStart = (downWeightOffset / 2) + ((uint64_t)d * intermediateDim);
    float down_dot = 0.0f;

    uint32_t num4 = intermediateDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t idx = i * 4;
        down_dot += bf16_to_fp32(rawDownBuffer[downRowStart + idx + 0]) * intermediateVector[idx + 0];
        down_dot += bf16_to_fp32(rawDownBuffer[downRowStart + idx + 1]) * intermediateVector[idx + 1];
        down_dot += bf16_to_fp32(rawDownBuffer[downRowStart + idx + 2]) * intermediateVector[idx + 2];
        down_dot += bf16_to_fp32(rawDownBuffer[downRowStart + idx + 3]) * intermediateVector[idx + 3];
    }

    outputAccumulator[d] += routingWeight * down_dot;
}

// =============================================================================
// MARK: - RMSNorm, Vector Addition & GEMV Compute Kernels
// =============================================================================

/// MSL Kernel: RMSNorm with BF16 Scale Factors
/// y_d = (x_d / sqrt( (1/D) * sum(x^2) + eps )) * gamma_d
kernel void rmsnorm_bf16(
    device const float* inVector [[buffer(0)]],
    device const ushort* gammaBuffer [[buffer(1)]],
    device float* outVector [[buffer(2)]],
    constant uint64_t& gammaOffset [[buffer(3)]],
    constant uint32_t& dim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    threadgroup float* sharedSum [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgSize [[threads_per_threadgroup]]
) {
    // 1. Parallel threadgroup reduction for sum of squares
    float localSum = 0.0f;
    for (uint i = tid; i < dim; i += tgSize) {
        float v = inVector[i];
        localSum += v * v;
    }
    sharedSum[tid] = localSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tgSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float meanSquare = sharedSum[0] / (float)dim;
    float invRms = rsqrt(meanSquare + eps);

    // 2. Normalize and scale by gamma
    uint64_t gammaStart = gammaOffset / 2;
    for (uint i = tid; i < dim; i += tgSize) {
        float gamma = bf16_to_fp32(gammaBuffer[gammaStart + i]);
        outVector[i] = inVector[i] * invRms * gamma;
    }
}

/// MSL Kernel: Parallel Element-Wise Vector Addition (Residual Connection)
/// out[d] = a[d] + b[d]
kernel void vector_add_f32(
    device const float* aVector [[buffer(0)]],
    device const float* bVector [[buffer(1)]],
    device float* outVector [[buffer(2)]],
    constant uint32_t& dim [[buffer(3)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= dim) return;
    outVector[d] = aVector[d] + bVector[d];
}

/// MSL Kernel: General MXFP8 Matrix-Vector GEMV (out = W * x)
kernel void mxfp8_gemv(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const float* inputVector [[buffer(1)]],
    device float* outputVector [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint64_t& scaleOffset [[buffer(4)]],
    constant uint32_t& inDim [[buffer(5)]],
    constant uint32_t& outDim [[buffer(6)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= outDim) return;

    uint32_t blocksPerRow = inDim / 32;
    uint64_t rowWeightStart = weightOffset + ((uint64_t)row * inDim);
    uint64_t rowScaleStart  = scaleOffset  + ((uint64_t)row * blocksPerRow);

    float dot = 0.0f;

    for (uint32_t b = 0; b < blocksPerRow; b++) {
        uint8_t scaleByte = rawWeightBuffer[rowScaleStart + b];
        int scaleExp = (int)scaleByte - 127;
        float scale = exp2((float)scaleExp);

        uint64_t blkStart = rowWeightStart + ((uint64_t)b * 32);
        uint32_t inStart = b * 32;

        for (uint32_t i = 0; i < 32; i++) {
            uint8_t byteVal = rawWeightBuffer[blkStart + i];
            float w = unpack_e4m3(byteVal) * scale;
            dot += w * inputVector[inStart + i];
        }
    }

    outputVector[row] = dot;
}

/// MSL Kernel: General BF16 Matrix-Vector GEMV (out = W * x)
kernel void bf16_gemv(
    device const ushort* rawWeightBuffer [[buffer(0)]],
    device const float* inputVector [[buffer(1)]],
    device float* outputVector [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint32_t& inDim [[buffer(4)]],
    constant uint32_t& outDim [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= outDim) return;

    uint64_t rowWeightStart = (weightOffset / 2) + ((uint64_t)row * inDim);
    device const ushort* wRow = rawWeightBuffer + rowWeightStart;

    float dot0 = 0.0f;
    float dot1 = 0.0f;
    uint32_t num4 = inDim / 4;
    for (uint32_t i = 0; i < num4; i++) {
        uint32_t base = i * 4;
        dot0 += bf16_to_fp32(wRow[base + 0]) * inputVector[base + 0];
        dot0 += bf16_to_fp32(wRow[base + 1]) * inputVector[base + 1];
        dot1 += bf16_to_fp32(wRow[base + 2]) * inputVector[base + 2];
        dot1 += bf16_to_fp32(wRow[base + 3]) * inputVector[base + 3];
    }

    outputVector[row] = dot0 + dot1;
}

/// MSL Kernel: Per-Head RMSNorm for Attention Heads (Q-Norm and K-Norm)
kernel void per_head_rmsnorm_bf16(
    device float* qkVector [[buffer(0)]],
    device const ushort* gammaBuffer [[buffer(1)]],
    constant uint64_t& gammaOffset [[buffer(2)]],
    constant uint32_t& numHeads [[buffer(3)]],
    constant uint32_t& headDim [[buffer(4)]],
    constant float& eps [[buffer(5)]],
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numHeads) return;

    uint32_t offset = headIdx * headDim;
    float sumSq = 0.0f;
    for (uint32_t d = 0; d < headDim; d++) {
        float v = qkVector[offset + d];
        sumSq += v * v;
    }

    float invRms = rsqrt((sumSq / (float)headDim) + eps);
    uint64_t gammaStart = gammaOffset / 2;

    for (uint32_t d = 0; d < headDim; d++) {
        float gamma = bf16_to_fp32(gammaBuffer[gammaStart + d]);
        qkVector[offset + d] = qkVector[offset + d] * invRms * gamma;
    }
}

