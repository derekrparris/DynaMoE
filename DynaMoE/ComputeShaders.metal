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

inline ushort read_u16_unaligned(device const uchar* p) {
    return (ushort)p[0] | ((ushort)p[1] << 8);
}

inline float read_bf16_unaligned(device const uchar* p) {
    ushort val = (ushort)p[0] | ((ushort)p[1] << 8);
    uint32_t bits = ((uint32_t)val) << 16;
    return as_type<float>(bits);
}

inline float read_f16_unaligned(device const uchar* p) {
    ushort val = (ushort)p[0] | ((ushort)p[1] << 8);
    return (float)as_type<half>(val);
}

inline uint32_t read_u32_unaligned(device const uchar* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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
    device const uchar* rawScaleBuffer [[buffer(3)]],
    constant uint64_t& weightOffset [[buffer(4)]],
    constant uint64_t& scaleOffset [[buffer(5)]],
    constant uint32_t& hiddenDim [[buffer(6)]], // e.g., 2048, 3072
    uint id [[thread_position_in_grid]]
) {
    uint tokenIdx = id / hiddenDim;
    uint dimIdx = id % hiddenDim;
    uint32_t targetTokenId = tokenIds[tokenIdx];
    
    // 1. Calculate row offsets for target token ID
    uint64_t byteOffset = weightOffset + ((uint64_t)targetTokenId * hiddenDim) + dimIdx;
    uchar rawFp8Byte = rawBaseBuffer[byteOffset];
    
    // 2. Fetch block scale (32 weights share 1 scale byte)
    uint32_t scalesPerRow = hiddenDim / 32;
    uint64_t scaleGlobalIdx = scaleOffset + ((uint64_t)targetTokenId * scalesPerRow) + (dimIdx / 32);
    uchar scaleByte = rawScaleBuffer[scaleGlobalIdx];
    
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

/// MSL Kernel: MoE Top-K Router with Q4 (4-bit affine quantized) weights
kernel void moe_router_topk_q4(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const uchar* rawScaleBuffer [[buffer(1)]],
    device const uchar* rawBiasBuffer [[buffer(2)]],
    device const float* inputHiddenState [[buffer(3)]],
    device uint32_t* outExpertIndices [[buffer(4)]],
    device float* outRoutingWeights [[buffer(5)]],
    constant uint64_t& gateWeightOffset [[buffer(6)]],
    constant uint64_t& scaleOffset [[buffer(7)]],
    constant uint64_t& biasOffset [[buffer(8)]],
    constant uint32_t& hiddenDim [[buffer(9)]],
    constant uint32_t& numExperts [[buffer(10)]],
    constant uint32_t& topK [[buffer(11)]],
    constant uint32_t& groupSize [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tokenIdx [[threadgroup_position_in_grid]]
) {
    threadgroup float sharedLogits[256];
    
    if (tid < numExperts && tid < 256) {
        uint32_t numGroups = hiddenDim / groupSize;
        device const uchar* wRow = rawWeightBuffer + gateWeightOffset + ((uint64_t)tid * (hiddenDim / 8) * 4);
        device const uchar* sRow = rawScaleBuffer + scaleOffset + ((uint64_t)tid * numGroups * 2);
        device const uchar* bRow = rawBiasBuffer + biasOffset + ((uint64_t)tid * numGroups * 2);
        device const float* tokenH = inputHiddenState + (tokenIdx * hiddenDim);
        
        float sum = 0.0f;
        uint32_t numU32 = hiddenDim / 8;
        for (uint32_t i = 0; i < numU32; i++) {
            uint32_t col = i * 8;
            uint32_t gIdx = col / groupSize;
            float scale = read_bf16_unaligned(sRow + gIdx * 2);
            float bias = read_bf16_unaligned(bRow + gIdx * 2);
            uint32_t u32 = read_u32_unaligned(wRow + i * 4);
            
            float x0 = tokenH[col + 0]; float x1 = tokenH[col + 1];
            float x2 = tokenH[col + 2]; float x3 = tokenH[col + 3];
            float x4 = tokenH[col + 4]; float x5 = tokenH[col + 5];
            float x6 = tokenH[col + 6]; float x7 = tokenH[col + 7];
            float xSum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            
            float w0 = float(u32 & 0x0F);
            float w1 = float((u32 >> 4) & 0x0F);
            float w2 = float((u32 >> 8) & 0x0F);
            float w3 = float((u32 >> 12) & 0x0F);
            float w4 = float((u32 >> 16) & 0x0F);
            float w5 = float((u32 >> 20) & 0x0F);
            float w6 = float((u32 >> 24) & 0x0F);
            float w7 = float((u32 >> 28) & 0x0F);
            
            float dot = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7;
            sum += scale * dot + bias * xSum;
        }
        sharedLogits[tid] = sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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

/// MSL Kernel: MoE Top-K Router with Q8 (8-bit affine quantized) weights
kernel void moe_router_topk_q8(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const uchar* rawScaleBuffer [[buffer(1)]],
    device const uchar* rawBiasBuffer [[buffer(2)]],
    device const float* inputHiddenState [[buffer(3)]],
    device uint32_t* outExpertIndices [[buffer(4)]],
    device float* outRoutingWeights [[buffer(5)]],
    constant uint64_t& gateWeightOffset [[buffer(6)]],
    constant uint64_t& scaleOffset [[buffer(7)]],
    constant uint64_t& biasOffset [[buffer(8)]],
    constant uint32_t& hiddenDim [[buffer(9)]],
    constant uint32_t& numExperts [[buffer(10)]],
    constant uint32_t& topK [[buffer(11)]],
    constant uint32_t& groupSize [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tokenIdx [[threadgroup_position_in_grid]]
) {
    threadgroup float sharedLogits[256];
    
    if (tid < numExperts && tid < 256) {
        uint32_t numGroups = hiddenDim / groupSize;
        device const uchar* wRow = rawWeightBuffer + gateWeightOffset + ((uint64_t)tid * hiddenDim);
        device const uchar* sRow = rawScaleBuffer + scaleOffset + ((uint64_t)tid * numGroups * 2);
        device const uchar* bRow = rawBiasBuffer + biasOffset + ((uint64_t)tid * numGroups * 2);
        device const float* tokenH = inputHiddenState + (tokenIdx * hiddenDim);
        
        float sum = 0.0f;
        for (uint32_t g = 0; g < numGroups; g++) {
            float scale = read_bf16_unaligned(sRow + g * 2);
            float bias = read_bf16_unaligned(bRow + g * 2);
            uint32_t colStart = g * groupSize;
            float groupSum = 0.0f;
            float groupXSum = 0.0f;
            for (uint32_t c = 0; c < groupSize; c++) {
                uint32_t col = colStart + c;
                float x = tokenH[col];
                float w8 = float(wRow[col]);
                groupSum += w8 * x;
                groupXSum += x;
            }
            sum += scale * groupSum + bias * groupXSum;
        }
        sharedLogits[tid] = sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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

    // 2. Vectorized dot products over hiddenDim in chunks of 8
    device const uchar* gRow = rawGateBuffer + gateWeightOffset + ((uint64_t)r * hiddenDim);
    device const uchar* uRow = rawUpBuffer + upWeightOffset + ((uint64_t)r * hiddenDim);

    float gate_dot0 = 0.0f;
    float gate_dot1 = 0.0f;
    float up_dot0   = 0.0f;
    float up_dot1   = 0.0f;

    uint32_t num8 = hiddenDim / 8;
    for (uint32_t i = 0; i < num8; i++) {
        uint32_t baseD = i * 8;
        uchar4 g4_0 = *(device const uchar4*)(gRow + baseD);
        uchar4 g4_1 = *(device const uchar4*)(gRow + baseD + 4);
        uchar4 u4_0 = *(device const uchar4*)(uRow + baseD);
        uchar4 u4_1 = *(device const uchar4*)(uRow + baseD + 4);
        float4 in4_0 = *(device const float4*)(inputVector + baseD);
        float4 in4_1 = *(device const float4*)(inputVector + baseD + 4);

        gate_dot0 += (unpack_e4m3(g4_0.x) * in4_0.x) + (unpack_e4m3(g4_0.y) * in4_0.y) +
                     (unpack_e4m3(g4_1.x) * in4_1.x) + (unpack_e4m3(g4_1.y) * in4_1.y);
        gate_dot1 += (unpack_e4m3(g4_0.z) * in4_0.z) + (unpack_e4m3(g4_0.w) * in4_0.w) +
                     (unpack_e4m3(g4_1.z) * in4_1.z) + (unpack_e4m3(g4_1.w) * in4_1.w);

        up_dot0 += (unpack_e4m3(u4_0.x) * in4_0.x) + (unpack_e4m3(u4_0.y) * in4_0.y) +
                   (unpack_e4m3(u4_1.x) * in4_1.x) + (unpack_e4m3(u4_1.y) * in4_1.y);
        up_dot1 += (unpack_e4m3(u4_0.z) * in4_0.z) + (unpack_e4m3(u4_0.w) * in4_0.w) +
                   (unpack_e4m3(u4_1.z) * in4_1.z) + (unpack_e4m3(u4_1.w) * in4_1.w);
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

    // 2. Vectorized dot product over intermediateDim in chunks of 8
    device const uchar* dRow = rawDownBuffer + downWeightOffset + ((uint64_t)d * intermediateDim);

    float down_dot0 = 0.0f;
    float down_dot1 = 0.0f;
    uint32_t num8 = intermediateDim / 8;
    for (uint32_t i = 0; i < num8; i++) {
        uint32_t baseI = i * 8;
        uchar4 d4_0 = *(device const uchar4*)(dRow + baseI);
        uchar4 d4_1 = *(device const uchar4*)(dRow + baseI + 4);
        float4 in4_0 = *(device const float4*)(intermediateVector + baseI);
        float4 in4_1 = *(device const float4*)(intermediateVector + baseI + 4);

        down_dot0 += (unpack_e4m3(d4_0.x) * in4_0.x) + (unpack_e4m3(d4_0.y) * in4_0.y) +
                     (unpack_e4m3(d4_1.x) * in4_1.x) + (unpack_e4m3(d4_1.y) * in4_1.y);
        down_dot1 += (unpack_e4m3(d4_0.z) * in4_0.z) + (unpack_e4m3(d4_0.w) * in4_0.w) +
                     (unpack_e4m3(d4_1.z) * in4_1.z) + (unpack_e4m3(d4_1.w) * in4_1.w);
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
    device const uchar* gammaBuffer [[buffer(1)]],
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
    for (uint i = tid; i < dim; i += tgSize) {
        float gamma = read_bf16_unaligned(gammaBuffer + gammaOffset + ((uint64_t)i * 2));
        outVector[i] = inVector[i] * invRms * gamma;
    }
}

/// MSL Kernel: RMSNorm with F16 (IEEE 754 half) Scale Factors
kernel void rmsnorm_f16(
    device const float* inVector [[buffer(0)]],
    device const uchar* gammaBuffer [[buffer(1)]],
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
    for (uint i = tid; i < dim; i += tgSize) {
        float gamma = read_f16_unaligned(gammaBuffer + gammaOffset + ((uint64_t)i * 2));
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
    device const uchar* rawScaleBuffer [[buffer(3)]],
    constant uint64_t& weightOffset [[buffer(4)]],
    constant uint64_t& scaleOffset [[buffer(5)]],
    constant uint32_t& inDim [[buffer(6)]],
    constant uint32_t& outDim [[buffer(7)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= outDim) return;

    uint32_t blocksPerRow = inDim / 32;
    uint64_t rowWeightStart = weightOffset + ((uint64_t)row * inDim);
    uint64_t rowScaleStart  = scaleOffset  + ((uint64_t)row * blocksPerRow);

    float dot = 0.0f;

    for (uint32_t b = 0; b < blocksPerRow; b++) {
        uint8_t scaleByte = rawScaleBuffer[rowScaleStart + b];
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

/// MSL Kernel: Fused MXFP8 SwiGLU Gate & Up Projections
kernel void mxfp8_swiglu_gate_up(
    device const uchar* gateWeightBuffer [[buffer(0)]],
    device const uchar* upWeightBuffer [[buffer(1)]],
    device const float* inputVector [[buffer(2)]],
    device float* intermediateState [[buffer(3)]],
    device const uchar* gateScaleBuffer [[buffer(4)]],
    device const uchar* upScaleBuffer [[buffer(5)]],
    constant uint64_t& gateWeightOffset [[buffer(6)]],
    constant uint64_t& gateScaleOffset [[buffer(7)]],
    constant uint64_t& upWeightOffset [[buffer(8)]],
    constant uint64_t& upScaleOffset [[buffer(9)]],
    constant uint32_t& inDim [[buffer(10)]],
    constant uint32_t& intermediateDim [[buffer(11)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= intermediateDim) return;

    uint32_t blocksPerRow = inDim / 32;
    uint64_t gRowStart = gateWeightOffset + ((uint64_t)d * inDim);
    uint64_t gScaleStart = gateScaleOffset + ((uint64_t)d * blocksPerRow);
    uint64_t uRowStart = upWeightOffset + ((uint64_t)d * inDim);
    uint64_t uScaleStart = upScaleOffset + ((uint64_t)d * blocksPerRow);

    float gate_dot = 0.0f;
    float up_dot = 0.0f;

    for (uint32_t b = 0; b < blocksPerRow; b++) {
        uint8_t gScaleByte = gateScaleBuffer[gScaleStart + b];
        float gScale = exp2((float)((int)gScaleByte - 127));

        uint8_t uScaleByte = upScaleBuffer[uScaleStart + b];
        float uScale = exp2((float)((int)uScaleByte - 127));

        uint64_t gBlk = gRowStart + ((uint64_t)b * 32);
        uint64_t uBlk = uRowStart + ((uint64_t)b * 32);
        uint32_t inStart = b * 32;

        for (uint32_t i = 0; i < 32; i++) {
            float inVal = inputVector[inStart + i];
            gate_dot += (unpack_e4m3(gateWeightBuffer[gBlk + i]) * gScale) * inVal;
            up_dot += (unpack_e4m3(upWeightBuffer[uBlk + i]) * uScale) * inVal;
        }
    }

    float silu_gate = gate_dot / (1.0f + exp(-gate_dot));
    intermediateState[d] = silu_gate * up_dot;
}

/// MSL Kernel: Fused MXFP8 Down Projection Accumulator
kernel void mxfp8_down_proj_accumulate(
    device const uchar* downWeightBuffer [[buffer(0)]],
    device const float* intermediateState [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    device const uchar* downScaleBuffer [[buffer(3)]],
    constant uint64_t& downWeightOffset [[buffer(4)]],
    constant uint64_t& downScaleOffset [[buffer(5)]],
    constant uint32_t& intermediateDim [[buffer(6)]],
    constant uint32_t& inDim [[buffer(7)]],
    constant float& routingWeight [[buffer(8)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= inDim) return;

    uint32_t blocksPerRow = intermediateDim / 32;
    uint64_t dRowStart = downWeightOffset + ((uint64_t)d * intermediateDim);
    uint64_t dScaleStart = downScaleOffset + ((uint64_t)d * blocksPerRow);

    float down_dot = 0.0f;

    for (uint32_t b = 0; b < blocksPerRow; b++) {
        uint8_t dScaleByte = downScaleBuffer[dScaleStart + b];
        float dScale = exp2((float)((int)dScaleByte - 127));

        uint64_t dBlk = dRowStart + ((uint64_t)b * 32);
        uint32_t interStart = b * 32;

        for (uint32_t i = 0; i < 32; i++) {
            down_dot += (unpack_e4m3(downWeightBuffer[dBlk + i]) * dScale) * intermediateState[interStart + i];
        }
    }

    outputAccumulator[d] += routingWeight * down_dot;
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
        ushort4 w4 = *(device const ushort4*)(wRow + base);
        float4 in4 = *(device const float4*)(inputVector + base);
        dot0 += (bf16_to_fp32(w4.x) * in4.x) + (bf16_to_fp32(w4.y) * in4.y);
        dot1 += (bf16_to_fp32(w4.z) * in4.z) + (bf16_to_fp32(w4.w) * in4.w);
    }

    outputVector[row] = dot0 + dot1;
}

/// MSL Kernel: Per-Head RMSNorm for Attention Heads (Q-Norm and K-Norm with custom stride)
kernel void per_head_rmsnorm_bf16(
    device float* qkVector [[buffer(0)]],
    device const uchar* gammaBuffer [[buffer(1)]],
    constant uint64_t& gammaOffset [[buffer(2)]],
    constant uint32_t& numHeads [[buffer(3)]],
    constant uint32_t& headDim [[buffer(4)]],
    constant uint32_t& headStride [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numHeads) return;

    uint32_t offset = headIdx * headStride;
    float sumSq = 0.0f;
    for (uint32_t d = 0; d < headDim; d++) {
        float v = qkVector[offset + d];
        sumSq += v * v;
    }

    float invRms = rsqrt((sumSq / (float)headDim) + eps);

    for (uint32_t d = 0; d < headDim; d++) {
        float gamma = read_bf16_unaligned(gammaBuffer + gammaOffset + ((uint64_t)d * 2));
        qkVector[offset + d] = qkVector[offset + d] * invRms * gamma;
    }
}

/// MSL Kernel: Per-Head RMSNorm with F16 Scale Factors
kernel void per_head_rmsnorm_f16(
    device float* qkVector [[buffer(0)]],
    device const uchar* gammaBuffer [[buffer(1)]],
    constant uint64_t& gammaOffset [[buffer(2)]],
    constant uint32_t& numHeads [[buffer(3)]],
    constant uint32_t& headDim [[buffer(4)]],
    constant uint32_t& headStride [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numHeads) return;

    uint32_t offset = headIdx * headStride;
    float sumSq = 0.0f;
    for (uint32_t d = 0; d < headDim; d++) {
        float v = qkVector[offset + d];
        sumSq += v * v;
    }

    float invRms = rsqrt((sumSq / (float)headDim) + eps);

    for (uint32_t d = 0; d < headDim; d++) {
        float gamma = read_f16_unaligned(gammaBuffer + gammaOffset + ((uint64_t)d * 2));
        qkVector[offset + d] = qkVector[offset + d] * invRms * gamma;
    }
}

/// MSL Kernel: Fused Rotary Position Embeddings (RoPE) for Qwen 3.5 / Ornith 1.5 with custom stride
kernel void apply_rope_qwen(
    device float* qkVector [[buffer(0)]],
    constant uint32_t& tokenPos [[buffer(1)]],
    constant uint32_t& numHeads [[buffer(2)]],
    constant uint32_t& headDim [[buffer(3)]],
    constant uint32_t& rotaryDim [[buffer(4)]],
    constant uint32_t& headStride [[buffer(5)]],
    constant float& ropeTheta [[buffer(6)]],
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numHeads) return;

    uint32_t headBase = headIdx * headStride;
    uint32_t halfRotary = rotaryDim / 2;

    for (uint32_t i = 0; i < halfRotary; i++) {
        float exponent = (2.0f * (float)i) / (float)rotaryDim;
        float freq = 1.0f / pow(ropeTheta, exponent);
        float angle = (float)tokenPos * freq;
        float cosVal = cos(angle);
        float sinVal = sin(angle);

        uint32_t idx0 = headBase + i;
        uint32_t idx1 = headBase + i + halfRotary;

        float v0 = qkVector[idx0];
        float v1 = qkVector[idx1];

        qkVector[idx0] = v0 * cosVal - v1 * sinVal;
        qkVector[idx1] = v0 * sinVal + v1 * cosVal;
    }
}

/// MSL Kernel: Stores current K and V vectors into the Layer's KV-Cache
kernel void store_kv_cache(
    device const float* kVector [[buffer(0)]],
    device const float* vVector [[buffer(1)]],
    device float* kCacheBuffer [[buffer(2)]],
    device float* vCacheBuffer [[buffer(3)]],
    constant uint32_t& tokenPos [[buffer(4)]],
    constant uint32_t& numKvHeads [[buffer(5)]],
    constant uint32_t& headDim [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    uint32_t kvStride = numKvHeads * headDim;
    if (id >= kvStride) return;

    uint32_t cacheOffset = (tokenPos * kvStride) + id;
    kCacheBuffer[cacheOffset] = kVector[id];
    vCacheBuffer[cacheOffset] = vVector[id];
}

///// MSL Kernel: Grouped-Query Attention (GQA) Autoregressive Decoding with Online Softmax & Sigmoid Output Gating
kernel void gqa_attention_decode_fused(
    device const float* qGateVector [[buffer(0)]], // [8192] = 16 heads * (Q[256] + Gate[256])
    device const float* kCacheBuffer [[buffer(1)]],
    device const float* vCacheBuffer [[buffer(2)]],
    device float* attnOutBuffer [[buffer(3)]],     // [4096] = 16 heads * 256
    constant uint32_t& seqLen [[buffer(4)]],
    constant uint32_t& numQHeads [[buffer(5)]],
    constant uint32_t& numKvHeads [[buffer(6)]],
    constant uint32_t& headDim [[buffer(7)]],
    uint qHeadIdx [[thread_position_in_grid]]
) {
    if (qHeadIdx >= numQHeads) return;

    uint32_t headsPerKv = numQHeads / numKvHeads;
    uint32_t kvHeadIdx = qHeadIdx / headsPerKv;

    uint32_t qHeadBase = qHeadIdx * 512;
    uint32_t gateBase = qHeadBase + 256;
    uint32_t kvStride = numKvHeads * headDim;
    uint32_t kvHeadBase = kvHeadIdx * headDim;

    float invSqrtHeadDim = rsqrt((float)headDim);

    // Online Softmax Accumulator (FlashAttention style, 1 pass, 0 stack arrays)
    float acc[256];
    for (uint32_t d = 0; d < headDim; d++) {
        acc[d] = 0.0f;
    }

    float m = -1e20f; // running max score
    float l = 0.0f;   // running sum of exponents

    uint32_t safeLen = min(seqLen, 2048u);

    for (uint32_t tau = 0; tau < safeLen; tau++) {
        uint32_t kBase = (tau * kvStride) + kvHeadBase;
        float dot = 0.0f;
        for (uint32_t d = 0; d < headDim; d++) {
            dot += qGateVector[qHeadBase + d] * kCacheBuffer[kBase + d];
        }
        float score = dot * invSqrtHeadDim;

        float m_prev = m;
        if (score > m) {
            m = score;
        }

        float alpha = exp(m_prev - m);
        float beta = exp(score - m);

        l = (l * alpha) + beta;

        uint32_t vBase = (tau * kvStride) + kvHeadBase;
        for (uint32_t d = 0; d < headDim; d++) {
            acc[d] = (acc[d] * alpha) + (beta * vCacheBuffer[vBase + d]);
        }
    }

    float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    uint32_t outOffset = qHeadIdx * headDim;

    for (uint32_t d = 0; d < headDim; d++) {
        float ctx = acc[d] * invL;
        float g = qGateVector[gateBase + d];
        float sig_g = 1.0f / (1.0f + exp(-g));
        attnOutBuffer[outOffset + d] = ctx * sig_g;
    }
}

/// MSL Kernel: Standard Grouped-Query Attention (GQA) Autoregressive Decoding for LLaMA / Mistral / Nanbeige
kernel void gqa_attention_decode_standard(
    device const float* qVector [[buffer(0)]],     // [numQHeads * headDim]
    device const float* kCacheBuffer [[buffer(1)]],
    device const float* vCacheBuffer [[buffer(2)]],
    device float* attnOutBuffer [[buffer(3)]],     // [numQHeads * headDim]
    constant uint32_t& seqLen [[buffer(4)]],
    constant uint32_t& numQHeads [[buffer(5)]],
    constant uint32_t& numKvHeads [[buffer(6)]],
    constant uint32_t& headDim [[buffer(7)]],
    uint qHeadIdx [[thread_position_in_grid]]
) {
    if (qHeadIdx >= numQHeads) return;

    uint32_t headsPerKv = numQHeads / numKvHeads;
    uint32_t kvHeadIdx = qHeadIdx / headsPerKv;

    uint32_t qHeadBase = qHeadIdx * headDim;
    uint32_t kvStride = numKvHeads * headDim;
    uint32_t kvHeadBase = kvHeadIdx * headDim;

    float invSqrtHeadDim = rsqrt((float)headDim);

    // Online Softmax Accumulator (float4 vectorized)
    float4 acc[64];
    uint32_t headDimVec = headDim / 4;
    for (uint32_t d = 0; d < headDimVec; d++) {
        acc[d] = float4(0.0f);
    }

    float m = -1e20f; // running max score
    float l = 0.0f;   // running sum of exponents

    uint32_t safeLen = min(seqLen, 2048u);
    device const float4* qHeadVec = (device const float4*)(qVector + qHeadBase);

    for (uint32_t tau = 0; tau < safeLen; tau++) {
        uint32_t kBase = (tau * kvStride) + kvHeadBase;
        device const float4* kVec = (device const float4*)(kCacheBuffer + kBase);
        
        float dot_val = 0.0f;
        for (uint32_t d = 0; d < headDimVec; d++) {
            dot_val += dot(qHeadVec[d], kVec[d]);
        }
        float score = dot_val * invSqrtHeadDim;

        float m_prev = m;
        if (score > m) {
            m = score;
        }

        float alpha = exp(m_prev - m);
        float beta = exp(score - m);

        l = (l * alpha) + beta;

        uint32_t vBase = (tau * kvStride) + kvHeadBase;
        device const float4* vVec = (device const float4*)(vCacheBuffer + vBase);
        for (uint32_t d = 0; d < headDimVec; d++) {
            acc[d] = (acc[d] * alpha) + (beta * vVec[d]);
        }
    }

    float invL = (l > 0.0f) ? (1.0f / l) : 0.0f;
    uint32_t outOffset = qHeadIdx * headDim;
    device float4* outVec = (device float4*)(attnOutBuffer + outOffset);

    for (uint32_t d = 0; d < headDimVec; d++) {
        outVec[d] = acc[d] * invL;
    }
}

///// MSL Kernel: Causal 1D Convolution with Shift Register State & SiLU Activation
/// Used in Ornith Gated DeltaNet recurrent layer prefix
kernel void causal_conv1d_silu(
    device const float* inRaw [[buffer(0)]],
    device const uchar* convWeight [[buffer(1)]],
    device float* convState [[buffer(2)]],
    device float* outQKV [[buffer(3)]],
    constant uint64_t& weightOffset [[buffer(4)]],
    constant uint32_t& numChannels [[buffer(5)]],
    uint c [[thread_position_in_grid]]
) {
    if (c >= numChannels) return;

    uint32_t stateBase = c * 4;
    float s0 = convState[stateBase + 1];
    float s1 = convState[stateBase + 2];
    float s2 = convState[stateBase + 3];
    float s3 = inRaw[c];

    convState[stateBase + 0] = s0;
    convState[stateBase + 1] = s1;
    convState[stateBase + 2] = s2;
    convState[stateBase + 3] = s3;

    device const uchar* wBase = convWeight + weightOffset + ((uint64_t)c * 4 * 2);
    float w0 = read_bf16_unaligned(wBase + 0 * 2);
    float w1 = read_bf16_unaligned(wBase + 1 * 2);
    float w2 = read_bf16_unaligned(wBase + 2 * 2);
    float w3 = read_bf16_unaligned(wBase + 3 * 2);

    float convVal = (w0 * s0) + (w1 * s1) + (w2 * s2) + (w3 * s3);
    float siluVal = convVal / (1.0f + exp(-convVal));
    outQKV[c] = siluVal;
}

/// MSL Kernel: Per-head L2 Normalization on Q and K vectors for Linear Attention
kernel void l2_norm_qk(
    device float* qkvVector [[buffer(0)]],
    constant uint32_t& numHeads [[buffer(1)]],
    constant uint32_t& headDim [[buffer(2)]],
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numHeads) return;

    // Q head at headIdx * 128
    uint32_t qBase = headIdx * headDim;
    float qSumSq = 0.0f;
    for (uint32_t i = 0; i < headDim; i++) {
        float v = qkvVector[qBase + i];
        qSumSq += v * v;
    }
    float invQNorm = rsqrt(qSumSq + 1e-6f);
    for (uint32_t i = 0; i < headDim; i++) {
        qkvVector[qBase + i] *= invQNorm;
    }

    // K head at 2048 + headIdx * 128
    uint32_t kBase = 2048 + (headIdx * headDim);
    float kSumSq = 0.0f;
    for (uint32_t i = 0; i < headDim; i++) {
        float v = qkvVector[kBase + i];
        kSumSq += v * v;
    }
    float invKNorm = rsqrt(kSumSq + 1e-6f);
    for (uint32_t i = 0; i < headDim; i++) {
        qkvVector[kBase + i] *= invKNorm;
    }
}

/// MSL Kernel: Gated Recurrent Linear Attention (DeltaNet with Gated Delta Rule)
kernel void linear_attention_recurrent_step(
    device const float* qkvVector [[buffer(0)]], // [8192] = normalized Q[2048] + normalized K[2048] + V[4096]
    device const float* zVector [[buffer(1)]],   // [4096] (Gate)
    device const float* aVector [[buffer(2)]],   // [32]
    device const float* bVector [[buffer(3)]],   // [32]
    device const uchar* aLogBuf [[buffer(4)]],   // [32] (BF16)
    device const uchar* dtBiasBuf [[buffer(5)]], // [32] (BF16)
    device const uchar* normBuf [[buffer(6)]],   // [128] (BF16)
    device float* stateMatrix [[buffer(7)]],     // [32 heads, 128 keyDim, 128 valDim]
    device float* outputVector [[buffer(8)]],    // [4096]
    constant uint64_t& aLogOffset [[buffer(9)]],
    constant uint64_t& dtBiasOffset [[buffer(10)]],
    constant uint64_t& normOffset [[buffer(11)]],
    constant uint32_t& numValHeads [[buffer(12)]],// 32
    constant uint32_t& numKeyHeads [[buffer(13)]],// 16
    constant uint32_t& headDim [[buffer(14)]],   // 128
    constant float& eps [[buffer(15)]],          // 1e-6
    uint headIdx [[thread_position_in_grid]]
) {
    if (headIdx >= numValHeads) return;

    uint32_t keyHeadIdx = headIdx / (numValHeads / numKeyHeads); // headIdx / 2

    // Offsets
    uint32_t qBase = keyHeadIdx * headDim;
    uint32_t kBase = 2048 + (keyHeadIdx * headDim);
    uint32_t vBase = 4096 + (headIdx * headDim);
    uint32_t zBase = headIdx * headDim;
    uint32_t outBase = headIdx * headDim;
    uint32_t stateBase = headIdx * headDim * headDim;

    // Decay rate computation: alpha = exp(-exp(A_log) * softplus(a + dt_bias))
    float aLogVal = read_bf16_unaligned(aLogBuf + aLogOffset + ((uint64_t)headIdx * 2));
    float dtBiasVal = read_bf16_unaligned(dtBiasBuf + dtBiasOffset + ((uint64_t)headIdx * 2));
    float aVal = aVector[headIdx];
    float bVal = bVector[headIdx];

    float x = aVal + dtBiasVal;
    float dt = (x > 20.0f) ? x : ((x < -20.0f) ? exp(x) : log(1.0f + exp(x))); // stable softplus
    float alpha = exp(-exp(aLogVal) * dt);
    float beta = 1.0f / (1.0f + exp(-bVal));     // sigmoid(b)

    // Delta Rule:
    // 1. Sk = S_prev * k
    // 2. u = v - alpha * Sk
    // 3. S_new = alpha * S_prev + beta * (u * k^T)
    // 4. y = S_new * q
    float Sk[128];
    for (uint32_t i = 0; i < headDim; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < headDim; j++) {
            uint32_t sIdx = stateBase + (i * headDim) + j;
            sum += stateMatrix[sIdx] * qkvVector[kBase + j];
        }
        Sk[i] = sum;
    }

    float y[128];
    float sumSq = 0.0f;
    float invSqrtHeadDim = rsqrt((float)headDim);

    for (uint32_t i = 0; i < headDim; i++) {
        float vi = qkvVector[vBase + i];
        float ui = vi - (alpha * Sk[i]);
        float betaUi = beta * ui;

        float yi = 0.0f;
        for (uint32_t j = 0; j < headDim; j++) {
            uint32_t sIdx = stateBase + (i * headDim) + j;
            float sOld = stateMatrix[sIdx];
            float kj = qkvVector[kBase + j];
            float sNew = (alpha * sOld) + (betaUi * kj);
            stateMatrix[sIdx] = sNew;

            yi += sNew * (qkvVector[qBase + j] * invSqrtHeadDim);
        }
        y[i] = yi;
        sumSq += yi * yi;
    }

    // Per-head RMSNorm
    float invRms = rsqrt((sumSq / (float)headDim) + eps);

    // Apply RMSNorm weight and SiLU Gating: y = (y * invRms * norm) * SiLU(z)
    for (uint32_t i = 0; i < headDim; i++) {
        float normW = read_bf16_unaligned(normBuf + normOffset + ((uint64_t)i * 2));
        float yNorm = y[i] * invRms * normW;

        float z = zVector[zBase + i];
        float silu_z = z / (1.0f + exp(-z));

        outputVector[outBase + i] = yNorm * silu_z;
    }
}

// ============================================================================
// SIMDgroup Cooperative Compute Kernels (32 Threads per Row Reduction)
// ============================================================================

/// MSL Kernel: 32-Thread SIMDgroup Cooperative BF16 SwiGLU Gate & Up Projections (128-bit Vectorized)
kernel void bf16_swiglu_gate_up_simd(
    device const ushort* rawGateBuffer [[buffer(0)]],
    device const ushort* rawUpBuffer [[buffer(1)]],
    device const float* inputVector [[buffer(2)]],
    device float* intermediateOutput [[buffer(3)]],
    constant uint64_t& gateWeightOffset [[buffer(4)]],
    constant uint64_t& upWeightOffset [[buffer(5)]],
    constant uint32_t& hiddenDim [[buffer(6)]],
    constant uint32_t& intermediateDim [[buffer(7)]],
    uint r [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (r >= intermediateDim) return;

    uint64_t gateRowStart = (gateWeightOffset / 2) + ((uint64_t)r * hiddenDim);
    uint64_t upRowStart   = (upWeightOffset / 2) + ((uint64_t)r * hiddenDim);

    device const ushort4* g4 = (device const ushort4*)(rawGateBuffer + gateRowStart);
    device const ushort4* u4 = (device const ushort4*)(rawUpBuffer + upRowStart);
    device const float4* in4 = (device const float4*)inputVector;

    float gate_dot = 0.0f;
    float up_dot   = 0.0f;

    uint32_t numChunks = hiddenDim / 8;
    for (uint32_t c = laneId; c < numChunks; c += 32) {
        ushort4 g_lo = g4[c * 2 + 0];
        ushort4 g_hi = g4[c * 2 + 1];
        ushort4 u_lo = u4[c * 2 + 0];
        ushort4 u_hi = u4[c * 2 + 1];

        float4 in_lo = in4[c * 2 + 0];
        float4 in_hi = in4[c * 2 + 1];

        gate_dot += (bf16_to_fp32(g_lo.x) * in_lo.x) +
                    (bf16_to_fp32(g_lo.y) * in_lo.y) +
                    (bf16_to_fp32(g_lo.z) * in_lo.z) +
                    (bf16_to_fp32(g_lo.w) * in_lo.w) +
                    (bf16_to_fp32(g_hi.x) * in_hi.x) +
                    (bf16_to_fp32(g_hi.y) * in_hi.y) +
                    (bf16_to_fp32(g_hi.z) * in_hi.z) +
                    (bf16_to_fp32(g_hi.w) * in_hi.w);

        up_dot   += (bf16_to_fp32(u_lo.x) * in_lo.x) +
                    (bf16_to_fp32(u_lo.y) * in_lo.y) +
                    (bf16_to_fp32(u_lo.z) * in_lo.z) +
                    (bf16_to_fp32(u_lo.w) * in_lo.w) +
                    (bf16_to_fp32(u_hi.x) * in_hi.x) +
                    (bf16_to_fp32(u_hi.y) * in_hi.y) +
                    (bf16_to_fp32(u_hi.z) * in_hi.z) +
                    (bf16_to_fp32(u_hi.w) * in_hi.w);
    }

    gate_dot = simd_sum(gate_dot);
    up_dot   = simd_sum(up_dot);

    if (laneId == 0) {
        float silu_gate = gate_dot / (1.0f + exp(-gate_dot));
        intermediateOutput[r] = silu_gate * up_dot;
    }
}

/// MSL Kernel: 32-Thread SIMDgroup Cooperative BF16 Down-Projection with Weighted Accumulation (128-bit Vectorized)
kernel void bf16_down_proj_accumulate_simd(
    device const ushort* rawDownBuffer [[buffer(0)]],
    device const float* intermediateVector [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    constant uint64_t& downWeightOffset [[buffer(3)]],
    constant uint32_t& intermediateDim [[buffer(4)]],
    constant uint32_t& hiddenDim [[buffer(5)]],
    constant float& routingWeight [[buffer(6)]],
    uint d [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (d >= hiddenDim) return;

    uint64_t downRowStart = (downWeightOffset / 2) + ((uint64_t)d * intermediateDim);
    device const ushort4* d4 = (device const ushort4*)(rawDownBuffer + downRowStart);
    device const float4* in4 = (device const float4*)intermediateVector;

    float down_dot = 0.0f;
    uint32_t numChunks = intermediateDim / 8;

    for (uint32_t c = laneId; c < numChunks; c += 32) {
        ushort4 d_lo = d4[c * 2 + 0];
        ushort4 d_hi = d4[c * 2 + 1];
        float4 in_lo = in4[c * 2 + 0];
        float4 in_hi = in4[c * 2 + 1];

        down_dot += (bf16_to_fp32(d_lo.x) * in_lo.x) +
                    (bf16_to_fp32(d_lo.y) * in_lo.y) +
                    (bf16_to_fp32(d_lo.z) * in_lo.z) +
                    (bf16_to_fp32(d_lo.w) * in_lo.w) +
                    (bf16_to_fp32(d_hi.x) * in_hi.x) +
                    (bf16_to_fp32(d_hi.y) * in_hi.y) +
                    (bf16_to_fp32(d_hi.z) * in_hi.z) +
                    (bf16_to_fp32(d_hi.w) * in_hi.w);
    }

    down_dot = simd_sum(down_dot);

    if (laneId == 0) {
        outputAccumulator[d] += routingWeight * down_dot;
    }
}

/// MSL Kernel: 32-Thread SIMDgroup Cooperative BF16 GEMV (out = W * x) (128-bit Vectorized)
kernel void bf16_gemv_simd(
    device const ushort* rawWeightBuffer [[buffer(0)]],
    device const float* inputVector [[buffer(1)]],
    device float* outputVector [[buffer(2)]],
    constant uint64_t& weightOffset [[buffer(3)]],
    constant uint32_t& inDim [[buffer(4)]],
    constant uint32_t& outDim [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (row >= outDim) return;

    uint64_t rowWeightStart = (weightOffset / 2) + ((uint64_t)row * inDim);
    device const ushort4* w4 = (device const ushort4*)(rawWeightBuffer + rowWeightStart);
    device const float4* in4 = (device const float4*)inputVector;

    float dot = 0.0f;
    uint32_t numChunks = inDim / 8;

    for (uint32_t c = laneId; c < numChunks; c += 32) {
        ushort4 w_lo = w4[c * 2 + 0];
        ushort4 w_hi = w4[c * 2 + 1];
        float4 in_lo = in4[c * 2 + 0];
        float4 in_hi = in4[c * 2 + 1];

        dot += (bf16_to_fp32(w_lo.x) * in_lo.x) +
               (bf16_to_fp32(w_lo.y) * in_lo.y) +
               (bf16_to_fp32(w_lo.z) * in_lo.z) +
               (bf16_to_fp32(w_lo.w) * in_lo.w) +
               (bf16_to_fp32(w_hi.x) * in_hi.x) +
               (bf16_to_fp32(w_hi.y) * in_hi.y) +
               (bf16_to_fp32(w_hi.z) * in_hi.z) +
               (bf16_to_fp32(w_hi.w) * in_hi.w);
    }

    dot = simd_sum(dot);

    if (laneId == 0) {
        outputVector[row] = dot;
    }
}

/// MSL Kernel: 32-Thread SIMDgroup Cooperative FP8 SwiGLU Gate & Up Projections
kernel void fp8_swiglu_gate_up_simd(
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
    uint r [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (r >= intermediateDim) return;

    float gateScale = bf16_to_fp32(rawGateScaleBuffer[(gateScaleOffset / 2) + r]);
    float upScale   = bf16_to_fp32(rawUpScaleBuffer[(upScaleOffset / 2) + r]);

    device const uchar* gRow = rawGateBuffer + gateWeightOffset + ((uint64_t)r * hiddenDim);
    device const uchar* uRow = rawUpBuffer + upWeightOffset + ((uint64_t)r * hiddenDim);

    float gate_dot = 0.0f;
    float up_dot   = 0.0f;

    for (uint32_t baseD = laneId * 8; baseD < hiddenDim; baseD += 32 * 8) {
        uchar4 g4_0 = *(device const uchar4*)(gRow + baseD);
        uchar4 g4_1 = *(device const uchar4*)(gRow + baseD + 4);
        uchar4 u4_0 = *(device const uchar4*)(uRow + baseD);
        uchar4 u4_1 = *(device const uchar4*)(uRow + baseD + 4);
        float4 in4_0 = *(device const float4*)(inputVector + baseD);
        float4 in4_1 = *(device const float4*)(inputVector + baseD + 4);

        gate_dot += (unpack_e4m3(g4_0.x) * in4_0.x) + (unpack_e4m3(g4_0.y) * in4_0.y) +
                    (unpack_e4m3(g4_1.x) * in4_1.x) + (unpack_e4m3(g4_1.y) * in4_1.y) +
                    (unpack_e4m3(g4_0.z) * in4_0.z) + (unpack_e4m3(g4_0.w) * in4_0.w) +
                    (unpack_e4m3(g4_1.z) * in4_1.z) + (unpack_e4m3(g4_1.w) * in4_1.w);

        up_dot   += (unpack_e4m3(u4_0.x) * in4_0.x) + (unpack_e4m3(u4_0.y) * in4_0.y) +
                    (unpack_e4m3(u4_1.x) * in4_1.x) + (unpack_e4m3(u4_1.y) * in4_1.y) +
                    (unpack_e4m3(u4_0.z) * in4_0.z) + (unpack_e4m3(u4_0.w) * in4_0.w) +
                    (unpack_e4m3(u4_1.z) * in4_1.z) + (unpack_e4m3(u4_1.w) * in4_1.w);
    }

    gate_dot = simd_sum(gate_dot);
    up_dot   = simd_sum(up_dot);

    if (laneId == 0) {
        float finalGate = gate_dot * gateScale;
        float finalUp   = up_dot * upScale;
        float silu_gate = finalGate / (1.0f + exp(-finalGate));
        intermediateOutput[r] = silu_gate * finalUp;
    }
}

/// MSL Kernel: 32-Thread SIMDgroup Cooperative FP8 Down-Projection with Weighted Accumulation
kernel void fp8_down_proj_accumulate_simd(
    device const uchar* rawDownBuffer [[buffer(0)]],
    device const float* intermediateVector [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    device const ushort* rawDownScaleBuffer [[buffer(3)]],
    constant uint64_t& downWeightOffset [[buffer(4)]],
    constant uint64_t& downScaleOffset [[buffer(5)]],
    constant uint32_t& intermediateDim [[buffer(6)]],
    constant uint32_t& hiddenDim [[buffer(7)]],
    constant float& routingWeight [[buffer(8)]],
    uint d [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (d >= hiddenDim) return;

    float downScale = bf16_to_fp32(rawDownScaleBuffer[(downScaleOffset / 2) + d]);
    device const uchar* dRow = rawDownBuffer + downWeightOffset + ((uint64_t)d * intermediateDim);

    float down_dot = 0.0f;
    for (uint32_t baseI = laneId * 8; baseI < intermediateDim; baseI += 32 * 8) {
        uchar4 d4_0 = *(device const uchar4*)(dRow + baseI);
        uchar4 d4_1 = *(device const uchar4*)(dRow + baseI + 4);
        float4 in4_0 = *(device const float4*)(intermediateVector + baseI);
        float4 in4_1 = *(device const float4*)(intermediateVector + baseI + 4);

        down_dot += (unpack_e4m3(d4_0.x) * in4_0.x) + (unpack_e4m3(d4_0.y) * in4_0.y) +
                    (unpack_e4m3(d4_1.x) * in4_1.x) + (unpack_e4m3(d4_1.y) * in4_1.y) +
                    (unpack_e4m3(d4_0.z) * in4_0.z) + (unpack_e4m3(d4_0.w) * in4_0.w) +
                    (unpack_e4m3(d4_1.z) * in4_1.z) + (unpack_e4m3(d4_1.w) * in4_1.w);
    }

    down_dot = simd_sum(down_dot);

    if (laneId == 0) {
        outputAccumulator[d] += routingWeight * (down_dot * downScale);
    }
}

// ============================================================================
// Q4 (4-Bit Affine) SIMDgroup Cooperative Compute Kernels
// ============================================================================

/// MSL Kernel: SIMDgroup Cooperative Q4 Affine SwiGLU Gate & Up Projections
kernel void q4_swiglu_gate_up(
    device const uchar* rawGateWeight [[buffer(0)]],
    device const uchar* rawGateScales [[buffer(1)]],
    device const uchar* rawGateBiases [[buffer(2)]],
    device const uchar* rawUpWeight [[buffer(3)]],
    device const uchar* rawUpScales [[buffer(4)]],
    device const uchar* rawUpBiases [[buffer(5)]],
    device const float* inputVector [[buffer(6)]],
    device float* intermediateOutput [[buffer(7)]],
    constant uint64_t& gateWeightOffset [[buffer(8)]],
    constant uint64_t& gateScaleOffset [[buffer(9)]],
    constant uint64_t& gateBiasOffset [[buffer(10)]],
    constant uint64_t& upWeightOffset [[buffer(11)]],
    constant uint64_t& upScaleOffset [[buffer(12)]],
    constant uint64_t& upBiasOffset [[buffer(13)]],
    constant uint32_t& hiddenDim [[buffer(14)]],
    constant uint32_t& intermediateDim [[buffer(15)]],
    constant uint32_t& groupSize [[buffer(16)]],
    uint r [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (r >= intermediateDim) return;

    uint32_t numGroups = hiddenDim / groupSize;
    device const uchar* gWRow = rawGateWeight + gateWeightOffset + ((uint64_t)r * (hiddenDim / 8) * 4);
    device const uchar* gSRow = rawGateScales + gateScaleOffset + ((uint64_t)r * numGroups * 2);
    device const uchar* gBRow = rawGateBiases + gateBiasOffset + ((uint64_t)r * numGroups * 2);

    device const uchar* uWRow = rawUpWeight + upWeightOffset + ((uint64_t)r * (hiddenDim / 8) * 4);
    device const uchar* uSRow = rawUpScales + upScaleOffset + ((uint64_t)r * numGroups * 2);
    device const uchar* uBRow = rawUpBiases + upBiasOffset + ((uint64_t)r * numGroups * 2);

    float gate_sum = 0.0f;
    float up_sum   = 0.0f;

    uint32_t numU32 = hiddenDim / 8;
    for (uint32_t i = laneId; i < numU32; i += 32) {
        uint32_t col = i * 8;
        uint32_t gIdx = col / groupSize;

        float gScale = read_bf16_unaligned(gSRow + gIdx * 2);
        float gBias  = read_bf16_unaligned(gBRow + gIdx * 2);
        float uScale = read_bf16_unaligned(uSRow + gIdx * 2);
        float uBias  = read_bf16_unaligned(uBRow + gIdx * 2);

        uint32_t gU32 = read_u32_unaligned(gWRow + i * 4);
        uint32_t uU32 = read_u32_unaligned(uWRow + i * 4);

        float x0 = inputVector[col + 0];
        float x1 = inputVector[col + 1];
        float x2 = inputVector[col + 2];
        float x3 = inputVector[col + 3];
        float x4 = inputVector[col + 4];
        float x5 = inputVector[col + 5];
        float x6 = inputVector[col + 6];
        float x7 = inputVector[col + 7];

        float xSum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        float gw0 = float(gU32 & 0x0F);
        float gw1 = float((gU32 >> 4) & 0x0F);
        float gw2 = float((gU32 >> 8) & 0x0F);
        float gw3 = float((gU32 >> 12) & 0x0F);
        float gw4 = float((gU32 >> 16) & 0x0F);
        float gw5 = float((gU32 >> 20) & 0x0F);
        float gw6 = float((gU32 >> 24) & 0x0F);
        float gw7 = float((gU32 >> 28) & 0x0F);

        float gDot = gw0*x0 + gw1*x1 + gw2*x2 + gw3*x3 + gw4*x4 + gw5*x5 + gw6*x6 + gw7*x7;
        gate_sum += gScale * gDot + gBias * xSum;

        float uw0 = float(uU32 & 0x0F);
        float uw1 = float((uU32 >> 4) & 0x0F);
        float uw2 = float((uU32 >> 8) & 0x0F);
        float uw3 = float((uU32 >> 12) & 0x0F);
        float uw4 = float((uU32 >> 16) & 0x0F);
        float uw5 = float((uU32 >> 20) & 0x0F);
        float uw6 = float((uU32 >> 24) & 0x0F);
        float uw7 = float((uU32 >> 28) & 0x0F);

        float uDot = uw0*x0 + uw1*x1 + uw2*x2 + uw3*x3 + uw4*x4 + uw5*x5 + uw6*x6 + uw7*x7;
        up_sum += uScale * uDot + uBias * xSum;
    }

    gate_sum = simd_sum(gate_sum);
    up_sum   = simd_sum(up_sum);

    if (laneId == 0) {
        float silu_gate = gate_sum / (1.0f + exp(-gate_sum));
        intermediateOutput[r] = silu_gate * up_sum;
    }
}

/// MSL Kernel: SIMDgroup Cooperative Q4 Affine Down-Projection with Weighted Accumulation
kernel void q4_down_proj_accumulate(
    device const uchar* rawDownWeight [[buffer(0)]],
    device const uchar* rawDownScales [[buffer(1)]],
    device const uchar* rawDownBiases [[buffer(2)]],
    device const float* intermediateVector [[buffer(3)]],
    device float* outputAccumulator [[buffer(4)]],
    constant uint64_t& downWeightOffset [[buffer(5)]],
    constant uint64_t& downScaleOffset [[buffer(6)]],
    constant uint64_t& downBiasOffset [[buffer(7)]],
    constant uint32_t& intermediateDim [[buffer(8)]],
    constant uint32_t& hiddenDim [[buffer(9)]],
    constant uint32_t& groupSize [[buffer(10)]],
    constant float& routingWeight [[buffer(11)]],
    uint d [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (d >= hiddenDim) return;

    uint32_t numGroups = intermediateDim / groupSize;
    device const uchar* dWRow = rawDownWeight + downWeightOffset + ((uint64_t)d * (intermediateDim / 8) * 4);
    device const uchar* dSRow = rawDownScales + downScaleOffset + ((uint64_t)d * numGroups * 2);
    device const uchar* dBRow = rawDownBiases + downBiasOffset + ((uint64_t)d * numGroups * 2);

    float down_sum = 0.0f;
    uint32_t numU32 = intermediateDim / 8;

    for (uint32_t i = laneId; i < numU32; i += 32) {
        uint32_t col = i * 8;
        uint32_t gIdx = col / groupSize;

        float dScale = read_bf16_unaligned(dSRow + gIdx * 2);
        float dBias  = read_bf16_unaligned(dBRow + gIdx * 2);

        uint32_t dU32 = read_u32_unaligned(dWRow + i * 4);

        float x0 = intermediateVector[col + 0];
        float x1 = intermediateVector[col + 1];
        float x2 = intermediateVector[col + 2];
        float x3 = intermediateVector[col + 3];
        float x4 = intermediateVector[col + 4];
        float x5 = intermediateVector[col + 5];
        float x6 = intermediateVector[col + 6];
        float x7 = intermediateVector[col + 7];

        float xSum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        float dw0 = float(dU32 & 0x0F);
        float dw1 = float((dU32 >> 4) & 0x0F);
        float dw2 = float((dU32 >> 8) & 0x0F);
        float dw3 = float((dU32 >> 12) & 0x0F);
        float dw4 = float((dU32 >> 16) & 0x0F);
        float dw5 = float((dU32 >> 20) & 0x0F);
        float dw6 = float((dU32 >> 24) & 0x0F);
        float dw7 = float((dU32 >> 28) & 0x0F);

        float dDot = dw0*x0 + dw1*x1 + dw2*x2 + dw3*x3 + dw4*x4 + dw5*x5 + dw6*x6 + dw7*x7;
        down_sum += dScale * dDot + dBias * xSum;
    }

    down_sum = simd_sum(down_sum);

    if (laneId == 0) {
        outputAccumulator[d] += routingWeight * down_sum;
    }
}

/// MSL Kernel: General Q4 Affine GEMV (out = (W_q4 * in))
kernel void q4_gemv(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const uchar* rawScaleBuffer [[buffer(1)]],
    device const uchar* rawBiasBuffer [[buffer(2)]],
    device const float* inputVector [[buffer(3)]],
    device float* outputVector [[buffer(4)]],
    constant uint64_t& weightOffset [[buffer(5)]],
    constant uint64_t& scaleOffset [[buffer(6)]],
    constant uint64_t& biasOffset [[buffer(7)]],
    constant uint32_t& inDim [[buffer(8)]],
    constant uint32_t& outDim [[buffer(9)]],
    constant uint32_t& groupSize [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (row >= outDim) return;

    uint32_t numGroups = inDim / groupSize;
    device const uchar* wRow = rawWeightBuffer + weightOffset + ((uint64_t)row * (inDim / 8) * 4);
    device const uchar* sRow = rawScaleBuffer + scaleOffset + ((uint64_t)row * numGroups * 2);
    device const uchar* bRow = rawBiasBuffer + biasOffset + ((uint64_t)row * numGroups * 2);

    float sum = 0.0f;
    uint32_t numU32 = inDim / 8;

    for (uint32_t i = laneId; i < numU32; i += 32) {
        uint32_t col = i * 8;
        uint32_t gIdx = col / groupSize;

        float scale = read_bf16_unaligned(sRow + gIdx * 2);
        float bias  = read_bf16_unaligned(bRow + gIdx * 2);

        uint32_t u32 = read_u32_unaligned(wRow + i * 4);

        float x0 = inputVector[col + 0];
        float x1 = inputVector[col + 1];
        float x2 = inputVector[col + 2];
        float x3 = inputVector[col + 3];
        float x4 = inputVector[col + 4];
        float x5 = inputVector[col + 5];
        float x6 = inputVector[col + 6];
        float x7 = inputVector[col + 7];

        float xSum = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;

        float w0 = float(u32 & 0x0F);
        float w1 = float((u32 >> 4) & 0x0F);
        float w2 = float((u32 >> 8) & 0x0F);
        float w3 = float((u32 >> 12) & 0x0F);
        float w4 = float((u32 >> 16) & 0x0F);
        float w5 = float((u32 >> 20) & 0x0F);
        float w6 = float((u32 >> 24) & 0x0F);
        float w7 = float((u32 >> 28) & 0x0F);

        float dot = w0*x0 + w1*x1 + w2*x2 + w3*x3 + w4*x4 + w5*x5 + w6*x6 + w7*x7;
        sum += scale * dot + bias * xSum;
    }

    sum = simd_sum(sum);

    if (laneId == 0) {
        outputVector[row] = sum;
    }
}

/// MSL Kernel: Lookup Embedding Token Vector for Q4 Affine Quantization
kernel void lookup_embeddings_q4(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const uchar* rawScaleBuffer [[buffer(1)]],
    device const uchar* rawBiasBuffer [[buffer(2)]],
    device float* outputVector [[buffer(3)]],
    constant uint32_t& tokenId [[buffer(4)]],
    constant uint64_t& weightOffset [[buffer(5)]],
    constant uint64_t& scaleOffset [[buffer(6)]],
    constant uint64_t& biasOffset [[buffer(7)]],
    constant uint32_t& hiddenDim [[buffer(8)]],
    constant uint32_t& groupSize [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= hiddenDim) return;

    uint32_t u32Idx = id / 8;
    uint32_t shift = (id % 8) * 4;
    uint32_t gIdx = id / groupSize;
    uint32_t numGroups = hiddenDim / groupSize;

    device const uchar* wRowStart = rawWeightBuffer + weightOffset + ((uint64_t)tokenId * (hiddenDim / 8) * 4);
    device const uchar* sRowStart = rawScaleBuffer + scaleOffset + ((uint64_t)tokenId * numGroups * 2);
    device const uchar* bRowStart = rawBiasBuffer + biasOffset + ((uint64_t)tokenId * numGroups * 2);

    uint32_t packed = read_u32_unaligned(wRowStart + u32Idx * 4);
    float w4 = float((packed >> shift) & 0x0F);

    float scale = read_bf16_unaligned(sRowStart + gIdx * 2);
    float bias  = read_bf16_unaligned(bRowStart + gIdx * 2);

    outputVector[id] = scale * w4 + bias;
}

/// MSL Kernel: Q8 (8-bit affine quantized) Matrix-Vector Multiplication (y = W * x)
kernel void q8_gemv(
    device const uchar* rawWeightBuffer [[buffer(0)]],
    device const uchar* rawScaleBuffer [[buffer(1)]],
    device const uchar* rawBiasBuffer [[buffer(2)]],
    device const float* inputVector [[buffer(3)]],
    device float* outputVector [[buffer(4)]],
    constant uint64_t& weightOffset [[buffer(5)]],
    constant uint64_t& scaleOffset [[buffer(6)]],
    constant uint64_t& biasOffset [[buffer(7)]],
    constant uint32_t& inDim [[buffer(8)]],
    constant uint32_t& outDim [[buffer(9)]],
    constant uint32_t& groupSize [[buffer(10)]],
    uint row [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]
) {
    if (row >= outDim) return;

    uint32_t numGroups = inDim / groupSize;
    device const uchar* wRow = rawWeightBuffer + weightOffset + ((uint64_t)row * inDim);
    device const uchar* sRow = rawScaleBuffer + scaleOffset + ((uint64_t)row * numGroups * 2);
    device const uchar* bRow = rawBiasBuffer + biasOffset + ((uint64_t)row * numGroups * 2);

    float threadSum = 0.0f;
    for (uint32_t g = laneId; g < numGroups; g += 32) {
        float scale = read_bf16_unaligned(sRow + g * 2);
        float bias  = read_bf16_unaligned(bRow + g * 2);
        uint32_t colStart = g * groupSize;
        float groupSum = 0.0f;
        float groupXSum = 0.0f;
        for (uint32_t c = 0; c < groupSize; c++) {
            uint32_t col = colStart + c;
            float x = inputVector[col];
            float w8 = float(wRow[col]);
            groupSum += w8 * x;
            groupXSum += x;
        }
        threadSum += scale * groupSum + bias * groupXSum;
    }

    float totalSum = simd_sum(threadSum);
    if (laneId == 0) {
        outputVector[row] = totalSum;
    }
}

