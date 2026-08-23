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

/// Converts a BF16 (bfloat16) word into an IEEE FP32 float
inline float bf16_to_fp32(ushort u) {
    uint32_t bits = ((uint32_t)u) << 16;
    return as_type<float>(bits);
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
        
        float sum = 0.0f;
        for (uint32_t d = 0; d < hiddenDim; d++) {
            float w = bf16_to_fp32(rawBaseBuffer[rowOffset + d]);
            sum += w * tokenH[d];
        }
        sharedLogits[tid] = sum;
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

/// MSL Kernel: Fused MXFP8 SwiGLU Gate & Up Projections with SiLU Activation and Hadamard Product
/// Computes inter[r] = SiLU(W_gate[r, :] * x) * (W_up[r, :] * x) for all intermediate dimensions r.
kernel void mxfp8_swiglu_gate_up(
    device const uchar* rawGateBuffer [[buffer(0)]],
    device const uchar* rawUpBuffer [[buffer(1)]],
    device const float* inputVector [[buffer(2)]],
    device float* intermediateOutput [[buffer(3)]],
    constant uint64_t& gateWeightOffset [[buffer(4)]],
    constant uint64_t& gateScaleOffset [[buffer(5)]],
    constant uint64_t& upWeightOffset [[buffer(6)]],
    constant uint64_t& upScaleOffset [[buffer(7)]],
    constant uint32_t& hiddenDim [[buffer(8)]],
    constant uint32_t& intermediateDim [[buffer(9)]],
    uint r [[thread_position_in_grid]]
) {
    if (r >= intermediateDim) return;

    uint32_t numBlocks = hiddenDim / 32;
    uint32_t gateWordsPerRow = hiddenDim / 4;
    uint32_t upWordsPerRow = hiddenDim / 4;
    uint32_t gateScalesPerRow = hiddenDim / 32;
    uint32_t upScalesPerRow = hiddenDim / 32;

    device const uint32_t* gateU32 = (device const uint32_t*)(rawGateBuffer + gateWeightOffset);
    device const uchar* gateScales = rawGateBuffer + gateScaleOffset;

    device const uint32_t* upU32 = (device const uint32_t*)(rawUpBuffer + upWeightOffset);
    device const uchar* upScales = rawUpBuffer + upScaleOffset;

    float gate_dot = 0.0f;
    float up_dot   = 0.0f;

    for (uint32_t b = 0; b < numBlocks; b++) {
        float gScale = decode_e8m0_scale(gateScales[r * gateScalesPerRow + b]);
        float uScale = decode_e8m0_scale(upScales[r * upScalesPerRow + b]);

        float gBlockSum = 0.0f;
        float uBlockSum = 0.0f;

        for (uint32_t w = 0; w < 8; w++) {
            uint32_t wordIdx = r * gateWordsPerRow + (b * 8 + w);
            uint32_t gWord = gateU32[wordIdx];
            uint32_t uWord = upU32[wordIdx];

            uint32_t baseD = (b * 32) + (w * 4);

            float g0 = unpack_e4m3((gWord >> 0) & 0xFF);
            float g1 = unpack_e4m3((gWord >> 8) & 0xFF);
            float g2 = unpack_e4m3((gWord >> 16) & 0xFF);
            float g3 = unpack_e4m3((gWord >> 24) & 0xFF);

            float u0 = unpack_e4m3((uWord >> 0) & 0xFF);
            float u1 = unpack_e4m3((uWord >> 8) & 0xFF);
            float u2 = unpack_e4m3((uWord >> 16) & 0xFF);
            float u3 = unpack_e4m3((uWord >> 24) & 0xFF);

            float in0 = inputVector[baseD + 0];
            float in1 = inputVector[baseD + 1];
            float in2 = inputVector[baseD + 2];
            float in3 = inputVector[baseD + 3];

            gBlockSum += (g0 * in0) + (g1 * in1) + (g2 * in2) + (g3 * in3);
            uBlockSum += (u0 * in0) + (u1 * in1) + (u2 * in2) + (u3 * in3);
        }

        gate_dot += gBlockSum * gScale;
        up_dot   += uBlockSum * uScale;
    }

    // SiLU activation: silu(x) = x / (1 + exp(-x))
    float silu_gate = gate_dot / (1.0f + exp(-gate_dot));
    intermediateOutput[r] = silu_gate * up_dot;
}

/// MSL Kernel: MXFP8 Down-Projection with Weighted Accumulation
/// Computes outputAccumulator[d] += routingWeight * (W_down[d, :] * intermediateVector) for all hidden dims d.
kernel void mxfp8_down_proj_accumulate(
    device const uchar* rawDownBuffer [[buffer(0)]],
    device const float* intermediateVector [[buffer(1)]],
    device float* outputAccumulator [[buffer(2)]],
    constant uint64_t& downWeightOffset [[buffer(3)]],
    constant uint64_t& downScaleOffset [[buffer(4)]],
    constant uint32_t& intermediateDim [[buffer(5)]],
    constant uint32_t& hiddenDim [[buffer(6)]],
    constant float& routingWeight [[buffer(7)]],
    uint d [[thread_position_in_grid]]
) {
    if (d >= hiddenDim) return;

    uint32_t numBlocks = intermediateDim / 32;
    uint32_t wordsPerRow = intermediateDim / 4;
    uint32_t scalesPerRow = intermediateDim / 32;

    device const uint32_t* downU32 = (device const uint32_t*)(rawDownBuffer + downWeightOffset);
    device const uchar* downScales = rawDownBuffer + downScaleOffset;

    float down_dot = 0.0f;

    for (uint32_t b = 0; b < numBlocks; b++) {
        float dScale = decode_e8m0_scale(downScales[d * scalesPerRow + b]);
        float blockSum = 0.0f;

        for (uint32_t w = 0; w < 8; w++) {
            uint32_t wordIdx = d * wordsPerRow + (b * 8 + w);
            uint32_t dWord = downU32[wordIdx];
            uint32_t baseI = (b * 32) + (w * 4);

            float d0 = unpack_e4m3((dWord >> 0) & 0xFF);
            float d1 = unpack_e4m3((dWord >> 8) & 0xFF);
            float d2 = unpack_e4m3((dWord >> 16) & 0xFF);
            float d3 = unpack_e4m3((dWord >> 24) & 0xFF);

            blockSum += (d0 * intermediateVector[baseI + 0])
                      + (d1 * intermediateVector[baseI + 1])
                      + (d2 * intermediateVector[baseI + 2])
                      + (d3 * intermediateVector[baseI + 3]);
        }

        down_dot += blockSum * dScale;
    }

    outputAccumulator[d] += routingWeight * down_dot;
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

    for (uint32_t d = 0; d < hiddenDim; d++) {
        float inVal = inputVector[d];
        gate_dot += bf16_to_fp32(rawGateBuffer[gateRowStart + d]) * inVal;
        up_dot   += bf16_to_fp32(rawUpBuffer[upRowStart + d]) * inVal;
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

    for (uint32_t i = 0; i < intermediateDim; i++) {
        down_dot += bf16_to_fp32(rawDownBuffer[downRowStart + i]) * intermediateVector[i];
    }

    outputAccumulator[d] += routingWeight * down_dot;
}

