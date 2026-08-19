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
