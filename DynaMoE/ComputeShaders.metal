//
//  ComputeShaders.metal
//  DynaMoE
//
//  Created by Derek Parris on 8/18/26.
//

#include <metal_stdlib>
using namespace metal;

/// Helper: Decodes FP8 (E4M3) byte to FP32 float in GPU registers
inline float unpack_e4m3(uchar u) {
    uint sign = (u >> 7) & 0x01;
    uint exp  = (u >> 3) & 0x0F;
    uint mant = u & 0x07;
    
    float val = 0.0f;
    if (exp == 0) {
        // Subnormal numbers (2^-6 = 0.015625)
        val = ((float)mant / 8.0f) * 0.015625f;
    } else {
        // Normalized numbers (Bias = 7)
        val = (1.0f + ((float)mant / 8.0f)) * pow(2.0f, (float)exp - 7.0f);
    }
    return sign ? -val : val;
}

/// MSL Kernel: Reads raw 8-bit MXFP8 weights directly off SSD and dequantizes them
kernel void dequantize_mxfp8_weights(
    device const uchar* rawBaseBuffer [[buffer(0)]],
    device float* outputPreview [[buffer(1)]],
    constant uint64_t& byteOffset [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // 1. Address raw byte stream at the exact tensor offset
    device const uchar* mxfp8Bytes = rawBaseBuffer + byteOffset;

    // 2. Fetch raw byte from virtual address space
    uchar rawByte = mxfp8Bytes[id];

    // 3. Dequantize FP8 -> FP32 in threadgroup registers
    outputPreview[id] = unpack_e4m3(rawByte);
}
