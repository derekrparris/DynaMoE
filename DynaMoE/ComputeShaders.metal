//
//  ComputeShaders.metal
//  DynaMoE
//
//  Created by Derek Parris on 8/18/26.
//

#include <metal_stdlib>
using namespace metal;

/// MSL Compute Kernel: Reads mapped FP16/Half weights directly off the SSD address space
kernel void inspect_tensor_weights(
    device const char* rawBaseBuffer [[buffer(0)]],
    device float* outputPreview [[buffer(1)]],
    constant uint64_t& byteOffset [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    // 1. Cast the raw byte pointer at the exact tensor offset into a half-precision (FP16) float pointer
    device const half* tensorWeights = (device const half*)(rawBaseBuffer + byteOffset);

    // 2. Read the weight value from GPU Unified Memory
    half rawWeight = tensorWeights[id];

    // 3. Convert FP16 -> FP32 and write to output buffer
    outputPreview[id] = float(rawWeight);
}
