#pragma once

#include "../common/cuda/cuda_utils.h"
#include <float.h>

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float shared[NUM_WARPS];

    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (laneId == 0) {
        shared[warpId] = value;
    }
    __syncthreads();

    value = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    value = warp_reduce_sum_f32<NUM_WARPS>(value);
    // broadcast value to all threads within warp.
    value = __shfl_sync(0xffffffff, value, 0);
    return value;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, stride));
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_max_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float shared[NUM_WARPS];

    float value = warp_reduce_max_f32<WARP_SIZE>(val);
    if (laneId == 0) {
        shared[warpId] = value;
    }
    __syncthreads();

    value = (laneId < NUM_WARPS) ? shared[laneId] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WARPS>(value);
    // broadcast value to all threads within warp.
    value = __shfl_sync(0xffffffff, value, 0);
    return value;
}
