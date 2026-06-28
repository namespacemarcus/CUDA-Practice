#pragma once

#include "../../../common/cuda/cuda_utils.h"
#include "md.cuh"
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

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger = value_bigger ? value : other;
        MD smaller = value_bigger ? other : value;

        value.m = bigger.m;
        value.d = bigger.d + smaller.d * expf(smaller.m - bigger.m);
    }
    return value;
}
