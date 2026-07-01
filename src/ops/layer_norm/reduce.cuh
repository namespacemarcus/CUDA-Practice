#include "../../common/cuda/cuda_utils.h"
#include <cuda_fp16.h>

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float shared[NUM_WARPS];

    val = warp_reduce_sum_f32<WARP_SIZE>(val);
    if (laneId == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    val = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    val = warp_reduce_sum_f32<NUM_WARPS>(val);
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f16(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = NUM_THREADS / WARP_SIZE;
    int laneId = NUM_THREADS % WARP_SIZE;
    __shared__ half shared[NUM_WARPS];

    val = warp_reduce_sum_f16_f16<WARP_SIZE>(val);
    if (laneId == 0) {
        shared[warpId] = val;
    }
    __syncthreads();

    val = (laneId < NUM_WARPS) ? shared[laneId] : __float2half(0.0f);
    val = warp_reduce_sum_f16_f16<NUM_WARPS>(val);
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

template <const int NUM_THREADS = 256>
__device__ half block_reduce_sum_f16_f32(half val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    __shared__ float shared[NUM_WARPS];

    float val_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(val);
    if(laneId==0){
        shared[warpId] = val_f32;
    }
    __syncthreads();

    val_f32 = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    val_f32 = warp_reduce_sum_f16_f32<NUM_WARPS>(val_f32);
    return val_f32;
}
