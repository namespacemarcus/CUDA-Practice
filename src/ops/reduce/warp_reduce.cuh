#include "../../common/cuda/cuda_utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

/* fp32 */
template <const int KWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32_f32(float val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

/* fp16 */
// f16 input, f16 accumulate.
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16_f16(half val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// f16 input, f32 accumulate
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f16_f32(half val) {
    float val_f32 = __half2float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

/* bf16 */
// bf16 input, bf16 accumulate
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ __nv_bfloat16
warp_reduce_sum_bf16_bf16(__nv_bfloat16 val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val = __hadd(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_bf16_f32(__nv_bfloat16 val) {
    float val_f32 = __bfloat162float(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f32 += __shfl_xor_sync(0xffffffff, val_f32, mask);
    }
    return val_f32;
}

/* fp8 */
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e4m3_f16(__nv_fp8_storage_t val) {
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E4M3);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    }
    return val_f16;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half
warp_reduce_sum_fp8_e5m2_f16(__nv_fp8_storage_t val) {
    half val_f16 = __nv_cvt_fp8_to_halfraw(val, __NV_E5M2);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_f16 = __hadd(val_f16, __shfl_xor_sync(0xffffffff, val_f16, mask));
    }
    return val_f16;
}

/* int8 */
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i8_i32(int8_t val) {
    int32_t val_i32 = static_cast<int32_t>(val);
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val_i32 += __shfl_xor_sync(0xffffffff, val_i32, mask);
    }
    return val_i32;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ int32_t warp_reduce_sum_i32_i32(int32_t val) {
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
