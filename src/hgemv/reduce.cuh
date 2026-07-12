#include "../common/cuda/cuda_utils.h"
#include <cuda_fp16.h>

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, stride);
    }
    return val;
}
