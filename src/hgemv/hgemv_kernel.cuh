#include "../common/defs.h"
#include "../common/pack.h"
#include "reduce.cuh"

__global__ void hgemv_k32_f16_kernel(half *A, half *x, half *y, int M, int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int laneId = threadIdx.x % WARP_SIZE;
    if (m < M) {
        float sum = 0.0f;
        int num_k_tiles = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
        for (int tile = 0; tile < num_k_tiles; ++tile) {
            int k = tile * WARP_SIZE + laneId;
            sum += __half2float(A[m * K + k]) * __half2float(x[k]);
        }
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
        if (laneId == 0) {
            y[m] = __float2half(sum);
        }
    }
}

__global__ void hgemv_k128_f16x4_kernel(half *A, half *x, half *y, int M,
                                        int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int laneId = threadIdx.x % WARP_SIZE;
    if (m < M) {
        float sum = 0.0f;
        int num_k_tiles = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
        for (int tile = 0; tile < num_k_tiles; ++tile) {
            int k = (tile * WARP_SIZE + laneId) * 4;
            float2 reg_x_0 = __half22float2(HALF2(x[k + 0]));
            float2 reg_x_1 = __half22float2(HALF2(x[k + 2]));
            float2 reg_A_0 = __half22float2(HALF2(A[m * K + k + 0]));
            float2 reg_A_1 = __half22float2(HALF2(A[m * K + k + 2]));
            sum += (reg_x_0.x * reg_A_0.x + reg_x_0.y * reg_A_0.y +
                    reg_x_1.x * reg_A_1.x + reg_x_1.y * reg_A_1.y);
        }
        sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
        if (laneId == 0) {
            y[m] = __float2half(sum);
        }
    }
}

template <const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *A, half *x, half *y, int M, int K) {
    constexpr int THREADS_PER_ROW =
        (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
    int laneId = threadIdx.x % WARP_SIZE;
    int k = laneId % THREADS_PER_ROW;
    int m = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_PER_WARP +
            laneId / THREADS_PER_ROW;
    float sum = 0.0f;
    if (m < M) {
        sum = __half2float(A[m * K + k]) * __half2float(x[k]);
    }
    sum = warp_reduce_sum_f32<THREADS_PER_ROW>(sum);
    if (k == 0 && m < M) {
        y[m] = __float2half(sum);
    }
}
