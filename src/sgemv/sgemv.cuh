#include "reduce.cuh"

// block(32, 4), grid(1, M/4)
__global__ void sgemv_k32_f32_kernel(float *A, float *x, float *y, int M,
                                     int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y; // m为当前warp要计算的行号
    int laneId = threadIdx.x % WARP_SIZE;
    if (m >= M) {
        return;
    }
    float sum = 0.0f;
    int num_k_tiles = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int k = tile * num_k_tiles + laneId;
        sum += A[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (laneId == 0) {
        y[m] = sum;
    }
}

__global__ void sgemv_k128_f32x4_kernel(float *A, float *x, float *y, int M,
                                        int K) {
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int laneId = threadIdx.x % WARP_SIZE;
    if (m >= M) {
        return;
    }
    float sum = 0.0f;
    int num_k_tiles = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int tile = 0; tile < num_k_tiles; ++tile) {
        int k = (tile * WARP_SIZE + laneId) * 4;
        float4 reg_x = FLOAT4(x[k]);
        float4 reg_a = FLOAT4(A[m * K + k]);
        sum += reg_a.x * reg_x.x + reg_a.y * reg_x.y + reg_a.z * reg_x.z +
               reg_a.w * reg_x.w;
    }
    sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
    if (laneId == 0) {
        y[m] = sum;
    }
}

template <const int ROW_PER_WARP = 2>
__global__ void sgemv_k16_f32_kernel(float *A, float *x, float *y, int M,
                                     int K) {
    constexpr int THREADS_PER_ROW =
        (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
    int laneId = threadIdx.x % WARP_SIZE;
    int k = laneId % THREADS_PER_ROW;
    int m = (blockIdx.y * blockDim.y + threadIdx.y) * ROW_PER_WARP +
            laneId / THREADS_PER_ROW;
    if (m >= M) {
        return;
    }
    float sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f32<THREADS_PER_ROW>(sum);
    if (k == 0) {
        y[m] = sum;
    }
}
