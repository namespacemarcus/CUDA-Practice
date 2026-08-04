#pragma once

#include "../common/cuda/cuda_utils.h"

__device__ __forceinline__ int swizzle_a(int x, int y) {
    return y ^ ((x >> 2) << 3);
}

// (M, K) @ (K, N) -> (M, N)

__global__ void sgemm_naive_kernel(float *a, float *b, float *c, int M, int N,
                                   int K) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            psum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = psum;
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_tiling_kernel(float *a, float *b, float *c, int m, int n,
                                    int k) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    // Each block loads 64x16 of A and 8x128 of B per pass, two passes → 128x16
    // and 16x128.
    int load_a_row = tid / 4;        // 0~63
    int load_a_col = (tid % 4) * 4;  // 0,4,8,12
    int load_b_row = tid / 32;       // tid / (128 / 4), 0~7
    int load_b_col = (tid % 32) * 4; // 0,4,8,12,16,20,...,124

    // warp tiling: every 4 warps cover the upper/lower 64x128 halves of C.
    int warp_row = warpId / 4;      // 0,1
    int warp_col = warpId % 4;      // 0,1,2,3
    int t_row_in_warp = laneId / 4; // 0~7
    int t_col_in_warp = laneId % 4; // 0~3

    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN]{0.0f};

    for (int bk = 0; bk < k; bk += BK) {
        // global -> shared
        FLOAT4(As[load_a_row][load_a_col]) =
            FLOAT4(a[(blockIdx.y * BM + load_a_row) * k + bk + load_a_col]);
        FLOAT4(As[load_a_row + 64][load_a_col]) = FLOAT4(
            a[(blockIdx.y * BM + load_a_row + 64) * k + bk + load_a_col]);

        FLOAT4(Bs[load_b_row][load_b_col]) =
            FLOAT4(b[(bk + load_b_row) * n + blockIdx.x * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            FLOAT4(b[(bk + load_b_row + 8) * n + blockIdx.x * BN + load_b_col]);

        __syncthreads();

        // 8x8
#pragma unroll
        for (int i = 0; i < BK; ++i) {
            float reg_a[TM], reg_b[TN];
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                reg_a[m] = As[c_row + m][i];
            }
            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    sum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col]) =
            FLOAT4(sum[i][0]);
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col +
                 4]) = FLOAT4(sum[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_kernel(float *a, float *b, float *c, int m,
                                       int n, int k) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    int warp_row = warpId / 4;
    int warp_col = warpId % 4;
    int t_row_in_warp = laneId / 4;
    int t_col_in_warp = laneId % 4;

    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN]{0.0f};

    for (int bk = 0; bk < k; bk += BK) {
        float4 tmp_a0 =
            FLOAT4(a[(blockIdx.y * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row] = tmp_a0.x;
        As_T[load_a_col + 1][load_a_row] = tmp_a0.y;
        As_T[load_a_col + 2][load_a_row] = tmp_a0.z;
        As_T[load_a_col + 3][load_a_row] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(
            a[(blockIdx.y * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][load_a_row + 64] = tmp_a1.x;
        As_T[load_a_col + 1][load_a_row + 64] = tmp_a1.y;
        As_T[load_a_col + 2][load_a_row + 64] = tmp_a1.z;
        As_T[load_a_col + 3][load_a_row + 64] = tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) =
            FLOAT4(b[(bk + load_b_row) * n + blockIdx.x * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            FLOAT4(b[(bk + load_b_row + 8) * n + blockIdx.x * BN + load_b_col]);

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; ++i) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][c_row]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][c_row + 4]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    sum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col]) =
            FLOAT4(sum[i][0]);
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col +
                 4]) = FLOAT4(sum[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_bcf_swizzling_kernel(float *a, float *b,
                                                     float *c, int m, int n,
                                                     int k) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    int warp_row = warpId / 4;
    int warp_col = warpId % 4;
    int t_row_in_warp = laneId / 4;
    int t_col_in_warp = laneId % 4;

    int c_row = warp_row * 64 + t_row_in_warp * 8;
    int c_col = warp_col * 32 + t_col_in_warp * 8;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN]{0.0f};

    for (int bk = 0; bk < k; bk += BK) {
        float4 tmp_a0 =
            FLOAT4(a[(blockIdx.y * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][swizzle_a(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][swizzle_a(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][swizzle_a(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][swizzle_a(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(
            a[(blockIdx.y * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][swizzle_a(load_a_col + 0, load_a_row + 64)] =
            tmp_a1.x;
        As_T[load_a_col + 1][swizzle_a(load_a_col + 1, load_a_row + 64)] =
            tmp_a1.y;
        As_T[load_a_col + 2][swizzle_a(load_a_col + 2, load_a_row + 64)] =
            tmp_a1.z;
        As_T[load_a_col + 3][swizzle_a(load_a_col + 3, load_a_row + 64)] =
            tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) =
            FLOAT4(b[(bk + load_b_row) * n + blockIdx.x * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            FLOAT4(b[(bk + load_b_row + 8) * n + blockIdx.x * BN + load_b_col]);

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; ++i) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][swizzle_a(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][swizzle_a(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col + 4]);

#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    sum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col]) =
            FLOAT4(sum[i][0]);
        FLOAT4(c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col +
                 4]) = FLOAT4(sum[i][4]);
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 16,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_at_tiling_bcf_swizzling_cstore_kernel(float *a, float *b,
                                                            float *c, int m,
                                                            int n, int k) {
    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    int t_row_in_warp = (laneId / 16) * 8; // 0 or 8.

    int c_row = warpId * 16 + t_row_in_warp;
    int c_col_base = (laneId % 16) * 4;
    int c_col_l = c_col_base;
    int c_col_r = c_col_base + 64;

    __shared__ float As_T[BK][BM];
    __shared__ float Bs[BK][BN];

    float sum[TM][TN]{0.0f};

    for (int bk = 0; bk < k; bk += BK) {
        float4 tmp_a0 =
            FLOAT4(a[(blockIdx.y * BM + load_a_row) * k + bk + load_a_col]);
        As_T[load_a_col + 0][swizzle_a(load_a_col + 0, load_a_row)] = tmp_a0.x;
        As_T[load_a_col + 1][swizzle_a(load_a_col + 1, load_a_row)] = tmp_a0.y;
        As_T[load_a_col + 2][swizzle_a(load_a_col + 2, load_a_row)] = tmp_a0.z;
        As_T[load_a_col + 3][swizzle_a(load_a_col + 3, load_a_row)] = tmp_a0.w;

        float4 tmp_a1 = FLOAT4(
            a[(blockIdx.y * BM + load_a_row + 64) * k + bk + load_a_col]);
        As_T[load_a_col + 0][swizzle_a(load_a_col + 0, load_a_row + 64)] =
            tmp_a1.x;
        As_T[load_a_col + 1][swizzle_a(load_a_col + 1, load_a_row + 64)] =
            tmp_a1.y;
        As_T[load_a_col + 2][swizzle_a(load_a_col + 2, load_a_row + 64)] =
            tmp_a1.z;
        As_T[load_a_col + 3][swizzle_a(load_a_col + 3, load_a_row + 64)] =
            tmp_a1.w;

        FLOAT4(Bs[load_b_row][load_b_col]) =
            FLOAT4(b[(bk + load_b_row) * n + blockIdx.x * BN + load_b_col]);
        FLOAT4(Bs[load_b_row + 8][load_b_col]) =
            FLOAT4(b[(bk + load_b_row + 8) * n + blockIdx.x * BN + load_b_col]);

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; ++i) {
            float reg_a[TM], reg_b[TN];

            FLOAT4(reg_a[0]) = FLOAT4(As_T[i][swizzle_a(i, c_row)]);
            FLOAT4(reg_a[4]) = FLOAT4(As_T[i][swizzle_a(i, c_row + 4)]);

            FLOAT4(reg_b[0]) = FLOAT4(Bs[i][c_col_l]);
            FLOAT4(reg_b[4]) = FLOAT4(Bs[i][c_col_r]);

#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    sum[m][n] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        FLOAT4(
            c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col_l]) =
            FLOAT4(sum[i][0]);
        FLOAT4(
            c[(blockIdx.y * BM + c_row + i) * n + blockIdx.x * BN + c_col_r]) =
            FLOAT4(sum[i][4]);
    }
}
