#pragma once

#include "../../common/cuda/cuda_utils.h"

// (M, K) @ (K, N) -> (M, N)

__global__ void sgemm_naive_f32_kernel(float *a, float *b, float *c, int M,
                                       int N, int K) {
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

template <const int BM = 32, const int BK = 32, const int BN = 32>
__global__ void sgemm_k_tiled_f32_kernel(float *a, float *b, float *c, int M,
                                         int N, int K) {
    /* BM = blockDim.y
       BN = blockDim.x */
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // tid within the block.

    int a_smem_row = tid / BK;
    int a_smem_col = tid % BK;
    int b_smem_row = tid / BN;
    int b_smem_col = tid % BN;

    int a_gmem_row = blockIdx.y * BM + a_smem_row;
    int b_gmem_col = blockIdx.x * BN + b_smem_col;

    float sum = 0.0f;
    for (int bk = 0; bk < ((K + BK - 1) / BK); ++bk) {
        int a_gmem_col = bk * BK + a_smem_col;
        int a_gmem_addr = a_gmem_row * K + a_gmem_col;
        a_smem[a_smem_row][a_smem_col] = a[a_gmem_addr];
        int b_gmem_row = bk * BK + b_smem_row;
        int b_gmem_addr = b_gmem_row * N + b_gmem_col;
        b_smem[b_smem_row][b_smem_col] = b[b_gmem_addr];
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            sum += a_smem[a_smem_row][k] * b_smem[k][b_smem_col];
        }
        __syncthreads();
    }
    int c_gmem_row = a_gmem_row;
    int c_gmem_col = b_gmem_col;
    int c_gmem_addr = c_gmem_row * N + c_gmem_col;
    c[c_gmem_addr] = sum;
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8>
__global__ void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_kernel(float *a, float *b,
                                                                  float *c, int M, int N,
                                                                  int K) {
    __shared__ float a_smem[BM][BK];
    __shared__ float b_smem[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int a_smem_row = tid / (BK / 4);
    int a_smem_col = tid % (BK / 4) * 4;
    int b_smem_row = tid / (BN / 4);
    int b_smem_col = tid % (BN / 4) * 4;

    int a_gmem_row = blockIdx.y * BM + a_smem_row;
    int b_gmem_col = blockIdx.x * BN + b_smem_col;

    float reg_c[TM][TN] = {0.0}; // 8x8
    for (int bk = 0; bk < ((K + BK - 1) / BK); ++bk) {
        int a_gmem_col = bk * BK + a_smem_col;
        int a_gmem_addr = a_gmem_row * K + a_gmem_col;
        FLOAT4(a_smem[a_smem_row][a_smem_col]) = FLOAT4(a[a_gmem_addr]);
        int b_gmem_row = bk * BK + b_smem_row;
        int b_gmem_addr = b_gmem_row * N + b_gmem_col;
        FLOAT4(b_smem[b_smem_row][b_smem_col]) = FLOAT4(b[b_gmem_addr]);
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    int a_smem_row_comp = threadIdx.y * TM + m;
                    int b_smem_col_comp = threadIdx.x * TN + n;
                    reg_c[m][n] +=
                        a_smem[a_smem_row_comp][k] * b_smem[k][b_smem_col_comp];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; ++m) {
        int c_gmem_row = blockIdx.y * BM + threadIdx.y * TM + m;
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            int c_gmem_col = blockIdx.x * BN + threadIdx.x * TN + n;
            int c_gmem_addr = c_gmem_row * N + c_gmem_col;
            FLOAT4(c[c_gmem_addr]) = FLOAT4(reg_c[m][n]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
    __shared__ float a_smem[BK][BM + OFFSET];
    __shared__ float b_smem[BK][BN + OFFSET];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int a_smem_m = tid / (BK / 4);
    int a_smem_k = (tid & 1) << 2;
    int b_smem_k = tid / (BN / 4);
    int b_smem_n = (tid & 31) << 2; // (tid & 31) = tid % 32, <<2 = *4

    int a_gmem_m = blockIdx.y * BM + a_smem_m;
    int b_gmem_n = blockIdx.x * BN + b_smem_n;
    if (a_gmem_m >= M || b_gmem_n >= N) {
        return;
    }

    // a_smem stored transposed; 4 consecutive global loads map to separate smem
    // rows, buffer via regs first
    float reg_a[TM / 2];  // 4
    float reg_comp_a[TM]; // 8个寄存器，从a_smem读出A的一小列
    float reg_comp_b[TN]; // 8个寄存器，从b_smem读出B的一小行
    float reg_c[TM][TN] = {0.0};

    for (int bk = 0; bk < ((K + BK - 1) / BK); ++bk) {
        /*
        a_smem[8][128] 每行128元素 -> 4 bank layer
            某一行的第0~31列 -> bank0~31
            第32~63列 -> 又一轮bank0~31
            共4 layer

        写 a_smem 时有bank conflict：a_smem[k][m] = reg_a[...]
            tid0和tid1负责m=0（一个搬k=0~3，一个搬k=4~7）,都是bank0
            tid2和tid3负责m=1 -> 都是bank1
            以此类推，每相邻两个线程有bank conflict
        */
        int a_gmem_k = bk * BK + a_smem_k;
        int a_gmem_addr = a_gmem_m * K + a_gmem_k;
        FLOAT4(reg_a[0]) = FLOAT4(a[a_gmem_addr]);
        a_smem[a_smem_k][a_smem_m] = reg_a[0];
        a_smem[a_smem_k + 1][a_smem_m] = reg_a[1];
        a_smem[a_smem_k + 2][a_smem_m] = reg_a[2];
        a_smem[a_smem_k + 3][a_smem_m] = reg_a[3];

        /*
        写 b_smem 时有bank conflict：用float4一个线程一次写4个连续列，占4个bank
            tid0写n=0~3 -> bank0~3
            tid1写n=4~7 -> bank4~7
            ...
            tid7写n=28~31 -> bank28~31
            （前8个线程刚好铺满32bank）
            tid8写n=32~35 -> bank0~3 与 tid0 有 bank conflict
            -> tid 0/8/16/24 都撞 bank0~3，4路冲突
        */
        int b_gmem_k = bk * BK + b_smem_k;
        int b_gmem_addr = b_gmem_k * N + b_gmem_n;
        FLOAT4(b_smem[b_smem_k][b_smem_n]) = FLOAT4(b[b_gmem_addr]);

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            /*
            这里计算时读a_smem/b_smem有冲突，以下面一句代码为例
                ty=0的线程读列0~3 -> bank0~3
                ty=1的线程读列4~7 -> bank4~7
            */
            FLOAT4(reg_comp_a[0]) = FLOAT4(a_smem[k][threadIdx.y * TM / 2]);
            FLOAT4(reg_comp_a[4]) =
                FLOAT4(a_smem[k][threadIdx.y * TM / 2 + BM / 2]);

            FLOAT4(reg_comp_b[0]) = FLOAT4(b_smem[k][threadIdx.x * TN / 2]);
            FLOAT4(reg_comp_b[4]) =
                FLOAT4(b_smem[k][threadIdx.x * TN / 2 + BN / 2]);
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                for (int n = 0; n < TN; ++n) {
                    // reg_c[m][n] += reg_comp_a[m] * reg_comp_b[n];
                    reg_c[m][n] =
                        __fmaf_rn(reg_comp_a[m], reg_comp_b[n], reg_c[m][n]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + i;
        int c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int c_gmem_addr = c_gmem_m * N + c_gmem_n;
        FLOAT4(c[c_gmem_addr]) = FLOAT4(reg_c[i][0]);
        FLOAT4(c[c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; ++i) {
        int c_gmem_m = blockIdx.y * BM + threadIdx.y * TM / 2 + BM / 2 + i;
        int c_gmem_n = blockIdx.x * BN + threadIdx.x * TN / 2;
        int c_gmem_addr = c_gmem_m * N + c_gmem_n;
        FLOAT4(c[c_gmem_addr]) = FLOAT4(reg_c[i + TM / 2][0]);
        FLOAT4(c[c_gmem_addr + BN / 2]) = FLOAT4(reg_c[i + TM / 2][4]);
    }
}
