#pragma once

#include "../common/cuda/cuda_utils.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define SWIZZLE_A(row, col) ((col) ^ (((row >> 1) & 0x3) << 3))

#define SWIZZLE_B(row, col) ((col) ^ (((row) & 0x7) << 3))

#define SWIZZLE_C(row, col) ((col) ^ (((row) & 0x7) << 3))

// cp.async
#define CP_ASYNC_CG(dst_smem_ptr_32b, src_global_ptr)                          \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"        \
                 :                                                             \
                 : "r"(dst_smem_ptr_32b), "l"(src_global_ptr))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP_0() asm volatile("cp.async.wait_group 0;\n" ::)

// ldmatrix
#define LDMATRIX_X4(R0, R1, R2, R3, PTR)                                       \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"     \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                               \
        : "r"(PTR))

#define LDMATRIX_X2(R0, R1, PTR)                                               \
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"    \
                 : "=r"(R0), "=r"(R1)                                          \
                 : "r"(PTR))

#define LDMATRIX_X2_TRANS(R0, R1, PTR)                                         \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"       \
        : "=r"(R0), "=r"(R1)                                                   \
        : "r"(PTR))

// mma.sync
#define M16N8K16_F16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                   \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "          \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                      \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),       \
                   "f"(C0), "f"(C1), "f"(C2), "f"(C3))

#define M16N8K16_BF16(C0, C1, C2, C3, A0, A1, A2, A3, B0, B1)                  \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "        \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n" \
                 : "=f"(C0), "=f"(C1), "=f"(C2), "=f"(C3)                      \
                 : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),       \
                   "f"(C0), "f"(C1), "f"(C2), "f"(C3))

template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_tiled_kernel(const T *__restrict__ A,
                                   const T *__restrict__ B, T *__restrict__ C,
                                   int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    int warp_row = warpId / 4; // 0, 1
    int warp_col = warpId % 4; // 0, 1, 2, 3

    float sum[4][4][4]{0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        uint32_t smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        const T *global_a0 = &A[(by * BM + load_a_row) * K + bk + load_a_col];
        const T *global_a1 =
            &A[(by * BM + load_a_row + 64) * K + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row][load_b_col]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row + 16][load_b_col]));

        const T *global_b0 = &B[(bk + load_b_row) * N + bx * BN + load_b_col];
        const T *global_b1 =
            &B[(bk + load_b_row + 16) * N + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_row * 64 + m * 16 + (laneId % 16);
                int a_col = k_offset + (laneId / 16) * 8;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            smem_addr);
            }

#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int b_row = k_offset + (laneId % 16);
                int b_col = warp_col * 32 + n * 8;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
            }

#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                     sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                     reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                     reg_b[n][1]);
                    } else {
                        M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                      sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                      reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                      reg_b[n][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    int t_row = laneId / 4;       // 0~7
    int t_col = (laneId % 4) * 2; // 0, 2, 4, 6
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c_base_row = by * BM + warp_row * 64 + m * 16;
            int c_base_col = bx * BN + warp_col * 32 + n * 8;
            int idx_0 = (c_base_row + t_row) * N + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * N + c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(C[idx_0]) = __float22half2_rn(FLOAT2(sum[m][n][0]));
                HALF2(C[idx_2]) = __float22half2_rn(FLOAT2(sum[m][n][2]));
            } else {
                BFLOAT2(C[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m][n][0]));
                BFLOAT2(C[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m][n][2]));
            }
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_gw_tiled_kernel(const T *__restrict__ A,
                                      const T *__restrict__ B,
                                      T *__restrict__ C, int M, int N, int K) {
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int bx = ((linear_block_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W) +
             (linear_block_id % SWIZZLE_W);
    int by = (linear_block_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    int warp_row = warpId / 4; // 0, 1
    int warp_col = warpId % 4; // 0, 1, 2, 3

    float sum[4][4][4]{0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        uint32_t smem_a0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row][load_a_col]));
        uint32_t smem_a1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&As[load_a_row + 64][load_a_col]));

        const T *global_a0 = &A[(by * BM + load_a_row) * K + bk + load_a_col];
        const T *global_a1 =
            &A[(by * BM + load_a_row + 64) * K + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        uint32_t smem_b0 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row][load_b_col]));
        uint32_t smem_b1 = static_cast<uint32_t>(
            __cvta_generic_to_shared(&Bs[load_b_row + 16][load_b_col]));

        const T *global_b0 = &B[(bk + load_b_row) * N + bx * BN + load_b_col];
        const T *global_b1 =
            &B[(bk + load_b_row + 16) * N + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_row * 64 + m * 16 + (laneId % 16);
                int a_col =
                    k_offset + (laneId / 16) * 8; // k_step = 0: a_col = 0/8
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&As[a_row][a_col]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            smem_addr);
            }

#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int b_row = k_offset + (laneId % 16);
                int b_col = warp_col * 32 + n * 8;
                uint32_t smem_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&Bs[b_row][b_col]));
                LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
            }

#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                     sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                     reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                     reg_b[n][1]);
                    } else {
                        M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                      sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                      reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                      reg_b[n][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    int t_row = laneId / 4;       // 0~7
    int t_col = (laneId % 4) * 2; // 0, 2, 4, 6
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c_base_row = by * BM + warp_row * 64 + m * 16;
            int c_base_col = bx * BN + warp_col * 32 + n * 8;
            int idx_0 = (c_base_row + t_row) * N + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * N + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(C[idx_0]) = __float22half2_rn(FLOAT2(sum[m][n][0]));
                HALF2(C[idx_2]) = __float22half2_rn(FLOAT2(sum[m][n][2]));
            } else {
                BFLOAT2(C[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m][n][0]));
                BFLOAT2(C[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m][n][2]));
            }
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void
hgemm_gw_tiled_bcf_kernel(const T *__restrict__ A, const T *__restrict__ B,
                          T *__restrict__ C, int M, int N, int K) {
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int bx = ((linear_block_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W) +
             (linear_block_id % SWIZZLE_W);
    int by = (linear_block_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    int warp_row = warpId / 4;
    int warp_col = warpId % 4;

    float sum[4][4][4]{0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &As[load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &As[load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

        const T *global_a0 = &A[(by * BM + load_a_row) * K + bk + load_a_col];
        const T *global_a1 =
            &A[(by * BM + load_a_row + 64) * K + bk + load_a_col];

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &Bs[load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &Bs[load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

        const T *global_b0 = &B[(bk + load_b_row) * N + bx * BN + load_b_col];
        const T *global_b1 =
            &B[(bk + load_b_row + 16) * N + bx * BN + load_b_col];

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_row * 64 + m * 16 + (laneId % 16);
                int a_col = k_offset + (laneId / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &As[a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            smem_addr);
            }
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int b_row = k_offset + (laneId % 16);
                int b_col = warp_col * 32 + n * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &Bs[b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
            }

#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                     sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                     reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                     reg_b[n][1]);
                    } else {
                        M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                      sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                      reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                      reg_b[n][1]);
                    }
                }
            }
        }
        __syncthreads();
    }

    int t_row = laneId / 4;
    int t_col = (laneId % 4) * 2;

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c_base_row = by * BM + warp_row * 64 + m * 16;
            int c_base_col = bx * BN + warp_col * 32 + n * 8;
            int idx_0 = (c_base_row + t_row) * N + c_base_col + t_col;
            int idx_2 = (c_base_row + t_row + 8) * N + c_base_col + t_col;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(C[idx_0]) = __float22half2_rn(FLOAT2(sum[m][n][0]));
                HALF2(C[idx_2]) = __float22half2_rn(FLOAT2(sum[m][n][2]));
            } else {
                BFLOAT2(C[idx_0]) = __float22bfloat162_rn(FLOAT2(sum[m][n][0]));
                BFLOAT2(C[idx_2]) = __float22bfloat162_rn(FLOAT2(sum[m][n][2]));
            }
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void
hgemm_gw_tiled_bcf_dbf_kernel(const T *__restrict__ A, const T *__restrict__ B,
                              T *__restrict__ C, int M, int N, int K) {
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int bx = ((linear_block_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W) +
             (linear_block_id % SWIZZLE_W);
    int by = (linear_block_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    __shared__ T As[2][BM][BK];
    __shared__ T Bs[2][BK][BN];

    int warp_row = warpId / 4;
    int warp_col = warpId % 4;

    float sum[4][4][4]{0.0f};

    /* prefetch As/Bs once. */
    // cp.async load A
    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    const T *global_a0 = &A[(by * BM + load_a_row) * K + load_a_col];
    const T *global_a1 = &A[(by * BM + load_a_row + 64) * K + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    // cp.async load B
    uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    const T *global_b0 = &B[(load_b_row + 0) * N + bx * BN + load_b_col];
    const T *global_b1 = &B[(load_b_row + 16) * N + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    for (int bk = 32; bk < K; bk += BK) {
        // 1. cp.async load A
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &As[write_idx][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &As[write_idx][load_a_row + 64]
               [SWIZZLE_A(load_a_row + 64, load_a_col)]));

        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        // 2. cp.async load B
        smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &Bs[write_idx][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &Bs[write_idx][load_b_row + 16]
               [SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * N;
        global_b1 += BK * N;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

        // 3. Tensor core:  two k_step, 16 elements each
#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

            // ldmatrix.x4: A(64, 16), (16, 16) per instruction
#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_row * 64 + m * 16 + (laneId % 16);
                int a_col = k_offset + (laneId / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            smem_addr);
            }

            // ldmatrix.x2: A(16, 32), (16, 8) per instruction
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int b_row = k_offset + (laneId % 16);
                int b_col = warp_col * 32 + n * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
            }

            // mma: 4x4 times m16n8k16
#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                     sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                     reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                     reg_b[n][1]);
                    } else {
                        M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                      sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                      reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                      reg_b[n][1]);
                    }
                }
            }
        }

        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();

        read_idx ^= 1;
        write_idx ^= 1;
    }

    // epilogue
#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

        // Four ldmatrix.issue for A (4 * 16 = 64 rows)
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            // ldmatrix x4 loads a 16x16 tile
            int a_row = warp_row * 64 + m * 16 + (laneId % 16);
            int a_col = k_offset + (laneId / 16) * 8;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                &As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                        smem_addr);
        }

        // Four ldmatrix.issue for B (4 * 8 = 32 columns)
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            // Lanes 0-15 load 16 rows (bases of two 8x8 tiles)
            int b_row = k_offset + (laneId % 16);
            int b_col = warp_col * 32 + n * 8;

            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                &Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
        }

        // MMA body: 4x4 m16n8k16
#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                 sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                 reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                 reg_b[n][1]);
                } else {
                    M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                  sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                  reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                  reg_b[n][1]);
                }
            }
        }
    }

    // store C
    int t_row = laneId / 4;
    int t_col = (laneId % 4) * 2;

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c_base_row = by * BM + warp_row * 64 + m * 16;
            int c_base_col = bx * BN + warp_col * 32 + n * 8;
            if constexpr (std::is_same_v<T, __half>) {
                HALF2(C[(c_base_row + t_row) * N + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m][n][0]));
                HALF2(C[(c_base_row + t_row + 8) * N + c_base_col + t_col]) =
                    __float22half2_rn(FLOAT2(sum[m][n][2]));
            } else {
                BFLOAT2(C[(c_base_row + t_row) * N + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m][n][0]));
                BFLOAT2(C[(c_base_row + t_row + 8) * N + c_base_col + t_col]) =
                    __float22bfloat162_rn(FLOAT2(sum[m][n][2]));
            }
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 32, typename T>
__global__ void hgemm_gw_tiled_bcf_dbf_cstore_kernel(const T *__restrict__ A,
                                                     const T *__restrict__ B,
                                                     T *__restrict__ C, int M,
                                                     int N, int K) {
    int linear_block_id = blockIdx.y * gridDim.x + blockIdx.x;
    const int SWIZZLE_W = 8;

    int bx = ((linear_block_id / (SWIZZLE_W * gridDim.y)) * SWIZZLE_W) +
             (linear_block_id % SWIZZLE_W);
    int by = (linear_block_id / SWIZZLE_W) % gridDim.y;

    int tid = threadIdx.x;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;

    int load_a_row = tid / 4;
    int load_a_col = (tid % 4) * 8;
    int load_b_row = tid / 16;
    int load_b_col = (tid % 16) * 8;

    __shared__ __align__(128) union {
        struct {
            T As[2][BM][BK];
            T Bs[2][BK][BN];
        };

        T Cs[BM][BN];
    } smem;

    int warp_row = warpId / 4;
    int warp_col = warpId % 4;

    float sum[4][4][4]{0.0f};

    uint32_t smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &smem.As[0][load_a_row][SWIZZLE_A(load_a_row, load_a_col)]));
    uint32_t smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &smem.As[0][load_a_row + 64][SWIZZLE_A(load_a_row + 64, load_a_col)]));

    const T *global_a0 = &A[(by * BM + load_a_row) * K + load_a_col];
    const T *global_a1 = &A[(by * BM + load_a_row + 64) * K + load_a_col];

    CP_ASYNC_CG(smem_a0, global_a0);
    CP_ASYNC_CG(smem_a1, global_a1);

    uint32_t smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &smem.Bs[0][load_b_row][SWIZZLE_B(load_b_row, load_b_col)]));
    uint32_t smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(
        &smem.Bs[0][load_b_row + 16][SWIZZLE_B(load_b_row + 16, load_b_col)]));

    const T *global_b0 = &B[(load_b_row + 0) * N + bx * BN + load_b_col];
    const T *global_b1 = &B[(load_b_row + 16) * N + bx * BN + load_b_col];

    CP_ASYNC_CG(smem_b0, global_b0);
    CP_ASYNC_CG(smem_b1, global_b1);

    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP_0();
    __syncthreads();

    int read_idx = 0;
    int write_idx = 1;

    for (int bk = 32; bk < K; bk += BK) {
        smem_a0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &smem.As[write_idx][load_a_row]
                    [SWIZZLE_A(load_a_row, load_a_col)]));
        smem_a1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &smem.As[write_idx][load_a_row + 64]
                    [SWIZZLE_A(load_a_row + 64, load_a_col)]));

        global_a0 += BK;
        global_a1 += BK;

        CP_ASYNC_CG(smem_a0, global_a0);
        CP_ASYNC_CG(smem_a1, global_a1);

        smem_b0 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &smem.Bs[write_idx][load_b_row]
                    [SWIZZLE_B(load_b_row, load_b_col)]));
        smem_b1 = static_cast<uint32_t>(__cvta_generic_to_shared(
            &smem.Bs[write_idx][load_b_row + 16]
                    [SWIZZLE_B(load_b_row + 16, load_b_col)]));

        global_b0 += BK * N;
        global_b1 += BK * N;

        CP_ASYNC_CG(smem_b0, global_b0);
        CP_ASYNC_CG(smem_b1, global_b1);

        CP_ASYNC_COMMIT_GROUP();

#pragma unroll
        for (int k_step = 0; k_step < 2; ++k_step) {
            int k_offset = k_step * 16;

            uint32_t reg_a[4][4];
            uint32_t reg_b[4][2];

#pragma unroll
            for (int m = 0; m < 4; ++m) {
                int a_row = warp_row * 64 + m * 16 + (laneId % 16);
                int a_col = k_offset + (laneId / 16) * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
                LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                            smem_addr);
            }

#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int b_row = k_offset + (laneId % 16);
                int b_col = warp_col * 32 + n * 8;
                uint32_t smem_addr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(
                        &smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
                LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
            }

#pragma unroll
            for (int m = 0; m < 4; ++m) {
#pragma unroll
                for (int n = 0; n < 4; ++n) {
                    if constexpr (std::is_same_v<T, __half>) {
                        M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                     sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                     reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                     reg_b[n][1]);
                    } else {
                        M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                      sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                      reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                      reg_b[n][1]);
                    }
                }
            }
        }
        CP_ASYNC_WAIT_GROUP_0();
        __syncthreads();
        read_idx ^= 1;
        write_idx ^= 1;
    }

#pragma unroll
    for (int k_step = 0; k_step < 2; ++k_step) {
        int k_offset = k_step * 16;

        uint32_t reg_a[4][4];
        uint32_t reg_b[4][2];

#pragma unroll
        for (int m = 0; m < 4; ++m) {
            int a_row = warp_row * 64 + m * 16 + (laneId % 16);
            int a_col = k_offset + (laneId / 16) * 8;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                &smem.As[read_idx][a_row][SWIZZLE_A(a_row, a_col)]));
            LDMATRIX_X4(reg_a[m][0], reg_a[m][1], reg_a[m][2], reg_a[m][3],
                        smem_addr);
        }

#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int b_row = k_offset + (laneId % 16);
            int b_col = warp_col * 32 + n * 8;
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(
                &smem.Bs[read_idx][b_row][SWIZZLE_B(b_row, b_col)]));
            LDMATRIX_X2_TRANS(reg_b[n][0], reg_b[n][1], smem_addr);
        }

#pragma unroll
        for (int m = 0; m < 4; ++m) {
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                if constexpr (std::is_same_v<T, __half>) {
                    M16N8K16_F16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                 sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                 reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                 reg_b[n][1]);
                } else {
                    M16N8K16_BF16(sum[m][n][0], sum[m][n][1], sum[m][n][2],
                                  sum[m][n][3], reg_a[m][0], reg_a[m][1],
                                  reg_a[m][2], reg_a[m][3], reg_b[n][0],
                                  reg_b[n][1]);
                }
            }
        }
    }

    /* store C */
    __syncthreads();

    int t_row = laneId / 4;
    int t_col = (laneId % 4) * 2;

#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            int c_base_row = warp_row * 64 + m * 16;
            int c_base_col = warp_col * 32 + n * 8;

            int c_row_0 = c_base_row + t_row;
            int c_row_2 = c_base_row + t_row + 8;
            int c_col = c_base_col + t_col;

            if constexpr (std::is_same_v<T, __half>) {
                HALF2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22half2_rn(FLOAT2(sum[m][n][0]));
                HALF2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22half2_rn(FLOAT2(sum[m][n][2]));
            } else {
                BFLOAT2(smem.Cs[c_row_0][SWIZZLE_C(c_row_0, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m][n][0]));
                BFLOAT2(smem.Cs[c_row_2][SWIZZLE_C(c_row_2, c_col)]) =
                    __float22bfloat162_rn(FLOAT2(sum[m][n][2]));
            }
        }
    }

    __syncthreads();

    T *c_block = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int step = 0; step < 8; ++step) {
        int row = step * 16 + tid / 16;
        int col = (tid % 16) * 8;

        FLOAT4(c_block[row * N + col]) =
            FLOAT4(smem.Cs[row][SWIZZLE_C(row, col)]);
    }
}
