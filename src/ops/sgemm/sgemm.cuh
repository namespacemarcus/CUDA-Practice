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
void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_kernel(float *a, float *b,
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

    float r_c[TM][TN] = {0.0}; // 8x8
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
                    r_c[m][n] +=
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
            FLOAT4(c[c_gmem_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

template <const int BM = 128, const int BN = 128, const int BK = 8,
          const int TM = 8, const int TN = 8, const int OFFSET = 0>
__global__ void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_kernel(
    float *a, float *b, float *c, const int M, const int N, const int K) {
    __shared__ float a_smem

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
}
