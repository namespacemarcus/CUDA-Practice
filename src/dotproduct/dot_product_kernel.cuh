#include "reduce.cuh"

template <const int NUM_THREADS = 256>
__global__ void dot_product_f32_f32_kernel(float *a, float *b, float *y,
                                           int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float shared[NUM_WARPS];

    float prod = (idx < N) ? a[idx] * b[idx] : 0.0f;
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    if (laneId == 0) {
        shared[warpId] = prod;
    }
    __syncthreads();

    prod = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    if (warpId == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (threadIdx.x == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void dot_product_f32x4_f32_kernel(float *a, float *b, float *y,
                                             int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float shared[NUM_WARPS];

    float prod;
    if (idx + 3 < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        prod = reg_a.x * reg_b.x + reg_a.y * reg_b.y + reg_a.z * reg_b.z +
               reg_a.w * reg_b.w;
    } else {
        prod = 0.0f;
        for (int i = idx; i < N; ++i) {
            prod += a[i] * b[i];
        }
    }
    prod = warp_reduce_sum_f32<WARP_SIZE>(prod);
    if (laneId == 0) {
        shared[warpId] = prod;
    }
    __syncthreads();

    prod = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    if (warpId == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (threadIdx.x == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256>
__global__ void dot_product_f16_f32_kernel(half *a, half *b, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float shared[NUM_WARPS];

    half prod_f16 = (idx < N) ? __hmul(a[idx], b[idx]) : __float2half(0.0f);
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    if (laneId == 0) {
        shared[warpId] = prod;
    }
    __syncthreads();

    prod = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    if (warpId == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (threadIdx.x == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void dot_product_f16x2_f32_kernel(half *a, half *b, float *y,
                                             int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float shared[NUM_WARPS];

    half prod_f16;
    if (idx + 1 < N) {
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        prod_f16 = __hadd(__hmul(reg_a.x, reg_b.x), __hmul(reg_a.y, reg_b.y));
    } else if (idx < N) {
        prod_f16 = __hmul(a[idx], b[idx]);
    } else {
        prod_f16 = __float2half(0.0f);
    }
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    if (laneId == 0) {
        shared[warpId] = prod;
    }
    __syncthreads();

    prod = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    if (warpId == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (threadIdx.x == 0) {
        atomicAdd(y, prod);
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void dot_product_f16x8_pack_f32_kernel(half *a, half *b, float *y,
                                                  int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    __shared__ float shared[NUM_WARPS];

    half pack_a[8], pack_b[8];
    half prod_f16;
    if (idx + 7 < N) {
        LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
        LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);
        prod_f16 = __float2half(0.0f);
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            half2 v = __hmul2(HALF2(pack_a[i]), HALF2(pack_b[i]));
            prod_f16 = __hadd(prod_f16, __hadd(v.x, v.y));
        }
    } else {
        prod_f16 = __float2half(0.0f);
        for (int i = idx; i < N; ++i) {
            prod_f16 = __hadd(prod_f16, __hmul(a[i], b[i]));
        }
    }
    float prod = warp_reduce_sum_f16_f32<WARP_SIZE>(prod_f16);
    if (laneId == 0) {
        shared[warpId] = prod;
    }
    __syncthreads();

    prod = (laneId < NUM_WARPS) ? shared[laneId] : 0.0f;
    if (warpId == 0) {
        prod = warp_reduce_sum_f32<NUM_WARPS>(prod);
    }

    if (threadIdx.x == 0) {
        atomicAdd(y, prod);
    }
}
