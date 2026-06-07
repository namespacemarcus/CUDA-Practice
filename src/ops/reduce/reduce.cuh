#pragma once

#include "warp_reduce.cuh"

// grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float sum = (idx < N) ? a[idx] : 0.0f;
    sum = warp_reduce_sum_f32_f32<WARP_SIZE>(sum);
    if (laneId == 0) {
        reduce_smem[warpId] = sum;
    }
    __syncthreads();

    // 256 threads in a block, 8 warps.
    // only 0~7 lane per warp have valid data.
    sum = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    // the first warp compute the final sum.
    if (warpId == 0) {
        sum = warp_reduce_sum_f32_f32<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

// grid(N/256), block(256/4=64), per thread compute 4 fp32
template <const int NUM_THREADS = 256 / 4>
__global__ void block_all_reduce_sum_f32x4_f32_kernel(float *a, float *y,
                                                      int N) {
    int tid = threadIdx.x;
    // 64 threads per block, but compute 256 numbers.
    int idx = (blockIdx.x * NUM_THREADS + tid) * 4;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    float4 reg = FLOAT4(a[idx]);
    float sum = (idx < N) ? (reg.x + reg.y + reg.z + reg.w) : 0.0f;
    sum = warp_reduce_sum_f32_f32<WARP_SIZE>(sum);

    if (laneId == 0) {
        reduce_smem[warpId] = sum;
    }
    __syncthreads();

    sum = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum = warp_reduce_sum_f32_f32<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

// grid(N/256), block(256)
template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f16_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = __half2float(sum_f16);
    }
    __syncthreads();

    float sum = (laneId < NUM_WARPS) ? reduce _smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum = warp_reduce_sum_f32_f32<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f16_f32_kernel(half *a, float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half sum_f16 = (idx < N) ? a[idx] : __float2half(0.0f);
    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f32_kernel(half *a, float *y,
                                                      int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half2 reg = HALF2(a[idx]);
    half sum_f16 = (idx < N) ? __hadd(reg.x, reg.y) : __float2half(0.0f);
    float sum_f32 = warp_reduce_sum_f16_f32<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_f16x2_f16_kernel(half *a, float *y,
                                                      int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half2 reg = HALF2(a[idx]);
    half sum_f16 = (idx < N) ? __hadd(reg.x, reg.y) : __float2half(0.0f);
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = __half2float(sum_f16);
    }
    __syncthreads();

    float sum = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum = warp_reduce_sum_f32_f32<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f16_kernel(half *a, float *y,
                                                           int N) {
    int tid = threadIdx.x;
    int idx =
        (blockIdx.x * NUM_THREADS + tid) * 8; // 8 half elements per thread.
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]); // load 128 bits.
    half sum_f16 = __float2half(0.0f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((idx + i) < N) {
            sum_f16 = __hadd(sum_f16, pack[i]);
        }
    }
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = __half2float(sum_f16);
    }
    __syncthreads();

    float sum = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum = warp_reduce_sum_f32_f32<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, sum);
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_f16x8_pack_f32_kernel(half *a, float *y,
                                                           int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    half pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    float sum_f32 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((idx + i) < N) {
            sum_f32 += __half2float(pack[i]);
        }
    }
    sum_f32 = warp_reduce_sum_f32_f32<WARP_SIZE>(sum_f32);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_bf16_kernel(__nv_bfloat16 *a,
                                                      float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_bf16;
    }
    __syncthreads();

    __nv_bfloat16 sum =
        (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2bfloat16(0.0f);
    if (warpId == 0) {
        sum = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum);
    }

    if (tid == 0) {
        atomicAdd(y, __bfloat162float(sum));
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_bf16_f32_kernel(__nv_bfloat16 *a, float *y,
                                                     int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    __nv_bfloat16 sum_bf16 = (idx < N) ? a[idx] : __float2bfloat16(0.0f);
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_bf16_kernel(__nv_bfloat16 *a,
                                                        float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    __nv_bfloat162 reg = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_bf16 =
        (idx < N) ? __hadd(reg.x, reg.y) : __float2bfloat16(0.0f);
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_bf16;
    }
    __syncthreads();

    sum_bf16 =
        (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2bfloat16(0.0f);
    if (warpId == 0) {
        sum_bf16 = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum_bf16);
    }

    if (tid == 0) {
        atomicAdd(y, __bfloat162float(sum_bf16));
    }
}

template <const int NUM_THREADS = 256 / 2>
__global__ void block_all_reduce_sum_bf16x2_f32_kernel(__nv_bfloat16 *a,
                                                       float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 2;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    __nv_bfloat162 reg = BFLOAT2(a[idx]);
    __nv_bfloat16 sum_bf16 =
        (idx < N) ? __hadd(reg.x, reg.y) : __float2bfloat16(0.0f);
    float sum_f32 = warp_reduce_sum_bf16_f32<WARP_SIZE>(sum_bf16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_bf16_kernel(__nv_bfloat16 *a,
                                                             float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ __nv_bfloat16 reduce_smem[NUM_WARPS];

    __nv_bfloat16 pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    __nv_bfloat16 sum_bf16 = __float2bfloat16(0.0f);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((idx + i) < N) {
            sum_bf16 = __hadd(sum_bf16, pack[i]);
        }
    }
    sum_bf16 = warp_reduce_sum_bf16_bf16<WARP_SIZE>(sum_bf16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_bf16;
    }
    __syncthreads();

    sum_bf16 =
        (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2bfloat16(0.0f);
    if (warpId == 0) {
        sum_bf16 = warp_reduce_sum_bf16_bf16<NUM_WARPS>(sum_bf16);
    }

    if (tid == 0) {
        atomicAdd(y, __bfloat162float(sum_bf16));
    }
}

template <const int NUM_THREADS = 256 / 8>
__global__ void block_all_reduce_sum_bf16x8_pack_f32_kernel(__nv_bfloat16 *a,
                                                            float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 8;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ float reduce_smem[NUM_WARPS];

    __nv_bfloat16 pack[8];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    float sum_f32 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((idx + i) < N) {
            sum_f32 += __bfloat162float(pack[i]);
        }
    }
    sum_f32 = warp_reduce_sum_f32_f32<WARP_SIZE>(sum_f32);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f32;
    }
    __syncthreads();

    sum_f32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0.0f;
    if (warpId == 0) {
        sum_f32 = warp_reduce_sum_f32_f32<NUM_WARPS>(sum_f32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_f32);
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e4m3_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

    __nv_fp8_storage_t sum_fp8 =
        (idx < N) ? a[idx]
                  : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E4M3);
    half sum_f16 = warp_reduce_sum_fp8_e4m3_f16<WARP_SIZE>(sum_fp8);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f16;
    }
    __syncthreads();

    sum_f16 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2half(0.0f);
    if (warpId == 0) {
        sum_f16 = warp_reduce_sum_f16_f16<NUM_WARPS>(sum_f16);
    }

    if (tid == 0) {
        atomicAdd(y, __half2float(sum_f16));
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_fp8_e5m2_f16_kernel(__nv_fp8_storage_t *a,
                                                         float *y, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

    __nv_fp8_storage_t sum_fp8 =
        (idx < N) ? a[idx]
                  : __nv_cvt_float_to_fp8(0.0f, __NV_SATFINITE, __NV_E5M2);
    half sum_f16 = warp_reduce_sum_fp8_e5m2_f16<WARP_SIZE>(sum_fp8);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f16;
    }
    __syncthreads();

    sum_f16 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2half(0.0f);
    if (warpId == 0) {
        sum_f16 = warp_reduce_sum_f16_f16<NUM_WARPS>(sum_f16);
    }

    if (tid == 0) {
        atomicAdd(y, __half2float(sum_f16));
    }
}

template <const int NUM_THREADS = 256 / 16>
__global__ void
block_all_reduce_sum_fp8_e4m3x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                 float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

    __nv_fp8_storage_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    half sum_f16 = __float2half(0.0f);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if ((idx + i) < N) {
            sum_f16 += __nv_cvt_fp8_to_halfraw(pack[i], __NV_E4M3);
        }
    }
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f16;
    }
    __syncthreads();

    sum_f16 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2half(0.0f);
    if (warpId == 0) {
        sum_f16 = warp_reduce_sum_f16_f16<NUM_WARPS>(sum_f16);
    }

    if (tid == 0) {
        atomicAdd(y, __half2float(sum_f16));
    }
}

template <const int NUM_THREADS = 256 / 16>
__global__ void
block_all_reduce_sum_fp8_e5m2x16_pack_f16_kernel(__nv_fp8_storage_t *a,
                                                 float *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ half reduce_smem[NUM_WARPS];

    __nv_fp8_storage_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    half sum_f16 = __float2half(0.0f);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        if ((idx + i) < N) {
            sum_f16 += __nv_cvt_fp8_to_halfraw(pack[i], __NV_E5M2);
        }
    }
    sum_f16 = warp_reduce_sum_f16_f16<WARP_SIZE>(sum_f16);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_f16;
    }
    __syncthreads();

    sum_f16 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : __float2half(0.0f);
    if (warpId == 0) {
        sum_f16 = warp_reduce_sum_f16_f16<NUM_WARPS>(sum_f16);
    }

    if (tid == 0) {
        atomicAdd(y, __half2float(sum_f16));
    }
}

template <const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_i8_i32_kernel(int8_t *a, int32_t *y,
                                                   int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * NUM_THREADS + tid;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ int32_t reduce_smem[NUM_WARPS];

    int8_t val_i8 = (idx < N) ? a[idx] : 0;
    int32_t sum_i32 = warp_reduce_sum_i8_i32<WARP_SIZE>(val_i8);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_i32;
    }
    __syncthreads();

    sum_i32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0;
    if (warpId == 0) {
        sum_i32 = warp_reduce_sum_i32_i32<NUM_WARPS>(sum_i32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_i32);
    }
}

template <const int NUM_THREADS = 256 / 16>
__global__ void block_all_reduce_sum_i8x16_pack_i32_kernel(int8_t *a,
                                                           int32_t *y, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * NUM_THREADS + tid) * 16;
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = tid / WARP_SIZE;
    int laneId = tid % WARP_SIZE;
    __shared__ int32_t reduce_smem[NUM_WARPS];

    int8_t pack[16];
    LDST128BITS(pack[0]) = LDST128BITS(a[idx]);
    int32_t sum_i32 = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        sum_i32 += static_cast<int32_t>(pack[i]);
    }
    sum_i32 = warp_reduce_sum_i32_i32<WARP_SIZE>(sum_i32);
    if (laneId == 0) {
        reduce_smem[warpId] = sum_i32;
    }
    __syncthreads();

    sum_i32 = (laneId < NUM_WARPS) ? reduce_smem[laneId] : 0;
    if (warpId == 0) {
        sum_i32 = warp_reduce_sum_i32_i32<NUM_WARPS>(sum_i32);
    }

    if (tid == 0) {
        atomicAdd(y, sum_i32);
    }
}
