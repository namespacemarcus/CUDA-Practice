#include "./common/reduce.cuh"
#include <cuda_fp16.h>

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < N) ? x[idx] : -FLT_MAX;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max
    float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (idx < N) {
        y[idx] = exp_val / exp_sum;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void safe_softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    float4 reg_x = FLOAT4(x[idx]);
    reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
    reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;
    reg_x.z = (idx + 2 < N) ? reg_x.z : -FLT_MAX;
    reg_x.w = (idx + 3 < N) ? reg_x.w : -FLT_MAX;

    float val = fmaxf(reg_x.x, reg_x.y);
    val = fmaxf(val, reg_x.z);
    val = fmaxf(val, reg_x.w);
    float max_val = block_reduce_max_f32<NUM_THREADS>(val); // block max

    float4 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;
    reg_exp.z = (idx + 2 < N) ? expf(reg_x.z - max_val) : 0.0f;
    reg_exp.w = (idx + 3 < N) ? expf(reg_x.w - max_val) : 0.0f;

    float local_exp_sum = (reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w);
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(local_exp_sum);

    if (idx + 3 < N) {
        float4 reg_y;
        reg_y.x = reg_exp.x / exp_sum;
        reg_y.y = reg_exp.y / exp_sum;
        reg_y.z = reg_exp.z / exp_sum;
        reg_y.w = reg_exp.w / exp_sum;
        FLOAT4(y[idx]) = reg_y;
    } else {
        if (idx + 0 < N) {
            y[idx + 0] = reg_exp.x / exp_sum;
        }
        if (idx + 1 < N) {
            y[idx + 1] = reg_exp.y / exp_sum;
        }
        if (idx + 2 < N) {
            y[idx + 2] = reg_exp.z / exp_sum;
        }
    }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16_f32_per_token_kernel(half *x, half *y, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < N) ? __half2float(x[idx]) : -FLT_MAX;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (idx < N) {
        y[idx] = __float2half(exp_val / exp_sum);
    }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x2_f32_per_token_kernel(half *x, half *y,
                                                        int N) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    float2 reg_x = __half22float2(HALF2(x[idx]));
    reg_x.x = (idx + 0 < N) ? reg_x.x : -FLT_MAX;
    reg_x.y = (idx + 1 < N) ? reg_x.y : -FLT_MAX;

    float max_val = fmaxf(reg_x.x, reg_x.y);
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val);

    float2 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x - max_val) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y - max_val) : 0.0f;

    float local_exp_sum = reg_exp.x + reg_exp.y;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(local_exp_sum);

    float2 reg_y;
    reg_y.x = reg_exp.x / exp_sum;
    reg_y.y = reg_exp.y / exp_sum;

    if (idx + 1 < N) {
        HALF2(y[idx]) = __float22half2_rn(reg_y);
    } else if (idx < N) {
        y[idx] = __float2half_rn(reg_y.x);
    }
}

template <const int NUM_THREADS = 256>
__global__ void safe_softmax_f16x8_pack_f32_per_token_kernel(half *x, half *y,
                                                             int N) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pack_x[i] = (idx + i < N) ? pack_x[i] : -FLT_MAX;
    }

    float max_val = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        max_val = fmaxf(max_val, __half2float(pack_x[i]));
    }
    max_val = block_reduce_max_f32<NUM_THREADS>(max_val);

    float local_exp_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float exp_val =
            (idx + i < N) ? expf(__half2float(pack_x[i]) - max_val) : 0.0f;
        local_exp_sum += exp_val;
    }
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(local_exp_sum);

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float exp_val =
            (idx + i < N) ? expf(__half2float(pack_x[i]) - max_val) : 0.0f;
        pack_y[i] = __float2half(exp_val / exp_sum);
    }

    if (idx + 7 < N) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (idx + i < N) {
                y[idx + i] = pack_y[i];
            }
        }
    }
}
