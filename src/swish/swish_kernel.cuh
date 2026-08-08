#include "../common/pack.h"
#include <cuda_fp16.h>

__device__ __forceinline__ float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swish_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = swish(x[idx]);
    }
}

__global__ void swish_f32x4_kernel(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 reg_x = FLOAT4(x[idx]);
        float4 reg_y;
        reg_y.x = swish(reg_x.x);
        reg_y.y = swish(reg_x.y);
        reg_y.z = swish(reg_x.z);
        reg_y.w = swish(reg_x.w);
        FLOAT4(y[idx]) = reg_y;
    } else {
        for (int i = 0; i < 4; ++i) {
            if (idx + i < N) {
                float reg_x = idx + i < N ? x[idx + i] : 0.0f;
                y[idx + i] = swish(reg_x);
            }
        }
    }
}

__device__ __forceinline__ half swish_half(half x) {
    return __hmul(x, __hdiv(__float2half(1.0f),
                            __hadd(__float2half(1.0f), hexp(__hneg(x)))));
}

__global__ void swish_f16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = swish_half(x[idx]);
    }
}

__global__ void swish_f16x2_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_y.x = swish_half(reg_x.x);
        reg_y.y = swish_half(reg_x.y);
        HALF2(y[idx]) = reg_y;
    } else if (idx < N) {
        half reg_x = x[idx];
        y[idx] = swish_half(reg_x);
    }
}

__global__ void swish_f16x8_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (idx + 7 < N) {
        half2 reg_x_0 = HALF2(x[idx + 0]);
        half2 reg_x_1 = HALF2(x[idx + 2]);
        half2 reg_x_2 = HALF2(x[idx + 4]);
        half2 reg_x_3 = HALF2(x[idx + 6]);
        half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
        reg_y_0.x = swish_half(reg_x_0.x);
        reg_y_0.y = swish_half(reg_x_0.y);
        reg_y_1.x = swish_half(reg_x_1.x);
        reg_y_1.y = swish_half(reg_x_1.y);
        reg_y_2.x = swish_half(reg_x_2.x);
        reg_y_2.y = swish_half(reg_x_2.y);
        reg_y_3.x = swish_half(reg_x_3.x);
        reg_y_3.y = swish_half(reg_x_3.y);
        HALF2(y[idx + 0]) = reg_y_0;
        HALF2(y[idx + 2]) = reg_y_1;
        HALF2(y[idx + 4]) = reg_y_2;
        HALF2(y[idx + 6]) = reg_y_3;
    } else {
        for (int i = idx; i < N; ++i) {
            y[i] = swish_half(x[i]);
        }
    }
}

__global__ void swish_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
    half pack_x[8], pack_y[8];
    if (idx + 7 < N) {
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pack_y[i] = swish_half(pack_x[i]);
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else {
        for (int i = idx; i < N; ++i) {
            y[i] = swish_half(x[i]);
        }
    }
}
