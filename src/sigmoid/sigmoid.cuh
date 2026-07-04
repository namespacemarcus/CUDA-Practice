#include "../common/cuda/cuda_utils.h"
#include <cuda_fp16.h>

__device__ __forceinline__ float clamp_exp_f32(float v) {
    return fminf(fmaxf(v, MIN_EXP_F32), MAX_EXP_F32);
}

__device__ __forceinline__ half clamp_exp_f16(half v) {
    return __hmin(__hmax(v, MIN_EXP_F16), MAX_EXP_F16);
}

__global__ void sigmoid_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = x[idx];
        v = clamp_exp_f32(v);
        y[idx] = 1.0f / (1.0f + expf(-v));
    }
}

__global__ void sigmoid_f32x4_kernel(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < N) {
        float4 reg_x = FLOAT4(x[idx]);
        reg_x.x = clamp_exp_f32(reg_x.x);
        reg_x.y = clamp_exp_f32(reg_x.y);
        reg_x.z = clamp_exp_f32(reg_x.z);
        reg_x.w = clamp_exp_f32(reg_x.w);

        float4 reg_y;
        reg_y.x = 1.0f / (1.0f + expf(-reg_x.x));
        reg_y.y = 1.0f / (1.0f + expf(-reg_x.y));
        reg_y.z = 1.0f / (1.0f + expf(-reg_x.z));
        reg_y.w = 1.0f / (1.0f + expf(-reg_x.w));

        FLOAT4(y[idx]) = reg_y;
    } else {
        for (int i = idx; i < N; ++i) {
            float v = x[i];
            v = clamp_exp_f32(v);
            y[i] = 1.0f / (1.0f + expf(-v));
        }
    }
}

__global__ void sigmoid_f16_kernel(half *x, half *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const half f = __float2half(1.0f);
    if (idx < N) {
        half v = x[idx];
        v = clamp_exp_f16(v);
        y[idx] = f / (f + hexp(-v));
    }
}

__global__ void sigmoid_f16x2_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const half f = __float2half(1.0f);
    if (idx + 1 < N) {
        half2 reg_x = HALF2(x[idx]);
        half2 reg_y;
        reg_x.x = clamp_exp_f16(reg_x.x);
        reg_x.y = clamp_exp_f16(reg_x.y);

        reg_y.x = f / (f + hexp(-reg_x.x));
        reg_y.y = f / (f + hexp(-reg_x.y));

        HALF2(y[idx]) = reg_y;
    } else if (idx < N) {
        half v = x[idx];
        v = clamp_exp_f16(v);
        y[idx] = f / (f + hexp(-v));
    }
}

__global__ void sigmoid_f16x8_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const half f = __float2half(1.0f);
    if (idx + 7 < N) {
        half2 reg_x_0 = HALF2(x[idx + 0]);
        half2 reg_x_1 = HALF2(x[idx + 2]);
        half2 reg_x_2 = HALF2(x[idx + 4]);
        half2 reg_x_3 = HALF2(x[idx + 6]);

        reg_x_0.x = clamp_exp_f16(reg_x_0.x);
        reg_x_0.y = clamp_exp_f16(reg_x_0.y);
        reg_x_1.x = clamp_exp_f16(reg_x_1.x);
        reg_x_1.y = clamp_exp_f16(reg_x_1.y);
        reg_x_2.x = clamp_exp_f16(reg_x_2.x);
        reg_x_2.y = clamp_exp_f16(reg_x_2.y);
        reg_x_3.x = clamp_exp_f16(reg_x_3.x);
        reg_x_3.y = clamp_exp_f16(reg_x_3.y);

        half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;

        reg_y_0.x = f / (f + hexp(-reg_x_0.x));
        reg_y_0.y = f / (f + hexp(-reg_x_0.y));
        reg_y_1.x = f / (f + hexp(-reg_x_1.x));
        reg_y_1.y = f / (f + hexp(-reg_x_1.y));
        reg_y_2.x = f / (f + hexp(-reg_x_2.x));
        reg_y_2.y = f / (f + hexp(-reg_x_2.y));
        reg_y_3.x = f / (f + hexp(-reg_x_3.x));
        reg_y_3.y = f / (f + hexp(-reg_x_3.y));

        HALF2(y[idx + 0]) = reg_y_0;
        HALF2(y[idx + 2]) = reg_y_1;
        HALF2(y[idx + 4]) = reg_y_2;
        HALF2(y[idx + 6]) = reg_y_3;
    } else {
        for (int i = idx; i < N; ++i) {
            half v = x[i];
            v = clamp_exp_f16(v);
            y[i] = f / (f + hexp(-v));
        }
    }
}

__global__ void sigmoid_f16x8_pack_kernel(half *x, half *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const half f = __float2half(1.0f);
    half pack_x[8], pack_y[8];
    if (idx + 7 < N) {
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            half v = clamp_exp_f16(pack_x[i]);
            pack_y[i] = f / (f + hexp(-v));
        }
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else {
        for (int i = idx; i < N; ++i) {
            half v = x[i];
            v = clamp_exp_f16(v);
            y[i] = f / (f + hexp(-v));
        }
    }
}
