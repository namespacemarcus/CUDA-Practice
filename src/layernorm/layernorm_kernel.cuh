#include "../common/pack.h"
#include "reduce.cuh"

// x, y: s*d
// grid(s), block(d)
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float gamma,
                                      float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float value = (idx < s * d) ? x[idx] : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon);
    }
    __syncthreads();

    if (idx < s * d) {
        y[idx] = ((value - s_mean) * s_variance) * gamma + beta;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float gamma,
                                        float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float4 reg_x;
    if (idx + 3 < s * d) {
        reg_x = FLOAT4(x[idx]);
    } else {
        reg_x.x = (idx + 0 < s * d) ? x[idx + 0] : 0.0f;
        reg_x.y = (idx + 1 < s * d) ? x[idx + 1] : 0.0f;
        reg_x.z = (idx + 2 < s * d) ? x[idx + 2] : 0.0f;
        reg_x.w = (idx + 3 < s * d) ? x[idx + 3] : 0.0f;
    }
    float local_sum = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
    float sum = block_reduce_sum_f32<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float4 reg_x_hat;
    reg_x_hat.x = (idx + 0 < s * d) ? (reg_x.x - s_mean) : 0.0f;
    reg_x_hat.y = (idx + 1 < s * d) ? (reg_x.y - s_mean) : 0.0f;
    reg_x_hat.z = (idx + 2 < s * d) ? (reg_x.z - s_mean) : 0.0f;
    reg_x_hat.w = (idx + 3 < s * d) ? (reg_x.w - s_mean) : 0.0f;
    float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                     reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon);
    }
    __syncthreads();

    float4 reg_y;
    reg_y.x = reg_x_hat.x * s_variance * gamma + beta;
    reg_y.y = reg_x_hat.y * s_variance * gamma + beta;
    reg_y.z = reg_x_hat.z * s_variance * gamma + beta;
    reg_y.w = reg_x_hat.w * s_variance * gamma + beta;
    if (idx + 3 < s * d) {
        FLOAT4(y[idx]) = reg_y;
    } else {
        if (idx + 0 < s * d) {
            y[idx + 0] = reg_y.x;
        }
        if (idx + 1 < s * d) {
            y[idx + 1] = reg_y.y;
        }
        if (idx + 2 < s * d) {
            y[idx + 2] = reg_y.z;
        }
        if (idx + 3 < s * d) {
            y[idx + 3] = reg_y.w;
        }
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float gamma,
                                          float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half value = (idx < s * d) ? x[idx] : __float2half(0.0f);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon);
    }
    __syncthreads();

    if (idx < s * d) {
        y[idx] = __hfma((value - s_mean) * s_variance, gamma_, beta_);
        // y[idx] = ((value - s_mean) * s_variance) * gamma_ + beta_;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x2_f16_kernel(half *x, half *y, float gamma,
                                            float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);
    const half zero_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half2 reg_x;
    if (idx + 1 < s * d) {
        reg_x = HALF2(x[idx]);
    } else {
        reg_x.x = (idx < s * d) ? x[idx] : zero_;
        reg_x.y = zero_;
    }
    half local_sum = reg_x.x + reg_x.y;
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half2 reg_x_hat;
    reg_x_hat.x = (idx + 0 < s * d) ? (reg_x.x - s_mean) : zero_;
    reg_x_hat.y = (idx + 1 < s * d) ? (reg_x.y - s_mean) : zero_;
    half variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon);
    }
    __syncthreads();

    half2 reg_y;
    reg_y.x = __hfma(reg_x_hat.x * s_variance, gamma_, beta_);
    reg_y.y = __hfma(reg_x_hat.y * s_variance, gamma_, beta_);
    if (idx + 1 < s * d) {
        HALF2(y[idx]) = reg_y;
    } else if (idx < s * d) {
        y[idx] = reg_y.x;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float gamma,
                                            float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);
    const half zero_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half2 reg_x_0, reg_x_1, reg_x_2, reg_x_3;
    if (idx + 1 < s * d) {
        reg_x_0 = HALF2(x[idx + 0]);
    } else {
        reg_x_0.x = (idx + 0 < s * d) ? x[idx + 0] : zero_;
        reg_x_0.y = zero_;
    }
    if (idx + 3 < s * d) {
        reg_x_1 = HALF2(x[idx + 2]);
    } else {
        reg_x_1.x = (idx + 2 < s * d) ? x[idx + 2] : zero_;
        reg_x_1.y = zero_;
    }
    if (idx + 5 < s * d) {
        reg_x_2 = HALF2(x[idx + 4]);
    } else {
        reg_x_2.x = (idx + 4 < s * d) ? x[idx + 4] : zero_;
        reg_x_2.y = zero_;
    }
    if (idx + 7 < s * d) {
        reg_x_3 = HALF2(x[idx + 6]);
    } else {
        reg_x_3.x = (idx + 6 < s * d) ? x[idx + 6] : zero_;
        reg_x_3.y = zero_;
    }

    half value = reg_x_0.x + reg_x_0.y + reg_x_1.x + reg_x_1.y + reg_x_2.x +
                 reg_x_2.y + reg_x_3.x + reg_x_3.y;
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
    reg_x_hat_0.x = (idx + 0 < s * d) ? (reg_x_0.x - s_mean) : zero_;
    reg_x_hat_0.y = (idx + 1 < s * d) ? (reg_x_0.y - s_mean) : zero_;
    reg_x_hat_1.x = (idx + 2 < s * d) ? (reg_x_1.x - s_mean) : zero_;
    reg_x_hat_1.y = (idx + 3 < s * d) ? (reg_x_1.y - s_mean) : zero_;
    reg_x_hat_2.x = (idx + 4 < s * d) ? (reg_x_2.x - s_mean) : zero_;
    reg_x_hat_2.y = (idx + 5 < s * d) ? (reg_x_2.y - s_mean) : zero_;
    reg_x_hat_3.x = (idx + 6 < s * d) ? (reg_x_3.x - s_mean) : zero_;
    reg_x_hat_3.y = (idx + 7 < s * d) ? (reg_x_3.y - s_mean) : zero_;

    half variance =
        reg_x_hat_0.x * reg_x_hat_0.x + reg_x_hat_0.y * reg_x_hat_0.y +
        reg_x_hat_1.x * reg_x_hat_1.x + reg_x_hat_1.y * reg_x_hat_1.y +
        reg_x_hat_2.x * reg_x_hat_2.x + reg_x_hat_2.y * reg_x_hat_2.y +
        reg_x_hat_3.x * reg_x_hat_3.x + reg_x_hat_3.y * reg_x_hat_3.y;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon);
    }
    __syncthreads();

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = __hfma(reg_x_hat_0.x * s_variance, gamma_, beta_);
    reg_y_0.y = __hfma(reg_x_hat_0.y * s_variance, gamma_, beta_);
    reg_y_1.x = __hfma(reg_x_hat_1.x * s_variance, gamma_, beta_);
    reg_y_1.y = __hfma(reg_x_hat_1.y * s_variance, gamma_, beta_);
    reg_y_2.x = __hfma(reg_x_hat_2.x * s_variance, gamma_, beta_);
    reg_y_2.y = __hfma(reg_x_hat_2.y * s_variance, gamma_, beta_);
    reg_y_3.x = __hfma(reg_x_hat_3.x * s_variance, gamma_, beta_);
    reg_y_3.y = __hfma(reg_x_hat_3.y * s_variance, gamma_, beta_);

    if ((idx + 1) < s * d) {
        HALF2(y[idx + 0]) = reg_y_0;
    } else if ((idx + 0) < s * d) {
        y[idx + 0] = reg_y_0.x;
    }
    if ((idx + 3) < s * d) {
        HALF2(y[idx + 2]) = reg_y_1;
    } else if ((idx + 2) < s * d) {
        y[idx + 2] = reg_y_1.x;
    }
    if ((idx + 5) < s * d) {
        HALF2(y[idx + 4]) = reg_y_2;
    } else if ((idx + 4) < s * d) {
        y[idx + 4] = reg_y_2.x;
    }
    if ((idx + 7) < s * d) {
        HALF2(y[idx + 6]) = reg_y_3;
    } else if ((idx + 6) < s * d) {
        y[idx + 6] = reg_y_3.x;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float gamma,
                                                 float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);
    const half zero_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half pack_x[8];
    if (idx + 7 < s * d) {
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pack_x[i] = ((idx + i) < s * d) ? x[idx + i] : zero_;
        }
    }
    half value = zero_;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value += pack_x[i];
    }
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half variance = zero_;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        half hat = ((idx + i) < s * d) ? (pack_x[i] - s_mean) : zero_;
        variance += hat * hat;
    }
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        half y_val =
            ((idx + i) < s * d)
                ? __hfma((pack_x[i] - s_mean) * s_variance, gamma_, beta_)
                : zero_;
        pack_y[i] = y_val;
    }
    if (idx + 7 < s * d) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((idx + i) < s * d) {
                y[idx + i] = pack_y[i];
            }
        }
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float gamma,
                                          float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float value = (idx < s * d) ? __half2float(x[idx]) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float variance = (value - s_mean) * (value - s_mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon);
    }
    __syncthreads();

    if (idx < s * d) {
        y[idx] =
            __float2half(__fmaf_rn((value - s_mean) * s_variance, gamma, beta));
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float gamma,
                                                 float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const float epsilon = 1e-5f;

    __shared__ float s_mean;
    __shared__ float s_variance;

    half pack_x[8];
    if (idx + 7 < s * d) {
        LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            pack_x[i] = ((idx + i) < s * d) ? x[idx + i] : __float2half(0.0f);
        }
    }
    float value = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value += __half2float(pack_x[i]);
    }
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float variance = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float x_hat =
            ((idx + i) < s * d) ? (__half2float(pack_x[i]) - s_mean) : 0.0f;
        variance += x_hat * x_hat;
    }
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float y_val =
            ((idx + i) < s * d)
                ? __fmaf_rn((__half2float(pack_x[i]) - s_mean) * s_variance,
                            gamma, beta)
                : 0.0f;
        pack_y[i] = __float2half(y_val);
    }
    if (idx + 7 < s * d) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            if ((idx + i) < s * d) {
                y[idx + i] = pack_y[i];
            }
        }
    }
}
