#include "reduce.cuh"

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f32_kernel(float *x, float *y, float gamma, int s,
                                    int d) {
    const float epsilon = 1e-5f;
    const int len = s * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_r_rms;
    float value = (idx < len) ? x[idx] : 0.0f;
    float sq = value * value;
    float sum_sq = block_reduce_sum_f32<NUM_THREADS>(sq);
    if (threadIdx.x == 0) {
        s_r_rms = rsqrtf(sum_sq / (float)d + epsilon);
    }
    __syncthreads();

    if (idx < len) {
        y[idx] = (value * s_r_rms) * gamma;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void rms_norm_f32x4_kernel(float *x, float *y, float gamma, int s,
                                      int d) {
    const float epsilon = 1e-5f;
    const int len = s * d;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    __shared__ float s_r_rms;
    float4 reg_x = FLOAT4(x[idx]);
    float sum_sq = reg_x.x * reg_x.x + reg_x.y * reg_x.y + reg_x.z * reg_x.z +
                   reg_x.w * reg_x.w;
    sum_sq = block_reduce_sum_f32<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = rsqrtf(sum_sq / (float)d + epsilon);
    }
    __syncthreads();

    float4 reg_y;
    reg_y.x = reg_x.x * s_r_rms * gamma;
    reg_y.y = reg_x.y * s_r_rms * gamma;
    reg_y.z = reg_x.z * s_r_rms * gamma;
    reg_y.w = reg_x.w * s_r_rms * gamma;
    if (idx < len) {
        FLOAT4(y[idx]) = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16_f16_kernel(half *x, half *y, float gamma, int s,
                                        int d) {
    const halfepsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half d_ = __int2half_rn(d);
    const int len = s * d;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ half s_r_rms;
    half value = (idx < len) ? x[idx] : __float2half(0.0f);
    half sq = value * value;
    half sum_sq = block_reduce_sum_f16_f16<NUM_THREADS>(sq);
    if (threadIdx.x == 0) {
        s_r_rms = hrsqrt(sum_sq / d_ + epsilon);
    }
    __syncthreads();

    if (idx < len) {
        y[idx] = (value * s_r_rms) * gamma_;
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16x2_f16_kernel(half *x, half *y, float gamma, int s,
                                          int d) {
    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half d_ = __int2half_rn(d);
    const int len = s * d;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

    __shared__ half s_r_rms;
    half2 reg_x = HALF2(x[idx]);
    half sum_sq = (reg_x.x * reg_x.x + reg_x.y * reg_x.y);
    sum_sq = block_reduce_sum_f16_f16<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = hrsqrt(sum_sq / d_ + epsilon);
    }
    __syncthreads();

    half2 reg_y;
    reg_y.x = reg_x.x * s_r_rms * gamma_;
    reg_y.y = reg_x.y * s_r_rms * gamma_;
    if (idx < len) {
        HALF2(y[idx]) = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16x8_f16_kernel(half *x, half *y, float gamma, int s,
                                          int d) {
    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half d_ = __int2half_rn(d);
    const int len = s * d;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    __shared__ half s_r_rms;
    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);
    half sum_sq = (reg_x_0.x * reg_x_0.x + reg_x_0.y * reg_x_0.y) +
                  (reg_x_1.x * reg_x_1.x + reg_x_1.y * reg_x_1.y) +
                  (reg_x_2.x * reg_x_2.x + reg_x_2.y * reg_x_2.y) +
                  (reg_x_3.x * reg_x_3.x + reg_x_3.y * reg_x_3.y);
    sum_sq = block_reduce_sum_f16_f16<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = hrsqrt(sum_sq / d_ + epsilon);
    }
    __syncthreads();

    half2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = reg_x_0.x * s_r_rms * gamma_;
    reg_y_0.y = reg_x_0.y * s_r_rms * gamma_;
    reg_y_1.x = reg_x_1.x * s_r_rms * gamma_;
    reg_y_1.y = reg_x_1.y * s_r_rms * gamma_;
    reg_y_2.x = reg_x_2.x * s_r_rms * gamma_;
    reg_y_2.y = reg_x_2.y * s_r_rms * gamma_;
    reg_y_3.x = reg_x_3.x * s_r_rms * gamma_;
    reg_y_3.y = reg_x_3.y * s_r_rms * gamma_;
    if (idx + 0 < len) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if (idx + 2 < len) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if (idx + 4 < len) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if (idx + 6 < len) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16x8_pack_f16_kernel(half *x, half *y, float gamma,
                                               int s, int d) {
    const half epsilon = __float2half(1e-5f);
    const half gamma_ = __float2half(gamma);
    const half d_ = __int2half_rn(d);
    const half zero_ = __float2half(0.0f);
    const int len = s * d;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    __shared__ half s_r_rms;
    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

    half sum_sq = zero_;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        sum_sq += idx + i < len ? pack_x[i] * pack_x[i] : zero_;
    }
    sum_sq = block_reduce_sum_f16_f16<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = hrsqrt(sum_sq / d_ + epsilon);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pack_y[i] = pack_x[i] * s_r_rms * gamma_;
    }
    if (idx + 7 < len) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16_f32_kernel(half *x, half *y, float gamma, int s,
                                        int d) {
    const float epsilon = 1e-5f;
    const int len = s * d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_r_rms;
    float value = idx < len ? __half2float(x[idx]) : 0.0f;
    float sq = value * value;
    float sum_sq = block_reduce_sum_f32<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = rsqrtf(sum_sq / (float)d + epsilon);
    }
    __syncthreads();

    if (idx < len) {
        y[idx] = __float2half(value * s_r_rms * gamma);
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16x8_f32_kernel(half *x, half *y, float gamma, int s,
                                          int d) {
    const float epsilon = 1e-5f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    __shared__ float s_r_rms;
    float2 reg_x_0 = __half22float2(HALF2(x[idx + 0]));
    float2 reg_x_1 = __half22float2(HALF2(x[idx + 2]));
    float2 reg_x_2 = __half22float2(HALF2(x[idx + 4]));
    float2 reg_x_3 = __half22float2(HALF2(x[idx + 6]));

    float sum_sq = (reg_x_0.x * reg_x_0.x + reg_x_0.y * reg_x_0.y) +
                   (reg_x_1.x * reg_x_1.x + reg_x_1.y * reg_x_1.y) +
                   (reg_x_2.x * reg_x_2.x + reg_x_2.y * reg_x_2.y) +
                   (reg_x_3.x * reg_x_3.x + reg_x_3.y * reg_x_3.y);
    sum_sq = block_reduce_sum_f32<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = rsqrtf(sum_sq / (float)d + epsilon);
    }
    __syncthreads();

    float2 reg_y_0, reg_y_1, reg_y_2, reg_y_3;
    reg_y_0.x = reg_x_0.x * s_r_rms * gamma;
    reg_y_0.y = reg_x_0.y * s_r_rms * gamma;
    reg_y_1.x = reg_x_1.x * s_r_rms * gamma;
    reg_y_1.y = reg_x_1.y * s_r_rms * gamma;
    reg_y_2.x = reg_x_2.x * s_r_rms * gamma;
    reg_y_2.y = reg_x_2.y * s_r_rms * gamma;
    reg_y_3.x = reg_x_3.x * s_r_rms * gamma;
    reg_y_3.y = reg_x_3.y * s_r_rms * gamma;
    if (idx + 0 < len) {
        HALF2(y[idx + 0]) = __float22half2_rn(reg_y_0);
    }
    if ((idx + 2) < len) {
        HALF2(y[idx + 2]) = __float22half2_rn(reg_y_1);
    }
    if ((idx + 4) < len) {
        HALF2(y[idx + 4]) = __float22half2_rn(reg_y_2);
    }
    if ((idx + 6) < len) {
        HALF2(y[idx + 6]) = __float22half2_rn(reg_y_3);
    }
}

template <const int NUM_THREADS = 256>
__global__ void rms_norm_f16x8_pack_f32_kernel(half *x, half *y, float gamma,
                                               int s, int d) {
    const float epsilon = 1e-5f;
    const int len = s * d;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    __shared__ float s_r_rms;
    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);

    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = __half2float(pack_x[i]);
        sum_sq += ((idx + i) < len ? v * v : 0.0f);
    }
    sum_sq = block_reduce_sum_f32<NUM_THREADS>(sum_sq);
    if (threadIdx.x == 0) {
        s_r_rms = rsqrtf(sum_sq / (float)d + epsilon);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pack_y[i] = __float2half(__half2float(pack_x[i]) * s_r_rms * gamma);
    }

    if (idx + 7 < len) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}
