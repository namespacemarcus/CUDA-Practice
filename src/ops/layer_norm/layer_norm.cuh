#include "reduce.cuh"

static const float epsilon_f32 = 1e-5f;
static const half epsilon_f16 = __float2half(1e-5f);

// x, y: s*d
// grid(s), block(d)
template <const int NUM_THREADS = 256>
__global__ void layer_norm_f32_kernel(float *x, float *y, float gamma,
                                      float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float value = (idx < s * d) ? x[idx] : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float variance = (value - mean) * (value - mean);
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon_f32);
    }
    __syncthreads();

    if (idx < s * k) {
        y[idx] = ((value - s_mean) * variance) * gamma + beta;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void layer_norm_f32x4_kernel(float *x, float *y, float gamma,
                                        float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    __shared__ float s_mean;
    __shared__ float s_variance;

    float4 reg_x = FLOAT4(x[idx]);
    float local_sum =
        (idx < s * d) ? (reg_x.x + reg_x.y + reg_x.z + reg_x.w) : 0.0f;
    float sum = block_reduce_sum_f32<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float4 reg_x_hat;
    reg_x_hat.x = reg_x.x - s_mean;
    reg_x_hat.y = reg_x.y - s_mean;
    reg_x_hat.z = reg_x.z - s_mean;
    reg_x_hat.w = reg_x.w - s_mean;
    float variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y +
                     reg_x_hat.z * reg_x_hat.z + reg_x_hat.w * reg_x_hat.w;
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon_f32);
    }
    __syncthreads();

    float reg_y;
    reg_y.x = reg_x_hat.x * s_variance * gamma + beta;
    reg_y.y = reg_x_hat.y * s_variance * gamma + beta;
    reg_y.z = reg_x_hat.z * s_variance * gamma + beta;
    reg_y.w = reg_x_hat.w * s_variance * gamma + beta;
    if (idx < s * d) {
        FLOAT4(y[idx]) = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f16_kernel(half *x, half *y, float gamma,
                                          float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
        s_variance = rsqrtf(variance / d_ + epsilon_f16);
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

    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);

    __shared__ float s_mean;
    __shared__ float s_variance;

    half2 reg_x = HALF2(x[idx]);
    half local_sum = (idx < s * d) ? (reg_x.x + reg_x.y) : __float2half(0.0f);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(local_sum);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half2 reg_x_hat;
    reg_x_hat.x = reg_x.x - s_mean;
    reg_x_hat.y = reg_x.y - s_mean;
    half variance = reg_x_hat.x * reg_x_hat.x + reg_x_hat.y * reg_x_hat.y;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / d_ + epsilon_f16);
    }
    __syncthreads();

    if (idx < s * d) {
        half2 reg_y;
        reg_y.x = __hfma(reg_x_hat.x * s_variance, gamma_, beta_);
        reg_y.y = __hfma(reg_x_hat.y * s_variance, gamma_, beta_);
        HALF2(y[idx]) = reg_y;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_f16_kernel(half *x, half *y, float gamma,
                                            float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);
    const half zero_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half2 reg_x_0 = HALF2(x[idx + 0]);
    half2 reg_x_1 = HALF2(x[idx + 2]);
    half2 reg_x_2 = HALF2(x[idx + 4]);
    half2 reg_x_3 = HALF2(x[idx + 6]);

    half value = (idx + 0 < s * d) ? (reg_x_0.x + reg_x_0.y);
    value += (idx + 2 < s * d)     ? (reg_x_1.x + reg_x_1.y);
    value += (idx + 4 < s * d)     ? (reg_x_2.x + reg_x_2.y);
    value += (idx + 6 < s * d)     ? (reg_x_3.x + reg_x_3.y);
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half2 reg_x_hat_0, reg_x_hat_1, reg_x_hat_2, reg_x_hat_3;
    reg_x_hat_0.x = reg_x_0.x - s_mean;
    reg_x_hat_0.y = reg_x_0.y - s_mean;
    reg_x_hat_1.x = reg_x_1.x - s_mean;
    reg_x_hat_1.y = reg_x_1.y - s_mean;
    reg_x_hat_2.x = reg_x_2.x - s_mean;
    reg_x_hat_2.y = reg_x_2.y - s_mean;
    reg_x_hat_3.x = reg_x_3.x - s_mean;
    reg_x_hat_3.y = reg_x_3.y - s_mean;

    half variance = zero_;
    variance +=
        (idx + 0 < s * d)
            ? (reg_x_hat_0.x * reg_x_hat_0.x + reg_x_hat_0.y * reg_x_hat_0.y)
            : zero_;
    variance +=
        (idx + 2 < s * d)
            ? (reg_x_hat_1.x * reg_x_hat_1.x + reg_x_hat_1.y * reg_x_hat_1.y)
            : zero_;
    variance +=
        (idx + 4 < s * d)
            ? (reg_x_hat_2.x * reg_x_hat_2.x + reg_x_hat_2.y * reg_x_hat_2.y)
            : zero_;
    variance +=
        (idx + 6 < s * d)
            ? (reg_x_hat_3.x * reg_x_hat_3.x + reg_x_hat_3.y * reg_x_hat_3.y)
            : zero_;
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon_f16);
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
    if ((idx + 0) < limit) {
        HALF2(y[idx + 0]) = reg_y_0;
    }
    if ((idx + 2) < limit) {
        HALF2(y[idx + 2]) = reg_y_1;
    }
    if ((idx + 4) < limit) {
        HALF2(y[idx + 4]) = reg_y_2;
    }
    if ((idx + 6) < limit) {
        HALF2(y[idx + 6]) = reg_y_3;
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f16_kernel(half *x, half *y, float gamma,
                                                 float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    const half gamma_ = __float2half(gamma);
    const half beta_ = __float2half(beta);
    const half d_ = __float2half(d);
    const half zero_ = __float2half(0.0f);

    __shared__ half s_mean;
    __shared__ half s_variance;

    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    half value = zero_;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value += ((idx + i) < s * k ? pack_x[i] : zero_);
    }
    half sum = block_reduce_sum_f16_f16<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / d_;
    }
    __syncthreads();

    half variance = zero_;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        variance += (idx + i) < s * d
                        ? (pack_x[i] - s_mean) * (pack_x[i] - s_mean)
                        : zero_;
    }
    variance = block_reduce_sum_f16_f16<NUM_THREADS>(variance);
    if (threadIdx.x == 0) {
        s_variance = hrsqrt(variance / d_ + epsilon_f16);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pack_y[i] = __hfma((pack_x[i] - s_mean) * s_variance, gamma_, beta_);
    }
    if ((idx + 7) < s * d) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16_f32_kernel(half *x, half *y, float gamma,
                                          float beta, int s, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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
    if (threadIdx.x) {
        s_variance = rsqrtf(variance / (float)d + epsilon_f32);
    }
    __syncthreads();

    if (idx < s * d) {
        y[idx] =
            __float2half(__fmaf((value - s_mean) * s_variance, gamma, beta));
    }
}

template <const int NUM_THREADS = 256>
__global__ void layer_norm_f16x8_pack_f32_kernel(half *x, half *y, float gamma,
                                                 float beta, int s, int d) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;

    __shared__ float s_mean;
    __shared__ float s_variance;

    half pack_x[8];
    LDST128BITS(pack_x[0]) = LDST128BITS(x[idx]);
    float value = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        value += ((idx + i) < s * d) ? __half2float(pack_x[i]) : 0.0f;
    }
    float sum = block_reduce_sum_f32<NUM_THREADS>(value);
    if (threadIdx.x == 0) {
        s_mean = sum / (float)d;
    }
    __syncthreads();

    float variance = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float x_hat = __half2float(pack_x[i]) - s_mean;
        variance += ((idx + i) < s * d) ? x_hat * x_hat : 0.0f;
    }
    variance = block_reduce_sum_f32<NUM_THREADS>(variance);
    if (thraedIdx.x == 0) {
        s_variance = rsqrtf(variance / (float)d + epsilon_f32);
    }
    __syncthreads();

    half pack_y[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        pack_y[i] =
            __float2half(__fmaf((__half2float(pack_x[i]) - s_mean) * s_variance,
                                gamma, beta);)
    }
    if (idx + 7 < s * d) {
        LDST128BITS(y[idx]) = LDST128BITS(pack_y[0]);
    }
}
