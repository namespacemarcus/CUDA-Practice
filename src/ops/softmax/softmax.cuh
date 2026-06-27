
#include "./common/reduce.cuh"

// x: (s, d), y: (s, d)
// grid(s), block(d)
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float *x, float *y, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float exp_val = (idx < N) ? expf(x[idx]) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (idx < N) {
        y[idx] = exp_val / exp_sum;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void softmax_f32x4_per_token_kernel(float *x, float *y, int N) {
    const int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    float4 reg_x = FLOAT4(x[idx]);
    float4 reg_exp;
    reg_exp.x = (idx + 0 < N) ? expf(reg_x.x) : 0.0f;
    reg_exp.y = (idx + 1 < N) ? expf(reg_x.y) : 0.0f;
    reg_exp.z = (idx + 2 < N) ? expf(reg_x.z) : 0.0f;
    reg_exp.w = (idx + 3 < N) ? expf(reg_x.w) : 0.0f;

    float exp_val = reg_exp.x + reg_exp.y + reg_exp.z + reg_exp.w;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    if (idx + 3 < N) {
        float4 reg_y;
        reg_y.x = reg_exp.x / (exp_sum);
        reg_y.y = reg_exp.y / (exp_sum);
        reg_y.z = reg_exp.z / (exp_sum);
        reg_y.w = reg_exp.w / (exp_sum);
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
