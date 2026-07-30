#include "../common/cuda/cuda_utils.h"

#define theta 10000.0f

__global__ void rope_f32_kernel(float *x, float *out, int seq_len,
                                int pairs_per_token) {
    int token_idx = blockIdx.x;
    int pair_idx = threadIdx.x;

    int base = token_idx * pairs_per_token * 2 + pair_idx * 2;
    float x1 = x[base];
    float x2 = x[base + 1];

    float inv_freq =
        1.0f / powf(theta, (float)(2 * pair_idx) / (pairs_per_token * 2));
    float sin_v = sinf(token_idx * inv_freq);
    float cos_v = cosf(token_idx * inv_freq);

    float out1 = x1 * cos_v - x2 * sin_v;
    float out2 = x1 * sin_v + x2 * cos_v;

    out[base] = out1;
    out[base + 1] = out2;
}

__global__ void rope_f32x4_kernel(float *x, float *out, int seq_len,
                                  int quarter_dim) {
    int token_idx = blockIdx.x;
    int i = threadIdx.x;

    int d = quarter_dim * 4;
    int base = token_idx * d + i * 4;

    float4 reg_x = FLOAT4(x[base]);

    float inv_freq_1 = 1.0f / powf(theta, (float)(2 * (2 * i)) / d);
    float inv_freq_2 = 1.0f / powf(theta, (float)(2 * (2 * i + 1)) / d);
    float sin_v_1 = sinf(token_idx * inv_freq_1);
    float cos_v_1 = cosf(token_idx * inv_freq_1);
    float sin_v_2 = sinf(token_idx * inv_freq_2);
    float cos_v_2 = cosf(token_idx * inv_freq_2);

    float4 reg_out;
    reg_out.x = reg_x.x * cos_v_1 - reg_x.y * sin_v_1;
    reg_out.y = reg_x.x * sin_v_1 + reg_x.y * cos_v_1;
    reg_out.z = reg_x.z * cos_v_2 - reg_x.w * sin_v_2;
    reg_out.w = reg_x.z * sin_v_2 + reg_x.w * cos_v_2;

    FLOAT4(out[base]) = reg_out;
}
