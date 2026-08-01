#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kKvTileSize = 16;
constexpr int kMaxHeadDim = 256;

template <typename scalar_t, int kMaxDim, bool kCausal>
__global__ void flash_attention_naive_kernel(const scalar_t *__restrict__ query,
                                             const scalar_t *__restrict__ key,
                                             const scalar_t *__restrict__ value,
                                             scalar_t *__restrict__ output,
                                             int q_seqlen, int kv_seqlen,
                                             int head_dim, int num_heads) {
    constexpr int kValuesPerLane = (kMaxDim + kWarpSize - 1) / kWarpSize;

    const int warpId = threadIdx.x / kWarpSize;
    const int laneId = threadIdx.x % kWarpSize;
    const int query_id = blockIdx.x * kWarpsPerBlock + warpId;
    const bool query_is_valid = query_id < q_seqlen;

    const int64_t batch_head_id =
        static_cast<int64_t>(blockIdx.z) * num_heads + blockIdx.y;
    const int64_t query_base =
        batch_head_id * static_cast<int64_t>(q_seqlen) * head_dim;
    const int64_t kv_base =
        batch_head_id * static_cast<int64_t>(kv_seqlen) * head_dim;

    float query_fragment[kValuesPerLane];
    float output_acc[kValuesPerLane];

#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
        const int dim = laneId + item * kWarpSize;
        query_fragment[item] =
            (query_is_valid && dim < head_dim)
                ? scalar_to_float(
                      query[query_base +
                            static_cast<int64_t>(query_id) * head_dim + dim])
                : 0.0f;
        output_acc[item] = 0.0f;
    }
    if (!query_is_valid) {
        return;
    }

    // Online Softmax:
    //     m: maximum scanned score.
    //     l: sum(exp(score - m)), normalization denominator.
    float m = -CUDART_INF_F;
    float l = 0.0f;
    const float softmax_scale = rsqrtf(static_cast<float>(head_dim));
    const int effective_kv_seqlen =
        (kCausal && query_id < kv_seqlen) ? query_id + 1 : kv_seqlen;

    for (int key_id = 0; key_id < effective_kv_seqlen; ++key_id) {
        float partial_score = 0.0f;
#pragma unroll
        for (int item = 0; item < kValuesPerLane; ++item) {
            const int dim = laneId + item * kWarpSize;
            if (dim < head_dim) {
                const float key_value = scalar_to_float(
                    key[kv_base + static_cast<int64_t>(key_id) * head_dim +
                        dim]);
                partial_score =
                    __fmaf_rn(query_fragment[item], key_value, partial_score);
            }
        }

        const float score = warp_reduce_sum(partial_score) * softmax_scale; // attention score

        // m_new = max(m_old, m_new)
        // alpha = exp(m_old - m_mew)
        // p = exp(score - m_new)
        // l_new = alpha * l_old + p
        // O_new = alpha * O_old + p * V[key_id]
        const float m_old = m;
        const float m_new = fmaxf(m, score);
        const float alpha = expf(m_old - m_new);
        const float p = expf(score - m_new);

#pragma unroll
        for (int item = 0; item < kValuesPerLane; ++item) {
            const int dim = laneId + item * kWarpSize;
            if (dim < head_dim) {
                const float value_element = scalar_to_float(
                    value[kv_base + static_cast<int64_t>(key_id) * head_dim +
                          dim]);
                output_acc[item] =
                    __fmaf_rn(p, value_element, alpha * output_acc[item]);
            }
        }
        l = alpha * l + p;
        m = m_new;
    }
    const float inv_l = 1.0f / l;
#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
        const int dim = laneId + item * kWarpSize;
        if (dim < head_dim) {
            output[query_base + static_cast<int64_t>(query_id) * head_dim +
                   dim] = float_to_scalar<scalar_t>(output_acc[item] * inv_l);
        }
    }
}
