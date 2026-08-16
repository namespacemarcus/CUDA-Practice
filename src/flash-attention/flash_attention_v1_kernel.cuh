#pragma once

#include "utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace fa1 {
constexpr int THREADS = 256;
constexpr int BQ = THREADS;
constexpr int BK = 16;
} // namespace fa1

/*
Q/O: [B,H,Nq,D]
K/V: [B,H,Nk,D]
m/l: [B,H,Nq]
*/

/*
for kv tile(16):
    load k/v tile to shared memory
    for q tile(256):
        read o/m/l from HBM, update online softmax, write o/m/l to HBM
*/
template <typename scalar_t, int kMaxDim, bool kCausal>
__global__
__launch_bounds__(fa1::THREADS) void flash_attention_v1_forward_kernel(
    const scalar_t *__restrict__ query, const scalar_t *__restrict__ key,
    const scalar_t *__restrict__ value, scalar_t *__restrict__ output,
    float *__restrict__ row_sum, float *__restrict__ row_max, int q_seqlen,
    int kv_seqlen, int head_dim, int num_heads) {
    __shared__ scalar_t shared_key[fa1::BK * kMaxDim];
    __shared__ scalar_t shared_value[fa1::BK * kMaxDim];
    __shared__ float shared_score[fa1::BQ * fa1::BK];

    const int thread_id = threadIdx.x;
    const int64_t batch_head_id =
        static_cast<int64_t>(blockIdx.z) * num_heads + blockIdx.y;
    const int64_t query_base =
        batch_head_id * static_cast<int64_t>(q_seqlen) * head_dim;
    const int64_t kv_base =
        batch_head_id * static_cast<int64_t>(kv_seqlen) * head_dim;
    const int64_t state_base = batch_head_id * static_cast<int64_t>(q_seqlen);
    const float scale = rsqrtf(static_cast<float>(head_dim));

    for (int query_id = thread_id; query_id < q_seqlen;
         query_id += fa1::THREADS) {
        row_sum[state_base + query_id] = 0.0f;
        row_max[state_base + query_id] = -CUDART_INF_F;
    }
    for (int task = thread_id; task < q_seqlen * head_dim;
         task += fa1::THREADS) {
        // 当前(batch,head)的输出形状是(Nq,D),共q_seqlen*head_dim个元素
        output[query_base + task] = float_to_scalar<scalar_t>(0.0f);
    }
    __syncthreads();

    for (int key_start = 0; key_start < kv_seqlen; key_start += fa1::BK) {
        const int key_rows = min(fa1::BK, kv_seqlen - key_start);

        // 256线程共同加载K和V. 线性空间前半段对应K,后半段对应V
        for (int task = thread_id; task < 2 * fa1::BK * kMaxDim;
             task += fa1::THREADS) {
            const bool load_value = task >= f1::BK * kMaxDim;
            const int index = load_value ? task - fa1::BK * kMaxDim : task;
            const int key_row = index / kMaxDim;
            const int dim = index % kMaxDim;
            scalar_t loaded = float_to_scalar<scalar_t>(0.0f);
            if (key_row < key_rows && dim < head_dim) {
                const int64_t offset =
                    kv_base +
                    static_cast<int64_t>(key_start + key_row) * head_dim + dim;
                loaded = load_value ? value[offset] : key[offset];
            }
            if (load_value) {
                shared_value[index] = loaded;
            } else {
                shared_key[index] = loaded;
            }
        }
        __syncthreads();

        for (int query_start = 0; query_start < q_seqlen;
             query_start += fa1::BQ) {
            if constexpr (kCausal) {
                if (query_start + fa1::BQ <= key_start) {
                    continue;
                }
            }

            const int query_id = query_start + thread_id;
            if (query_id < q_seqlen) {
                /*
                计算 S_tile=Q_tile @ K_tile * scale
                    Q_tile: [Br, D]
                    K_tile: [D, key_rows]
                    S_tile: [Br, key_rows]

                当前线程只算一行q：q_row[D] @ K_tile[D, key_rows] ->
                score_row[key_rows]
                */
                float tile_max =
                    -CUDART_INF_F; // 当前这一行q对 k
                                   // tile逐个算分，tile_max是其中分的最大值
                for (int key_row = 0; key_row < key_rows; ++key_row) {
                    const int key_id = key_start + key_row;
                    float score = -CUDART_INF_F;
                    if (!kCausal || key_id <= query_id) {
                        score = 0.0f;
                        for (int dim = 0; dim < head_dim; ++dim) {
                            score = __fmaf_rn(
                                scalar_to_float(
                                    query[query_base +
                                          static_cast<int64_t>(query_id) *
                                              head_dim +
                                          dim]),
                                scalar_to_float(
                                    shared_key[key_row * kMaxDim + dim]),
                                score);
                        }
                        score *= scale;
                        tile_max = fmaxf(tile_max, score);
                    }
                    shared_score[thread_id * fa1::BK + key_row] = score;
                }

                if (isfinite(tile_max)) {
                    float tile_sum = 0.0f;
                    for (int key_row = 0; key_row < key_rows; ++key_row) {
                        const int score_offset = thread_id * fa1::BK + key_row;
                        const float score = shared_score[score_offset];
                        const float probability =
                            isfinite(score) ? expf(score - tile_max) : 0.0f;
                        shared_score[score_offset] = probability;
                        tile_sum += probability;
                    }

                    const int64_t state_offset = state_base + query_id;
                    const float old_max = row_max[state_offset];
                    const float old_sum = row_sum[state_offset];
                    const float new_max = fmaxf(old_max, tile_max);

                    const float old_weight =
                        old_sum > 0.0f ? expf(old_max - new_max) * old_sum
                                       : 0.0f;
                    const float tile_weight = expf(tile_max - new_max);
                    const float new_sum =
                        old_weight +
                        tile_weight *
                            tile_sum; // tile_sum是在上面基于tile_max算的，这里更新为new_max后也需要更新

                    for (int dim = 0; dim < head_dim; ++dim) {
                        float tile_output = 0.0f;
                        for (int key_row = 0; key_row < key_rows; ++key_row) {
                            tile_output = __fmaf_rn(
                                shared_score[thread_id * fa1::BK + key_row],
                                scalar_to_float(
                                    shared_value[key_row * kMaxDim + dim]),
                                tile_output);
                        }
                        const int64_t output_offset =
                            query_base +
                            static_cast<int64_t>(query_id) * head_dim + dim;
                        const float old_output =
                            scalar_to_float(output[output_offset]);
                        output[output_offset] = float_to_scalar<scalar_t>(
                            (old_weight * old_output +
                             tile_weight * tile_output) /
                            new_sum);
                    }
                    row_sum[state_offset] = new_sum;
                    row_max[state_offset] = new_max;
                }
            }
        }
        __syncthreads();
    }
}

// todo: backward
