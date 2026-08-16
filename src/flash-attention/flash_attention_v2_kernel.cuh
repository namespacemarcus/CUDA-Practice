#pragma once

#include "../common/defs.h"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace fa2 {
constexpr int NUM_WARPS = 8;
constexpr int THREADS = WARP_SIZE * NUM_WARPS;
constexpr int BQ = NUM_WARPS;
constexpr int BK = 16;
} // namespace fa2

// 一个 block 256线程算一个 BQ，一个 warp 算一行 query

template <typename scalar_t, int kMaxDim, bool kCausal>
__global__
__launch_bounds__(fa2::THREADS) void flash_attention_v2_forward_kernel(
    const scalar_t *__restrict__ query, const scalar_t *__restrict__ key,
    const scalar_t *__restrict__ value, scalar_t *__restrict__ output,
    float *__restrict__ logsumexp, int q_seqlen, int kv_seqlen, int head_dim,
    int num_heads) {
    constexpr int kValuesPerLane = (kMaxDim + WARP_SIZE - 1) / WARP_SIZE;

    __shared__ scalar_t shared_key[fa2::BK * kMaxDim];
    __shared__ scalar_t shared_value[fa2::BK * kMaxDim];

    const int thread_id = threadIdx.x;
    const int warpId = thread_id / WARP_SIZE;
    const int laneId = thread_id % WARP_SIZE;

    const int query_start = blockIdx.x * fa2::BQ;
    const int query_id = query_start + warpId;
    const int64_t batch_head_id =
        static_cast<int64_t>(blockIdx.z) * num_heads + blockIdx.y;
    const int64_t query_base =
        batch_head_id * static_cast<int64_t>(q_seqlen) * head_dim;
    const int64_t kv_base =
        batch_head_id * static_cast<int64_t>(kv_seqlen) * head_dim;
    const int64_t lse_base = batch_head_id * static_cast<int64_t>(q_seqlen);

    float query_fragment[kValuesPerLane];
    float output_acc[kValuesPerLane];
#pragma unroll
    for (int item = 0; item < kValuesPerLane; ++item) {
        const int dim = laneId + item * WARP_SIZE;
        query_fragment[item] =
            query_id < q_seqlen && dim < head_dim
                ? scalar_to_float(
                      query[query_base +
                            static_cast<int64_t>(query_id) * head_dim + dim])
                : 0.0f;
        output_acc[item] = 0.0f;
    }

    float row_max = -CUDART_INF_F;
    float row_sum = 0.0f;
    const float softmax_scale = rsqrtf(static_cast<float>(head_dim));

    const int query_end = min(q_seqlen, query_start + fa2::BQ);
    const int block_kv_end = kCausal ? min(kv_seqlen, query_end) : kv_seqlen;

    for (int tile_start = 0; tile_start < block_kv_end; tile_start += fa2::BK) {
        const int rows_in_tile = min(fa2::BK, block_kv_end - tile_start);
        const int tile_elements = rows_in_tile * head_dim;

        for (int task = thread_id; task < 2 * tile_elements;
             task += fa2::THREADS) {
            const bool loading_value = task >= tile_elements;
            const int element = loading_value ? task - tile_elements : task;
            const int tile_row = element / head_dim;
            const int dim = element % head_dim;
            const int64_t global_offset =
                kv_base +
                static_cast<int64_t>(tile_start + tile_row) * head_dim + dim;
            if (loading_value) {
                shared_value[tile_row * kMaxDim + dim] = value[global_offset];
            } else {
                shared_key[tile_row * kMaxDim + dim] = key[global_offset];
            }
        }
        __syncthreads();

        if (query_id < q_seqlen) {
            int valid_keys = rows_in_tile;
            if constexpr (kCausal) {
                valid_keys =
                    max(0, min(rows_in_tile, query_id - tile_start + 1));
            }

            float lane_score = -CUDART_INF_F;
            for (int tile_row = 0; tile_row < valid_keys; ++tile_row) {
                float partial_score = 0.0f;
#pragma unroll
                for (int item = 0; item < kValuesPerLane; ++item) {
                    const int dim = laneId + item * WARP_SIZE;
                    if (dim < head_dim) {
                        partial_score =
                            __fmaf_rn(query_fragment[item],
                                      scalar_to_float(
                                          shared_key[tile_row * kMaxDim + dim]),
                                      partial_score);
                    }
                }
                const float score =
                    warp_reduce_sum(partial_score) * softmax_scale;
                if (laneId ==
                    tile_row) { // 每个lane寄存器里，保存下标等于自己的laneId的那个key
                                // - score
                    lane_score = score;
                }
            }

            // 当前q对当前BK=16个key的score的最大值
            const float tile_max = warp_reduce_max(lane_score);
            const float new_row_max = fmaxf(row_max, tile_max);
            const float alpha =
                isfinite(row_max) ? expf(row_max - new_row_max) : 0.0f;
            const float p =
                laneId < valid_keys ? expf(lane_score - new_row_max) : 0.0f;
            const float tile_sum = warp_reduce_sum(p);

#pragma unroll
            for (int item = 0; item < kValuesPerLane; ++item) {
                const int dim = laneId + item * WARP_SIZE;
                float tile_output = 0.0f;
                for (int tile_row = 0; tile_row < valid_keys; ++tile_row) {
                    const float tile_p = __shfl_sync(
                        0xffffffff, p,
                        tile_row); // 每个线程都拿到了q对当前k未归一化的权重，分别去算对应v向量的分量
                    if (dim < head_dim) {
                        tile_output = __fmaf_rn(
                            tile_p,
                            scalar_to_float(
                                shared_value[tile_row * kMaxDim + dim]),
                            tile_output);
                    }
                }
                if (dim < head_dim) {
                    output_acc[item] =
                        __fmaf_rn(alpha, output_acc[item], tile_output);
                }
            }
            row_sum = __fmaf_rn(alpha, row_sum, tile_sum);
            row_max = new_row_max;
        }
        __syncthreads();
    }

    if (query_id < q_seqlen) {
        if (laneId == 0) {
            logsumexp[lse_base + query_id] = row_max + logf(row_sum);
        }
        const float inverse_row_sum = 1.0f / row_sum;
#pragma unroll
        for (int item = 0; item < kValuesPerLane; ++item) {
            const int dim = laneId + item * WARP_SIZE;
            if (dim < head_dim) {
                output[query_base + static_cast<int64_t>(query_id) * head_dim +
                       dim] = float_to_scalar<scalar_t>(output_acc[item] *
                                                        inverse_row_sum);
            }
        }
    }
}

// todo: backward kernel
// template <typename scalar_t, int kMaxDim, bool kCausal>
// __global__
// __launch_bounds__(fa2::THREADS) void flash_attention_v2_backward_kernel(
//     const scalar_t *__restrict__ query, const scalar_t *__restrict__ key,
//     const scalar_t *__restrict__ value,
//     const scalar_t *__restrict__ output,          // [B,H,Nq,D]
//     const scalar_t *__restrict__ output_gradient, // [B,H,Nq,D]
//     const float *__restrict__ logsumexp,          // [B,H,Nq]
//     float *__restrict__ query_gradient,           // [B,H,Nq,D]
//     float *__restrict__ key_gradient,             // [B,H,Nk,D]
//     float *__restrict__ value_gradient,           // [B,H,Nk,D]
//     int q_seqlen, int kv_seqlen, int head_dim, int num_heads) {
//     constexpr int kValuesPerLane = (kMaxDim + WARP_SIZE - 1) / WARP_SIZE;

//     __shared__ scalar_t shared_key[fa2::BK * kMaxDim];
//     __shared__ scalar_t shared_value[fa2::BK * kMaxDim];

//     const int thread_id = threadIdx.x;
//     const int warpId = thread_id / WARP_SIZE;
//     const int laneId = thread_id % WARP_SIZE;

//     const int query_start = blockIdx.x * fa2::BQ;
//     const int query_id = query_start + warpId;
//     const int64_t batch_head_id =
//         static_cast<int64_t>(blockIdx.z) * num_heads + blockIdx.y;
//     const int64_t query_base =
//         batch_head_id * static_cast<int64_t>(q_seqlen) * head_dim;
//     const int64_t kv_base =
//         batch_head_id * static_cast<int64_t>(kv_seqlen) * head_dim;
//     const int64_t lse_base = batch_head_id * static_cast<int64_t>(q_seqlen);

//     float query_fragment[kValuesPerLane];
//     float gradient_fragment[kValuesPerLane];
// }
