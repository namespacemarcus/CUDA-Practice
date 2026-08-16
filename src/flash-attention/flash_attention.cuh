#pragma once
#include "../common/tensor_utils.h"
#include "flash_attention_v1_kernel.cuh"
#include "flash_attention_v2_kernel.cuh"
#include "utils.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

namespace fa {
constexpr int kMaxHeadDim = 256;
} // namespace fa

// flash attention v1
template <typename scalar_t, int kMaxDim, bool kCausal>
void launch_flash_attention_v1(
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, torch::Tensor &output, torch::Tensor &row_sum,
    torch::Tensor &row_max, const AttentionShape &shape, cudaStream_t stream) {
    const dim3 block(fa1::THREADS);
    const dim3 grid(1, shape.num_heads, shape.batch_size);
    flash_attention_v1_forward_kernel<scalar_t, kMaxDim, kCausal>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const scalar_t *>(query.data_ptr()),
            reinterpret_cast<const scalar_t *>(key.data_ptr()),
            reinterpret_cast<const scalar_t *>(value.data_ptr()),
            reinterpret_cast<scalar_t *>(output.data_ptr()),
            reinterpret_cast<float *>(row_sum.data_ptr()),
            reinterpret_cast<float *>(row_max.data_ptr()), shape.q_seqlen,
            shape.kv_seqlen, shape.head_dim, shape.num_heads);
}

template <typename scalar_t, bool kCausal>
void dispatch_flash_attention_v1_head_dim(
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, torch::Tensor &output, torch::Tensor &row_sum,
    torch::Tensor &row_max, const AttentionShape &shape, cudaStream_t stream) {
    if (shape.head_dim <= 32) {
        launch_flash_attention_v1<scalar_t, 32, kCausal>(
            query, key, value, output, row_sum, row_max, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_flash_attention_v1<scalar_t, 64, kCausal>(
            query, key, value, output, row_sum, row_max, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_flash_attention_v1<scalar_t, 128, kCausal>(
            query, key, value, output, row_sum, row_max, shape, stream);
    } else {
        launch_flash_attention_v1<scalar_t, 256, kCausal>(
            query, key, value, output, row_sum, row_max, shape, stream);
    }
}

torch::Tensor flash_attention_v1_forward(torch::Tensor query, torch::Tensor key,
                                         torch::Tensor value, bool causal) {
    const AttentionShape shape =
        check_attention_inputs(query, key, value, fa::kMaxHeadDim);
    const CudaDeviceGuard device_guard(query.get_device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor output = torch::zeros_like(query);
    const auto state_options = query.options().dtype(torch::kFloat32);
    torch::Tensor row_sum = torch::zeros(
        {shape.batch_size, shape.num_heads, shape.q_seqlen}, state_options);
    torch::Tensor row_max =
        torch::full({shape.batch_size, shape.num_heads, shape.q_seqlen},
                    -std::numeric_limits<float>::infinity(), state_options);
    if (query.scalar_type() == torch::kFloat16) {
        if (causal) {
            dispatch_flash_attention_v1_head_dim<half, true>(
                query, key, value, output, row_sum, row_max, shape, stream);
        } else {
            dispatch_flash_attention_v1_head_dim<half, false>(
                query, key, value, output, row_sum, row_max, shape, stream);
        }
    } else if (causal) {
        dispatch_flash_attention_v1_head_dim<float, true>(
            query, key, value, output, row_sum, row_max, shape, stream);
    } else {
        dispatch_flash_attention_v1_head_dim<float, false>(
            query, key, value, output, row_sum, row_max, shape, stream);
    }
    check_cuda(cudaGetLastError(), "flash_attention_v1 failed.");
    return output;
}

// flash attention v2
template <typename scalar_t, int kMaxDim, bool kCausal>
void launch_flash_attention_v2(const torch::Tensor &query,
                               const torch::Tensor &key,
                               const torch::Tensor &value,
                               torch::Tensor &output, torch::Tensor &logsumexp,
                               const AttentionShape &shape,
                               cudaStream_t stream) {
    const dim3 block(fa2::THREADS);
    const int query_blocks = (shape.q_seqlen + fa2::BQ - 1) / fa2::BQ;
    const dim3 grid(query_blocks, shape.num_heads, shape.batch_size);
    flash_attention_v2_forward_kernel<scalar_t, kMaxDim, kCausal>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const scalar_t *>(query.data_ptr()),
            reinterpret_cast<const scalar_t *>(key.data_ptr()),
            reinterpret_cast<const scalar_t *>(value.data_ptr()),
            reinterpret_cast<scalar_t *>(output.data_ptr()),
            reinterpret_cast<float *>(logsumexp.data_ptr()), shape.q_seqlen,
            shape.kv_seqlen, shape.head_dim, shape.num_heads);
}

template <typename scalar_t, bool kCausal>
void dispatch_flash_attention_v2_head_dim(
    const torch::Tensor &query, const torch::Tensor &key,
    const torch::Tensor &value, torch::Tensor &output, torch::Tensor &logsumexp,
    const AttentionShape &shape, cudaStream_t stream) {
    if (shape.head_dim <= 32) {
        launch_flash_attention_v2<scalar_t, 32, kCausal>(
            query, key, value, output, logsumexp, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_flash_attention_v2<scalar_t, 64, kCausal>(
            query, key, value, output, logsumexp, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_flash_attention_v2<scalar_t, 128, kCausal>(
            query, key, value, output, logsumexp, shape, stream);
    } else {
        launch_flash_attention_v2<scalar_t, 256, kCausal>(
            query, key, value, output, logsumexp, shape, stream);
    }
}

torch::Tensor flash_attention_v2_forward(torch::Tensor query, torch::Tensor key,
                                         torch::Tensor value, bool causal) {
    const AttentionShape shape =
        check_attention_inputs(query, key, value, fa::kMaxHeadDim);
    const CudaDeviceGuard device_guard(query.get_device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor output = torch::empty_like(query);
    torch::Tensor logsumexp =
        torch::empty({shape.batch_size, shape.num_heads, shape.q_seqlen},
                     query.options().dtype(torch::kFloat32));
    if (query.scalar_type() == torch::kFloat16) {
        if (causal) {
            dispatch_flash_attention_v2_head_dim<half, true>(
                query, key, value, output, logsumexp, shape, stream);
        } else {
            dispatch_flash_attention_v2_head_dim<half, false>(
                query, key, value, output, logsumexp, shape, stream);
        }
    } else if (causal) {
        dispatch_flash_attention_v2_head_dim<float, true>(
            query, key, value, output, logsumexp, shape, stream);
    } else {
        dispatch_flash_attention_v2_head_dim<float, false>(
            query, key, value, output, logsumexp, shape, stream);
    }
    check_cuda(cudaGetLastError(), "flash_attention_v2 failed.");
    return output;
}
