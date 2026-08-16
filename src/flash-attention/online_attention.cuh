#include "../common/tensor_utils.h"
#include "online_attention_kernel.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

// naive
template <typename scalar_t, int kMaxDim, bool kCausal>
void launch_online_attention_naive(const torch::Tensor &query,
                                   const torch::Tensor &key,
                                   const torch::Tensor &value,
                                   torch::Tensor &output,
                                   const AttentionShape &shape,
                                   cudaStream_t stream) {
    const dim3 block(kThreadsPerBlock);
    const int query_blocks =
        (shape.q_seqlen + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid(query_blocks, shape.num_heads, shape.batch_size);

    online_attention_naive_kernel<scalar_t, kMaxDim, kCausal>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const scalar_t *>(query.data_ptr()),
            reinterpret_cast<const scalar_t *>(key.data_ptr()),
            reinterpret_cast<const scalar_t *>(value.data_ptr()),
            reinterpret_cast<scalar_t *>(output.data_ptr()), shape.q_seqlen,
            shape.kv_seqlen, shape.head_dim, shape.num_heads);
}

template <typename scalar_t, bool kCausal>
void dispatch_naive_head_dim(const torch::Tensor &query,
                             const torch::Tensor &key,
                             const torch::Tensor &value, torch::Tensor &output,
                             const AttentionShape &shape, cudaStream_t stream) {
    if (shape.head_dim <= 32) {
        launch_online_attention_naive<scalar_t, 32, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_online_attention_naive<scalar_t, 64, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_online_attention_naive<scalar_t, 128, kCausal>(
            query, key, value, output, shape, stream);
    } else {
        launch_online_attention_naive<scalar_t, 256, kCausal>(
            query, key, value, output, shape, stream);
    }
}

torch::Tensor online_attention_naive(torch::Tensor query, torch::Tensor key,
                                     torch::Tensor value, bool causal) {
    const AttentionShape shape =
        check_attention_inputs(query, key, value, kMaxHeadDim);
    const CudaDeviceGuard device_guard(query.get_device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor output = torch::empty_like(query);

    if (query.scalar_type() == torch::kFloat16) {
        if (causal) {
            dispatch_naive_head_dim<half, true>(query, key, value, output,
                                                shape, stream);
        } else {
            dispatch_naive_head_dim<half, false>(query, key, value, output,
                                                 shape, stream);
        }
    } else if (causal) {
        dispatch_naive_head_dim<float, true>(query, key, value, output, shape,
                                             stream);
    } else {
        dispatch_naive_head_dim<float, false>(query, key, value, output, shape,
                                              stream);
    }

    check_cuda(cudaGetLastError(), "flash_attention_naive failed.");
    return output;
}

// kv tiled
template <typename scalar_t, int kMaxDim, bool kCausal>
void launch_online_attention_tiled(const torch::Tensor &query,
                                   const torch::Tensor &key,
                                   const torch::Tensor &value,
                                   torch::Tensor &output,
                                   const AttentionShape &shape,
                                   cudaStream_t stream) {
    const dim3 block(kThreadsPerBlock);
    const int query_blocks = 1 + (shape.q_seqlen - 1) / kWarpsPerBlock;
    const dim3 grid(query_blocks, shape.num_heads, shape.batch_size);
    const size_t shared_bytes =
        2ull * kKvTileSize * shape.head_dim * sizeof(scalar_t);

    online_attention_kv_tiled_kernel<scalar_t, kMaxDim, kCausal>
        <<<grid, block, shared_bytes, stream>>>(
            reinterpret_cast<const scalar_t *>(query.data_ptr()),
            reinterpret_cast<const scalar_t *>(key.data_ptr()),
            reinterpret_cast<const scalar_t *>(value.data_ptr()),
            reinterpret_cast<scalar_t *>(output.data_ptr()), shape.q_seqlen,
            shape.kv_seqlen, shape.head_dim, shape.num_heads);
}

template <typename scalar_t, bool kCausal>
void dispatch_kv_tiled_head_dim(const torch::Tensor &query,
                                const torch::Tensor &key,
                                const torch::Tensor &value,
                                torch::Tensor &output,
                                const AttentionShape &shape,
                                cudaStream_t stream) {
    if (shape.head_dim <= 32) {
        launch_online_attention_tiled<scalar_t, 32, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_online_attention_tiled<scalar_t, 64, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_online_attention_tiled<scalar_t, 128, kCausal>(
            query, key, value, output, shape, stream);
    } else {
        launch_online_attention_tiled<scalar_t, 256, kCausal>(
            query, key, value, output, shape, stream);
    }
}

torch::Tensor online_attention_kv_tiled(torch::Tensor query, torch::Tensor key,
                                        torch::Tensor value, bool causal) {
    const AttentionShape shape =
        check_attention_inputs(query, key, value, kMaxHeadDim);
    const CudaDeviceGuard device_guard(query.get_device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor output = torch::empty_like(query);

    if (query.scalar_type() == torch::kFloat16) {
        if (causal) {
            dispatch_kv_tiled_head_dim<half, true>(query, key, value, output,
                                                   shape, stream);
        } else {
            dispatch_kv_tiled_head_dim<half, false>(query, key, value, output,
                                                    shape, stream);
        }
    } else if (causal) {
        dispatch_kv_tiled_head_dim<float, true>(query, key, value, output,
                                                shape, stream);
    } else {
        dispatch_kv_tiled_head_dim<float, false>(query, key, value, output,
                                                 shape, stream);
    }

    check_cuda(cudaGetLastError(), "flash_attention_kv_tiled failed.");
    return output;
}
