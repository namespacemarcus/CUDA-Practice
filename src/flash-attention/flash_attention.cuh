#include "flash_attention_kernel.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

struct AttentionShape {
    int batch_size;
    int num_heads;
    int q_seqlen;
    int kv_seqlen;
    int head_dim;
};

AttentionShape check_attention_inputs(const torch::Tensor &query,
                                      const torch::Tensor &key,
                                      const torch::Tensor &value) {
    TORCH_CHECK(query.is_cuda() && key.is_cuda() && value.is_cuda(),
                "Q, K, V must all be CUDA tensors");
    TORCH_CHECK(query.device() == key.device() &&
                    query.device() == value.device(),
                "Q, K, V must be on the same CUDA device");
    TORCH_CHECK(
        query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "Q, K, V must be 4-dimensional tensors with layout [B, H, N, D]");
    TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                    value.is_contiguous(),
                "Q, K, V must be contiguous");
    TORCH_CHECK(query.scalar_type() == key.scalar_type() &&
                    query.scalar_type() == value.scalar_type(),
                "Q, K, V must have the same scalar type");
    TORCH_CHECK(query.scalar_type() == torch::kFloat16 ||
                    query.scalar_type() == torch::kFloat32,
                "Only torch.float16 and torch.float32 dtypes are supported");
    TORCH_CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0),
                "Q, K, V must have the same batch size");
    TORCH_CHECK(query.size(1) == key.size(1) && query.size(1) == value.size(1),
                "Q, K, V must have the same number of heads (GQA/MQA is not "
                "supported)");
    TORCH_CHECK(key.size(2) == value.size(2),
                "K and V must have the same sequence length");
    TORCH_CHECK(query.size(3) == key.size(3) && query.size(3) == value.size(3),
                "Q, K, V must have the same head dimension");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && query.size(2) > 0 &&
                    key.size(2) > 0,
                "batch, head, query sequence length and kv sequence length "
                "must all be greater than 0");
    TORCH_CHECK(query.size(3) > 0 && query.size(3) <= kMaxHeadDim,
                "head_dim must be in range [1, ", kMaxHeadDim, "], but got ",
                query.size(3));
    TORCH_CHECK(
        query.size(0) <= 65535 && query.size(1) <= 65535,
        "batch size and num_heads must not exceed CUDA grid limit 65535");
    TORCH_CHECK(query.size(2) <= std::numeric_limits<int>::max() &&
                    key.size(2) <= std::numeric_limits<int>::max() &&
                    query.size(3) <= std::numeric_limits<int>::max(),
                "sequence length or head_dim exceeds the maximum limit of int "
                "index type");

    return {static_cast<int>(query.size(0)), static_cast<int>(query.size(1)),
            static_cast<int>(query.size(2)), static_cast<int>(key.size(2)),
            static_cast<int>(query.size(3))};
}

// naive
template <typename scalar_t, int kMaxDim, bool kCausal>
void launch_flash_attention_naive(const torch::Tensor &query,
                                  const torch::Tensor &key,
                                  const torch::Tensor &value,
                                  torch::Tensor &output,
                                  const AttentionShape &shape,
                                  cudaStream_t stream) {
    const dim3 block(kThreadsPerBlock);
    const int query_blocks =
        (shape.q_seqlen + kWarpsPerBlock - 1) / kWarpsPerBlock;
    const dim3 grid(query_blocks, shape.num_heads, shape.batch_size);

    flash_attention_naive_kernel<scalar_t, kMaxDim, kCausal>
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
        launch_flash_attention_naive<scalar_t, 32, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_flash_attention_naive<scalar_t, 64, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_flash_attention_naive<scalar_t, 128, kCausal>(
            query, key, value, output, shape, stream);
    } else {
        launch_flash_attention_naive<scalar_t, 256, kCausal>(
            query, key, value, output, shape, stream);
    }
}

torch::Tensor flash_attention_naive(torch::Tensor query, torch::Tensor key,
                                    torch::Tensor value, bool causal) {
    const AttentionShape shape = check_attention_inputs(query, key, value);
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
void launch_flash_attention_tiled(const torch::Tensor &query,
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

    flash_attention_kv_tiled_kernel<scalar_t, kMaxDim, kCausal>
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
        launch_flash_attention_tiled<scalar_t, 32, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 64) {
        launch_flash_attention_tiled<scalar_t, 64, kCausal>(
            query, key, value, output, shape, stream);
    } else if (shape.head_dim <= 128) {
        launch_flash_attention_tiled<scalar_t, 128, kCausal>(
            query, key, value, output, shape, stream);
    } else {
        launch_flash_attention_tiled<scalar_t, 256, kCausal>(
            query, key, value, output, shape, stream);
    }
}

torch::Tensor flash_attention_kv_tiled(torch::Tensor query, torch::Tensor key,
                                       torch::Tensor value, bool causal) {
    const AttentionShape shape = check_attention_inputs(query, key, value);
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
