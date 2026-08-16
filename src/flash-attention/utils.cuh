#include "../common/defs.h"
#include "../common/tensor_utils.h"
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value) {
    return value;
}

template <>
__forceinline__ float scalar_to_float<half>(half value) {
    return __half2float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
    return __float2half(value);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, stride);
    }
    return value;
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, stride));
    }
    return val;
}

void check_cuda(cudaError_t error, const char *operation) {
    TORCH_CHECK(error == cudaSuccess, operation, ": ",
                cudaGetErrorString(error));
}

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device)
        : previous_device_(0),
          device_changed_(false) {
        check_cuda(cudaGetDevice(&previous_device_),
                   "Failed to get current CUDA device");
        if (device != previous_device_) {
            check_cuda(cudaSetDevice(device), "Failed to set CUDA device");
            device_changed_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (device_changed_) {
            check_cuda(cudaSetDevice(previous_device_),
                       "Failed to restore previous CUDA device");
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
    int previous_device_;
    bool device_changed_;
};

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
    CHECK_CUDA(query);
    CHECK_CUDA(key);
    CHECK_CUDA(value);
    TORCH_CHECK(query.device() == key.device() &&
                    query.device() == value.device(),
                "Q, K, V must be on the same CUDA device");
    TORCH_CHECK(
        query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
        "Q, K, V must be 4-dimensional tensors with layout [B, H, N, D]");
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(value);
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

void check_attention_backward_inputs(const torch::Tensor &query,
                                     const torch::Tensor &output,
                                     const torch::Tensor &output_gradient,
                                     const torch::Tensor &state,
                                     const AttentionShape &shape,
                                     const char *state_name) {
    CHECK_CUDA_CONTIGUOUS(output);
    CHECK_CUDA_CONTIGUOUS(output_gradient);
    CHECK_CUDA_CONTIGUOUS(state);
    TORCH_CHECK(output.device() == query.device() &&
                    output_gradient.device() == query.device() &&
                    state.device() == query.device(),
                "output, output_gradient and state must be on the Q device");
    TORCH_CHECK(output.sizes() == query.sizes() &&
                    output_gradient.sizes() == query.sizes(),
                "output and output_gradient must have the same shape as Q");
    TORCH_CHECK(output.scalar_type() == query.scalar_type() &&
                    output_gradient.scalar_type() == query.scalar_type(),
                "output and output_gradient must have the same dtype as Q");
    TORCH_CHECK(state.scalar_type() == torch::kFloat32 && state.dim() == 3 &&
                    state.size(0) == shape.batch_size &&
                    state.size(1) == shape.num_heads &&
                    state.size(2) == shape.q_seqlen,
                state_name, " must be float32 with shape [B, H, Nq]");
}
