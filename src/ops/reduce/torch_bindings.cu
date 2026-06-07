#include "../../common/torch/torch_binding_utils.h"
#include "reduce.cuh"
#include <cuda_runtime.h>
#include <torch/version.h>

#if (TORCH_VERSION_MAJOR > 2) ||                                               \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 1)
# define REDUCE_HAS_FP8 1
#else
# define REDUCE_HAS_FP8 0
#endif

#define TORCH_BINDING_REDUCE(packed_type, acc_type, th_type, element_type,     \
                             n_elements, out_type)                             \
    torch::Tensor block_all_reduce_sum_##packed_type##_##acc_type(             \
        torch::Tensor x) {                                                     \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                 \
        auto y_th_type =                                                       \
            (th_type) == torch::kInt8 ? torch::kInt32 : torch::kFloat32;       \
        auto options =                                                         \
            torch::TensorOptions().dtype(y_th_type).device(x.device());        \
        auto y = torch::zeros({1}, options);                                   \
        const int ndim = x.dim();                                              \
        int N = 1;                                                             \
        for (int i = 0; i < ndim; ++i) {                                       \
            N *= x.size(i);                                                    \
        }                                                                      \
        constexpr int NT = 256;                                                \
        dim3 block(NT);                                                        \
        dim3 grid((N + NT * (n_elements)-1) / (NT * (n_elements)));            \
        block_all_reduce_sum_##packed_type##_##acc_type##_kernel<NT>           \
            <<<grid, block>>>(reinterpret_cast<element_type *>(x.data_ptr()),  \
                              reinterpret_cast<out_type *>(y.data_ptr()), N);  \
        return y;                                                              \
    }

// packed_type, acc_type, th_type, element_type, n_elements_per_pack, out_type
TORCH_BINDING_REDUCE(f32, f32, torch::kFloat32, float, 1, float)
TORCH_BINDING_REDUCE(f32x4, f32, torch::kFloat32, float, 4, float)
TORCH_BINDING_REDUCE(f16, f16, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16, f32, torch::kHalf, half, 1, float)
TORCH_BINDING_REDUCE(f16x2, f16, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x2, f32, torch::kHalf, half, 2, float)
TORCH_BINDING_REDUCE(f16x8_pack, f16, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(f16x8_pack, f32, torch::kHalf, half, 8, float)
TORCH_BINDING_REDUCE(bf16, bf16, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16, f32, torch::kBFloat16, __nv_bfloat16, 1, float)
TORCH_BINDING_REDUCE(bf16x2, bf16, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x2, f32, torch::kBFloat16, __nv_bfloat16, 2, float)
TORCH_BINDING_REDUCE(bf16x8_pack, bf16, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
TORCH_BINDING_REDUCE(bf16x8_pack, f32, torch::kBFloat16, __nv_bfloat16, 8,
                     float)
#if REDUCE_HAS_FP8
TORCH_BINDING_REDUCE(fp8_e4m3, f16, torch::kFloat8_e4m3fn, __nv_fp8_storage_t,
                     1, float)
TORCH_BINDING_REDUCE(fp8_e4m3x16_pack, f16, torch::kFloat8_e4m3fn,
                     __nv_fp8_storage_t, 16, float)
TORCH_BINDING_REDUCE(fp8_e5m2, f16, torch::kFloat8_e5m2, __nv_fp8_storage_t, 1,
                     float)
TORCH_BINDING_REDUCE(fp8_e5m2x16_pack, f16, torch::kFloat8_e5m2,
                     __nv_fp8_storage_t, 16, float)
#endif
TORCH_BINDING_REDUCE(i8, i32, torch::kInt8, int8_t, 1, int32_t)
TORCH_BINDING_REDUCE(i8x16_pack, i32, torch::kInt8, int8_t, 16, int32_t)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "block-all-reduce sum CUDA kernels (returns a scalar tensor).";

    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f32x4_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x2_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_f16x8_pack_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x2_f32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_bf16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_bf16x8_pack_f32)
#if REDUCE_HAS_FP8
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e4m3x16_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2_f16)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_fp8_e5m2x16_pack_f16)
#endif
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8_i32)
    TORCH_BINDING_COMMON_EXTENSION(block_all_reduce_sum_i8x16_pack_i32)
}
