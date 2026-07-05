#include "../common/torch/torch_binding_utils.h"
#include "dot_product.cuh"

#define LAUNCH_DOT_PRODUCT_KERNEL(NT, packed_type, acc_type, element_type)     \
    dot_product_##packed_type##_##acc_type##_kernel<(NT)>                      \
        <<<grid, block>>>(reinterpret_cast<element_type *>(a.data_ptr()),      \
                          reinterpret_cast<element_type *>(b.data_ptr()),      \
                          prod.data_ptr<float>(), N);

#define DISPATCH_DOT_PRODUCT_KERNEL(K, packed_type, acc_type, element_type,    \
                                    n_elements)                                \
    const int NT = (K) / (n_elements);                                         \
    dim3 block(NT);                                                            \
    dim3 grid((S));                                                            \
    switch (NT) {                                                              \
    case 32:                                                                   \
        LAUNCH_DOT_PRODUCT_KERNEL(32, packed_type, acc_type, element_type)     \
        break;                                                                 \
    case 64:                                                                   \
        LAUNCH_DOT_PRODUCT_KERNEL(64, packed_type, acc_type, element_type)     \
        break;                                                                 \
    case 128:                                                                  \
        LAUNCH_DOT_PRODUCT_KERNEL(128, packed_type, acc_type, element_type)    \
        break;                                                                 \
    case 256:                                                                  \
        LAUNCH_DOT_PRODUCT_KERNEL(256, packed_type, acc_type, element_type)    \
        break;                                                                 \
    case 512:                                                                  \
        LAUNCH_DOT_PRODUCT_KERNEL(512, packed_type, acc_type, element_type)    \
        break;                                                                 \
    case 1024:                                                                 \
        LAUNCH_DOT_PRODUCT_KERNEL(1024, packed_type, acc_type, element_type)   \
        break;                                                                 \
    default:                                                                   \
        throw std::runtime_error(                                              \
            "only support (K)/(n_elements): 32/64/128/256/512/1024");          \
        break;                                                                 \
    }

#define TORCH_BINDING_DOT_PRODUCT(packed_type, acc_type, th_type,              \
                                  element_type, n_elements)                    \
    torch::Tensor dot_product_##packed_type##_##acc_type(torch::Tensor a,      \
                                                         torch::Tensor b) {    \
        CHECK_TORCH_TENSOR_DTYPE(a, (th_type))                                 \
        CHECK_TORCH_TENSOR_DTYPE(b, (th_type))                                 \
        auto options = torch::TensorOptions()                                  \
                           .dtype(torch::kFloat32)                             \
                           .device(torch::kCUDA, 0);                           \
        auto prod = torch::zeros({1}, options);                                \
        const int ndim = a.dim();                                              \
        if (ndim != 2) {                                                       \
            int N = 1;                                                         \
            for (int i = 0; i < ndim; ++i) {                                   \
                N *= a.size(i);                                                \
            }                                                                  \
            dim3 block(256);                                                   \
            dim3 grid(((N + 256 * (n_elements)-1) / (256 * (n_elements))));    \
            dot_product_##packed_type##_##acc_type##_kernel<256>               \
                <<<grid, block>>>(                                             \
                    reinterpret_cast<element_type *>(a.data_ptr()),            \
                    reinterpret_cast<element_type *>(b.data_ptr()),            \
                    prod.data_ptr<float>(), N);                                \
        } else {                                                               \
            const int S = a.size(0);                                           \
            const int K = a.size(1);                                           \
            const int N = S * K;                                               \
            if ((K / (n_elements)) <= 1024) {                                  \
                DISPATCH_DOT_PRODUCT_KERNEL(K, packed_type, acc_type,          \
                                            element_type, n_elements)          \
            } else {                                                           \
                int N = 1;                                                     \
                for (int i = 0; i < ndim; ++i) {                               \
                    N *= a.size(i);                                            \
                }                                                              \
                dim3 block(256);                                               \
                dim3 grid(                                                     \
                    ((N + 256 * (n_elements)-1) / (256 * (n_elements))));      \
                dot_product_##packed_type##_##acc_type##_kernel<256>           \
                    <<<grid, block>>>(                                         \
                        reinterpret_cast<element_type *>(a.data_ptr()),        \
                        reinterpret_cast<element_type *>(b.data_ptr()),        \
                        prod.data_ptr<float>(), N);                            \
            }                                                                  \
        }                                                                      \
        return prod;                                                           \
    }

TORCH_BINDING_DOT_PRODUCT(f32, f32, torch::kFloat32, float, 1)
TORCH_BINDING_DOT_PRODUCT(f32x4, f32, torch::kFloat32, float, 4)
TORCH_BINDING_DOT_PRODUCT(f16, f32, torch::kHalf, half, 1)
TORCH_BINDING_DOT_PRODUCT(f16x2, f32, torch::kHalf, half, 2)
TORCH_BINDING_DOT_PRODUCT(f16x8_pack, f32, torch::kHalf, half, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(dot_product_f32_f32)
    TORCH_BINDING_COMMON_EXTENSION(dot_product_f32x4_f32)
    TORCH_BINDING_COMMON_EXTENSION(dot_product_f16_f32)
    TORCH_BINDING_COMMON_EXTENSION(dot_product_f16x2_f32)
    TORCH_BINDING_COMMON_EXTENSION(dot_product_f16x8_pack_f32)
}
