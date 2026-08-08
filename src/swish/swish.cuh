#pragma once

#include "../common/tensor_utils.h"
#include "swish_kernel.cuh"

#define TORCH_BINDING_SWISH(packed_type, th_type, element_type, n_elements)    \
    void swish_##packed_type(torch::Tensor x, torch::Tensor y) {               \
        CHECK_TORCH_TENSOR_DTYPE(x, (th_type))                                 \
        CHECK_TORCH_TENSOR_DTYPE(y, (th_type))                                 \
        const int ndim = x.dim();                                              \
        if (ndim != 2) {                                                       \
            int N = 1;                                                         \
            for (int i = 0; i < ndim; ++i) {                                   \
                N *= x.size(i);                                                \
            }                                                                  \
            dim3 block(256 / (n_elements));                                    \
            dim3 grid((N + 256 - 1) / 256);                                    \
            swish_##packed_type##_kernel<<<grid, block>>>(                     \
                reinterpret_cast<element_type *>(x.data_ptr()),                \
                reinterpret_cast<element_type *>(y.data_ptr()), N);            \
        } else {                                                               \
            const int S = x.size(0);                                           \
            const int D = x.size(1);                                           \
            const int N = S * D;                                               \
            if ((D / (n_elements)) <= 1024) {                                  \
                dim3 block(D / (n_elements));                                  \
                dim3 grid(S);                                                  \
                swish_##packed_type##_kernel<<<grid, block>>>(                 \
                    reinterpret_cast<element_type *>(x.data_ptr()),            \
                    reinterpret_cast<element_type *>(y.data_ptr()), N);        \
            } else {                                                           \
                int N = 1;                                                     \
                for (int i = 0; i < ndim; ++i) {                               \
                    N *= x.size(i);                                            \
                }                                                              \
                dim3 block(256 / (n_elements));                                \
                dim3 grid((N + 256 - 1) / 256);                                \
                swish_##packed_type##_kernel<<<grid, block>>>(                 \
                    reinterpret_cast<element_type *>(x.data_ptr()),            \
                    reinterpret_cast<element_type *>(y.data_ptr()), N);        \
            }                                                                  \
        }                                                                      \
    }

TORCH_BINDING_SWISH(f32, torch::kFloat32, float, 1)
TORCH_BINDING_SWISH(f32x4, torch::kFloat32, float, 4)
TORCH_BINDING_SWISH(f16, torch::kHalf, half, 1)
TORCH_BINDING_SWISH(f16x2, torch::kHalf, half, 2)
TORCH_BINDING_SWISH(f16x8, torch::kHalf, half, 8)
TORCH_BINDING_SWISH(f16x8_pack, torch::kHalf, half, 8)
