#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/types.h>

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info: " << (T).options() << std::endl;            \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
    if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                      \
        throw std::runtime_error("Tensor size mismatch!");                     \
    }

#define CHECK_TORCH_TENSOR_SAME_SHAPE(T1, T2)                                  \
    assert((T1).dim() == (T2).dim());                                          \
    for (int i = 0; i < (T1).dim(); ++i) {                                     \
        if ((T2).size(i) != (T1).size(i)) {                                    \
            throw std::runtime_error("Tensor size mismatch!");                 \
        }                                                                      \
    }

#define CHECK_CUDA(T)                                                          \
    TORCH_CHECK((T).is_cuda(), #T " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(T)                                                    \
    TORCH_CHECK((T).is_contiguous(), #T " must be contiguous")

#define CHECK_CUDA_CONTIGUOUS(T)                                               \
    TORCH_CHECK((T).is_cuda() && (T).is_contiguous(),                          \
                #T " must be a contiguous CUDA tensor")
