#pragma once

#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
    if (((T).options().dtype() != (th_type))) {                                \
        std::cout << "Tensor Info: " << (T).options() << std::endl;            \
        throw std::runtime_error("values must be " #th_type);                  \
    }

#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
    m.def(STRINGFY(func), &func, STRINGFY(func));

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
