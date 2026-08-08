#pragma once

#include "../common/defs.h"
#include "../common/tensor_utils.h"
#include "hgemv_kernel.cuh"

void hgemv_k32_f16(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
    if (K % 32 != 0) {
        throw std::runtime_error("K must be multiple of 32.");
    }

    dim3 block(32, 4);
    dim3 grid(1, (M + 4 - 1) / 4);
    hgemv_k32_f16_kernel<<<grid, block>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(x.data_ptr()),
        reinterpret_cast<half *>(y.data_ptr()), M, K);
}

void hgemv_k128_f16x4(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
    if (K % 128 != 0) {
        throw std::runtime_error("K must be multiple of 128.");
    }

    dim3 block(32, 4);
    dim3 grid(1, (M + 4 - 1) / 4);
    hgemv_k128_f16x4_kernel<<<grid, block>>>(
        reinterpret_cast<half *>(A.data_ptr()),
        reinterpret_cast<half *>(x.data_ptr()),
        reinterpret_cast<half *>(y.data_ptr()), M, K);
}

void hgemv_k16_f16(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
    if (K != 16) {
        throw std::runtime_error("K must be 16.");
    }

    constexpr int NUM_THREADS = 128;
    constexpr int ROW_PER_WARP = 2;
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP;

    dim3 block(32, NUM_WARPS);
    dim3 grid(1, (M + NUM_ROWS - 1) / NUM_ROWS);
    hgemv_k16_f16_kernel<ROW_PER_WARP>
        <<<grid, block>>>(reinterpret_cast<half *>(A.data_ptr()),
                          reinterpret_cast<half *>(x.data_ptr()),
                          reinterpret_cast<half *>(y.data_ptr()), M, K);
}
