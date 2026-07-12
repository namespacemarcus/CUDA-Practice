#pragma once

#include "../common/cuda/cuda_utils.h"
#include "sgemv.cuh"

void sgemv_k32_f32(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(y, M, 1)

    dim3 block(32, 4);
    dim3 grid(1, (M + 4 - 1) / 4);
    sgemv_k32_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(A.data_ptr()),
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, K);
}

void sgemv_k128_f32x4(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    const int M = A.size(0);
    const int K = A.size(1);
    CHECK_TORCH_TENSOR_SHAPE(A, M, K)
    CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
    CHECK_TORCH_TENSOR_SHAPE(y, M, 1)

    dim3 block(32, 4);
    dim3 grid(1, (M + 4 - 1) / 4);
    sgemv_k128_f32x4_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(A.data_ptr()),
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, K);
}

void sgemv_k16_f32(torch::Tensor A, torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(A, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
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
    sgemv_k16_f32_kernel<ROW_PER_WARP>
        <<<grid, block>>>(reinterpret_cast<float *>(A.data_ptr()),
                          reinterpret_cast<float *>(x.data_ptr()),
                          reinterpret_cast<float *>(y.data_ptr()), M, K);
}
