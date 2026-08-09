#include "../common/tensor_utils.h"
#include "matrix_transpose_loadcoal_kernel.cuh"

constexpr int kThreadsPerBlock = 256;

void matrix_transpose_f32_loadcoal(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(kThreadsPerBlock);
    dim3 grid((M * N + kThreadsPerBlock - 1) / kThreadsPerBlock);
    matrix_transpose_f32_loadcoal_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}

void matrix_transpose_f32x4_loadcoal(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(kThreadsPerBlock);
    dim3 grid((M * N + kThreadsPerBlock - 1) / kThreadsPerBlock / 4);
    matrix_transpose_f32x4_loadcoal_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}

void matrix_transpose_f32x4_loadcoal_smem(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(kThreadsPerBlock);
    dim3 grid(((N + kTileSize - 1) / kTileSize),
              ((M + kTileSize - 1) / kTileSize));
    matrix_transpose_f32x4_loadcoal_smem_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}

void matrix_transpose_f32x4_loadcoal_smem_bcf(torch::Tensor x,
                                              torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(kThreadsPerBlock);
    dim3 grid(((N + kTileSize - 1) / kTileSize),
              ((M + kTileSize - 1) / kTileSize));
    matrix_transpose_f32x4_loadcoal_smem_bcf_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}

// 2d block
constexpr int ThreadsPerDim = 16;

void matrix_transpose_f32_loadcoal_2d(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(ThreadsPerDim, ThreadsPerDim);
    dim3 grid((N + ThreadsPerDim - 1) / ThreadsPerDim,
              (M + ThreadsPerDim - 1) / ThreadsPerDim);
    matrix_transpose_f32_loadcoal_2d_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}

void matrix_transpose_f32x4_loadcoal_2d(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32);
    const int M = x.size(0);
    const int N = x.size(1);

    dim3 block(kTileSize, ThreadsPerDim);
    dim3 grid((N + ThreadsPerDim - 1) / ThreadsPerDim / 4,
              (M + ThreadsPerDim - 1) / ThreadsPerDim);
    matrix_transpose_f32x4_loadcoal_2d_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), M, N);
}
