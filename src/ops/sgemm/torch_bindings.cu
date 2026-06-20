#include "../../common/torch/torch_binding_utils.h"
#include "sgemm.cuh"

void sgemm_naive_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    constexpr int BM = 32;
    constexpr int BN = 32;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN, BM);

    sgemm_naive_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(a.data_ptr()),
        reinterpret_cast<float *>(b.data_ptr()),
        reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_k_tiled_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN, BM);
    sgemm_k_tiled_f32_kernel<BM, BN, BK>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_thread_tiled_8x8_and_k_tiled_f32x4(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    sgemm_thread_tiled_8x8_and_k_tiled_f32x4_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf(torch::Tensor a,
                                                  torch::Tensor b,
                                                  torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

void sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_offset(torch::Tensor a,
                                                         torch::Tensor b,
                                                         torch::Tensor c) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    CHECK_TORCH_TENSOR_SHAPE(a, M, K)
    CHECK_TORCH_TENSOR_SHAPE(b, K, N)
    CHECK_TORCH_TENSOR_SHAPE(c, M, N)

    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int OFFSET = 4;

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);
    sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_kernel<BM, BN, BK, TM, TN,
                                                        OFFSET>
        <<<grid, block>>>(reinterpret_cast<float *>(a.data_ptr()),
                          reinterpret_cast<float *>(b.data_ptr()),
                          reinterpret_cast<float *>(c.data_ptr()), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_k_tiled_f32)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_thread_tiled_8x8_and_k_tiled_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf)
    TORCH_BINDING_COMMON_EXTENSION(
        sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_offset)
}
