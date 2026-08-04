#pragma once

#include "../common/cuda/cuda_utils.h"
#include "sgemm_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>

void sgemm_naive(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    const dim3 block(16, 16);
    const dim3 grid((N + 16 - 1) / 16, (M + 16 - 1) / 16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_naive_kernel<<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
}

constexpr int kThreadsPerBlock = 256;

void sgemm_tiling(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    TORCH_CHECK(M % BM == 0, "M must be divisible by 128.");
    TORCH_CHECK(N % BN == 0, "N must be divisible by 128.");
    TORCH_CHECK(K % BK == 0, "K must be divisible by 128.");

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_tiling_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
}

void sgemm_at_tiling(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    TORCH_CHECK(M % BM == 0, "M must be divisible by 128.");
    TORCH_CHECK(N % BN == 0, "N must be divisible by 128.");
    TORCH_CHECK(K % BK == 0, "K must be divisible by 128.");

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_at_tiling_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
}

void sgemm_at_tiling_bcf_swizzling(torch::Tensor a, torch::Tensor b,
                                   torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    TORCH_CHECK(M % BM == 0, "M must be divisible by 128.");
    TORCH_CHECK(N % BN == 0, "N must be divisible by 128.");
    TORCH_CHECK(K % BK == 0, "K must be divisible by 128.");

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_at_tiling_bcf_swizzling_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                     c.data_ptr<float>(), M, N, K);
}

void sgemm_at_tiling_bcf_swizzling_cstore(torch::Tensor a, torch::Tensor b,
                                          torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    TORCH_CHECK(M % BM == 0, "M must be divisible by 128.");
    TORCH_CHECK(N % BN == 0, "N must be divisible by 128.");
    TORCH_CHECK(K % BK == 0, "K must be divisible by 128.");

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_at_tiling_bcf_swizzling_cstore_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                     c.data_ptr<float>(), M, N, K);
}

void sgemm_at_tiling_bcf_swizzling_cstore_dbf(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(),
                "a must be contiguous CUDA tensor.");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(),
                "b must be contiguous CUDA tensor.");
    TORCH_CHECK(c.is_cuda() && c.is_contiguous(),
                "c must be contiguous CUDA tensor.");

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 16;
    const int TM = 8;
    const int TN = 8;

    TORCH_CHECK(M % BM == 0, "M must be divisible by 128.");
    TORCH_CHECK(N % BN == 0, "N must be divisible by 128.");
    TORCH_CHECK(K % BK == 0, "K must be divisible by 128.");

    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    const dim3 block(kThreadsPerBlock);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    sgemm_at_tiling_bcf_swizzling_cstore_dbf_kernel<BM, BN, BK, TM, TN>
        <<<grid, block, 0, stream>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                     c.data_ptr<float>(), M, N, K);
}
