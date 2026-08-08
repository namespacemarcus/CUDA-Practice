#pragma once

#include "../common/tensor_utils.h"
#include "hgemm_kernel.cuh"
#include <ATen/cuda/CUDAContext.h>

constexpr int kThreadsPerBlock = 256;

void hgemm_tiled(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_CUDA_CONTIGUOUS(a);
    CHECK_CUDA_CONTIGUOUS(b);
    CHECK_CUDA_CONTIGUOUS(c);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == torch::kHalf) {
        hgemm_tiled_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__half *>(a.data_ptr()),
            reinterpret_cast<__half *>(b.data_ptr()),
            reinterpret_cast<__half *>(c.data_ptr()), M, N, K);
    } else {
        hgemm_tiled_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
    }
}

void hgemm_gw_tiled(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_CUDA_CONTIGUOUS(a);
    CHECK_CUDA_CONTIGUOUS(b);
    CHECK_CUDA_CONTIGUOUS(c);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == torch::kHalf) {
        hgemm_gw_tiled_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__half *>(a.data_ptr()),
            reinterpret_cast<__half *>(b.data_ptr()),
            reinterpret_cast<__half *>(c.data_ptr()), M, N, K);
    } else {
        hgemm_gw_tiled_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
    }
}

void hgemm_gw_tiled_bcf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_CUDA_CONTIGUOUS(a);
    CHECK_CUDA_CONTIGUOUS(b);
    CHECK_CUDA_CONTIGUOUS(c);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == torch::kHalf) {
        hgemm_gw_tiled_bcf_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__half *>(a.data_ptr()),
            reinterpret_cast<__half *>(b.data_ptr()),
            reinterpret_cast<__half *>(c.data_ptr()), M, N, K);
    } else {
        hgemm_gw_tiled_bcf_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
    }
}

void hgemm_gw_tiled_bcf_dbf(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    CHECK_CUDA_CONTIGUOUS(a);
    CHECK_CUDA_CONTIGUOUS(b);
    CHECK_CUDA_CONTIGUOUS(c);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == torch::kHalf) {
        hgemm_gw_tiled_bcf_dbf_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__half *>(a.data_ptr()),
            reinterpret_cast<__half *>(b.data_ptr()),
            reinterpret_cast<__half *>(c.data_ptr()), M, N, K);
    } else {
        hgemm_gw_tiled_bcf_dbf_kernel<BM, BN, BK><<<grid, block, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
    }
}

void hgemm_gw_tiled_bcf_dbf_cstore(torch::Tensor a, torch::Tensor b,
                                   torch::Tensor c) {
    CHECK_CUDA_CONTIGUOUS(a);
    CHECK_CUDA_CONTIGUOUS(b);
    CHECK_CUDA_CONTIGUOUS(c);

    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const int BM = 128;
    const int BN = 128;
    const int BK = 32;

    const dim3 block(kThreadsPerBlock);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (a.dtype() == torch::kHalf) {
        hgemm_gw_tiled_bcf_dbf_cstore_kernel<BM, BN, BK>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<__half *>(a.data_ptr()),
                reinterpret_cast<__half *>(b.data_ptr()),
                reinterpret_cast<__half *>(c.data_ptr()), M, N, K);
    } else {
        hgemm_gw_tiled_bcf_dbf_cstore_kernel<BM, BN, BK>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(a.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(b.data_ptr()),
                reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()), M, N, K);
    }
}
