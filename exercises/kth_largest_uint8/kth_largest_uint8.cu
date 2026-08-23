#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int kBlockThreads = 256;
constexpr int kWarpsPerBlock = kBlockThreads / 32; // =8

__global__ void histogramKernel(const uint8_t *data, size_t n,
                                unsigned long long *hist) {
    __shared__ unsigned int warpHist[kWarpsPerBlock][256];

    int tid = threadIdx.x;
    int warpId = tid / 32;

    for (int i = tid; i < kWarpsPerBlock * 256; i += kBlockThreads) {
        reinterpret_cast<unsigned int *>(warpHist)[i] = 0U;
    }
    __syncthreads();

    size_t gid = static_cast<size_t>(blockIdx.x) * kBlockThreads + tid;
    if (gid < n) {
        uint8_t val = data[gid];
        atomicAdd(&warpHist[warpId][val], 1U);
    }
    __syncthreads();

    int bin = tid;
    unsigned int cnt = 0U;
    if (bin < 256) {
#pragma unroll
        for (int w = 0; w < kWarpsPerBlock; ++w) {
            cnt += warpHist[w][bin];
        }
        if (cnt > 0) {
            atomicAdd(&hist[bin], static_cast<unsigned long long>(cnt));
        }
    }
}

__global__ void findKthLargestKernel(const unsigned long long *hist,
                                     unsigned long long k, int *res) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    unsigned long long acc = 0ULL;
    for (int v = 255; v >= 0; --v) {
        acc += hist[v];
        if (acc >= k) {
            *res = v;
            return;
        }
    }
    *res = -1;
}

int kthLargestUint8(const uint8_t *d_data, size_t n, unsigned long long k) {
    if (d_data == nullptr || n == 0 || k == 0 || k > n) {
        return -1;
    }

    unsigned long long *d_hist = nullptr;
    int *d_res = nullptr;

    CUDA_CHECK(cudaMalloc(&d_hist, 256 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, 256 * sizeof(unsigned long long)));

    int blocks = static_cast<int>((n + kBlockThreads - 1) / kBlockThreads);

    histogramKernel<<<blocks, kBlockThreads>>>(d_data, n, d_hist);
    CUDA_CHECK(cudaGetLastError());

    findKthLargestKernel<<<1, 1>>>(d_hist, k, d_res);
    CUDA_CHECK(cudaGetLastError());

    int ret = -1;
    CUDA_CHECK(cudaMemcpy(&ret, d_res, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_hist));
    return ret;
}

int main() {
    const size_t n = 1024;
    uint8_t *h_data = new uint8_t[n];
    for (size_t i = 0; i < n; i++) {
        h_data[i] = static_cast<uint8_t>(i % 200);
    }

    uint8_t *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(uint8_t),
                          cudaMemcpyHostToDevice));

    int ans = kthLargestUint8(d_data, n, 5ULL);
    printf("%d\n", ans);

    CUDA_CHECK(cudaFree(d_data));
    delete[] h_data;
    return 0;
}
