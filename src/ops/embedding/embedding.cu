#include "../../common/cuda/cuda_utils.h"
#include <cuda_fp16.h>

// grid(n), block(emb_size)
__global__ void embedding_f32_kernel(const int *idx, float *weight,
                                     float *output, int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

// grid(n), block(emb_size/4)
__global__ void embedding_f32x4_kernel(const int *idx, float *weight,
                                       float *output, int n, int emb_size) {
    int tx = threadIdx.x * 4;
    int bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
    output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
    output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
    output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
}

__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight,
                                            float *output, int n,
                                            int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    LDST128BITS(output[bx * emb_size + 4 * tx]) =
        LDST128BITS(weight[offset + 4 * tx]);
}

__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output,
                                     int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
}

__global__ void embedding_f16x8_kernel(const int *idx, half *weight,
                                       half *output, int n, int emb_size) {
    int tx = threadIdx.x * 8;
    int bx = blockIdx.x;
    int offset = idx[bx] * emb_size;
    output[bx * emb_size + tx] = weight[offset + tx];
    output[bx * emb_size + tx + 1] = weight[offset + tx + 1];
    output[bx * emb_size + tx + 2] = weight[offset + tx + 2];
    output[bx * emb_size + tx + 3] = weight[offset + tx + 3];
    output[bx * emb_size + tx + 4] = weight[offset + tx + 4];
    output[bx * emb_size + tx + 5] = weight[offset + tx + 5];
    output[bx * emb_size + tx + 6] = weight[offset + tx + 6];
    output[bx * emb_size + tx + 7] = weight[offset + tx + 7];
}

__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight,
                                            half *output, int n, int emb_size) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int tid = bx * blockDim.x + tx;
    int offset = idx[bx] * emb_size;
    LDST128BITS(output[bx * emb_size + 8 * tx]) =
        LDST128BITS(weight[offset + 8 * tx]);
}
