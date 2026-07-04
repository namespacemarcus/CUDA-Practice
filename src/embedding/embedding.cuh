#pragma once

#include <cuda_fp16.h>

__global__ void embedding_f32_kernel(const int *idx, float *weight, float *output, int n, int emb_size);
__global__ void embedding_f32x4_kernel(const int *idx, float *weight, float *output, int n, int emb_size);
__global__ void embedding_f32x4_pack_kernel(const int *idx, float *weight, float *output, int n, int emb_size);

__global__ void embedding_f16_kernel(const int *idx, half *weight, half *output, int n, int emb_size);
__global__ void embedding_f16x8_kernel(const int *idx, half *weight, half *output, int n, int emb_size);
__global__ void embedding_f16x8_pack_kernel(const int *idx, half *weight, half *output, int n, int emb_size);
