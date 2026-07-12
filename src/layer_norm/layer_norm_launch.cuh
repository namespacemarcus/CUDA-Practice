#pragma once

#include "../common/cuda/cuda_utils.h"
#include "layer_norm.cuh"

// f32
#define LAUNCH_LAYER_NORM_F32_KERNEL(d)                                        \
    layer_norm_f32_kernel<(d)><<<grid, block>>>(                               \
        reinterpret_cast<float *>(x.data_ptr()),                               \
        reinterpret_cast<float *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F32_KERNEL(s, d)                                   \
    dim3 block((d));                                                           \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F32_KERNEL(64) break;                         \
    case 128:  LAUNCH_LAYER_NORM_F32_KERNEL(128) break;                        \
    case 256:  LAUNCH_LAYER_NORM_F32_KERNEL(256) break;                        \
    case 512:  LAUNCH_LAYER_NORM_F32_KERNEL(512) break;                        \
    case 1024: LAUNCH_LAYER_NORM_F32_KERNEL(1024) break;                       \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void layer_norm_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F32_KERNEL(s, d)
}

// f32x4
#define LAUNCH_LAYER_NORM_F32x4_KERNEL(d)                                      \
    layer_norm_f32x4_kernel<(d) / 4><<<grid, block>>>(                         \
        reinterpret_cast<float *>(x.data_ptr()),                               \
        reinterpret_cast<float *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F32x4_KERNEL(s, d)                                 \
    dim3 block((d) / 4);                                                       \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F32x4_KERNEL(64) break;                       \
    case 128:  LAUNCH_LAYER_NORM_F32x4_KERNEL(128) break;                      \
    case 256:  LAUNCH_LAYER_NORM_F32x4_KERNEL(256) break;                      \
    case 512:  LAUNCH_LAYER_NORM_F32x4_KERNEL(512) break;                      \
    case 1024: LAUNCH_LAYER_NORM_F32x4_KERNEL(1024) break;                     \
    case 2048: LAUNCH_LAYER_NORM_F32x4_KERNEL(2048) break;                     \
    case 4096: LAUNCH_LAYER_NORM_F32x4_KERNEL(4096) break;                     \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/.../1024*4");         \
        break;                                                                 \
    }

void layer_norm_f32x4(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F32x4_KERNEL(s, d)
}

// f16 f16
#define LAUNCH_LAYER_NORM_F16F16_KERNEL(d)                                     \
    layer_norm_f16_f16_kernel<(d)><<<grid, block>>>(                           \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16F16_KERNEL(s, d)                                \
    dim3 block((d));                                                           \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16F16_KERNEL(64) break;                      \
    case 128:  LAUNCH_LAYER_NORM_F16F16_KERNEL(128) break;                     \
    case 256:  LAUNCH_LAYER_NORM_F16F16_KERNEL(256) break;                     \
    case 512:  LAUNCH_LAYER_NORM_F16F16_KERNEL(512) break;                     \
    case 1024: LAUNCH_LAYER_NORM_F16F16_KERNEL(1024) break;                    \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void layer_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16F16_KERNEL(s, d)
}

// f16x2 f16
#define LAUNCH_LAYER_NORM_F16x2F16_KERNEL(d)                                   \
    layer_norm_f16x2_f16_kernel<(d) / 2><<<grid, block>>>(                     \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16x2F16_KERNEL(s, d)                              \
    dim3 block((d) / 2);                                                       \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16x2F16_KERNEL(64) break;                    \
    case 128:  LAUNCH_LAYER_NORM_F16x2F16_KERNEL(128) break;                   \
    case 256:  LAUNCH_LAYER_NORM_F16x2F16_KERNEL(256) break;                   \
    case 512:  LAUNCH_LAYER_NORM_F16x2F16_KERNEL(512) break;                   \
    case 1024: LAUNCH_LAYER_NORM_F16x2F16_KERNEL(1024) break;                  \
    case 2048: LAUNCH_LAYER_NORM_F16x2F16_KERNEL(2048) break;                  \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/.../1024*2");         \
        break;                                                                 \
    }

void layer_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16x2F16_KERNEL(s, d)
}

// f16x8 f16
#define LAUNCH_LAYER_NORM_F16x8F16_KERNEL(d)                                   \
    layer_norm_f16x8_f16_kernel<(d) / 8><<<grid, block>>>(                     \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16x8F16_KERNEL(s, d)                              \
    dim3 block((d) / 8);                                                       \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16x8F16_KERNEL(64) break;                    \
    case 128:  LAUNCH_LAYER_NORM_F16x8F16_KERNEL(128) break;                   \
    case 256:  LAUNCH_LAYER_NORM_F16x8F16_KERNEL(256) break;                   \
    case 512:  LAUNCH_LAYER_NORM_F16x8F16_KERNEL(512) break;                   \
    case 1024: LAUNCH_LAYER_NORM_F16x8F16_KERNEL(1024) break;                  \
    case 2048: LAUNCH_LAYER_NORM_F16x8F16_KERNEL(2048) break;                  \
    case 4096: LAUNCH_LAYER_NORM_F16x8F16_KERNEL(4096) break;                  \
    case 8192: LAUNCH_LAYER_NORM_F16x8F16_KERNEL(8192) break;                  \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void layer_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16x8F16_KERNEL(s, d)
}

// f16x8_pack f16
#define LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(d)                             \
    layer_norm_f16x8_pack_f16_kernel<(d) / 8><<<grid, block>>>(                \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(s, d)                        \
    dim3 block((d) / 8);                                                       \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(64) break;              \
    case 128:  LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(128) break;             \
    case 256:  LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(256) break;             \
    case 512:  LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(512) break;             \
    case 1024: LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(1024) break;            \
    case 2048: LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(2048) break;            \
    case 4096: LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(4096) break;            \
    case 8192: LAUNCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(8192) break;            \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void layer_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16x8_PACK_F16_KERNEL(s, d)
}

// f16 f32
#define LAUNCH_LAYER_NORM_F16F32_KERNEL(d)                                     \
    layer_norm_f16_f32_kernel<(d)><<<grid, block>>>(                           \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16F32_KERNEL(s, d)                                \
    dim3 block((d));                                                           \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16F32_KERNEL(64) break;                      \
    case 128:  LAUNCH_LAYER_NORM_F16F32_KERNEL(128) break;                     \
    case 256:  LAUNCH_LAYER_NORM_F16F32_KERNEL(256) break;                     \
    case 512:  LAUNCH_LAYER_NORM_F16F32_KERNEL(512) break;                     \
    case 1024: LAUNCH_LAYER_NORM_F16F32_KERNEL(1024) break;                    \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void layer_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g, float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16F32_KERNEL(s, d)
}

// f16x8_pack f32
#define LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(d)                             \
    layer_norm_f16x8_pack_f32_kernel<(d) / 8><<<grid, block>>>(                \
        reinterpret_cast<half *>(x.data_ptr()),                                \
        reinterpret_cast<half *>(y.data_ptr()), g, b, s, (d));

#define DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(s, d)                        \
    dim3 block((d) / 8);                                                       \
    dim3 grid((s));                                                            \
    switch ((d)) {                                                             \
    case 64:   LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(64) break;              \
    case 128:  LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(128) break;             \
    case 256:  LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(256) break;             \
    case 512:  LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(512) break;             \
    case 1024: LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(1024) break;            \
    case 2048: LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(2048) break;            \
    case 4096: LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(4096) break;            \
    case 8192: LAUNCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(8192) break;            \
    default:                                                                   \
        throw std::runtime_error("only support d: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void layer_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g,
                               float b) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int s = x.size(0);
    const int d = x.size(1);
    DISPATCH_LAYER_NORM_F16x8_PACK_F32_KERNEL(s, d)
}
