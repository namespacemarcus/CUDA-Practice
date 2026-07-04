#include "../common/torch/torch_binding_utils.h"
#include "rms_norm.cuh"

// f32
#define LAUNCH_RMS_NORM_F32_KERNEL(K)                                          \
    rms_norm_f32_kernel<(K)>                                                   \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32_KERNEL(N, K)                                     \
    dim3 block((K));                                                           \
    dim3 grid((N));                                                            \
                                                                               \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F32_KERNEL(64) break;                           \
    case 128:  LAUNCH_RMS_NORM_F32_KERNEL(128) break;                          \
    case 256:  LAUNCH_RMS_NORM_F32_KERNEL(256) break;                          \
    case 512:  LAUNCH_RMS_NORM_F32_KERNEL(512) break;                          \
    case 1024: LAUNCH_RMS_NORM_F32_KERNEL(1024) break;                         \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void rms_norm_f32(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F32_KERNEL(N, K)
}

// f32x4
#define LAUNCH_RMS_NORM_F32x4_KERNEL(K)                                        \
    rms_norm_f32x4_kernel<(K) / 4>                                             \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F32x4_KERNEL(N, K)                                   \
    dim3 block((K) / 4);                                                       \
    dim3 grid((N));                                                            \
                                                                               \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F32x4_KERNEL(64) break;                         \
    case 128:  LAUNCH_RMS_NORM_F32x4_KERNEL(128) break;                        \
    case 256:  LAUNCH_RMS_NORM_F32x4_KERNEL(256) break;                        \
    case 512:  LAUNCH_RMS_NORM_F32x4_KERNEL(512) break;                        \
    case 1024: LAUNCH_RMS_NORM_F32x4_KERNEL(1024) break;                       \
    case 2048: LAUNCH_RMS_NORM_F32x4_KERNEL(2048) break;                       \
    case 4096: LAUNCH_RMS_NORM_F32x4_KERNEL(4096) break;                       \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/.../512/1024*4");         \
        break;                                                                 \
    }

void rms_norm_f32x4(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F32x4_KERNEL(N, K)
}

// f16 f16
#define LAUNCH_RMS_NORM_F16F16_KERNEL(K)                                       \
    rms_norm_f16_f16_kernel<(K)>                                               \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16F16_KERNEL(N, K)                                  \
    dim3 block((K));                                                           \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16F16_KERNEL(64) break;                        \
    case 128:  LAUNCH_RMS_NORM_F16F16_KERNEL(128) break;                       \
    case 256:  LAUNCH_RMS_NORM_F16F16_KERNEL(256) break;                       \
    case 512:  LAUNCH_RMS_NORM_F16F16_KERNEL(512) break;                       \
    case 1024: LAUNCH_RMS_NORM_F16F16_KERNEL(1024) break;                      \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void rms_norm_f16_f16(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16F16_KERNEL(N, K)
}

// f16x2 f16
#define LAUNCH_RMS_NORM_F16x2F16_KERNEL(K)                                     \
    rms_norm_f16x2_f16_kernel<(K) / 2>                                         \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16x2F16_KERNEL(N, K)                                \
    dim3 block((K) / 2);                                                       \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16x2F16_KERNEL(64) break;                      \
    case 128:  LAUNCH_RMS_NORM_F16x2F16_KERNEL(128) break;                     \
    case 256:  LAUNCH_RMS_NORM_F16x2F16_KERNEL(256) break;                     \
    case 512:  LAUNCH_RMS_NORM_F16x2F16_KERNEL(512) break;                     \
    case 1024: LAUNCH_RMS_NORM_F16x2F16_KERNEL(1024) break;                    \
    case 2048: LAUNCH_RMS_NORM_F16x2F16_KERNEL(2048) break;                    \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/.../1024*2");         \
        break;                                                                 \
    }

void rms_norm_f16x2_f16(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16x2F16_KERNEL(N, K)
}

// f16x8 f16
#define LAUNCH_RMS_NORM_F16x8F16_KERNEL(K)                                     \
    rms_norm_f16x8_f16_kernel<(K) / 8>                                         \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16x8F16_KERNEL(N, K)                                \
    dim3 block((K) / 8);                                                       \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16x8F16_KERNEL(64) break;                      \
    case 128:  LAUNCH_RMS_NORM_F16x8F16_KERNEL(128) break;                     \
    case 256:  LAUNCH_RMS_NORM_F16x8F16_KERNEL(256) break;                     \
    case 512:  LAUNCH_RMS_NORM_F16x8F16_KERNEL(512) break;                     \
    case 1024: LAUNCH_RMS_NORM_F16x8F16_KERNEL(1024) break;                    \
    case 2048: LAUNCH_RMS_NORM_F16x8F16_KERNEL(2048) break;                    \
    case 4096: LAUNCH_RMS_NORM_F16x8F16_KERNEL(4096) break;                    \
    case 8192: LAUNCH_RMS_NORM_F16x8F16_KERNEL(8192) break;                    \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void rms_norm_f16x8_f16(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16x8F16_KERNEL(N, K)
}

// f16x8_pack f16
#define LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(K)                               \
    rms_norm_f16x8_pack_f16_kernel<(K) / 8>                                    \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16x8_PACK_F16_KERNEL(N, K)                          \
    dim3 block((K) / 8);                                                       \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(64) break;                \
    case 128:  LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(128) break;               \
    case 256:  LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(256) break;               \
    case 512:  LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(512) break;               \
    case 1024: LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(1024) break;              \
    case 2048: LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(2048) break;              \
    case 4096: LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(4096) break;              \
    case 8192: LAUNCH_RMS_NORM_F16x8_PACK_F16_KERNEL(8192) break;              \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void rms_norm_f16x8_pack_f16(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16x8_PACK_F16_KERNEL(N, K)
}

// f16 f32
#define LAUNCH_RMS_NORM_F16F32_KERNEL(K)                                       \
    rms_norm_f16_f32_kernel<(K)>                                               \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16F32_KERNEL(N, K)                                  \
    dim3 block((K));                                                           \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16F32_KERNEL(64) break;                        \
    case 128:  LAUNCH_RMS_NORM_F16F32_KERNEL(128) break;                       \
    case 256:  LAUNCH_RMS_NORM_F16F32_KERNEL(256) break;                       \
    case 512:  LAUNCH_RMS_NORM_F16F32_KERNEL(512) break;                       \
    case 1024: LAUNCH_RMS_NORM_F16F32_KERNEL(1024) break;                      \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/256/512/1024");       \
        break;                                                                 \
    }

void rms_norm_f16_f32(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16F32_KERNEL(N, K)
}

// f16x8 f32
#define LAUNCH_RMS_NORM_F16x8F32_KERNEL(K)                                     \
    rms_norm_f16x8_f32_kernel<(K) / 8>                                         \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16x8F32_KERNEL(N, K)                                \
    dim3 block((K) / 8);                                                       \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16x8F32_KERNEL(64) break;                      \
    case 128:  LAUNCH_RMS_NORM_F16x8F32_KERNEL(128) break;                     \
    case 256:  LAUNCH_RMS_NORM_F16x8F32_KERNEL(256) break;                     \
    case 512:  LAUNCH_RMS_NORM_F16x8F32_KERNEL(512) break;                     \
    case 1024: LAUNCH_RMS_NORM_F16x8F32_KERNEL(1024) break;                    \
    case 2048: LAUNCH_RMS_NORM_F16x8F32_KERNEL(2048) break;                    \
    case 4096: LAUNCH_RMS_NORM_F16x8F32_KERNEL(4096) break;                    \
    case 8192: LAUNCH_RMS_NORM_F16x8F32_KERNEL(8192) break;                    \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void rms_norm_f16x8_f32(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16x8F32_KERNEL(N, K)
}

// f16x8_pack f32
#define LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(K)                               \
    rms_norm_f16x8_pack_f32_kernel<(K) / 8>                                    \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), g, N, (K));

#define DISPATCH_RMS_NORM_F16x8_PACK_F32_KERNEL(N, K)                          \
    dim3 block((K) / 8);                                                       \
    dim3 grid((N));                                                            \
    switch ((K)) {                                                             \
    case 64:   LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(64) break;                \
    case 128:  LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(128) break;               \
    case 256:  LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(256) break;               \
    case 512:  LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(512) break;               \
    case 1024: LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(1024) break;              \
    case 2048: LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(2048) break;              \
    case 4096: LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(4096) break;              \
    case 8192: LAUNCH_RMS_NORM_F16x8_PACK_F32_KERNEL(8192) break;              \
    default:                                                                   \
        throw std::runtime_error("only support K: 64/128/.../1024*8");         \
        break;                                                                 \
    }

void rms_norm_f16x8_pack_f32(torch::Tensor x, torch::Tensor y, float g) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int N = x.size(0);
    const int K = x.size(1);
    DISPATCH_RMS_NORM_F16x8_PACK_F32_KERNEL(N, K)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16_f16)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x2_f16)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x8_f16)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x8_pack_f16)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x8_f32)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16x8_pack_f32)
    TORCH_BINDING_COMMON_EXTENSION(rms_norm_f16_f32)
}
