#include "../../common/torch/torch_binding_utils.h"
#include "online_softmax.cuh"
#include "safe_softmax.cuh"
#include "softmax.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                                 \
    softmax_f32_per_token_kernel<(H)>                                          \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                            \
    dim3 block((H));                                                           \
    dim3 grid((S));                                                            \
    switch ((H)) {                                                             \
    case 32:   LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(32) break;                  \
    case 64:   LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(64) break;                  \
    case 128:  LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(128) break;                 \
    case 256:  LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(256) break;                 \
    case 512:  LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(512) break;                 \
    case 1024: LANUCH_SOFTMAX_F32_PER_TOKEN_KERNEL(1024) break;                \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/256/512/1024");       \
        break;                                                                 \
    }

#define LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                               \
    softmax_f32x4_per_token_kernel<(H) / 4>                                    \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                          \
    const int NT = (H) / 4;                                                    \
    dim3 block(NT);                                                            \
    dim3 grid((S));                                                            \
    switch (H) {                                                               \
    case 32:   LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;                \
    case 64:   LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;                \
    case 128:  LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;               \
    case 256:  LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;               \
    case 512:  LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;               \
    case 1024: LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;              \
    case 2048: LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;              \
    case 4096: LANUCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;              \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/.../1024*4");         \
        break;                                                                 \
    }

#define LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                            \
    safe_softmax_f32_per_token_kernel<(H)>                                     \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                       \
    dim3 block((H));                                                           \
    dim3 grid((S));                                                            \
    switch ((H)) {                                                             \
    case 32:   LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(32) break;             \
    case 64:   LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(64) break;             \
    case 128:  LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(128) break;            \
    case 256:  LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(256) break;            \
    case 512:  LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(512) break;            \
    case 1024: LANUCH_SAFE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024) break;           \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/256/512/1024");       \
        break;                                                                 \
    }

#define LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(H)                          \
    safe_softmax_f32x4_per_token_kernel<(H) / 4>                               \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)                     \
    const int NT = (H) / 4;                                                    \
    dim3 block(NT);                                                            \
    dim3 grid((S));                                                            \
    switch (H) {                                                               \
    case 32:   LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(32) break;           \
    case 64:   LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(64) break;           \
    case 128:  LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(128) break;          \
    case 256:  LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(256) break;          \
    case 512:  LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(512) break;          \
    case 1024: LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(1024) break;         \
    case 2048: LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(2048) break;         \
    case 4096: LANUCH_SAFE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(4096) break;         \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/.../1024*4");         \
        break;                                                                 \
    }

#define LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(H)                        \
    safe_softmax_f16_f32_per_token_kernel<(H)>                                 \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)                   \
    dim3 block((H));                                                           \
    dim3 grid((S));                                                            \
    switch ((H)) {                                                             \
    case 32:   LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(32) break;         \
    case 64:   LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(64) break;         \
    case 128:  LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(128) break;        \
    case 256:  LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(256) break;        \
    case 512:  LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(512) break;        \
    case 1024: LANUCH_SAFE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(1024) break;       \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/256/512/1024");       \
        break;                                                                 \
    }

#define LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(H)                      \
    safe_softmax_f16x2_f32_per_token_kernel<(H) / 2>                           \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)                 \
    const int NT = (H) / 2;                                                    \
    dim3 block(NT);                                                            \
    dim3 grid((S));                                                            \
    switch (H) {                                                               \
    case 32:   LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(32) break;       \
    case 64:   LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(64) break;       \
    case 128:  LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(128) break;      \
    case 256:  LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(256) break;      \
    case 512:  LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(512) break;      \
    case 1024: LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(1024) break;     \
    case 2048: LANUCH_SAFE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(2048) break;     \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/.../1024*2");         \
        break;                                                                 \
    }

#define LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(H)                 \
    safe_softmax_f16x8_pack_f32_per_token_kernel<(H) / 8>                      \
        <<<grid, block>>>(reinterpret_cast<half *>(x.data_ptr()),              \
                          reinterpret_cast<half *>(y.data_ptr()), N);

#define DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)            \
    const int NT = (H) / 8;                                                    \
    dim3 block(NT);                                                            \
    dim3 grid((S));                                                            \
    switch (H) {                                                               \
    case 32:  LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(32) break;   \
    case 64:  LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(64) break;   \
    case 128: LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(128) break;  \
    case 256: LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(256) break;  \
    case 512: LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(512) break;  \
    case 1024:                                                                 \
        LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(1024) break;       \
    case 2048:                                                                 \
        LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(2048) break;       \
    case 4096:                                                                 \
        LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(4096) break;       \
    case 8192:                                                                 \
        LANUCH_SAFE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(8192) break;       \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/.../1024*8");         \
        break;                                                                 \
    }

#define LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(H)                          \
    online_safe_softmax_f32_per_token_kernel<(H)>                              \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)                     \
    dim3 block((H));                                                           \
    dim3 grid((S));                                                            \
    switch ((H)) {                                                             \
    case 32:   LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(32) break;           \
    case 64:   LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(64) break;           \
    case 128:  LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(128) break;          \
    case 256:  LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(256) break;          \
    case 512:  LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(512) break;          \
    case 1024: LANUCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(1024) break;         \
    default:                                                                   \
        throw std::runtime_error("only support H: 64/128/256/512/1024");       \
        break;                                                                 \
    }

#define LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(H)                   \
    online_safe_softmax_f32x4_pack_per_token_kernel<(H / 4)>                   \
        <<<grid, block>>>(reinterpret_cast<float *>(x.data_ptr()),             \
                          reinterpret_cast<float *>(y.data_ptr()), N);

#define DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)              \
    dim3 block((H / 4));                                                       \
    dim3 grid((S));                                                            \
    switch ((H)) {                                                             \
    case 128:  LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(128) break;   \
    case 256:  LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(256) break;   \
    case 512:  LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(512) break;   \
    case 1024: LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(1024) break;  \
    case 2048: LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(2048) break;  \
    case 4096: LANUCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(4096) break;  \
    default:                                                                   \
        throw std::runtime_error("only support H: 128/256/.../4096;");         \
        break;                                                                 \
    }

void softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32);
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f32x4_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F32x4_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)

    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F16_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x2_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F16x2_F32_PER_TOKEN_KERNEL(S, H)
}

void safe_softmax_f16x8_pack_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int S = x.size(0);
    const int H = x.size(1);
    const int N = S * H;
    DISPATCH_SATE_SOFTMAX_F16x8_PACK_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32_per_token(torch::Tensor x, torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int S = x.size(0);
    const int H = x.size(1);
    // online kernel uses per-row indexing: row_start = blockIdx.x * N
    const int N = S * H;
    DISPATCH_ONLINE_SOFTMAX_F32_PER_TOKEN_KERNEL(S, H)
}

void online_safe_softmax_f32x4_pack_per_token(torch::Tensor x,
                                              torch::Tensor y) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(y, torch::kFloat32)
    CHECK_TORCH_TENSOR_SAME_SHAPE(x, y)
    const int S = x.size(0);
    const int H = x.size(1);
    // online kernel uses per-row indexing: row_start = blockIdx.x * N
    const int N = S * H;
    DISPATCH_ONLINE_SOFTMAX_F32X4_PACK_PER_TOKEN_KERNEL(S, H)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(softmax_f32x4_per_token)
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f32x4_per_token)
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x2_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(safe_softmax_f16x8_pack_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32_per_token)
    TORCH_BINDING_COMMON_EXTENSION(online_safe_softmax_f32x4_pack_per_token)
}
