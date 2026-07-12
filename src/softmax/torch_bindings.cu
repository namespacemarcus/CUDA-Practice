#include "softmax_launch.cuh"

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
