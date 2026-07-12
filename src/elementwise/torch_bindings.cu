#include "elementwise_add.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x2)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8)
    TORCH_BINDING_COMMON_EXTENSION(elementwise_add_f16x8_pack)
}
