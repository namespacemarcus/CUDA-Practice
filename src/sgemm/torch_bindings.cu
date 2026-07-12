#include "sgemm.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(sgemm_naive_f32)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_k_tiled_f32)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_thread_tiled_8x8_and_k_tiled_f32x4)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf)
    TORCH_BINDING_COMMON_EXTENSION(
        sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_offset)
}
