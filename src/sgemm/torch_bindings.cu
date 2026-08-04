#include "sgemm.cuh"
#include "sgemm_cublas.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(sgemm_naive)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_tiling)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_at_tiling)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_at_tiling_bcf_swizzling)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_at_tiling_bcf_swizzling_cstore)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_at_tiling_bcf_swizzling_cstore_dbf)

    TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas)
    TORCH_BINDING_COMMON_EXTENSION(sgemm_cublas_tf32)
}
