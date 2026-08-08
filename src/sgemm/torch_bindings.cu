#include "sgemm.cuh"
#include "sgemm_cublas.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_naive", &sgemm_naive, "sgemm_naive");
    m.def("sgemm_tiling", &sgemm_tiling, "sgemm_tiling");
    m.def("sgemm_at_tiling", &sgemm_at_tiling, "sgemm_at_tiling");
    m.def("sgemm_at_tiling_bcf_swizzling", &sgemm_at_tiling_bcf_swizzling,
          "sgemm_at_tiling_bcf_swizzling");
    m.def("sgemm_at_tiling_bcf_swizzling_cstore",
          &sgemm_at_tiling_bcf_swizzling_cstore,
          "sgemm_at_tiling_bcf_swizzling_cstore");
    m.def("sgemm_at_tiling_bcf_swizzling_cstore_dbf",
          &sgemm_at_tiling_bcf_swizzling_cstore_dbf,
          "sgemm_at_tiling_bcf_swizzling_cstore_dbf");

    m.def("sgemm_cublas", &sgemm_cublas, "sgemm_cublas");
    m.def("sgemm_cublas_tf32", &sgemm_cublas_tf32, "sgemm_cublas_tf32");
}
