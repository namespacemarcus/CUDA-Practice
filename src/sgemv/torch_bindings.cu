#include "sgemv.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemv_k32_f32", &sgemv_k32_f32, "sgemv_k32_f32");
    m.def("sgemv_k128_f32x4", &sgemv_k128_f32x4, "sgemv_k128_f32x4");
    m.def("sgemv_k16_f32", &sgemv_k16_f32, "sgemv_k16_f32");
}
