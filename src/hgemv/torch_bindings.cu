#include "hgemv.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemv_k32_f16", &hgemv_k32_f16, "hgemv_k32_f16");
    m.def("hgemv_k128_f16x4", &hgemv_k128_f16x4, "hgemv_k128_f16x4");
    m.def("hgemv_k16_f16", &hgemv_k16_f16, "hgemv_k16_f16");
}
