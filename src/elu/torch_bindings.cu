#include "elu.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elu_f32", &elu_f32, "elu_f32");
    m.def("elu_f32x4", &elu_f32x4, "elu_f32x4");
    m.def("elu_f16", &elu_f16, "elu_f16");
    m.def("elu_f16x2", &elu_f16x2, "elu_f16x2");
    m.def("elu_f16x8", &elu_f16x8, "elu_f16x8");
    m.def("elu_f16x8_pack", &elu_f16x8_pack, "elu_f16x8_pack");
}
