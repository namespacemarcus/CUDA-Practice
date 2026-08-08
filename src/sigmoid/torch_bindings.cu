#include "sigmoid.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_f32", &sigmoid_f32, "sigmoid_f32");
    m.def("sigmoid_f32x4", &sigmoid_f32x4, "sigmoid_f32x4");
    m.def("sigmoid_f16", &sigmoid_f16, "sigmoid_f16");
    m.def("sigmoid_f16x2", &sigmoid_f16x2, "sigmoid_f16x2");
    m.def("sigmoid_f16x8", &sigmoid_f16x8, "sigmoid_f16x8");
    m.def("sigmoid_f16x8_pack", &sigmoid_f16x8_pack, "sigmoid_f16x8_pack");
}
