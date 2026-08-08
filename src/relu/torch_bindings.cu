#include "relu.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_f32", &relu_f32, "relu_f32");
    m.def("relu_f32x4", &relu_f32x4, "relu_f32x4");
    m.def("relu_f16", &relu_f16, "relu_f16");
    m.def("relu_f16x2", &relu_f16x2, "relu_f16x2");
    m.def("relu_f16x8", &relu_f16x8, "relu_f16x8");
    m.def("relu_f16x8_pack", &relu_f16x8_pack, "relu_f16x8_pack");
}
