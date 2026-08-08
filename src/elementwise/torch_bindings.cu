#include "elementwise_add.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add_f32", &elementwise_add_f32, "elementwise_add_f32");
    m.def("elementwise_add_f32x4", &elementwise_add_f32x4,
          "elementwise_add_f32x4");
    m.def("elementwise_add_f16", &elementwise_add_f16, "elementwise_add_f16");
    m.def("elementwise_add_f16x2", &elementwise_add_f16x2,
          "elementwise_add_f16x2");
    m.def("elementwise_add_f16x8", &elementwise_add_f16x8,
          "elementwise_add_f16x8");
    m.def("elementwise_add_f16x8_pack", &elementwise_add_f16x8_pack,
          "elementwise_add_f16x8_pack");
}
