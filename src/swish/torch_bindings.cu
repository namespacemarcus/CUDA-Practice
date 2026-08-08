#include "swish.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_f32", &swish_f32, "swish_f32");
    m.def("swish_f32x4", &swish_f32x4, "swish_f32x4");
    m.def("swish_f16", &swish_f16, "swish_f16");
    m.def("swish_f16x2", &swish_f16x2, "swish_f16x2");
    m.def("swish_f16x8", &swish_f16x8, "swish_f16x8");
    m.def("swish_f16x8_pack", &swish_f16x8_pack, "swish_f16x8_pack");
}
