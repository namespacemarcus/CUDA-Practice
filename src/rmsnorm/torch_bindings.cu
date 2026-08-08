#include "rmsnorm.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_f32", &rms_norm_f32, "rms_norm_f32");
    m.def("rms_norm_f32x4", &rms_norm_f32x4, "rms_norm_f32x4");
    m.def("rms_norm_f16_f16", &rms_norm_f16_f16, "rms_norm_f16_f16");
    m.def("rms_norm_f16x2_f16", &rms_norm_f16x2_f16, "rms_norm_f16x2_f16");
    m.def("rms_norm_f16x8_f16", &rms_norm_f16x8_f16, "rms_norm_f16x8_f16");
    m.def("rms_norm_f16x8_pack_f16", &rms_norm_f16x8_pack_f16,
          "rms_norm_f16x8_pack_f16");
    m.def("rms_norm_f16x8_f32", &rms_norm_f16x8_f32, "rms_norm_f16x8_f32");
    m.def("rms_norm_f16x8_pack_f32", &rms_norm_f16x8_pack_f32,
          "rms_norm_f16x8_pack_f32");
    m.def("rms_norm_f16_f32", &rms_norm_f16_f32, "rms_norm_f16_f32");
}
