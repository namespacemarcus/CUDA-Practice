#include "layernorm.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_f32", &layer_norm_f32, "layer_norm_f32");
    m.def("layer_norm_f32x4", &layer_norm_f32x4, "layer_norm_f32x4");
    m.def("layer_norm_f16_f16", &layer_norm_f16_f16, "layer_norm_f16_f16");
    m.def("layer_norm_f16_f32", &layer_norm_f16_f32, "layer_norm_f16_f32");
    m.def("layer_norm_f16x2_f16", &layer_norm_f16x2_f16,
          "layer_norm_f16x2_f16");
    m.def("layer_norm_f16x8_f16", &layer_norm_f16x8_f16,
          "layer_norm_f16x8_f16");
    m.def("layer_norm_f16x8_pack_f16", &layer_norm_f16x8_pack_f16,
          "layer_norm_f16x8_pack_f16");
    m.def("layer_norm_f16x8_pack_f32", &layer_norm_f16x8_pack_f32,
          "layer_norm_f16x8_pack_f32");
}
