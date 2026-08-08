#include "softmax.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_f32_per_token", &softmax_f32_per_token,
          "softmax_f32_per_token");
    m.def("softmax_f32x4_per_token", &softmax_f32x4_per_token,
          "softmax_f32x4_per_token");
    m.def("safe_softmax_f32_per_token", &safe_softmax_f32_per_token,
          "safe_softmax_f32_per_token");
    m.def("safe_softmax_f32x4_per_token", &safe_softmax_f32x4_per_token,
          "safe_softmax_f32x4_per_token");
    m.def("safe_softmax_f16_f32_per_token", &safe_softmax_f16_f32_per_token,
          "safe_softmax_f16_f32_per_token");
    m.def("safe_softmax_f16x2_f32_per_token", &safe_softmax_f16x2_f32_per_token,
          "safe_softmax_f16x2_f32_per_token");
    m.def("safe_softmax_f16x8_pack_f32_per_token",
          &safe_softmax_f16x8_pack_f32_per_token,
          "safe_softmax_f16x8_pack_f32_per_token");
    m.def("online_safe_softmax_f32_per_token",
          &online_safe_softmax_f32_per_token,
          "online_safe_softmax_f32_per_token");
    m.def("online_safe_softmax_f32x4_pack_per_token",
          &online_safe_softmax_f32x4_pack_per_token,
          "online_safe_softmax_f32x4_pack_per_token");
}
