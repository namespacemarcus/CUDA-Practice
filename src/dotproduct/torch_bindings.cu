#include "dot_product.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dot_product_f32_f32", &dot_product_f32_f32, "dot_product_f32_f32");
    m.def("dot_product_f32x4_f32", &dot_product_f32x4_f32,
          "dot_product_f32x4_f32");
    m.def("dot_product_f16_f32", &dot_product_f16_f32, "dot_product_f16_f32");
    m.def("dot_product_f16x2_f32", &dot_product_f16x2_f32,
          "dot_product_f16x2_f32");
    m.def("dot_product_f16x8_pack_f32", &dot_product_f16x8_pack_f32,
          "dot_product_f16x8_pack_f32");
}
