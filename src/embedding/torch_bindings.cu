#include "embedding.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_f32", &embedding_f32, "embedding_f32");
    m.def("embedding_f32x4", &embedding_f32x4, "embedding_f32x4");
    m.def("embedding_f32x4_pack", &embedding_f32x4_pack,
          "embedding_f32x4_pack");
    m.def("embedding_f16", &embedding_f16, "embedding_f16");
    m.def("embedding_f16x8", &embedding_f16x8, "embedding_f16x8");
    m.def("embedding_f16x8_pack", &embedding_f16x8_pack,
          "embedding_f16x8_pack");
}
