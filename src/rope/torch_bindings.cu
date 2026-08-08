#include "rope.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_f32", &rope_f32, "rope_f32");
    m.def("rope_f32x4", &rope_f32x4, "rope_f32x4");
}
