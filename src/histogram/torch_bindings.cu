#include "histogram.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("histogram_i32", &histogram_i32, "histogram_i32");
    m.def("histogram_i32x4", &histogram_i32x4, "histogram_i32x4");
}
