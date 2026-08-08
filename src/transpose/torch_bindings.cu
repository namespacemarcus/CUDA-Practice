#include "matrix_transpose.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    namespace py = pybind11;
    m.def("matrix_transpose_f32_loadcoal", &matrix_transpose_f32_loadcoal,
          py::arg("x"), py::arg("y"));
    m.def("matrix_transpose_f32x4_loadcoal", &matrix_transpose_f32x4_loadcoal,
          py::arg("x"), py::arg("y"));
    m.def("matrix_transpose_f32x4_loadcoal_smem",
          &matrix_transpose_f32x4_loadcoal_smem, py::arg("x"), py::arg("y"));
    m.def("matrix_transpose_f32x4_loadcoal_smem_bcf",
          &matrix_transpose_f32x4_loadcoal_smem_bcf, py::arg("x"),
          py::arg("y"));

    m.def("matrix_transpose_f32_loadcoal_2d", &matrix_transpose_f32_loadcoal_2d,
          py::arg("x"), py::arg("y"));
    m.def("matrix_transpose_f32x4_loadcoal_2d",
          &matrix_transpose_f32x4_loadcoal_2d, py::arg("x"), py::arg("y"));
}
