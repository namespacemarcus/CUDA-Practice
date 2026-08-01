#include "flash_attention.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    namespace py = pybind11;
    module.doc() = "CUDA Core Flash Attention";
    module.def("flash_attention_naive", &flash_attention_naive,
               py::arg("query"), py::arg("key"), py::arg("value"),
               py::arg("causal") = false);
}
