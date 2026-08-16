#include "online_attention.cuh"
#include "flash_attention.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    namespace py = pybind11;
    module.def("online_attention_naive", &online_attention_naive,
               py::arg("query"), py::arg("key"), py::arg("value"),
               py::arg("causal") = false);
    module.def("online_attention_kv_tiled", &online_attention_kv_tiled,
               py::arg("query"), py::arg("key"), py::arg("value"),
               py::arg("causal") = false);

    module.def("flash_attention_v1_forward", &flash_attention_v1,
               py::arg("query"), py::arg("key"), py::arg("value"),
               py::arg("causal") = false);
    module.def("flash_attention_v2_forward", &flash_attention_v2,
               py::arg("query"), py::arg("key"), py::arg("value"),
               py::arg("causal") = false);
}
