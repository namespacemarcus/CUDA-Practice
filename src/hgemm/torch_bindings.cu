#include "hgemm.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgemm_gw_tiled", &hgemm_gw_tiled, "hgemm_gw_tiled");
    m.def("hgemm_gw_tiled_bcf", &hgemm_gw_tiled_bcf, "hgemm_gw_tiled_bcf");
    m.def("hgemm_gw_tiled_bcf_dbf", &hgemm_gw_tiled_bcf_dbf,
          "hgemm_gw_tiled_bcf_dbf");
}
