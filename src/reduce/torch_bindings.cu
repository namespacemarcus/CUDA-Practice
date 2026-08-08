#include "reduce.cuh"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "block-all-reduce sum CUDA kernels (returns a scalar tensor).";

    m.def("block_all_reduce_sum_f32_f32", &block_all_reduce_sum_f32_f32,
          "block_all_reduce_sum_f32_f32");
    m.def("block_all_reduce_sum_f32x4_f32", &block_all_reduce_sum_f32x4_f32,
          "block_all_reduce_sum_f32x4_f32");
    m.def("block_all_reduce_sum_f16_f16", &block_all_reduce_sum_f16_f16,
          "block_all_reduce_sum_f16_f16");
    m.def("block_all_reduce_sum_f16_f32", &block_all_reduce_sum_f16_f32,
          "block_all_reduce_sum_f16_f32");
    m.def("block_all_reduce_sum_f16x2_f16", &block_all_reduce_sum_f16x2_f16,
          "block_all_reduce_sum_f16x2_f16");
    m.def("block_all_reduce_sum_f16x2_f32", &block_all_reduce_sum_f16x2_f32,
          "block_all_reduce_sum_f16x2_f32");
    m.def("block_all_reduce_sum_f16x8_pack_f16",
          &block_all_reduce_sum_f16x8_pack_f16,
          "block_all_reduce_sum_f16x8_pack_f16");
    m.def("block_all_reduce_sum_f16x8_pack_f32",
          &block_all_reduce_sum_f16x8_pack_f32,
          "block_all_reduce_sum_f16x8_pack_f32");
    m.def("block_all_reduce_sum_bf16_bf16", &block_all_reduce_sum_bf16_bf16,
          "block_all_reduce_sum_bf16_bf16");
    m.def("block_all_reduce_sum_bf16_f32", &block_all_reduce_sum_bf16_f32,
          "block_all_reduce_sum_bf16_f32");
    m.def("block_all_reduce_sum_bf16x2_bf16", &block_all_reduce_sum_bf16x2_bf16,
          "block_all_reduce_sum_bf16x2_bf16");
    m.def("block_all_reduce_sum_bf16x2_f32", &block_all_reduce_sum_bf16x2_f32,
          "block_all_reduce_sum_bf16x2_f32");
    m.def("block_all_reduce_sum_bf16x8_pack_bf16",
          &block_all_reduce_sum_bf16x8_pack_bf16,
          "block_all_reduce_sum_bf16x8_pack_bf16");
    m.def("block_all_reduce_sum_bf16x8_pack_f32",
          &block_all_reduce_sum_bf16x8_pack_f32,
          "block_all_reduce_sum_bf16x8_pack_f32");
#if REDUCE_HAS_FP8
    m.def("block_all_reduce_sum_fp8_e4m3_f16",
          &block_all_reduce_sum_fp8_e4m3_f16,
          "block_all_reduce_sum_fp8_e4m3_f16");
    m.def("block_all_reduce_sum_fp8_e4m3x16_pack_f16",
          &block_all_reduce_sum_fp8_e4m3x16_pack_f16,
          "block_all_reduce_sum_fp8_e4m3x16_pack_f16");
    m.def("block_all_reduce_sum_fp8_e5m2_f16",
          &block_all_reduce_sum_fp8_e5m2_f16,
          "block_all_reduce_sum_fp8_e5m2_f16");
    m.def("block_all_reduce_sum_fp8_e5m2x16_pack_f16",
          &block_all_reduce_sum_fp8_e5m2x16_pack_f16,
          "block_all_reduce_sum_fp8_e5m2x16_pack_f16");
#endif
    m.def("block_all_reduce_sum_i8_i32", &block_all_reduce_sum_i8_i32,
          "block_all_reduce_sum_i8_i32");
    m.def("block_all_reduce_sum_i8x16_pack_i32",
          &block_all_reduce_sum_i8x16_pack_i32,
          "block_all_reduce_sum_i8x16_pack_i32");
}
