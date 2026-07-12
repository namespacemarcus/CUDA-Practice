#include "../common/torch/torch_binding_utils.h"
#include "histogram.cuh"

torch::Tensor histogram_i32(torch::Tensor a) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kInt32)
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    const int N = a.size(0);
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);
    torch::Tensor max_val = std::get<0>(max_a).cpu();
    const int M = max_val.item().to<int>();
    auto y = torch::zeros({M + 1}, options);

    const int NUM_THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + 256 - 1) / 256;
    dim3 block(NUM_THREADS_PER_BLOCK);
    dim3 grid(NUM_BLOCKS);
    histogram_i32_kernel<<<grid, block>>>(reinterpret_cast<int *>(a.data_ptr()),
                                          reinterpret_cast<int *>(y.data_ptr()),
                                          N);
    return y;
}

torch::Tensor histogram_i32x4(torch::Tensor a) {
    CHECK_TORCH_TENSOR_DTYPE(a, torch::kInt32)
    auto options =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, 0);
    const int N = a.size(0);
    std::tuple<torch::Tensor, torch::Tensor> max_a = torch::max(a, 0);
    torch::Tensor max_val = std::get<0>(max_a).cpu();
    const int M = max_val.item().to<int>();
    auto y = torch::zeros({M + 1}, options);

    const int NUM_THREADS_PER_BLOCK = 64;
    const int NUM_BLOCKS = (N + 256 - 1) / 256;
    dim3 block(NUM_THREADS_PER_BLOCK);
    dim3 grid(NUM_BLOCKS);
    histogram_i32x4_kernel<<<grid, block>>>(
        reinterpret_cast<int *>(a.data_ptr()),
        reinterpret_cast<int *>(y.data_ptr()), N);
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32)
    TORCH_BINDING_COMMON_EXTENSION(histogram_i32x4)
}
