#include "rope_kernel.cuh"

#define BLOCK_SIZE 256

void rope_f32(torch::Tensor x, torch::Tensor out) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
    int seq_len = x.size(0);
    int hidden_size = x.size(1);
    int pairs_per_token = (int)(hidden_size / 2);
    dim3 grid(seq_len);
    dim3 block(pairs_per_token);
    rope_f32_kernel<<<grid, block>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                     seq_len, pairs_per_token);
}

void rope_f32x4(torch::Tensor x, torch::Tensor out) {
    CHECK_TORCH_TENSOR_DTYPE(x, torch::kFloat32)
    CHECK_TORCH_TENSOR_DTYPE(out, torch::kFloat32)
    int seq_len = x.size(0);
    int hidden_size = x.size(1);
    int quarter_dim = hidden_size / 4;
    dim3 grid(seq_len);
    dim3 block(quarter_dim);
    rope_f32x4_kernel<<<grid, block>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), seq_len, quarter_dim);
}
