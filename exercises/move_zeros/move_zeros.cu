#include <iostream>
#include <vector>

constexpr int BLOCK_SIZE = 256;

__global__ void build_flags(const int *input, int *flags, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        flags[index] = input[index] != 0 ? 1 : 0;
    }
}

// Hillis-Steele Scan
__global__ void scan_step(const int *input, int *output, int offset, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    int value = input[index];
    if (index >= offset) {
        value += input[index - offset];
    }

    output[index] = value;
}

__global__ void scatter_non_zeros(const int *input, const int *prefix_sum,
                                  int *output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }

    int non_zero_count = prefix_sum[n - 1];

    if (input[index] != 0) {
        int target_index = prefix_sum[index] - 1;
        output[target_index] = input[index];
    }

    if (index >= non_zero_count) {
        output[index] = 0;
    }
}

void move_zeros_to_end(int *array, int n) {
    if (array == nullptr || n <= 1) {
        return;
    }

    int *buffer1 = nullptr;
    int *buffer2 = nullptr;
    int *output = nullptr;

    cudaMalloc(&buffer1, n * sizeof(int));
    cudaMalloc(&buffer2, n * sizeof(int));
    cudaMalloc(&output, n * sizeof(int));

    int block_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    build_flags<<<block_count, BLOCK_SIZE>>>(array, buffer1, n);

    // inclusive scan
    int *scan_input = buffer1;
    int *scan_output = buffer2;
    for (int offset = 1; offset < n; offset *= 2) {
        scan_step<<<block_count, BLOCK_SIZE>>>(scan_input, scan_output, offset,
                                               n);
        std::swap(scan_input, scan_output);
    }

    scatter_non_zeros<<<block_count, BLOCK_SIZE>>>(array, scan_input, output,
                                                   n);
    cudaMemcpy(array, output, n * sizeof(int), cudaMemcpyDeviceToDevice);

    cudaFree(buffer1);
    cudaFree(buffer2);
    cudaFree(output);
}

int main() {
    std::vector<int> host_array{0, 1, 0, 3, 12, 0, 5};
    int n = host_array.size();

    int *device_array = nullptr;
    cudaMalloc(&device_array, n * sizeof(int));

    cudaMemcpy(device_array, host_array.data(), n * sizeof(int),
               cudaMemcpyHostToDevice);

    move_zeros_to_end(device_array, n);

    cudaMemcpy(host_array.data(), device_array, n * sizeof(int),
               cudaMemcpyDeviceToHost);

    for (int value: host_array) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    cudaFree(device_array);

    return 0;
}
