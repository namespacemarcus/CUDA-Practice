#include "../common/defs.h"
#include "../common/tensor_utils.h"
#include <cuda_fp16.h>

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t value);

template <>
__device__ __forceinline__ float scalar_to_float<float>(float value) {
    return value;
}

template <>
__forceinline__ float scalar_to_float<half>(half value) {
    return __half2float(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float value);

template <>
__device__ __forceinline__ float float_to_scalar<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ half float_to_scalar<half>(float value) {
    return __float2half(value);
}

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, stride);
    }
    return value;
}

void check_cuda(cudaError_t error, const char *operation) {
    TORCH_CHECK(error == cudaSuccess, operation, ": ",
                cudaGetErrorString(error));
}

class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int device)
        : previous_device_(0),
          device_changed_(false) {
        check_cuda(cudaGetDevice(&previous_device_),
                   "Failed to get current CUDA device");
        if (device != previous_device_) {
            check_cuda(cudaSetDevice(device), "Failed to set CUDA device");
            device_changed_ = true;
        }
    }

    ~CudaDeviceGuard() {
        if (device_changed_) {
            check_cuda(cudaSetDevice(previous_device_),
                       "Failed to restore previous CUDA device");
        }
    }

    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &operator=(const CudaDeviceGuard &) = delete;

private:
    int previous_device_;
    bool device_changed_;
};
