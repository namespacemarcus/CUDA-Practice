#include "../common/cuda/cuda_utils.h"

/*
transpose:
(i, j) -> x[i * col + j]
(j, i) -> y[j * row + i]
*/

__global__ void matrix_transpose_f32_loadcoal_kernel(float *x, float *y,
                                                     const int row,
                                                     const int col) {
    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_idx < row * col) {
        y[global_col * row + global_row] = x[global_idx];
    }
}

__global__ void matrix_transpose_f32x4_loadcoal_kernel(float *x, float *y,
                                                       const int row,
                                                       const int col) {
    const int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int global_row = global_idx / col;
    const int global_col = global_idx % col;
    if (global_row < row && global_col + 3 < col) {
        float4 reg_x = FLOAT4(x[global_idx]);
        y[(global_col + 0) * row + global_row] = reg_x.x;
        y[(global_col + 1) * row + global_row] = reg_x.y;
        y[(global_col + 2) * row + global_row] = reg_x.z;
        y[(global_col + 3) * row + global_row] = reg_x.w;
    }
}

// row == col
__global__ void matrix_transpose_f32_loadcoal_2d_kernel(float *x, float *y,
                                                        const int row,
                                                        const int col) {
    const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x < col && global_y < row) {
        y[global_x * row + global_y] = x[global_y * col + global_x];
    }
}

__global__ void matrix_transpose_f32x4_loadcoal_2d_kernel(float *x, float *y,
                                                          const int row,
                                                          const int col) {
    const int global_x = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (global_x + 3 < col && global_y < row) {
        float4 reg_x = FLOAT4(x[global_y * col + global_x]);
        y[(global_x + 0) * row + global_y] = reg_x.x;
        y[(global_x + 1) * row + global_y] = reg_x.y;
        y[(global_x + 2) * row + global_y] = reg_x.z;
        y[(global_x + 3) * row + global_y] = reg_x.w;
    }
}
