#include "../common/pack.h"

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

constexpr int kTileSize = 32;

__global__ void matrix_transpose_f32x4_loadcoal_smem_kernel(float *x, float *y,
                                                            const int row,
                                                            const int col) {
    __shared__ float tile[32][32];

    const int tid = threadIdx.x;

    const int sx = (tid % 8) * 4;
    const int sy = tid / 8;

    const int tiles_x = (col + 32 - 1) / 32;
    const int bx = blockIdx.x % tiles_x;
    const int by = blockIdx.x / tiles_x;

    const int input_x = bx * 32 + sx;
    const int input_y = by * 32 + sy;

    const bool read_full =
        (bx + 1) * 32 <= col && (by + 1) * 32 <= row && col % 4 == 0;
    if (read_full) {
        const float4 value = FLOAT4(x[input_y * col + input_x]);
        tile[sy][sx + 0] = value.x;
        tile[sy][sx + 1] = value.y;
        tile[sy][sx + 2] = value.z;
        tile[sy][sx + 3] = value.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (input_y < row && input_x + i < col) {
                tile[sy][sx + i] = x[input_y * col + input_x + i];
            }
        }
    }
    __syncthreads();

    // transpose: tile (bx, by) -> tile (by, bx)
    const int output_x = by * 32 + sx;
    const int output_y = bx * 32 + sy;

    const bool write_full =
        (by + 1) * 32 <= row && (bx + 1) * 32 <= col && row % 4 == 0;
    if (write_full) {
        float4 value;
        value.x = tile[sx + 0][sy];
        value.y = tile[sx + 1][sy];
        value.z = tile[sx + 2][sy];
        value.w = tile[sx + 3][sy];

        FLOAT4(y[output_y * row + output_x]) = value;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (output_y < col && output_x + i < row) {
                y[output_y * row + output_x + i] = tile[sx + i][sy];
            }
        }
    }
}

__global__ void matrix_transpose_f32x4_loadcoal_smem_bcf_kernel(float *x,
                                                                float *y,
                                                                const int row,
                                                                const int col) {
    __shared__ float tile[32][33];

    const int tid = threadIdx.x;

    const int sx = tid % 8;
    const int sy = tid / 8;

    const int tiles_x = (col + 32 - 1) / 32;
    const int bx = blockIdx.x % tiles_x;
    const int by = blockIdx.x / tiles_x;

    const int input_x = bx * 32 + sx * 4;
    const int input_y = by * 32 + sy;

    const bool read_full =
        (bx + 1) * 32 <= col && (by + 1) * 32 <= row && col % 4 == 0;

    if (read_full) {
        const float4 value = FLOAT4(x[input_y * col + input_x]);

        tile[sy][sx * 4 + 0] = value.x;
        tile[sy][sx * 4 + 1] = value.y;
        tile[sy][sx * 4 + 2] = value.z;
        tile[sy][sx * 4 + 3] = value.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (input_y < row && input_x + i < col) {
                tile[sy][sx * 4 + i] = x[input_y * col + input_x + i];
            }
        }
    }
    __syncthreads();

    const int output_x = by * 32 + sx * 4;
    const int output_y = bx * 32 + sy;

    const bool write_full =
        (by + 1) * 32 <= row && (bx + 1) * 32 <= col && row % 4 == 0;

    if (write_full) {
        float4 value;

        value.x = tile[sx * 4 + 0][sy];
        value.y = tile[sx * 4 + 1][sy];
        value.z = tile[sx * 4 + 2][sy];
        value.w = tile[sx * 4 + 3][sy];

        FLOAT4(y[output_y * row + output_x]) = value;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (output_y < col && output_x + i < row) {
                y[output_y * row + output_x + i] = tile[sx * 4 + i][sy];
            }
        }
    }
}

// when row == col, 2d block.
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
