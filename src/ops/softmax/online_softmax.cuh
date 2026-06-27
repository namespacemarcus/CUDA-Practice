#include "./common/reduce.cuh"
#include <float.h>

template <const int NUM_THREADS = 256>
__global__ void online_safe_softmax_f32_per_token_kernel(const float *x,
                                                         float *y, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    MD val;
    val.m = (idx < N) ? x[idx] : -FLI_MAX;
    val.d = (idx < N) ? 1.0f : 0.0f;

    __shared__ MD shared[NUM_WARPS];
    MD res = warp_reduce_md_op<WARP_SIZE>(val);
    if (laneId == 0) {
        shared[warpId] = res;
    }
    __syncthreads();

    if (warpId == 0) {
        MD block_res =
            threadIdx.x < NUM_WARPS ? shared[threadIdx.x] : MD{-FLT_MAX, 0.0f};
        block_res = warp_reduce_md_op<NUM_WARPS>(block_res);
        if (threadIdx.x == 0) {
            shared[0] = block_res;
        }
    }
    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (idx < N) {
        y[idx] = expf(x[idx] - final_res.m) * d_total_inverse;
    }
}

template <const int NUM_THREADS = 256 / 4>
__global__ void
online_safe_softmax_f32x4_pack_per_token_kernel(float *x, float *y, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    float4 val = FLOAT4(x[idx]);
    float local_m = fmaxf(fmaxf(val.x, val.y), fmaxf(val.z, val.w));
    float local_d = expf(val.x - local_m) + expf(val.y - local_m) +
                    expf(val.z - local_m) + expf(val.w - local_m);
    MD local_md = {local_m, local_d};

    MD res = warp_reduce_md_op<WARP_SIZE>(local_md);
    __shared__ MD shared[NUM_WARPS];
    if (laneId == 0) {
        shared[warpId] = res;
    }
    __syncthreads();

    if (warpId == 0) {
        MD block_res =
            threadIdx.x < NUM_WARPS ? shared[threadidx.x] : MD{-FLT_MAX, 0.0f};
        block_res = warp_reduce_md_op<NUM_WARPS>(block_res);
        if (threadIdx.x == 0) {
            shared[0] = block_res;
        }
    }
    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d);
    if (idx < N) {
        float4 reg_y;
        reg_y.x = expf(val.x - final_res.m) * d_total_inverse;
        reg_y.y = expf(val.y - final_res.m) * d_total_inverse;
        reg_y.z = expf(val.z - final_res.m) * d_total_inverse;
        reg_y.w = expf(val.w - final_res.m) * d_total_inverse;
        FLOAT4(y[idx]) = reg_y;
    }
}
