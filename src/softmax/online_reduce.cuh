#include "../common/cuda/cuda_utils.h"

struct __align__(8) MD {
    float m; // current partial max
    float d; // current partial sum of exp(x−m)
};

template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value) {
    unsigned int mask = 0xffffffff;
#pragma unroll
    for (int stride = kWarpSize >> 1; stride >= 1; stride >>= 1) {
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger = value_bigger ? value : other;
        MD smaller = value_bigger ? other : value;

        value.m = bigger.m;
        value.d = bigger.d + smaller.d * expf(smaller.m - bigger.m);
    }
    return value;
}
