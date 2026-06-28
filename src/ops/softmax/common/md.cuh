#pragma once

#include <cuda_runtime.h>

struct __align__(8) MD {
    float m; // current partial max
    float d; // current partial sum of exp(x−m)
};
