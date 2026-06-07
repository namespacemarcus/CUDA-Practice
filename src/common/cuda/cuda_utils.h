#pragma once

#define WARP_SIZE          32
#define INT4(value)        (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value)      (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value)       (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value)     (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
