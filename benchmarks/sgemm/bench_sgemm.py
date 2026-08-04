import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import torch

from conftest import load_op

lib = load_op(name="sgemm_lib", op_subdir="sgemm", sources=["torch_bindings.cu"])

FNS = [
    "sgemm_naive",
    "sgemm_tiling",
    "sgemm_at_tiling",
    "sgemm_at_tiling_bcf_swizzling",
    "sgemm_at_tiling_bcf_swizzling_cstore",
    "sgemm_at_tiling_bcf_swizzling_cstore_dbf",
    "sgemm_cublas",
    "sgemm_cublas_tf32",
]
SHAPES = [128, 256, 512, 1024, 2048, 4096]
WARMUP = 3
ITERS = 10


def bench(fn_name, n):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    c = torch.zeros(n, n, device="cuda", dtype=torch.float32).contiguous()
    fn = getattr(lib, fn_name)
    for _ in range(WARMUP):
        fn(a, b, c)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(a, b, c)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main():
    for n in SHAPES:
        print(f"=== M=N=K={n} ===")
        for fn_name in FNS:
            ms = bench(fn_name, n)
            gflops = 2 * n ** 3 / (ms * 1e-3) / 1e9
            print(f"  {fn_name:50s} {ms:8.3f} ms  {gflops:8.1f} GFLOPS")
        print()


if __name__ == "__main__":
    main()
