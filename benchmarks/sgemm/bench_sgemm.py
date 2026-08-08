import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import gflops, load_op, print_row, time_ms, torch

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


def bench(fn_name, n):
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    c = torch.zeros(n, n, device="cuda", dtype=torch.float32).contiguous()
    fn = getattr(lib, fn_name)
    ms = time_ms(fn, a, b, c)
    return ms, gflops(2 * n**3, ms)


def main():
    for n in SHAPES:
        print(f"=== M=N=K={n} ===")
        for fn_name in FNS:
            ms, flops = bench(fn_name, n)
            print_row(fn_name, ms, flops, unit="GFLOPS")
        print()


if __name__ == "__main__":
    main()
