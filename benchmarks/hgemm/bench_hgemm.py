import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))

import torch

from conftest import load_op

lib = load_op(name="hgemm_lib", op_subdir="hgemm", sources=["torch_bindings.cu"])

FNS = [
    "hgemm_gw_tiled",
    "hgemm_gw_tiled_bcf",
    "hgemm_gw_tiled_bcf_dbf",
    "hgemm_gw_tiled_bcf_dbf_cstore",
    "hgemm_cublas",
    "torch_hgemm",
]

SHAPES = [1024, 2048, 4096, 8192]
WARMUP = 3
ITERS = 10


def bench(fn_name, n):
    a = torch.randn(n, n, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(n, n, device="cuda", dtype=torch.float16).contiguous()
    c = torch.zeros(n, n, device="cuda", dtype=torch.float16).contiguous()

    if fn_name == "torch_hgemm":
        fn = lambda a_in, b_in, c_out: torch.mm(a_in, b_in, out=c_out)
    else:
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
            gflops = 2 * (n**3) / (ms * 1e-3) / 1e9
            print(f"  {fn_name:50s} {ms:8.3f} ms  {gflops:8.1f} GFLOPS")
        print()


if __name__ == "__main__":
    main()
