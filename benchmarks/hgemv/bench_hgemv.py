import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import gflops, load_op, print_row, time_ms, torch

lib = load_op(name="hgemv_lib", op_subdir="hgemv", sources=["torch_bindings.cu"])

CONFIGS = [
    (131072, 16, ["hgemv_k16_f16", "torch_hgemv"]),
    (262144, 16, ["hgemv_k16_f16", "torch_hgemv"]),
    (262140, 128, ["hgemv_k32_f16", "hgemv_k128_f16x4", "torch_hgemv"]),
    (131072, 512, ["hgemv_k32_f16", "hgemv_k128_f16x4", "torch_hgemv"]),
    (65536, 2048, ["hgemv_k32_f16", "hgemv_k128_f16x4", "torch_hgemv"]),
]


def bench(fn_name, m, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float16).contiguous()
    x = torch.randn(k, 1, device="cuda", dtype=torch.float16).contiguous()
    out = torch.zeros(m, 1, device="cuda", dtype=torch.float16)

    if fn_name == "torch_hgemv":
        fn = lambda a_in, x_in, o: torch.mm(a_in, x_in, out=o)
    else:
        fn = getattr(lib, fn_name)

    ms = time_ms(fn, a, x, out)
    return ms, gflops(2 * m * k, ms)


def main():
    for m, k, fns in CONFIGS:
        print(f"=== M={m}, K={k} ===")
        for fn_name in fns:
            ms, flops = bench(fn_name, m, k)
            print_row(fn_name, ms, flops, unit="GFLOPS")
        print()


if __name__ == "__main__":
    main()
