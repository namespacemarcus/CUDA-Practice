import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import gflops, load_op, print_row, time_ms, torch

lib = load_op(name="sgemv_lib", op_subdir="sgemv", sources=["torch_bindings.cu"])

CONFIGS = [
    (131072, 16, ["sgemv_k16_f32", "torch_sgemv"]),
    (262144, 16, ["sgemv_k16_f32", "torch_sgemv"]),
    (262140, 128, ["sgemv_k32_f32", "sgemv_k128_f32x4", "torch_sgemv"]),
    (131072, 512, ["sgemv_k32_f32", "sgemv_k128_f32x4", "torch_sgemv"]),
    (65536, 2048, ["sgemv_k32_f32", "sgemv_k128_f32x4", "torch_sgemv"]),
]


def bench(fn_name, m, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.float32).contiguous()
    x = torch.randn(k, 1, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(m, 1, device="cuda", dtype=torch.float32)

    if fn_name == "torch_sgemv":
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
