import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import dtype_for, load_op, print_row, time_ms, torch

lib = load_op(
    name="dot_product_lib", op_subdir="dotproduct", sources=["torch_bindings.cu"]
)

FNS = [
    "dot_product_f32_f32",
    "dot_product_f32x4_f32",
    "dot_product_f16_f32",
    "dot_product_f16x2_f32",
    "dot_product_f16x8_pack_f32",
    "torch_dot",
]

SIZES = [16777216, 33554432, 67108864, 134217728]


def bench(fn_name, n):
    dt = dtype_for(fn_name)
    a = torch.randn(n, device="cuda", dtype=dt).contiguous()
    b = torch.randn(n, device="cuda", dtype=dt).contiguous()

    if fn_name == "torch_dot":
        fn = lambda a_in, b_in: torch.dot(a_in, b_in)
    else:
        fn = getattr(lib, fn_name)

    return time_ms(fn, a, b)


def main():
    for n in SIZES:
        print(f"=== N={n} ===")
        for fn_name in FNS:
            ms = bench(fn_name, n)
            print_row(fn_name, ms)
        print()


if __name__ == "__main__":
    main()
