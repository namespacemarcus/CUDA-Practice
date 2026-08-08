import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import dtype_for, load_op, print_row, time_ms, torch

lib = load_op(
    name="elementwise_add_lib",
    op_subdir="elementwise",
    sources=["elementwise_add_kernel.cu", "torch_bindings.cu"],
)

FNS = [
    "elementwise_add_f32",
    "elementwise_add_f32x4",
    "elementwise_add_f16",
    "elementwise_add_f16x2",
    "elementwise_add_f16x8",
    "elementwise_add_f16x8_pack",
    "torch_add",
]

SHAPES = [(4096, 4096), (8192, 4096), (8192, 8192), (16384, 4096)]


def bench(fn_name, shape):
    dt = dtype_for(fn_name)
    a = torch.randn(shape, device="cuda", dtype=dt).contiguous()
    b = torch.randn(shape, device="cuda", dtype=dt).contiguous()
    out = torch.zeros_like(a)

    if fn_name == "torch_add":
        fn = lambda a_in, b_in, o: torch.add(a_in, b_in, out=o)
    else:
        fn = getattr(lib, fn_name)

    return time_ms(fn, a, b, out)


def main():
    for shape in SHAPES:
        print(f"=== shape={shape} ({shape[0] * shape[1]} elems) ===")
        for fn_name in FNS:
            ms = bench(fn_name, shape)
            print_row(fn_name, ms)
        print()


if __name__ == "__main__":
    main()
