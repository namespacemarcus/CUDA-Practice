import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import dtype_for, load_op, print_row, time_ms, torch

lib = load_op(name="sigmoid_lib", op_subdir="sigmoid", sources=["torch_bindings.cu"])

FNS = [
    "sigmoid_f32",
    "sigmoid_f32x4",
    "sigmoid_f16",
    "sigmoid_f16x2",
    "sigmoid_f16x8",
    "sigmoid_f16x8_pack",
    "torch_sigmoid",
]

SHAPES = [(4096, 4096), (8192, 4096), (8192, 8192), (16384, 4096)]


def bench(fn_name, shape):
    dt = dtype_for(fn_name)
    x = torch.randn(shape, device="cuda", dtype=dt).contiguous()
    y = torch.zeros_like(x)

    if fn_name == "torch_sigmoid":
        fn = lambda x_in, y_out: torch.sigmoid(x_in, out=y_out)
    else:
        fn = getattr(lib, fn_name)

    return time_ms(fn, x, y)


def main():
    for shape in SHAPES:
        print(f"=== shape={shape} ({shape[0] * shape[1]} elems) ===")
        for fn_name in FNS:
            ms = bench(fn_name, shape)
            print_row(fn_name, ms)
        print()


if __name__ == "__main__":
    main()
