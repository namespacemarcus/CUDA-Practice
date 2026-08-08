import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import load_op, print_row, time_ms, torch

lib = load_op(
    name="matrix_transpose_lib", op_subdir="transpose", sources=["torch_bindings.cu"]
)

FNS = [
    "matrix_transpose_f32_loadcoal",
    "matrix_transpose_f32x4_loadcoal",
    "matrix_transpose_f32x4_loadcoal_smem",
    "matrix_transpose_f32x4_loadcoal_smem_bcf",
    "torch_transpose",
]

SHAPES = [(2048, 4096), (4096, 4096), (4096, 8192), (8192, 4096), (8192, 8192)]


def bench(fn_name, shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros(n, m, device="cuda", dtype=torch.float32).contiguous()

    if fn_name == "torch_transpose":
        fn = lambda x_in, y_out: y_out.copy_(x_in.t())
    else:
        fn = getattr(lib, fn_name)

    return time_ms(fn, x, y)


def main():
    for shape in SHAPES:
        m, n = shape
        print(f"=== shape=({m}, {n}) ===")
        for fn_name in FNS:
            ms = bench(fn_name, shape)
            print_row(fn_name, ms)
        print()


if __name__ == "__main__":
    main()
