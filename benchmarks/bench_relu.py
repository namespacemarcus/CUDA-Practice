"""relu benchmark.

y = max(x, 0)  → 每次迭代读 1 写 1，共 2 * N * dtype_bytes。
以 ``torch.relu`` 为 baseline 计算 speedup。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.bench_utils import bench, load_op, print_table  # noqa: E402

torch.manual_seed(0)
torch.set_grad_enabled(False)

lib = load_op(
    name="relu_lib",
    op_subdir="relu",
    sources=["relu.cu", "torch_bindings.cu"],
)


def _torch_relu(x, out):
    out.copy_(torch.relu(x))


def run_shape(s: int, k: int) -> None:
    n_elems = s * k
    print(f"\n=== shape=({s}, {k})  N={n_elems} ===")

    x32 = torch.randn((s, k), device="cuda", dtype=torch.float32).contiguous()
    y32 = torch.zeros_like(x32)
    bytes_f32 = 2 * n_elems * 4

    f32_results = [
        bench(lib.relu_f32, (x32,), "f32", y32, bytes_f32),
        bench(lib.relu_f32x4, (x32,), "f32x4", y32, bytes_f32),
        bench(_torch_relu, (x32,), "torch.relu f32", y32, bytes_f32),
    ]
    print_table(f32_results, baseline_tag="torch.relu f32")

    x16 = x32.half().contiguous()
    y16 = y32.half().contiguous()
    bytes_f16 = 2 * n_elems * 2

    f16_results = [
        bench(lib.relu_f16, (x16,), "f16", y16, bytes_f16),
        bench(lib.relu_f16x2, (x16,), "f16x2", y16, bytes_f16),
        bench(lib.relu_f16x8, (x16,), "f16x8", y16, bytes_f16),
        bench(lib.relu_f16x8_pack, (x16,), "f16x8_pack", y16, bytes_f16),
        bench(_torch_relu, (x16,), "torch.relu f16", y16, bytes_f16),
    ]
    print_table(f16_results, baseline_tag="torch.relu f16")


def main() -> None:
    for s, k in [(1024, 1024), (2048, 2048), (4096, 4096)]:
        run_shape(s, k)


if __name__ == "__main__":
    main()
