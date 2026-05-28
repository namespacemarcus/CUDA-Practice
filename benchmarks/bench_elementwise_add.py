"""element-wise add benchmark.

c = a + b  → 每次迭代读 2 写 1，共 3 * N * dtype_bytes。
以 ``torch.add`` 为 baseline 计算 speedup。
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.bench_utils import bench, load_op, print_table  # noqa: E402

torch.manual_seed(0)
torch.set_grad_enabled(False)

lib = load_op(
    name="elementwise_add_lib",
    op_subdir="elementwise",
    sources=["elementwise_add.cu", "torch_bindings.cu"],
)


def _torch_add(a, b, out):
    torch.add(a, b, out=out)


def run_shape(s: int, k: int) -> None:
    n_elems = s * k
    print(f"\n=== shape=({s}, {k})  N={n_elems} ===")

    a32 = torch.randn((s, k), device="cuda", dtype=torch.float32).contiguous()
    b32 = torch.randn_like(a32)
    out32 = torch.zeros_like(a32)
    bytes_f32 = 3 * n_elems * 4

    f32_results = [
        bench(lib.elementwise_add_f32, (a32, b32), "f32", out32, bytes_f32),
        bench(lib.elementwise_add_f32x4, (a32, b32), "f32x4", out32, bytes_f32),
        bench(_torch_add, (a32, b32), "torch.add f32", out32, bytes_f32),
    ]
    print_table(f32_results, baseline_tag="torch.add f32")

    a16 = a32.half().contiguous()
    b16 = b32.half().contiguous()
    out16 = out32.half().contiguous()
    bytes_f16 = 3 * n_elems * 2

    f16_results = [
        bench(lib.elementwise_add_f16, (a16, b16), "f16", out16, bytes_f16),
        bench(lib.elementwise_add_f16x2, (a16, b16), "f16x2", out16, bytes_f16),
        bench(lib.elementwise_add_f16x8, (a16, b16), "f16x8", out16, bytes_f16),
        bench(
            lib.elementwise_add_f16x8_pack, (a16, b16), "f16x8_pack", out16, bytes_f16
        ),
        bench(_torch_add, (a16, b16), "torch.add f16", out16, bytes_f16),
    ]
    print_table(f16_results, baseline_tag="torch.add f16")


def main() -> None:
    for s, k in [(1024, 1024), (2048, 2048), (4096, 4096)]:
        run_shape(s, k)


if __name__ == "__main__":
    main()
