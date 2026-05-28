import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch
from torch.utils.cpp_extension import load

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

DEFAULT_CUDA_FLAGS: List[str] = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]
DEFAULT_CFLAGS: List[str] = ["-std=c++17"]


def load_op(name: str, op_subdir: str, sources: Sequence[str]):
    """JIT-compile an op under src/ops/<op_dir>"""
    op_dir = os.path.join(SRC_DIR, "ops", op_subdir)
    full_sources = [os.path.join(op_dir, s) for s in sources]
    return load(
        name=name,
        sources=full_sources,
        extra_cuda_cflags=DEFAULT_CUDA_FLAGS,
        extra_cflags=DEFAULT_CFLAGS,
    )


@dataclass
class BenchResult:
    tag: str
    median_ms: float
    gb_per_s: Optional[float]


def bench(
    fn: Callable,
    inputs: Sequence[torch.Tensor],
    tag: str,
    out: Optional[torch.Tensor] = None,
    bytes_per_iter: Optional[int] = None,
    warmup: int = 20,
    iters: int = 100,
) -> BenchResult:

    def _call():
        if out is not None:
            fn(*inputs, out)
        else:
            return fn(*inputs)

    for _ in range(warmup):
        _call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        _call()
        ends[i].record()
    torch.cuda.synchronize()

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times[len(times) // 2]

    gb_per_s = None
    if bytes_per_iter is not None and median_ms > 0:
        gb_per_s = bytes_per_iter / (median_ms * 1e-3) / 1e9

    return BenchResult(tag=tag, median_ms=median_ms, gb_per_s=gb_per_s)


def print_table(results: Sequence[BenchResult], baseline_tag: str) -> None:
    base = next((r for r in results if r.tag == baseline_tag), None)
    base_ms = base.median_ms if base is not None else None

    print(f"  {'kernel':<22} {'median(ms)':>12} {'GB/s':>10} {'vs baseline':>14}")
    print("  " + "-" * 60)
    for r in results:
        bw = f"{r.gb_per_s:9.1f}" if r.gb_per_s is not None else "      n/a"
        if base_ms and r.median_ms > 0:
            speedup = base_ms / r.median_ms
            sp = f"{speedup:6.2f}x"
            if r.tag == baseline_tag:
                sp += " *"
        else:
            sp = "      -"
        print(f"  {r.tag:<22} {r.median_ms:12.5f} {bw} {sp:>14}")
