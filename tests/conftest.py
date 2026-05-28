import os
from typing import Sequence

from torch.utils.cpp_extension import load

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

_CUDA_FLAGS = [
    "-O3",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
]
_CFLAGS = ["-std=c++17"]


def load_op(name: str, op_subdir: str, sources: Sequence[str]):
    """JIT-compile an op under src/ops/<op_dir>"""
    op_dir = os.path.join(SRC_DIR, "ops", op_subdir)
    full_sources = [os.path.join(op_dir, s) for s in sources]
    return load(
        name=name,
        sources=full_sources,
        extra_cuda_cflags=_CUDA_FLAGS,
        extra_cflags=_CFLAGS,
    )
