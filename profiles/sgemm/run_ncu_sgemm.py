import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from run_ncu import run


def make_input():
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    c = torch.zeros(n, n, device="cuda", dtype=torch.float32).contiguous()
    return a, b, c


def call(fn, inp):
    fn(inp[0], inp[1], inp[2])


run(
    op_subdir="sgemm",
    lib_name="sgemm_lib",
    sources=["torch_bindings.cu"],
    kernel_filter="sgemm",
    make_input=make_input,
    call=call,
    fns=[
        "sgemm_naive",
        "sgemm_tiling",
        "sgemm_at_tiling",
        "sgemm_at_tiling_bcf_swizzling",
        "sgemm_at_tiling_bcf_swizzling_cstore",
        "sgemm_at_tiling_bcf_swizzling_cstore_dbf",
        "sgemm_cublas",
        "sgemm_cublas_tf32",
    ],
    report=os.path.join(os.path.dirname(__file__), "ncu_reports", "sgemm.ncu-rep"),
)
