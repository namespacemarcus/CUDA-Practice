import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from run_ncu import run


def make_input():
    n = 4096
    a = torch.randn(n, n, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(n, n, device="cuda", dtype=torch.float16).contiguous()
    c = torch.zeros(n, n, device="cuda", dtype=torch.float16).contiguous()
    return a, b, c


def call(fn, inp):
    fn(inp[0], inp[1], inp[2])


run(
    op_subdir="hgemm",
    lib_name="hgemm_lib",
    sources=["torch_bindings.cu"],
    kernel_filter="gemm",
    make_input=make_input,
    call=call,
    fns=[
        "hgemm_gw_tiled",
        "hgemm_gw_tiled_bcf",
        "hgemm_gw_tiled_bcf_dbf",
        "hgemm_gw_tiled_bcf_dbf_cstore",
        "hgemm_cublas",
    ],
    report=os.path.join(os.path.dirname(__file__), "ncu_reports", "hgemm.ncu-rep"),
)
