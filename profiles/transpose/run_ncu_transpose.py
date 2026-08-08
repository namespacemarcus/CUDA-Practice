import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from run_ncu import run


def make_input():
    m, n = 4096, 4096
    x = torch.randn(m, n, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros(n, m, device="cuda", dtype=torch.float32).contiguous()
    return x, y


def call(fn, inp):
    fn(inp[0], inp[1])


run(
    op_subdir="transpose",
    lib_name="matrix_transpose_lib",
    sources=["torch_bindings.cu"],
    kernel_filter="matrix_transpose",
    make_input=make_input,
    call=call,
    fns=[
        "matrix_transpose_f32_loadcoal",
        "matrix_transpose_f32x4_loadcoal",
        "matrix_transpose_f32x4_loadcoal_smem",
        "matrix_transpose_f32x4_loadcoal_smem_bcf",
        "matrix_transpose_f32_loadcoal_2d",
        "matrix_transpose_f32x4_loadcoal_2d",
    ],
    report=os.path.join(os.path.dirname(__file__), "ncu_reports", "transpose.ncu-rep"),
)
