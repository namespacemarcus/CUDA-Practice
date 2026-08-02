import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import run_op

OP_SUBDIR = "transpose"
SOURCES = ["torch_bindings.cu"]
KERNEL_FILTER = "matrix_transpose"
DEFAULT_FN = "matrix_transpose_f32_loadcoal_2d"
DEFAULT_N = 64


def make_input(fn_name, n):
    x = torch.randn(n, n, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros(n, n, device="cuda", dtype=torch.float32).contiguous()
    return x, y


def call(fn, inp):
    fn(inp[0], inp[1])


CONFIG = {
    "op_subdir": OP_SUBDIR,
    "sources": SOURCES,
    "kernel_filter": KERNEL_FILTER,
    "default_fn": DEFAULT_FN,
    "default_n": DEFAULT_N,
    "make_input": make_input,
    "call": call,
}


if __name__ == "__main__":
    run_op(CONFIG, __file__)
