import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import run_op

OP_SUBDIR = "flash-attention"
SOURCES = ["torch_bindings.cu"]
KERNEL_FILTER = "flash_attention"
DEFAULT_FN = "flash_attention_naive"
DEFAULT_N = 64


def make_input(fn_name, n):
    q = torch.randn(1, 2, n, 64, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    return q, k, v


def call(fn, inp):
    fn(inp[0], inp[1], inp[2], causal=False)


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
