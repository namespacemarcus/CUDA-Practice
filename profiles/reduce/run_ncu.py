import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from common import run_op

OP_SUBDIR = "reduce"
SOURCES = ["torch_bindings.cu"]
KERNEL_FILTER = "block_all_reduce_sum"
DEFAULT_FN = "block_all_reduce_sum_f32x4_f32"
DEFAULT_N = 4096


def make_input(fn_name, n):
    if "fp8_e4m3" in fn_name:
        return (torch.rand(n, device="cuda", dtype=torch.float32) + 0.5).to(
            torch.float8_e4m3fn
        )
    if "fp8_e5m2" in fn_name:
        return (torch.rand(n, device="cuda", dtype=torch.float32) + 0.5).to(
            torch.float8_e5m2
        )
    if "bf16" in fn_name:
        return torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
    if "f16" in fn_name:
        return torch.rand(n, device="cuda", dtype=torch.float16) + 0.5
    if "i8" in fn_name:
        return torch.randint(-8, 8, (n,), device="cuda", dtype=torch.int8)
    return torch.rand(n, device="cuda", dtype=torch.float32) + 0.5


def call(fn, inp):
    fn(inp)


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
