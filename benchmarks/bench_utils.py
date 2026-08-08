import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests")
)

import torch
from conftest import load_op

WARMUP = 3
ITERS = 10


def time_ms(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def dtype_for(fn_name):
    return torch.float16 if "f16" in fn_name else torch.float32


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def gflops(flops, ms):
    return flops / (ms * 1e-3) / 1e9


def print_row(name, ms, throughput=None, unit="GFLOPS"):
    if throughput is None:
        print(f"  {name:50s} {ms:8.3f} ms")
    else:
        print(f"  {name:50s} {ms:8.3f} ms  {throughput:8.1f} {unit}")
