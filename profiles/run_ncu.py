import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

import torch

from conftest import load_op


def run(op_subdir, lib_name, sources, kernel_filter, make_input, call, fns, report, metric_set="full"):
    if "--target" in sys.argv:
        lib = load_op(name=lib_name, op_subdir=op_subdir, sources=sources)
        inp = make_input()
        for fn_name in fns:
            call(getattr(lib, fn_name), inp)
        torch.cuda.synchronize()
        return

    os.makedirs(os.path.dirname(report), exist_ok=True)
    target_cmd = [sys.executable, os.path.abspath(sys.argv[0]), "--target"]
    ncu_cmd = [
        "ncu",
        "--set",
        metric_set,
        "--kernel-name",
        f"regex:{kernel_filter}",
        "-o",
        report,
    ] + target_cmd
    print(" ".join(ncu_cmd), flush=True)
    subprocess.run(ncu_cmd)
