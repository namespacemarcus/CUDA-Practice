"""Shared helpers for per-op ncu profiling.

Each op has its own folder profiles/<op>/profile.py defining a CONFIG dict
(op_subdir, sources, kernel_filter, default_fn, default_n, make_input, call)
and calling run_op(CONFIG, __file__). This module does the rest.
"""
import argparse
import os
import subprocess
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "..", "tests")))

from conftest import load_op


def load_lib(op_subdir, sources, name=None):
    if name is None:
        name = f"{op_subdir.replace('-', '_')}_lib"
    return load_op(name=name, op_subdir=op_subdir, sources=sources)


def profile_with_ncu(target_script, kernel_filter, report_path, fn, n, warmup=3, metric_set="full"):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    target_cmd = [
        sys.executable,
        target_script,
        "--target",
        "--fn",
        fn,
        "--n",
        str(n),
        "--warmup",
        str(warmup),
    ]
    ncu_cmd = [
        "ncu",
        "--set",
        metric_set,
        "--kernel-name",
        f"regex:{kernel_filter}",
        "--launch-skip",
        str(warmup),
        "--launch-count",
        "1",
        "--target-processes",
        "all",
        "-o",
        report_path,
    ] + target_cmd
    print(" ".join(ncu_cmd), flush=True)
    completed = subprocess.run(ncu_cmd)
    if completed.returncode != 0 or not os.path.exists(report_path):
        print("\nncu did not produce a report.")
        print("If you saw ERR_NVGPUCTRPERM, enable profiling once:")
        print(
            "  echo 'options nvidia NVreg_RestrictProfilingAdminOnly=0' "
            "| sudo tee /etc/modprobe.d/nvidia-profiling.conf"
        )
        print(
            "  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia "
            "&& sudo modprobe nvidia"
        )
        print("or rerun with sudo.")
        sys.exit(1)
    print(f"\nReport: {report_path}")
    print(f'Open with: ncu-ui "{report_path}"')


def run_op(config, script_path):
    parser = argparse.ArgumentParser(
        description=f"Profile {config['op_subdir']} kernels with ncu."
    )
    parser.add_argument("--fn", default=config["default_fn"])
    parser.add_argument("--n", type=int, default=config["default_n"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--set", default="full")
    parser.add_argument(
        "--target",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.target:
        lib = load_lib(config["op_subdir"], config["sources"])
        fn = getattr(lib, args.fn)
        inp = config["make_input"](args.fn, args.n)
        for _ in range(args.warmup):
            config["call"](fn, inp)
        torch.cuda.synchronize()
        config["call"](fn, inp)
        torch.cuda.synchronize()
        return

    report = os.path.join(
        os.path.dirname(os.path.abspath(script_path)),
        f"{args.fn}_n{args.n}.ncu-rep",
    )
    profile_with_ncu(
        script_path,
        config["kernel_filter"],
        report,
        args.fn,
        args.n,
        args.warmup,
        args.set,
    )
