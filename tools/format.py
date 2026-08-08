#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def collect_files(base_path, extensions):
    files = []
    for sub in ("src", "tests", "profiles", "benchmarks"):
        root = os.path.join(base_path, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _, names in os.walk(root):
            for name in names:
                if any(name.endswith(ext) for ext in extensions):
                    files.append(os.path.join(dirpath, name))
    return sorted(files)


def resolve_black():
    if shutil.which("black"):
        return ["black"]
    probe = subprocess.run(
        [sys.executable, "-m", "black", "--version"], capture_output=True
    )
    if probe.returncode == 0:
        return [sys.executable, "-m", "black"]
    return None


def resolve_clang_format():
    if shutil.which("clang-format"):
        return ["clang-format"]
    return None


def run(label, cmd, files):
    if not files:
        print(f"[{label}] no files found")
        return 0
    print(f"[{label}] {len(files)} file(s)")
    return subprocess.run(cmd + files).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="dry-run; exit non-zero if any file would be reformatted",
    )
    args = ap.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_path, ".."))

    rc = 0

    cu_files = collect_files(repo_root, (".cu", ".cuh"))
    cf = resolve_clang_format()
    if cf is None:
        print(
            "[clang-format] not found; install e.g. `conda install clang-format`",
            file=sys.stderr,
        )
        rc = 1
    else:
        cmd = cf + (["--dry-run", "--Werror"] if args.check else ["-i"])
        rc |= run("clang-format", cmd, cu_files)

    py_files = collect_files(repo_root, (".py",))
    bk = resolve_black()
    if bk is None:
        print("[black] not found; install e.g. `pip install black`", file=sys.stderr)
        rc = 1
    else:
        cmd = bk + (["--check", "--diff"] if args.check else [])
        rc |= run("black", cmd, py_files)

    return rc


if __name__ == "__main__":
    sys.exit(main())
