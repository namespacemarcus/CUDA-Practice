import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_utils import dtype_for, load_op, print_row, time_ms, torch

lib = load_op(name="rmsnorm_lib", op_subdir="rmsnorm", sources=["torch_bindings.cu"])

EPS = 1e-5
GAMMA = 2.0

FNS = [
    "rms_norm_f32",
    "rms_norm_f32x4",
    "rms_norm_f16_f16",
    "rms_norm_f16x2_f16",
    "rms_norm_f16x8_f16",
    "rms_norm_f16x8_pack_f16",
    "rms_norm_f16_f32",
    "rms_norm_f16x8_f32",
    "rms_norm_f16x8_pack_f32",
    "torch_rmsnorm",
]

SHAPES = [(16384, 1024), (65536, 1024), (32768, 512), (65536, 256)]


def _torch_rms_norm(x, y, gamma):
    xf = x.float()
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + EPS) * gamma).to(x.dtype)


def bench(fn_name, shape):
    s, d = shape
    dt = dtype_for(fn_name)
    x = torch.randn(s, d, device="cuda", dtype=dt).contiguous()
    y = torch.zeros_like(x)

    if fn_name == "torch_rmsnorm":
        fn = _torch_rms_norm
    else:
        fn = getattr(lib, fn_name)

    return time_ms(fn, x, y, GAMMA)


def main():
    for shape in SHAPES:
        s, d = shape
        print(f"=== S={s}, D={d} ===")
        for fn_name in FNS:
            ms = bench(fn_name, shape)
            print_row(fn_name, ms)
        print()


if __name__ == "__main__":
    main()
