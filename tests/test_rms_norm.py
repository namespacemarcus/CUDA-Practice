import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="rms_norm_lib",
    op_subdir="rms_norm",
    sources=["torch_bindings.cu"],
)

GAMMA = 2.0
EPS = 1e-5

SHAPES = [(4, 256), (4, 1024), (16, 512), (1, 1024)]


def rms_norm_ref(x, gamma=GAMMA, eps=EPS):
    xf = x.float()
    ms = (xf * xf).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(ms + eps) * gamma).to(x.dtype)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn_name", ["rms_norm_f32", "rms_norm_f32x4"])
def test_rms_norm_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y, GAMMA)
    torch.testing.assert_close(y, rms_norm_ref(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["rms_norm_f16_f32", "rms_norm_f16x8_f32", "rms_norm_f16x8_pack_f32"],
)
def test_rms_norm_f16_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y, GAMMA)
    torch.testing.assert_close(y, rms_norm_ref(x), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    [
        "rms_norm_f16_f16",
        "rms_norm_f16x2_f16",
        "rms_norm_f16x8_f16",
        "rms_norm_f16x8_pack_f16",
    ],
)
def test_rms_norm_f16_f16(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y, GAMMA)
    torch.testing.assert_close(y, rms_norm_ref(x), rtol=1e-2, atol=1e-2)
