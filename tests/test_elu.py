import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="elu_lib",
    op_subdir="elu",
    sources=["torch_bindings.cu"],
)

ALPHA = 1.0

SHAPES = [(1024, 1024), (1024, 2048), (2048, 4096)]
FALLBACK_SHAPES = [(1025,), (2053,), (7,), (13,)]


def elu_ref(x, alpha=ALPHA):
    return torch.where(x > 0, x, alpha * torch.expm1(x))


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize("fn_name", ["elu_f32", "elu_f32x4"])
def test_elu_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, elu_ref(x))


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["elu_f16", "elu_f16x2", "elu_f16x8", "elu_f16x8_pack"],
)
def test_elu_f16(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, elu_ref(x), rtol=1e-3, atol=1e-3)
