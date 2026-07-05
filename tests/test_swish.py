import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="swish_lib",
    op_subdir="swish",
    sources=["torch_bindings.cu"],
)


SHAPES = [(1024, 1024), (1024, 2048), (2048, 4096)]
FALLBACK_SHAPES = [(1025,), (2053,)]


def swish_ref(x):
    return x * torch.sigmoid(x)


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize("fn_name", ["swish_f32", "swish_f32x4"])
def test_swish_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, swish_ref(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["swish_f16", "swish_f16x2", "swish_f16x8", "swish_f16x8_pack"],
)
def test_swish_f16(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, swish_ref(x), rtol=1e-3, atol=1e-3)
