import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="sigmoid_lib",
    op_subdir="sigmoid",
    sources=["torch_bindings.cu"],
)

# 2D shapes with last dim divisible by 8 (full-vector path for every pack variant)
SHAPES = [(1024, 1024), (1024, 2048), (2048, 4096)]
# 1D shapes with N not a multiple of 8 (exercises the scalar tail/fallback branch)
FALLBACK_SHAPES = [(1025,), (2053,)]


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize("fn_name", ["sigmoid_f32", "sigmoid_f32x4"])
def test_sigmoid_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, torch.sigmoid(x))


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["sigmoid_f16", "sigmoid_f16x2", "sigmoid_f16x8", "sigmoid_f16x8_pack"],
)
def test_sigmoid_f16(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, torch.sigmoid(x), rtol=1e-3, atol=1e-3)
