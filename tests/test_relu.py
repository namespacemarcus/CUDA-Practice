import pytest
import torch

from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="relu_lib",
    op_subdir="relu",
    sources=["relu.cu", "torch_bindings.cu"],
)


SHAPES = [(1024, 1024), (1024, 2048), (2048, 4096)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn_name", ["relu_f32", "relu_f32x4"])
def test_relu_f32(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, torch.relu(x))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "fn_name", ["relu_f16", "relu_f16x2", "relu_f16x8", "relu_f16x8_pack"]
)
def test_relu_f16(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, torch.relu(x), rtol=1e-3, atol=1e-3)
