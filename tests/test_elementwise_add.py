import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="elementwise_add_lib",
    op_subdir="elementwise",
    sources=["elementwise_add.cu", "torch_bindings.cu"],
)


SHAPES = [(1024, 1024), (1024, 2048), (2048, 4096)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["elementwise_add_f32", "elementwise_add_f32x4"],
)
def test_elementwise_add_f32(shape, fn_name):
    a = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros_like(a)
    getattr(lib, fn_name)(a, b, out)
    torch.testing.assert_close(out, a + b)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    [
        "elementwise_add_f16",
        "elementwise_add_f16x2",
        "elementwise_add_f16x8",
        "elementwise_add_f16x8_pack",
    ],
)
def test_elementwise_add_f16(shape, fn_name):
    a = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    out = torch.zeros_like(a)
    getattr(lib, fn_name)(a, b, out)
    torch.testing.assert_close(out, a + b, rtol=1e-3, atol=1e-3)
