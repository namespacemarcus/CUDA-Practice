import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="dot_product_lib",
    op_subdir="dotproduct",
    sources=["torch_bindings.cu"],
)


SHAPES = [(4, 256), (4, 1024), (16, 512)]
FALLBACK_SHAPES = [(4096,), (8193,)]


def dot_product_ref(a, b):
    return (a.float() * b.float()).sum()


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize("fn_name", ["dot_product_f32_f32", "dot_product_f32x4_f32"])
def test_dot_product_f32(shape, fn_name):
    a = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = getattr(lib, fn_name)(a, b)
    ref = dot_product_ref(a, b)
    torch.testing.assert_close(y, ref.view(1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize("fn_name", ["dot_product_f16_f32"])
def test_dot_product_f16_f32(shape, fn_name):
    a = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = getattr(lib, fn_name)(a, b)
    ref = dot_product_ref(a, b)
    torch.testing.assert_close(y, ref.view(1), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("shape", SHAPES + FALLBACK_SHAPES)
@pytest.mark.parametrize(
    "fn_name",
    ["dot_product_f16x2_f32", "dot_product_f16x8_pack_f32"],
)
def test_dot_product_f16x2_f32(shape, fn_name):
    a = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    b = torch.randn(shape, device="cuda", dtype=torch.float16).contiguous()
    y = getattr(lib, fn_name)(a, b)
    ref = dot_product_ref(a, b)
    torch.testing.assert_close(y, ref.view(1), rtol=1e-2, atol=1e-2)
