import pytest
import torch
import torch.nn.functional as F
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="flash_attention_lib",
    op_subdir="flash-attention",
    sources=["torch_bindings.cu"],
)

SHAPES = [
    (4, 2, 64, 32),
    (4, 2, 64, 64),
    (2, 4, 128, 64),
    (1, 4, 256, 64),
    (1, 2, 64, 128),
    (1, 1, 32, 256),
]

CROSS_SHAPES = [
    (1, 2, 48, 64, 64),
    (2, 4, 32, 96, 64),
]


def _ref_attention(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_naive_f32(shape, causal):
    B, H, N, D = shape
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32).contiguous()
    out = lib.flash_attention_naive(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_naive_f16(shape, causal):
    B, H, N, D = shape
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16).contiguous()
    out = lib.flash_attention_naive(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", CROSS_SHAPES)
def test_flash_attention_cross_attention(shape):
    B, H, Nq, Nkv, D = shape
    q = torch.randn(B, H, Nq, D, device="cuda", dtype=torch.float32).contiguous()
    k = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float32).contiguous()
    v = torch.randn(B, H, Nkv, D, device="cuda", dtype=torch.float32).contiguous()
    out = lib.flash_attention_naive(q, k, v, causal=False)
    ref = _ref_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)
