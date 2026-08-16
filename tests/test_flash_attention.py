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


def _make_qkv(dtype, batch, heads, q_seqlen, kv_seqlen, head_dim):
    q = torch.randn(
        batch, heads, q_seqlen, head_dim, device="cuda", dtype=dtype
    ).contiguous()
    k = torch.randn(
        batch, heads, kv_seqlen, head_dim, device="cuda", dtype=dtype
    ).contiguous()
    v = torch.randn(
        batch, heads, kv_seqlen, head_dim, device="cuda", dtype=dtype
    ).contiguous()
    return q, k, v


# online attention naive
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_online_attention_naive_f32(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out = lib.online_attention_naive(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_online_attention_naive_f16(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float16, B, H, N, N, D)
    out = lib.online_attention_naive(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", CROSS_SHAPES)
def test_online_attention_naive_cross_attention(shape):
    B, H, Nq, Nkv, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, Nq, Nkv, D)
    out = lib.online_attention_naive(q, k, v, causal=False)
    ref = _ref_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


# online attention kv tiled
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_online_attention_kv_tiled_f32(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out = lib.online_attention_kv_tiled(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_online_attention_kv_tiled_f16(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float16, B, H, N, N, D)
    out = lib.online_attention_kv_tiled(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", CROSS_SHAPES)
def test_online_attention_kv_tiled_cross_attention(shape):
    B, H, Nq, Nkv, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, Nq, Nkv, D)
    out = lib.online_attention_kv_tiled(q, k, v, causal=False)
    ref = _ref_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_online_attention_kv_tiled_matches_naive(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out_naive = lib.online_attention_naive(q, k, v, causal=causal)
    out_tiled = lib.online_attention_kv_tiled(q, k, v, causal=causal)
    torch.testing.assert_close(out_tiled, out_naive, rtol=1e-3, atol=1e-5)


# flash attention v1
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v1_f32(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out = lib.flash_attention_v1_forward(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v1_f16(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float16, B, H, N, N, D)
    out = lib.flash_attention_v1_forward(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-3)


@pytest.mark.parametrize("shape", CROSS_SHAPES)
def test_flash_attention_v1_cross_attention(shape):
    B, H, Nq, Nkv, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, Nq, Nkv, D)
    out = lib.flash_attention_v1_forward(q, k, v, causal=False)
    ref = _ref_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


# flash attention v2
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v2_f32(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out = lib.flash_attention_v2_forward(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v2_f16(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float16, B, H, N, N, D)
    out = lib.flash_attention_v2_forward(q, k, v, causal=causal)
    ref = _ref_attention(q, k, v, causal)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", CROSS_SHAPES)
def test_flash_attention_v2_cross_attention(shape):
    B, H, Nq, Nkv, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, Nq, Nkv, D)
    out = lib.flash_attention_v2_forward(q, k, v, causal=False)
    ref = _ref_attention(q, k, v, causal=False)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_v2_matches_v1(shape, causal):
    B, H, N, D = shape
    q, k, v = _make_qkv(torch.float32, B, H, N, N, D)
    out_v1 = lib.flash_attention_v1_forward(q, k, v, causal=causal)
    out_v2 = lib.flash_attention_v2_forward(q, k, v, causal=causal)
    torch.testing.assert_close(out_v2, out_v1, rtol=1e-3, atol=1e-5)
