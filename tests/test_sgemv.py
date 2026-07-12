import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="sgemv_lib",
    op_subdir="sgemv",
    sources=["torch_bindings.cu"],
)


def _check(fn_name, M, K, rtol=1e-3, atol=1e-2):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    x = torch.randn(K, 1, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(M, 1, device="cuda", dtype=torch.float32)
    getattr(lib, fn_name)(a, x, out)
    ref = a @ x
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


K32_SHAPES = [
    (1, 32),
    (4, 32),
    (32, 32),
    (8, 64),
    (64, 128),
    (17, 128),
    (100, 256),
    (1024, 1024),
]

K128_SHAPES = [
    (1, 128),
    (4, 128),
    (8, 128),
    (32, 128),
    (64, 256),
    (17, 256),
    (100, 512),
    (1024, 1024),
    (8, 4),
    (8, 5),
    (17, 6),
    (33, 7),
    (100, 132),
    (17, 130),
    (33, 129),
    (8, 131),
]

K16_SHAPES = [
    (1, 16),
    (8, 16),
    (16, 16),
    (17, 16),
    (33, 16),
    (100, 16),
    (1024, 16),
]


@pytest.mark.parametrize("shape", K32_SHAPES)
def test_sgemv_k32_f32(shape):
    M, K = shape
    _check("sgemv_k32_f32", M, K)


@pytest.mark.parametrize("shape", K128_SHAPES)
def test_sgemv_k128_f32x4(shape):
    M, K = shape
    _check("sgemv_k128_f32x4", M, K)


@pytest.mark.parametrize("shape", K16_SHAPES)
def test_sgemv_k16_f32(shape):
    M, K = shape
    _check("sgemv_k16_f32", M, K)
