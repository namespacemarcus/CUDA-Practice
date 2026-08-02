import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="matrix_transpose_lib",
    op_subdir="transpose",
    sources=["torch_bindings.cu"],
)

SHAPES = [
    (64, 64),
    (128, 64),
    (32, 128),
    (64, 256),
]

SMALL_SHAPES = [
    (4, 4),
    (17, 13),
    (1, 1),
    (13, 7),
]

ALL_FNS = [
    "matrix_transpose_f32_loadcoal",
    "matrix_transpose_f32x4_loadcoal",
    "matrix_transpose_f32_loadcoal_2d",
    "matrix_transpose_f32x4_loadcoal_2d",
]

F32_FNS = [
    "matrix_transpose_f32_loadcoal",
    "matrix_transpose_f32_loadcoal_2d",
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn_name", ALL_FNS)
def test_matrix_transpose(shape, fn_name):
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros(N, M, device="cuda", dtype=torch.float32).contiguous()
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, x.t().contiguous(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", SMALL_SHAPES)
@pytest.mark.parametrize("fn_name", F32_FNS)
def test_matrix_transpose_f32_small(shape, fn_name):
    M, N = shape
    x = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros(N, M, device="cuda", dtype=torch.float32).contiguous()
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, x.t().contiguous(), rtol=0, atol=0)
