import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="sgemm_lib",
    op_subdir="sgemm",
    sources=["torch_bindings.cu"],
)

SHAPES = [
    (128, 128, 128),
    (256, 128, 256),
    (128, 256, 128),
    (256, 256, 256),
]

FNS = [
    "sgemm_naive",
    "sgemm_tiling",
    "sgemm_at_tiling",
    "sgemm_at_tiling_bcf_swizzling",
    "sgemm_at_tiling_bcf_swizzling_cstore",
    "sgemm_at_tiling_bcf_swizzling_cstore_dbf",
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn_name", FNS)
def test_sgemm(shape, fn_name):
    M, K, N = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32).contiguous()
    getattr(lib, fn_name)(a, b, out)
    ref = a @ b
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
