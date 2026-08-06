import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="hgemm_lib",
    op_subdir="hgemm",
    sources=["torch_bindings.cu"],
)

SHAPES = [
    (128, 32, 1024),
    (256, 160, 1024),
    (512, 96, 1024),
    (1024, 256, 1024),
    (512, 512, 2048),
    (1024, 1024, 1024),
]

FNS = [
    "hgemm_gw_tiled",
    "hgemm_gw_tiled_bcf",
    "hgemm_gw_tiled_bcf_dbf",
]

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fn_name", FNS)
@pytest.mark.parametrize("shape", SHAPES)
def test_hgemm(shape, fn_name, dtype):
    M, K, N = shape
    a = torch.randn(M, K, device="cuda", dtype=dtype).contiguous()
    b = torch.randn(K, N, device="cuda", dtype=dtype).contiguous()
    out = torch.zeros(M, N, device="cuda", dtype=dtype).contiguous()
    getattr(lib, fn_name)(a, b, out)
    ref = a.float() @ b.float()
    rtol = 1e-3 if dtype == torch.float16 else 1e-2
    atol = 1e-2 if dtype == torch.float16 else 5e-2
    torch.testing.assert_close(out.float(), ref, rtol=rtol, atol=atol)
