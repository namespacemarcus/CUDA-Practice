import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="histogram_lib",
    op_subdir="histogram",
    sources=["torch_bindings.cu"],
)

SIZES = [1, 2, 3, 4, 5, 7, 8, 65, 255, 256, 257, 1024, 4096, 10000]
NUM_BINS = 16


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", ["histogram_i32", "histogram_i32x4"])
def test_histogram(fn_name, n):
    a = torch.randint(0, NUM_BINS, (n,), device="cuda", dtype=torch.int32).contiguous()
    y = getattr(lib, fn_name)(a)
    ref = torch.bincount(a).to(torch.int32)
    assert y.dtype == torch.int32
    assert torch.equal(y, ref)
