import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="rope_lib",
    op_subdir="rope",
    sources=["torch_bindings.cu"],
)

THETA = 10000.0
SHAPES = [(4, 256), (4, 1024), (16, 512), (1, 1024), (8, 64), (32, 128)]


def rope_ref(x, theta=THETA):
    seq_len, hidden = x.shape
    d = hidden
    half = d // 2
    pair_idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (2.0 * pair_idx / d))
    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x1 = x[..., 0::2].float()
    x2 = x[..., 1::2].float()
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn_name", ["rope_f32", "rope_f32x4"])
def test_rope(shape, fn_name):
    x = torch.randn(shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    torch.testing.assert_close(y, rope_ref(x), rtol=1e-4, atol=1e-4)
