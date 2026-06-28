import pytest
import torch
import torch.nn.functional as F
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="softmax_lib",
    op_subdir="softmax",
    sources=["torch_bindings.cu"],
)

# softmax / safe_softmax: flat-index, N=S*H, block=H. Reduce functions require
# H to be a multiple of 32 (warp size). H must match a dispatch case.
SOFTMAX_SAFE_SHAPES_F32 = [
    (1, 32),
    (1, 128),
    (1, 256),
    (4, 64),
    (4, 256),
    (128, 256),
    (1, 512),
    (1, 1024),
]

SOFTMAX_SAFE_SHAPES_F32X4 = [
    (1, 32),
    (1, 128),
    (1, 256),
    (4, 64),
    (4, 256),
    (128, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
]

SOFTMAX_SAFE_SHAPES_F16 = [
    (1, 32),
    (1, 128),
    (1, 256),
    (4, 256),
    (1, 512),
    (1, 1024),
]

SOFTMAX_SAFE_SHAPES_F16X2 = [
    (1, 32),
    (1, 128),
    (1, 256),
    (4, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
]

SOFTMAX_SAFE_SHAPES_F16X8 = [
    (1, 32),
    (1, 128),
    (1, 256),
    (4, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),
]

# online_softmax: per-row indexing, N=H, block always 256 (f32) or 64 (f32x4).
# f32 dispatch supports H ∈ {32,64,128,256,512,1024}.
# f32x4 dispatch supports H ∈ {128,256,512,1024,2048,4096}.
ONLINE_SHAPES_F32 = [
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 1024),
    (4, 64),
    (4, 128),
    (128, 256),
]

ONLINE_SHAPES_F32X4 = [
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (4, 128),
    (4, 256),
    (128, 256),
]

ALL_F32_FNS = [
    "softmax_f32_per_token",
    "softmax_f32x4_per_token",
    "safe_softmax_f32_per_token",
    "safe_softmax_f32x4_per_token",
    "online_safe_softmax_f32_per_token",
    "online_safe_softmax_f32x4_pack_per_token",
]

ALL_F16_FNS = [
    "safe_softmax_f16_f32_per_token",
    "safe_softmax_f16x2_f32_per_token",
    "safe_softmax_f16x8_pack_f32_per_token",
]


def _ref_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x.float(), dim=-1).to(x.dtype)


# ---- f32 softmax (naive) ----
@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F32)
def test_softmax_f32_per_token(shape):
    S, H = shape
    x = (torch.randn(S, H, device="cuda", dtype=torch.float32) * 0.1).contiguous()
    y = torch.zeros_like(x)
    lib.softmax_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F32X4)
def test_softmax_f32x4_per_token(shape):
    S, H = shape
    x = (torch.randn(S, H, device="cuda", dtype=torch.float32) * 0.1).contiguous()
    y = torch.zeros_like(x)
    lib.softmax_f32x4_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


# ---- f32 safe softmax ----
@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F32)
def test_safe_softmax_f32_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.safe_softmax_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F32X4)
def test_safe_softmax_f32x4_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.safe_softmax_f32x4_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


# ---- f32 online safe softmax ----
@pytest.mark.parametrize("shape", ONLINE_SHAPES_F32)
def test_online_safe_softmax_f32_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.online_safe_softmax_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("shape", ONLINE_SHAPES_F32X4)
def test_online_safe_softmax_f32x4_pack_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.online_safe_softmax_f32x4_pack_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-3, atol=1e-5)


# ---- f16 safe softmax ----
@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F16)
def test_safe_softmax_f16_f32_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.safe_softmax_f16_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F16X2)
def test_safe_softmax_f16x2_f32_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.safe_softmax_f16x2_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("shape", SOFTMAX_SAFE_SHAPES_F16X8)
def test_safe_softmax_f16x8_pack_f32_per_token(shape):
    S, H = shape
    x = torch.randn(S, H, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.safe_softmax_f16x8_pack_f32_per_token(x, y)
    torch.testing.assert_close(y, _ref_softmax(x), rtol=1e-2, atol=1e-3)


# ---- smoke test: all fns with H=256 ----
@pytest.mark.parametrize("fn_name", ALL_F32_FNS + ALL_F16_FNS)
def test_softmax_all_smoke(fn_name):
    S, H = 4, 256
    dtype = torch.float16 if "f16" in fn_name else torch.float32
    x = torch.randn(S, H, device="cuda", dtype=dtype).contiguous()
    y = torch.zeros_like(x)
    getattr(lib, fn_name)(x, y)
    rtol = 1e-2 if dtype == torch.float16 else 1e-3
    atol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(y, _ref_softmax(x), rtol=rtol, atol=atol)
