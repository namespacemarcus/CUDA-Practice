import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="layer_norm_lib",
    op_subdir="layer_norm",
    sources=["torch_bindings.cu"],
)

EPS = 1e-5

F32_SHAPES = [
    (1, 64),
    (4, 128),
    (16, 256),
    (2, 512),
    (8, 1024),
]

F32X4_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
    (1, 2048),
    (2, 4096),
]

F16_F16_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
]

F16X2_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
    (1, 2048),
]

F16X8_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
    (1, 2048),
    (2, 4096),
    (1, 8192),
]

F16_F32_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
]

F16X8_PACK_F32_SHAPES = [
    (1, 64),
    (4, 128),
    (8, 256),
    (2, 512),
    (4, 1024),
    (1, 2048),
    (2, 4096),
    (1, 8192),
]


def _ref_layer_norm(x: torch.Tensor, gamma: float, beta: float,
                    eps: float = EPS) -> torch.Tensor:
    """Reference layer norm: y = (x - mean) / sqrt(var + eps) * gamma + beta."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


# ---- f32 ----
@pytest.mark.parametrize("shape", F32_SHAPES)
def test_layer_norm_f32(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f32(x, y, gamma, beta)
    expected = _ref_layer_norm(x, gamma, beta)
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("shape", F32X4_SHAPES)
def test_layer_norm_f32x4(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f32x4(x, y, gamma, beta)
    expected = _ref_layer_norm(x, gamma, beta)
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("shape", F16_F16_SHAPES)
def test_layer_norm_f16_f16(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16_f16(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", F16X2_SHAPES)
def test_layer_norm_f16x2_f16(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16x2_f16(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", F16X8_SHAPES)
def test_layer_norm_f16x8_f16(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16x8_f16(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", F16X8_SHAPES)
def test_layer_norm_f16x8_pack_f16(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16x8_pack_f16(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", F16_F32_SHAPES)
def test_layer_norm_f16_f32(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16_f32(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", F16X8_PACK_F32_SHAPES)
def test_layer_norm_f16x8_pack_f32(shape):
    S, D = shape
    gamma, beta = 1.0, 0.0
    x = torch.randn(S, D, device="cuda", dtype=torch.float16).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f16x8_pack_f32(x, y, gamma, beta)
    expected = _ref_layer_norm(x.float(), gamma, beta).half()
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(4, 256), (2, 1024)])
@pytest.mark.parametrize("gamma,beta", [(2.0, -1.0), (0.5, 3.0)])
def test_layer_norm_affine(shape, gamma, beta):
    S, D = shape
    x = torch.randn(S, D, device="cuda", dtype=torch.float32).contiguous()
    y = torch.zeros_like(x)
    lib.layer_norm_f32(x, y, gamma, beta)
    expected = _ref_layer_norm(x, gamma, beta)
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-5)


ALL_FNS = [
    "layer_norm_f32",
    "layer_norm_f32x4",
    "layer_norm_f16_f16",
    "layer_norm_f16x2_f16",
    "layer_norm_f16x8_f16",
    "layer_norm_f16x8_pack_f16",
    "layer_norm_f16_f32",
    "layer_norm_f16x8_pack_f32",
]


@pytest.mark.parametrize("fn_name", ALL_FNS)
def test_layer_norm_all_smoke(fn_name):
    S, D = 4, 256
    dtype = torch.float16 if "f16" in fn_name else torch.float32
    x = torch.randn(S, D, device="cuda", dtype=dtype).contiguous()
    y = torch.zeros_like(x)
    gamma, beta = 1.0, 0.0
    getattr(lib, fn_name)(x, y, gamma, beta)
    rtol = 1e-2 if dtype == torch.float16 else 1e-4
    atol = 1e-2 if dtype == torch.float16 else 1e-5
    expected = _ref_layer_norm(
        x.float() if dtype == torch.float16 else x, gamma, beta
    )
    if dtype == torch.float16:
        expected = expected.half()
    torch.testing.assert_close(y, expected, rtol=rtol, atol=atol)
