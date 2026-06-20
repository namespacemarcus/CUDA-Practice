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

# Shapes valid for all kernels (naive and k_tiled work with any shape,
# thread-tiled kernels require M, N >= BM=128).
# (M, K, N)
# The k_tiled kernel requires K to be a multiple of BK=32 (no bounds check for
# partial tiles). The naive kernel handles arbitrary shapes.
SHAPES_ALL = [
    (64, 64, 64),
    (128, 128, 128),
    (256, 128, 256),
    (128, 256, 128),
    (1024, 1024, 256),
]

SHAPES_TILED = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (256, 128, 256),
    (128, 256, 128),
    (1024, 1024, 256),
    (2048, 256, 1024),
]

NAIVE_AND_KTILED_FNS = ["sgemm_naive_f32", "sgemm_k_tiled_f32"]
TILED_FNS = [
    "sgemm_thread_tiled_8x8_and_k_tiled_f32x4",
    "sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf",
    "sgemm_thread_tiled_8x8_and_k_tiled_f32x4_bcf_offset",
]

# sgemm kernels use a different accumulation order than cuBLAS (PyTorch's @),
# so small floating-point differences are expected. Tolerances account for
# error growth with K (each of K FMAs adds ~1 ulp of potential drift).
# rtol=1e-3, atol=1e-4 is generous enough for K up to ~2048 with float32.


@pytest.mark.parametrize("shape", SHAPES_ALL)
@pytest.mark.parametrize("fn_name", NAIVE_AND_KTILED_FNS)
def test_sgemm_naive_and_ktiled(shape, fn_name):
    M, K, N = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    getattr(lib, fn_name)(a, b, out)

    ref = a @ b
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("shape", SHAPES_TILED)
@pytest.mark.parametrize("fn_name", TILED_FNS)
def test_sgemm_thread_tiled(shape, fn_name):
    M, K, N = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    getattr(lib, fn_name)(a, b, out)

    ref = a @ b
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("fn_name", NAIVE_AND_KTILED_FNS + TILED_FNS)
def test_sgemm_large(fn_name):
    """Test with larger matrices to exercise multi-block execution."""
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    out = torch.zeros(M, N, device="cuda", dtype=torch.float32)

    getattr(lib, fn_name)(a, b, out)

    ref = a @ b
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)
