import pytest
import torch

from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="reduce_lib",
    op_subdir="reduce",
    sources=["torch_bindings.cu"],
)

SIZES = [1024, 4096, 10000, 1000, 1234, 4095]

_HAS_FP8_E4M3 = hasattr(torch, "float8_e4m3fn")
_HAS_FP8_E5M2 = hasattr(torch, "float8_e5m2")


def _ref_sum(x: torch.Tensor):
    return x.double().sum()


def _check(fn_name, x, rtol, atol):
    out = getattr(lib, fn_name)(x)
    assert out.numel() == 1
    got = out.double().item()
    ref = _ref_sum(x).item()
    torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)


F32_F32_FNS = ["block_all_reduce_sum_f32_f32", "block_all_reduce_sum_f32x4_f32"]
F16_F32_FNS = [
    "block_all_reduce_sum_f16_f32",
    "block_all_reduce_sum_f16x2_f32",
    "block_all_reduce_sum_f16x8_pack_f32",
]
F16_F16_FNS = [
    "block_all_reduce_sum_f16_f16",
    "block_all_reduce_sum_f16x2_f16",
    "block_all_reduce_sum_f16x8_pack_f16",
]
BF16_F32_FNS = [
    "block_all_reduce_sum_bf16_f32",
    "block_all_reduce_sum_bf16x2_f32",
    "block_all_reduce_sum_bf16x8_pack_f32",
]
BF16_BF16_FNS = [
    "block_all_reduce_sum_bf16_bf16",
    "block_all_reduce_sum_bf16x2_bf16",
    "block_all_reduce_sum_bf16x8_pack_bf16",
]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", F32_F32_FNS)
def test_reduce_f32_f32(n, fn_name):
    x = torch.rand(n, device="cuda", dtype=torch.float32) + 0.5
    _check(fn_name, x, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", F16_F32_FNS)
def test_reduce_f16_f32(n, fn_name):
    x = torch.rand(n, device="cuda", dtype=torch.float16) + 0.5
    _check(fn_name, x, rtol=2e-3, atol=2e-3 * n)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", F16_F16_FNS)
def test_reduce_f16_f16(n, fn_name):
    x = torch.rand(n, device="cuda", dtype=torch.float16) + 0.5
    _check(fn_name, x, rtol=1e-2, atol=5e-3 * n)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", BF16_F32_FNS)
def test_reduce_bf16_f32(n, fn_name):
    x = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
    _check(fn_name, x, rtol=1e-2, atol=1e-2 * n)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", BF16_BF16_FNS)
def test_reduce_bf16_bf16(n, fn_name):
    x = torch.rand(n, device="cuda", dtype=torch.bfloat16) + 0.5
    _check(fn_name, x, rtol=5e-2, atol=5e-2 * n)


I8_I32_FNS = ["block_all_reduce_sum_i8_i32", "block_all_reduce_sum_i8x16_pack_i32"]


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", I8_I32_FNS)
def test_reduce_i8_i32(n, fn_name):
    x = torch.randint(-8, 8, (n,), device="cuda", dtype=torch.int8)
    out = getattr(lib, fn_name)(x)
    assert out.numel() == 1
    assert out.item() == int(x.int().sum().item())


FP8_E4M3_FNS = [
    "block_all_reduce_sum_fp8_e4m3_f16",
    "block_all_reduce_sum_fp8_e4m3x16_pack_f16",
]
FP8_E5M2_FNS = [
    "block_all_reduce_sum_fp8_e5m2_f16",
    "block_all_reduce_sum_fp8_e5m2x16_pack_f16",
]


@pytest.mark.skipif(not _HAS_FP8_E4M3, reason="fp8 e4m3 dtype not available.")
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", FP8_E4M3_FNS)
def test_reduce_fp8_e4m3(n, fn_name):
    x = (torch.rand(n, device="cuda", dtype=torch.float32) + 0.5).to(
        torch.float8_e4m3fn
    )
    _check(fn_name, x, rtol=1e-2, atol=5e-3 * n)


@pytest.mark.skipif(not _HAS_FP8_E5M2, reason="fp8 e5m2 dtype not available.")
@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("fn_name", FP8_E5M2_FNS)
def test_reduce_fp8_e4m3(n, fn_name):
    x = (torch.rand(n, device="cuda", dtype=torch.float32) + 0.5).to(
        torch.float8_e5m2
    )
    _check(fn_name, x, rtol=5e-2, atol=2e-2 * n)
