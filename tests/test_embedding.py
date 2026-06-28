import pytest
import torch
from conftest import load_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

lib = load_op(
    name="embedding_lib",
    op_subdir="embedding",
    sources=["embedding.cu", "torch_bindings.cu"],
)

# N: number of indices to look up, emb_size: embedding dimension
# emb_size values must be divisible by 8 for the f16x8 / f16x8_pack kernels.
NS = [1, 100, 1024, 4096, 12345]
EMB_SIZES = [64, 128, 256, 512]

F32_FNS = ["embedding_f32", "embedding_f32x4", "embedding_f32x4_pack"]
F16_FNS = ["embedding_f16", "embedding_f16x8", "embedding_f16x8_pack"]


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("emb_size", EMB_SIZES)
@pytest.mark.parametrize("fn_name", F32_FNS)
def test_embedding_f32(n, emb_size, fn_name):
    vocab_size = max(n * 2, 1024)
    weight = torch.randn(vocab_size, emb_size, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, vocab_size, (n,), device="cuda", dtype=torch.int32)
    out = torch.zeros(n, emb_size, device="cuda", dtype=torch.float32)

    getattr(lib, fn_name)(idx, weight, out)

    ref = weight[idx]
    torch.testing.assert_close(out, ref)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("emb_size", EMB_SIZES)
@pytest.mark.parametrize("fn_name", F16_FNS)
def test_embedding_f16(n, emb_size, fn_name):
    vocab_size = max(n * 2, 1024)
    weight = torch.randn(vocab_size, emb_size, device="cuda", dtype=torch.float16)
    idx = torch.randint(0, vocab_size, (n,), device="cuda", dtype=torch.int32)
    out = torch.zeros(n, emb_size, device="cuda", dtype=torch.float16)

    getattr(lib, fn_name)(idx, weight, out)

    ref = weight[idx]
    torch.testing.assert_close(out, ref)
