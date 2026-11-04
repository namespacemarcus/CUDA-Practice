import torch
from torch.utils.cpp_extension import load

mod = load(
    name="mega", sources=["mega_minimal.cu"], verbose=True, extra_cuda_cflags=["-O3"]
)

if __name__ == "__main__":
    M, N, K = 128, 128, 128
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C1 = torch.empty(M, N, device="cuda")
    D = torch.randn(N, K, device="cuda")
    C2 = torch.empty(M, K, device="cuda")

    # dep 初始值：task0:0就绪, task1:依赖0→dep=1, task2依赖1→dep=1
    dep_host = torch.tensor([0, 1, 1], dtype=torch.int32)

    out = mod.run(A, B, C1, D, C2, dep_host)

    # 验证：C2 = (A@B) @ D
    ref = (A @ B) @ D
    print("max diff:", (out - ref).abs().max().item())
