import torch
import ofk


def test_gemm():
    print("开始测试 gemm...")
    M, N, K = 4096, 24576, 1536
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")

    for _ in range(10):
        torch.ops.ofk.gemm(A, B)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    out = torch.ops.ofk.gemm(A, B)
    torch.cuda.synchronize()
    ref = torch.mm(A, B)
    torch.cuda.cudart().cudaProfilerStop()

    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    print(f"gemm [{M}, {K}] x [{K}, {N}]: PASS")


def test_gemm_shapes():
    print("开始测试 gemm 不同形状...")
    shapes = [
        (4096, 24576,  1536),
        (4096, 32768,   512),
        (4096,  7168, 16384),
        (4096,  4096,  7168),
        (4096,  7168,  2048),
    ]
    for M, N, K in shapes:
        A = torch.randn(M, K, dtype=torch.float32, device="cuda")
        B = torch.randn(K, N, dtype=torch.float32, device="cuda")
        out = torch.ops.ofk.gemm(A, B)
        ref = torch.mm(A, B)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
        print(f"  (m={M:5d}, n={N:5d}, k={K:5d}): PASS")
    print("gemm shape 测试: PASS")


if __name__ == "__main__":
    test_gemm()
    test_gemm_shapes()
