import torch
import ofk

def test_my_cuda_reduce():
    print("🚀 开始测试 reduce...")
    size = (10240 * 10240)
    x = torch.randn(size, dtype=torch.float32, device="cuda")

    # warmup
    for _ in range(10):
        torch.ops.ofk.reduce(x)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    my_output = torch.ops.ofk.reduce(x)
    torch.cuda.synchronize()
    ref = x.sum().unsqueeze(0)
    torch.cuda.cudart().cudaProfilerStop()

    torch.testing.assert_close(my_output, ref, atol=1e-1, rtol=1e-3)
    print("reduce (1M): PASS")

def test_reduce_edge_cases():
    print("🚀 开始测试 reduce 边界...")

    for n in [1, 255, 256, 257, 1023, 1024, 100003]:
        x = torch.ones(n, dtype=torch.float32, device="cuda")
        out = torch.ops.ofk.reduce(x)
        ref = x.sum().unsqueeze(0)
        torch.testing.assert_close(out, ref, atol=0.5, rtol=0.0)

    print("reduce edge cases: PASS")

if __name__ == "__main__":
    test_my_cuda_reduce()
    test_reduce_edge_cases()
