import torch
import ofk


def test_softmax():
    print("开始测试 softmax...")
    N, D = 10240, 10240
    x = torch.randn(N, D, dtype=torch.float32, device="cuda")

    for _ in range(10):
        torch.ops.ofk.softmax(x)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    my_output = torch.ops.ofk.softmax(x)
    torch.cuda.synchronize()
    ref = torch.softmax(x, dim=-1)
    torch.cuda.cudart().cudaProfilerStop()

    torch.testing.assert_close(my_output, ref, atol=1e-5, rtol=1e-5)
    print("softmax [1024, 1024]: PASS")


def test_softmax_edge_cases():
    print("开始测试 softmax 边界...")
    for D in [1, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048]:
        x = torch.randn(16, D, dtype=torch.float32, device="cuda")
        out = torch.ops.ofk.softmax(x)
        ref = torch.softmax(x, dim=-1)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    print("softmax edge cases: PASS")


if __name__ == "__main__":
    test_softmax()
    test_softmax_edge_cases()
