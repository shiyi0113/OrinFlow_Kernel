import torch
import ofk 

def test_my_cuda_add():
    print("🚀 开始测试自定义 CUDA 算子...")
    
    size = (1024, 1024)
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    
    # 这里的 my_lib 和 add 对应你在 entry.cc 里写的 TORCH_LIBRARY(my_lib, m) { m.def("add...") }
    my_output = torch.ops.my_lib.add(a, b)
    torch_output = a + b
    torch.testing.assert_close(my_output, torch_output)
    
    print("✅ 测试完美通过！你的自定义 CUDA 算子计算结果与 PyTorch 原生算子完全一致！")

if __name__ == "__main__":
    test_my_cuda_add()