import torch
import ofk 

def test_my_cuda_add():
    print("🚀 开始测试...")
    
    size = (1024, 1024)
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    # warmup
    for _ in range(10):
        torch.ops.my_lib.add(a, b) # 这里的 my_lib 和 add 对应你在 entry.cc 里写的 TORCH_LIBRARY(my_lib, m) { m.def("add...") }
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()   # ← profile 从这里开始
    my_output = torch.ops.my_lib.add(a, b)
    torch.cuda.synchronize()
    torch_output = a + b
    torch.cuda.cudart().cudaProfilerStop()    # ← profile 到这里结束
    
    torch.testing.assert_close(my_output, torch_output)
    
    print("✅ 测试通过！")

if __name__ == "__main__":
    test_my_cuda_add()