import torch
import ofk 

def test_my_cuda_add():
    print("🚀 开始测试add...")
    
    size = (10240, 10240)
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')

    # warmup
    for _ in range(10):
        torch.ops.ofk.add(a, b)
    torch.cuda.synchronize()
    
    torch.cuda.cudart().cudaProfilerStart()   
    my_output = torch.ops.ofk.add(a, b)
    torch.cuda.synchronize()
    torch_output = a + b
    torch.cuda.cudart().cudaProfilerStop()   
    
    torch.testing.assert_close(my_output, torch_output)
    
    print("✅ 测试通过！")

if __name__ == "__main__":
    test_my_cuda_add()