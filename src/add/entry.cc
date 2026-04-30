#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_add_kernel_float(const float* a, const float* b, float* out, size_t size, cudaStream_t stream);

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    // 输入校验
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only Float32 is supported for now");
    // stream 管理
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    // 创建tensor
    auto out = torch::empty_like(a);
    // 调用kernel
    launch_add_kernel_float(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel(), stream);
    return out;
}
// 算子注册宏 用 torch.ops.ofk.add(a, b)
TORCH_LIBRARY(ofk, m) {
    m.def("add(Tensor a, Tensor b) -> Tensor"); // 声明 schema
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("add", &my_add); // 绑定 CUDA dispatch
}