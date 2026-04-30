#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void launch_add_kernel_float(const float* a, const float* b, float* out, size_t size, cudaStream_t stream);

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only Float32 is supported for now");
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());
    auto out = torch::empty_like(a);
    launch_add_kernel_float(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel(), stream);
    return out;
}

TORCH_LIBRARY(ofk, m) {
    m.def("add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("add", &my_add);
}