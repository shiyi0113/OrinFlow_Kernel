#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void reduce(const float* d_x, float* d_y, int n, cudaStream_t stream);

torch::Tensor my_reduce(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only Float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto out = torch::zeros({1}, x.options());
    reduce(x.data_ptr<float>(), out.data_ptr<float>(), x.numel(), stream);
    return out;
}

TORCH_LIBRARY_FRAGMENT(ofk, m) {
    m.def("reduce(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("reduce", &my_reduce);
}
