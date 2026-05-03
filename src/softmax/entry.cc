#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void softmax(const float* d_x, float* d_out, int m, int n, cudaStream_t stream);

torch::Tensor my_softmax(torch::Tensor x) {
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Only Float32 is supported");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D [N, D]");

    auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
    auto out = torch::empty_like(x);

    softmax(x.data_ptr<float>(), out.data_ptr<float>(),
            x.size(0), x.size(1), stream);
    return out;
}

TORCH_LIBRARY_FRAGMENT(ofk, m) {
    m.def("softmax(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("softmax", &my_softmax);
}
