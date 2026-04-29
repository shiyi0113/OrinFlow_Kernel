#include <torch/extension.h>

void launch_add_kernel_float(const float* a, const float* b, float* out, size_t size);

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "Only Float32 is supported for now");
    auto out = torch::empty_like(a);
    launch_add_kernel_float(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), a.numel());
    return out;
}

TORCH_LIBRARY(my_lib, m) {
    m.def("add(Tensor a, Tensor b) -> Tensor", &my_add);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
}