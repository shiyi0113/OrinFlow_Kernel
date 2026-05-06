#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, cudaStream_t stream);

torch::Tensor my_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Only Float32 is supported");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A columns must match B rows");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());
    auto C = torch::empty({M, N}, A.options());

    gemm(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
         M, N, K, stream);
    return C;
}

TORCH_LIBRARY_FRAGMENT(ofk, m) {
    m.def("gemm(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("gemm", &my_gemm);
}
