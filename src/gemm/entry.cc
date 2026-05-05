#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, cudaStream_t stream);

void gemm_cute(void* Dptr, const void* Aptr, const void* Bptr,
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

// Accepts standard A (M,K) and B (K,N); computes D = A @ B as BF16 output.
// M and N must be multiples of 128, K must be a multiple of 32.
torch::Tensor my_gemm_cute(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "gemm_cute requires BFloat16 input A");
    TORCH_CHECK(B.scalar_type() == torch::kBFloat16, "gemm_cute requires BFloat16 input B");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A columns must match B rows");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 32 == 0,
                "M and N must be multiples of 128, K must be a multiple of 32");

    // Kernel expects B in (N, K) row-major layout
    auto B_T = B.t().contiguous();
    auto stream = at::cuda::getCurrentCUDAStream(A.get_device());
    auto D = torch::empty({M, N}, A.options());

    gemm_cute(D.data_ptr(), A.data_ptr(), B_T.data_ptr(), M, N, K, stream);
    return D;
}

TORCH_LIBRARY_FRAGMENT(ofk, m) {
    m.def("gemm(Tensor A, Tensor B) -> Tensor");
    m.def("gemm_cute(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("gemm", &my_gemm);
    m.impl("gemm_cute", &my_gemm_cute);
}
