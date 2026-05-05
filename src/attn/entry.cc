#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

void run_flash_attn_v2(const float* Q, const float* K, const float* V,
                       float* O, int B, int H, int M, int N, int d,
                       cudaStream_t stream);

// q, k, v: (B, H, seq, d) contiguous float32
torch::Tensor my_flash_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v) {
    TORCH_CHECK(q.scalar_type() == torch::kFloat32, "flash_attn requires Float32");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "q, k, v must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "q, k, v must be 4-D (B, H, seq, d)");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch size must match");
    TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "num heads must match");
    TORCH_CHECK(k.size(2) == v.size(2), "k and v seq length must match");
    TORCH_CHECK(q.size(3) == k.size(3) && q.size(3) == v.size(3), "head dim must match");

    int B = q.size(0), H = q.size(1);
    int M = q.size(2), N = k.size(2), d = q.size(3);

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device());
    auto out = torch::empty_like(q);

    run_flash_attn_v2(
        q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
        out.data_ptr<float>(), B, H, M, N, d, stream);
    return out;
}

TORCH_LIBRARY_FRAGMENT(ofk, m) {
    m.def("flash_attn(Tensor q, Tensor k, Tensor v) -> Tensor");
}

TORCH_LIBRARY_IMPL(ofk, CUDA, m) {
    m.impl("flash_attn", &my_flash_attn);
}
